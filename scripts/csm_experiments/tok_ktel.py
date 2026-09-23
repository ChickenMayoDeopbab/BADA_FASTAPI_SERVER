# -*- coding: utf-8 -*-
"""상담 음성(AIHub 100, 파일명 KtelSpeech) → 세션·턴 구조를 지킨 Mimi 32코드북 토큰. 집 PC(4060)에서 돈다. zip 을 풀지 않는다.

  라벨 zip  D60/J91/S00000001/S00000001.json (dataSet.typeInfo.speakers[] · dataSet.dialogs[] = 발화 순서) + 0001.txt …  ← 전부 줘도 wav zip 의 도메인 것만 연다
  wav zip   D60/J91/S00000001/0001.wav (8 kHz · 16 bit · mono)                                                      ← 갖고 있는 것만. 샤드 = wav zip 당 100세션
  세션 → dialogs 순서 → 같은 화자의 연속 발화를 한 턴으로(사이 --gap-s 무음, --max-turn-s 넘으면 새 턴)
       → 8 k→24 k(0 채우기 ×3 + Kaiser-sinc 저역 통과, 4 kHz 위는 빈 채로) → Mimi 로 **턴 하나씩** 인코딩
       → codes/<샤드>.npz(codes[32, 합] + offsets) + manifest/<샤드>.jsonl(행 = 턴). 전사는 tok_kspon.parse_text(raw/spell/pron/flags).
  화자 태그([0]/[1])는 여기서 정하지 않는다 — 행의 role(상담원/고객)·spk_id 로 로더가 정한다.
  출력 폴더는 .lock 으로 잠근다(겹쳐 돌리면 두 번째가 거부). 끝난 샤드는 건너뛴다. 입력 zip 이 바뀌면 --out 을 새로 잡는다(샤드 경계가 달라진다).

실물 확인(2026-09-23, 집 PC zip 6+1개): json 키 위와 같음 · audioPath 의 `KtelSpeech/` 접두어는 zip 안 경로에 없음 · 같은 화자 연속 파일 있음(D62 S00009253 0001~0005)
· 전사 표기는 KsponSpeech 와 같은 규칙(n/ 어/ (1권)/(한 권)) · wav 헤더 8000 Hz·16 bit·mono.

사용:
  python tok_ktel.py --label ~/aihub/012.상담_음성_데이터/01.데이터/*/라벨링데이터_1129_add/*.zip \
                     --wav ~/aihub/012.상담_음성_데이터/01.데이터/2.Validation/원천데이터_1129_add/KtelSpeech_valid_D60_wav_0.zip --out ~/tok/ktel
"""
import argparse, collections, fcntl, io, json, math, os, re, sys, time, wave, zipfile
import numpy as np, torch, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tok_kspon import parse_text                                   # 같은 폴더의 tok_kspon.py 가 필요하다

SR_IN, SR_OUT, UP = 8000, 24000, 3
HALF, CUTOFF_HZ, BETA = 224, 3825.0, 8.6                            # 449탭 @24k · 통과대역 ~3.7 kHz · 저지대역 4.0 kHz 부터 ≈ -85 dB(계산값 — test_tok_ktel.py 가 잰다)
FRAME = int(SR_OUT / 12.5)                                          # 1,920 샘플 = Mimi 한 프레임. 실제 Mimi 프레임 수 = ceil(N/1920)(2026-09-23 맥 CPU 실측 4건 일치)
PREFIX = "KtelSpeech/"


def resample_kernel(device):
    n = torch.arange(-HALF, HALF + 1, dtype=torch.float64)
    fc = CUTOFF_HZ / SR_OUT                                         # 24 kHz 격자에서의 정규화 차단 주파수
    h = 2 * fc * torch.sinc(2 * fc * n) * torch.kaiser_window(2 * HALF + 1, periodic=False, beta=BETA, dtype=torch.float64)
    return (h * UP).to(torch.float32).to(device)                    # 0 채우기로 줄어든 이득(1/3)을 되돌린다


def resample_8k_to_24k(x, kernel):                                  # x [N] float32 → [3N]
    up = torch.zeros(x.numel() * UP, device=x.device, dtype=x.dtype); up[::UP] = x
    return F.conv1d(up[None, None], kernel[None, None], padding=HALF)[0, 0]


class SessionSkip(Exception):
    def __init__(self, reason): super().__init__(reason); self.reason = reason


def strip_prefix(p):
    p = p.replace("\\", "/")
    return p[len(PREFIX):] if p.startswith(PREFIX) else p


def read_wav(b):
    with wave.open(io.BytesIO(b)) as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (1, 2, SR_IN):
            raise SessionSkip("wav 형식 아님(8000 Hz·16 bit·mono 가 아니다)")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")


def read_text(b):
    try: return b.decode("utf-8-sig").strip()
    except UnicodeDecodeError: return b.decode("cp949", "replace").strip()


def label_index(paths, domains):
    """세션 키('D60/J91/S00000001') → (라벨 zip, json ZipInfo). wav zip 의 도메인(D60…)이 이름에 든 zip 만 연다(도메인을 모르면 전부)."""
    idx, dup = {}, 0
    for p in paths:
        m = re.search(r"_(D\d+)_", os.path.basename(p))
        if domains and m and m.group(1) not in domains:
            continue
        zf = zipfile.ZipFile(p)
        for i in zf.infolist():
            if i.filename.lower().endswith(".json"):
                k = os.path.dirname(i.filename)
                if k in idx: dup += 1
                else: idx[k] = (zf, i)
    return idx, dup


def load_session(sess, labels, wz):
    """세션 하나 → (발화 목록 [(번호, 화자 id, 전사 raw, pcm int16)] 순서대로, 메타). 빠지거나 형식이 다르면 SessionSkip."""
    if sess not in labels:
        raise SessionSkip("라벨 없음")
    lz, ji = labels[sess]; d = json.loads(lz.read(ji).decode("utf-8-sig"))["dataSet"]
    info = d.get("typeInfo", {}); spk = {s["id"]: s for s in info.get("speakers", [])}
    utts = []
    for dl in d.get("dialogs", []):
        if dl["speaker"] not in spk:
            raise SessionSkip("dialogs 의 speaker 가 speakers 에 없다")
        ap_, tp = strip_prefix(dl["audioPath"]), strip_prefix(dl["textPath"])
        try: b_wav, b_txt = wz.read(ap_), lz.read(tp)
        except KeyError: raise SessionSkip("파일 빠짐(wav 또는 txt)")
        utts.append((os.path.splitext(os.path.basename(ap_))[0], dl["speaker"], read_text(b_txt), read_wav(b_wav)))
    if not utts:
        raise SessionSkip("dialogs 비어 있음")
    try: order_ok = all(int(u[0]) < int(v[0]) for u, v in zip(utts, utts[1:]))
    except ValueError: order_ok = False
    return utts, dict(category=info.get("category"), spk=spk, order_ok=order_ok)


def build_turns(utts, gap_s, max_turn_s):
    """같은 화자의 연속 발화를 한 턴으로 잇는다. 사이에 gap_s 무음, 합이 max_turn_s 를 넘으면 새 턴(혼자 넘는 발화는 그대로 한 턴)."""
    gap = np.zeros(int(round(gap_s * SR_IN)), dtype=np.int16); turns = []
    for uid, spk, raw, pcm in utts:
        t = turns[-1] if turns else None
        if t and t["spk"] == spk and t["pcm"].size + gap.size + pcm.size <= max_turn_s * SR_IN:
            t["utts"].append(uid); t["raws"].append(raw); t["utt_s"].append(round(pcm.size / SR_IN, 3)); t["pcm"] = np.concatenate([t["pcm"], gap, pcm])
        else:
            turns.append(dict(spk=spk, utts=[uid], raws=[raw], utt_s=[round(pcm.size / SR_IN, 3)], pcm=pcm))
    return turns


def make_codec(a, dev):
    if a.codec == "fake":                                           # 시험용: 길이만 맞는 결정적 가짜 코드(모델 없음). 학습에 쓰면 안 된다
        return lambda x: np.random.default_rng(x.numel()).integers(0, 2048, size=(32, math.ceil(x.numel() / FRAME)), dtype=np.int16)
    from transformers import MimiModel
    mimi = MimiModel.from_pretrained(a.mimi).to(dev).eval()
    return lambda x: mimi.encode(x[None, None]).audio_codes[0].to(torch.int16).cpu().numpy()      # [32, T] — 턴 하나씩


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description="상담 음성(KtelSpeech) → 세션·턴 구조 Mimi 토큰")
    ap.add_argument("--label", nargs="+", required=True, help="라벨 zip 들(전부 줘도 된다 — wav zip 의 도메인 것만 연다)")
    ap.add_argument("--wav", nargs="+", required=True, help="갖고 있는 wav zip 들. 샤드는 wav zip 당 --sessions-per-shard 세션씩")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mimi", default="kyutai/mimi")
    ap.add_argument("--codec", choices=("mimi", "fake"), default="mimi", help="fake = 시험용 가짜 코드(모델 없음)")
    ap.add_argument("--gap-s", type=float, default=0.3, help="같은 화자 발화를 이을 때 사이 무음(초)")
    ap.add_argument("--max-turn-s", type=float, default=20.0, help="이어 붙인 턴의 최대 길이(초)")
    ap.add_argument("--sessions-per-shard", type=int, default=100)
    ap.add_argument("--limit-shards", type=int, default=0, help="시험용: 샤드 n개만")
    a = ap.parse_args()
    dev = torch.device(a.device)
    os.makedirs(os.path.join(a.out, "codes"), exist_ok=True); os.makedirs(os.path.join(a.out, "manifest"), exist_ok=True)
    lock = open(os.path.join(a.out, ".lock"), "a+")                 # 프로세스가 사는 동안 잠긴다(죽으면 커널이 푼다)
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.seek(0); print(f"다른 프로세스가 {a.out} 를 쓰고 있다(pid {lock.read().strip() or '?'}). 겹쳐 돌리지 않는다.", file=sys.stderr); sys.exit(2)
    lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()

    domains = {m.group(1) for p in a.wav for m in [re.search(r"_(D\d+)_", os.path.basename(p))] if m}
    labels, dup = label_index(a.label, domains); print(f"라벨 세션 {len(labels):,}개 색인(도메인 {sorted(domains) or '전부'})" + (f" · 중복 {dup}" if dup else ""))
    encode = make_codec(a, dev); kernel = resample_kernel(dev)
    json.dump(dict(source="AIHub 100 상담 음성(KtelSpeech)", row="turn", mimi=a.mimi, codec=a.codec, codebooks=32, frame_hz=12.5, sr_src=SR_IN, sr_mimi=SR_OUT, dtype="int16",
                   resampler=f"zero-stuff x{UP} + kaiser-sinc {2*HALF+1}tap cutoff {CUTOFF_HZ:.0f}Hz beta {BETA}", gap_s=a.gap_s, max_turn_s=a.max_turn_s,
                   sessions_per_shard=a.sessions_per_shard, layout="codes[32, sum(frames)] + offsets[N+1]"),
              open(os.path.join(a.out, "meta.json"), "w"), ensure_ascii=False, indent=1)
    skipped, roles, done_sess, done_turn, done_s, n_shard, order_bad, n_spk_odd = collections.Counter(), collections.Counter(), 0, 0, 0.0, 0, 0, 0; t0 = time.time()
    for wpath in a.wav:
        wz = zipfile.ZipFile(wpath); stem = os.path.splitext(os.path.basename(wpath))[0]
        sessions = sorted({os.path.dirname(i.filename) for i in wz.infolist() if i.filename.lower().endswith(".wav")})
        print(f"{stem}: 세션 {len(sessions):,}개 → 샤드 {math.ceil(len(sessions) / a.sessions_per_shard)}개")
        for si in range(0, len(sessions), a.sessions_per_shard):
            shard = f"{stem}_{si // a.sessions_per_shard + 1:04d}"
            npz, man = os.path.join(a.out, "codes", shard + ".npz"), os.path.join(a.out, "manifest", shard + ".jsonl")
            if os.path.exists(npz) and os.path.exists(man):
                continue
            codes, offsets, rows = [], [0], []; sk = collections.Counter()
            for sess in sessions[si:si + a.sessions_per_shard]:
                try: utts, meta = load_session(sess, labels, wz)
                except SessionSkip as e:
                    sk[e.reason] += 1; continue
                turns = build_turns(utts, a.gap_s, a.max_turn_s)
                order_bad += not meta["order_ok"]; n_spk_odd += len(meta["spk"]) != 2
                for ti, t in enumerate(turns):
                    x = resample_8k_to_24k(torch.from_numpy(t["pcm"].astype(np.float32) / 32768.0).to(dev), kernel)
                    c = encode(x); s = meta["spk"][t["spk"]]
                    raw = " ".join(r for r in t["raws"] if r); spell, pron, flags = parse_text(raw)
                    rows.append(dict(id=f"{sess}/t{ti:03d}", shard=shard, idx=len(rows), session=sess, domain=sess.split("/")[0], category=meta["category"],
                                     turn=ti, n_turns=len(turns), role=s.get("type"), spk_id=t["spk"], gender=s.get("gender"), age=s.get("age"),
                                     utts=t["utts"], n_utt=len(t["utts"]), utt_s=t["utt_s"], frames=int(c.shape[1]), dur_s=round(t["pcm"].size / SR_IN, 3),
                                     raw=raw, spell=spell, pron=pron, flags=flags, has_text=bool(raw), order_ok=meta["order_ok"]))
                    codes.append(c); offsets.append(offsets[-1] + c.shape[1]); done_s += t["pcm"].size / SR_IN; roles[s.get("type")] += 1
                done_sess += 1; done_turn += len(turns)
            skipped.update(sk)
            if not rows:
                print(f"[{shard}] 쓸 세션 없음" + (f" · 건너뜀 {dict(sk)}" if sk else ""), flush=True); continue
            tmp = npz + ".tmp.npz"
            np.savez(tmp, codes=np.concatenate(codes, 1), offsets=np.asarray(offsets, dtype=np.int64), ids=np.asarray([r["id"] for r in rows]))
            with open(man + ".tmp", "w", encoding="utf-8") as f:
                for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(man + ".tmp", man); os.replace(tmp, npz)      # npz 가 마지막 — 둘 다 있어야 끝난 샤드
            n_shard += 1; el = time.time() - t0
            print(f"[{shard}] {len(rows):>5}턴 · 누적 세션 {done_sess:,} · 턴 {done_turn:,} · {done_s/3600:.2f} h · 실시간의 {done_s/el:.0f}배 · {el/60:.1f}분"
                  + (f" · 건너뜀 {dict(sk)}" if sk else ""), flush=True)
            if a.limit_shards and n_shard >= a.limit_shards:
                break
        else:
            continue
        break
    print(f"끝. 세션 {done_sess:,} · 턴 {done_turn:,} · {done_s/3600:.2f} h · {(time.time()-t0)/60:.1f}분 → {a.out}")
    print(f"  역할별 턴 {dict(roles)} · 번호 순서 어긋난 세션 {order_bad} · 화자가 2명이 아닌 세션 {n_spk_odd}" + (f" · 건너뜀 {dict(skipped)}" if skipped else ""))


if __name__ == "__main__":
    main()
