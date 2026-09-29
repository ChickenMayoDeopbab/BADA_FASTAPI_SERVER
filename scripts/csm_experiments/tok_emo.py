# -*- coding: utf-8 -*-
"""감정이 태깅된 자유대화(AIHub 71631 성인 / 71632 청소년) → 세션·턴 구조를 지킨 Mimi 32코드북 토큰. 집 PC(4060)에서 돈다. zip 을 풀지 않는다.

  라벨 zip  <이름>.json — Wav(헤더 표기, 믿지 않음) · File · Noise(화자별 SNR·잡음) · Speaker1/2(ID·Age·Gender·RecDevice·Mask·Condition) · Conversation[](TextNo·SpeakerNo·StartTime·EndTime·Text·감정)
  wav zip   /<이름>.wav — 실측 16 kHz · 16 bit · **stereo**(ch0 = Speaker1, ch1 = Speaker2; 상대 음성이 2~24 dB 작게 새어 들어 있다 — 2026-09-29 검증 실외 3파일 실측). 라벨 헤더의 48000 은 틀렸다.
  세션(= 파일 하나) → Conversation 을 StartTime 순으로 → 같은 화자의 연속 발화를 한 턴으로(사이 --merge-gap-s 이하 · 합 --max-turn-s 이하; 자연 간격은 그대로 둔다)
       → 턴 오디오 = **그 화자 채널**의 [첫 발화 시작 − pad, 마지막 발화 끝 + pad]
       → 상대 발화와 겹친 비율(overlap_ratio) 이 --overlap-max 를 넘으면 flags.overlap(문맥엔 쓰고 목표 턴에선 뺀다 — csm_data_b)
       → 전사의 #@이름# 같은 마스킹 태그는 지우고 flags.masked(목표 턴에서 뺀다)
       → (--level-target) 턴 말소리 RMS 를 목표 dBFS 로(level.analyze, 16 kHz 에서 재고 리샘플 전에 이득)
       → 16 k→24 k(tok_kspon 의 Kaiser-sinc) → Mimi 로 턴 하나씩 → codes/<샤드>.npz + manifest/<샤드>.jsonl (tok_ktel 과 같은 행 형식)
  role: Speaker1 → "상담원", Speaker2 → "고객" 으로 적는다(로더의 태그 규칙에 맞추기 위한 자리 표시 — 둘 다 또래 대화라 의미 없음). 학습은 --tag-by random 으로.

사용:
  python tok_emo.py --label <TL_01.실내.zip> <TL_02.실외.zip> --wav <TS_01.실내_5.zip> --out ~/tok/emo71631 --level-target -26
"""
import argparse, collections, fcntl, io, json, math, os, re, sys, time, wave, zipfile
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tok_kspon import resample_kernel, resample_16k_to_24k         # 16 k→24 k(Kspon 과 같은 필터)
from tok_ktel import make_codec, FRAME                              # Mimi / fake 코덱
from level import analyze, apply_gain

SR = 16000
MASK = re.compile(r"#@[^#]*#")                                      # '#@이름#' 같은 마스킹 태그(검증 실외 186파일 중 552발화)
CHANNEL = {"Speaker1": 0, "Speaker2": 1}
ROLE = {"Speaker1": "상담원", "Speaker2": "고객"}


class SessionSkip(Exception):
    def __init__(self, reason): super().__init__(reason); self.reason = reason


def fl(s):
    """라벨의 시간 문자열 → float. '1,000.73' 처럼 천 단위 쉼표가 있다(실측)."""
    return float(str(s).replace(",", "").strip())


def read_stereo_wav(b):
    with wave.open(io.BytesIO(b)) as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (2, 2, SR):
            raise SessionSkip(f"wav 형식 아님({w.getframerate()} Hz·{w.getsampwidth()*8} bit·{w.getnchannels()} ch)")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, 2)


def label_index(label_zips):
    """<이름> → (ZipFile, ZipInfo). 라벨 zip 전부 색인(작다: 100 MB 안팎)."""
    idx, dup = {}, 0
    for p in label_zips:
        z = zipfile.ZipFile(p)
        for i in z.infolist():
            if i.filename.lower().endswith(".json"):
                k = os.path.splitext(os.path.basename(i.filename))[0]; dup += k in idx; idx[k] = (z, i)
    return idx, dup


def clean_text(t):
    """마스킹 태그 제거·공백 정리 → (text, masked)"""
    masked = bool(MASK.search(t)); t = MASK.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip(), masked


def utterances(d):
    """Conversation → StartTime 순 [(spk, s, e, text, emotion)]"""
    us = []
    for u in d["Conversation"]:
        s, e = fl(u["StartTime"]), fl(u["EndTime"])
        if u["SpeakerNo"] not in CHANNEL or e <= s: continue
        us.append((u["SpeakerNo"], s, e, u.get("Text", "") or "", u.get("VerifyEmotionTarget") or u.get("SpeakerEmotionTarget") or ""))
    us.sort(key=lambda x: (x[1], x[2]))
    return us


def build_turns(us, merge_gap_s, max_turn_s):
    """같은 화자의 연속 발화를 한 턴으로. 사이 간격 ≤ merge_gap_s 이고 합이 ≤ max_turn_s 일 때만 잇는다(자연 간격 유지)."""
    turns = []
    for spk, s, e, text, emo in us:
        t = turns[-1] if turns else None
        if t and t["spk"] == spk and s - t["end"] <= merge_gap_s and e - t["start"] <= max_turn_s:
            t["end"] = max(t["end"], e); t["texts"].append(text); t["emos"].append(emo); t["n_utt"] += 1
        else:
            turns.append(dict(spk=spk, start=s, end=e, texts=[text], emos=[emo], n_utt=1))
    return turns


def overlap_ratio(turn, us):
    """턴 구간 중 상대 화자 발화와 겹친 시간의 비율"""
    ov = 0.0
    for spk, s, e, _, _ in us:
        if spk == turn["spk"]: continue
        ov += max(0.0, min(e, turn["end"]) - max(s, turn["start"]))
    return ov / max(turn["end"] - turn["start"], 1e-6)


def channel_map(x, us):
    """파일마다 화자별로 그 화자 소리가 큰 채널을 잰다(상대와 겹치지 않는 발화만). → ({spk: ch}, {spk: 분리 dB})
    실측(2026-09-29 실내 3파일): 고정 매핑(ch0=Speaker1)이 틀리는 파일이 있다 — 한 파일은 두 채널이 거의 같은 소리(+0.3 dB), 한 파일은 반대 채널이 더 큼(−3.1 dB)."""
    e = {s: np.zeros(2) for s in CHANNEL}; n = {s: 0 for s in CHANNEL}
    iv = [(s, t, spk) for spk, s, t, _, _ in us]
    for spk, s, t, _, _ in us:
        if any(o != spk and os_ < t and oe > s for os_, oe, o in iv): continue
        a, b = int(s * SR), int(t * SR)
        if b <= a or b > len(x): continue
        e[spk] += (x[a:b].astype(np.float32) ** 2).mean(0); n[spk] += 1
    chan, sep = {}, {}
    for spk in CHANNEL:
        if n[spk] == 0 or e[spk].sum() == 0: chan[spk], sep[spk] = CHANNEL[spk], 0.0; continue
        c = int(np.argmax(e[spk])); chan[spk] = c; sep[spk] = float(10 * np.log10((e[spk][c] + 1e-12) / (e[spk][1 - c] + 1e-12)))
    return chan, sep


def cut(x, turn, pad_s, chan=None):
    """그 화자 채널의 [start − pad, end + pad] → float32 [N]. chan 이 없으면 고정 매핑."""
    a = max(0, int(round((turn["start"] - pad_s) * SR))); b = min(len(x), int(round((turn["end"] + pad_s) * SR)))
    return x[a:b, (chan or CHANNEL)[turn["spk"]]].astype(np.float32) / 32768.0


def load_session(name, labels, wz, winfo):
    if name not in labels: raise SessionSkip("라벨 없음")
    lz, ji = labels[name]; d = json.loads(lz.read(ji).decode("utf-8-sig"))
    x = read_stereo_wav(wz.read(winfo))
    us = utterances(d)
    if len({u[0] for u in us}) < 2: raise SessionSkip("한쪽 화자만 있음")
    return d, x, us


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", nargs="+", required=True, help="라벨 zip 들(TL_*/VL_* 전부 줘도 된다)")
    ap.add_argument("--wav", nargs="+", required=True, help="원천 zip 들(TS_01.실내_5.zip …). 샤드 = wav zip 당 --sessions-per-shard 파일씩")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mimi", default="kyutai/mimi"); ap.add_argument("--codec", choices=("mimi", "fake"), default="mimi")
    ap.add_argument("--merge-gap-s", type=float, default=1.0, help="같은 화자 발화를 한 턴으로 이을 최대 간격(초)")
    ap.add_argument("--max-turn-s", type=float, default=20.0); ap.add_argument("--pad-s", type=float, default=0.05)
    ap.add_argument("--overlap-max", type=float, default=0.3, help="상대 발화와 겹친 비율이 이보다 크면 flags.overlap(목표 턴 제외)")
    ap.add_argument("--sessions-per-shard", type=int, default=20, help="파일 하나 ≈ 18분(중앙값)이라 20파일 ≈ 6 h")
    ap.add_argument("--level-target", type=float, default=None); ap.add_argument("--level-peak-cap", type=float, default=-2.0); ap.add_argument("--level-gain-max", type=float, default=30.0)
    ap.add_argument("--min-turn-s", type=float, default=0.4, help="이보다 짧은 턴은 버린다(추임새)")
    ap.add_argument("--mixed-db", type=float, default=1.0, help="화자 채널 분리가 이보다 작으면(두 채널에 거의 같은 소리) flags.mixed(목표 턴 제외)")
    ap.add_argument("--limit-shards", type=int, default=0)
    a = ap.parse_args(); dev = torch.device(a.device)
    os.makedirs(os.path.join(a.out, "codes"), exist_ok=True); os.makedirs(os.path.join(a.out, "manifest"), exist_ok=True)
    lock = open(os.path.join(a.out, ".lock"), "a+")
    try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.seek(0); print(f"다른 프로세스가 {a.out} 를 쓰고 있다(pid {lock.read().strip() or '?'}).", file=sys.stderr); sys.exit(2)
    lock.seek(0); lock.truncate(); lock.write(str(os.getpid())); lock.flush()
    labels, dup = label_index(a.label); print(f"라벨 {len(labels):,}개 색인" + (f" · 중복 {dup}" if dup else ""))
    encode = make_codec(a, dev); kernel = resample_kernel(dev)
    json.dump(dict(source="AIHub 71631/71632 감정이 태깅된 자유대화", row="turn", mimi=a.mimi, codec=a.codec, codebooks=32, frame_hz=12.5, sr_src=SR, sr_mimi=24000, dtype="int16",
                   channel="per-file: speaker -> channel with more energy in non-overlapped utterances (fallback ch0=Speaker1 ch1=Speaker2)", mixed_db=a.mixed_db, merge_gap_s=a.merge_gap_s, max_turn_s=a.max_turn_s, pad_s=a.pad_s, overlap_max=a.overlap_max, min_turn_s=a.min_turn_s,
                   sessions_per_shard=a.sessions_per_shard, layout="codes[32, sum(frames)] + offsets[N+1]", role_map=ROLE,
                   level=dict(fn="level.analyze: 20 ms frames, gate max(p90-20, p10+10)", target=a.level_target, peak_cap=a.level_peak_cap, gain_max=a.level_gain_max)),
              open(os.path.join(a.out, "meta.json"), "w"), ensure_ascii=False, indent=1)
    skipped, done_sess, done_turn, done_s, n_shard = collections.Counter(), 0, 0, 0.0, 0; n_ov = n_mask = n_capped = n_weak = n_short = n_oob = n_mixed_sess = n_swapped = 0; t0 = time.time()
    for wpath in a.wav:
        wz = zipfile.ZipFile(wpath); stem = os.path.splitext(os.path.basename(wpath))[0]
        wavs = sorted((i for i in wz.infolist() if i.filename.lower().endswith(".wav")), key=lambda i: i.filename)
        print(f"{stem}: 파일 {len(wavs):,}개 → 샤드 {math.ceil(len(wavs) / a.sessions_per_shard)}개")
        for si in range(0, len(wavs), a.sessions_per_shard):
            shard = f"{stem}_{si // a.sessions_per_shard + 1:04d}"
            npz, man = os.path.join(a.out, "codes", shard + ".npz"), os.path.join(a.out, "manifest", shard + ".jsonl")
            if os.path.exists(npz) and os.path.exists(man): continue
            codes, offsets, rows = [], [0], []; sk = collections.Counter()
            for winfo in wavs[si:si + a.sessions_per_shard]:
                name = os.path.splitext(os.path.basename(winfo.filename))[0]
                try: d, x, us = load_session(name, labels, wz, winfo)
                except SessionSkip as e: sk[e.reason] += 1; continue
                turns = build_turns(us, a.merge_gap_s, a.max_turn_s); info = d.get("ConversationInfo", {}); noise = d.get("Noise", {})
                chan, sep = channel_map(x, us); n_mixed_sess += any(v < a.mixed_db for v in sep.values()); n_swapped += any(chan[s] != CHANNEL[s] for s in CHANNEL)
                ti_out = 0
                for turn in turns:
                    if turn["end"] - turn["start"] < a.min_turn_s: n_short += 1; continue
                    x16 = cut(x, turn, a.pad_s, chan)
                    if len(x16) < a.min_turn_s * SR: n_oob += 1; continue                       # 라벨 시간이 wav 길이 밖(실측: 검증 실외에서 발생) → 빈/잘린 오디오는 버린다
                    lv = analyze(x16, SR, a.level_target if a.level_target is not None else 0.0, a.level_peak_cap, a.level_gain_max)
                    if a.level_target is None: lv["gain_db"], lv["capped"] = 0.0, False
                    else: x16 = apply_gain(x16, lv["gain_db"])
                    n_capped += lv["capped"]; n_weak += lv["weak"]
                    c = encode(resample_16k_to_24k(torch.from_numpy(np.ascontiguousarray(x16)).to(dev), kernel))
                    raw = " ".join(t for t in turn["texts"] if t); text, masked = clean_text(raw)
                    ov = overlap_ratio(turn, us); flags = dict(overlap=int(ov > a.overlap_max), masked=int(masked), mixed=int(sep[turn["spk"]] < a.mixed_db)); n_ov += flags["overlap"]; n_mask += masked
                    sp = d[turn["spk"]]; k = turn["spk"][-1]
                    rows.append(dict(id=f"{name}/t{ti_out:03d}", shard=shard, idx=len(rows), session=name, domain=info.get("Domain"), category=info.get("Step1Subject"),
                                     turn=ti_out, n_turns=len(turns), role=ROLE[turn["spk"]], spk_no=turn["spk"], spk_id=sp.get("ID"), gender=sp.get("Gender"), age=sp.get("Age"),
                                     device=sp.get("RecDevice"), mask=sp.get("Mask"), condition=sp.get("Condition"), snr_label=noise.get(f"Speaker{k}SNR"), noise_cat=noise.get(f"Speaker{k}NoiseCategory"),
                                     start_s=round(turn["start"], 3), end_s=round(turn["end"], 3), n_utt=turn["n_utt"], emotion=collections.Counter(e for e in turn["emos"] if e).most_common(1)[0][0] if any(turn["emos"]) else None,
                                     overlap_ratio=round(ov, 3), chan=chan[turn["spk"]], sep_db=round(sep[turn["spk"]], 1), frames=int(c.shape[1]), dur_s=round(len(x16) / SR, 3),
                                     raw=raw, spell=text, pron=text, flags=flags, has_text=bool(text), order_ok=True, **lv))
                    codes.append(c); offsets.append(offsets[-1] + c.shape[1]); done_s += len(x16) / SR; ti_out += 1
                done_sess += 1; done_turn += ti_out
            skipped.update(sk)
            if not rows: print(f"[{shard}] 쓸 파일 없음" + (f" · 건너뜀 {dict(sk)}" if sk else ""), flush=True); continue
            tmp = npz + ".tmp.npz"
            np.savez(tmp, codes=np.concatenate(codes, 1), offsets=np.asarray(offsets, dtype=np.int64), ids=np.asarray([r["id"] for r in rows]))
            with open(man + ".tmp", "w", encoding="utf-8") as f:
                for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(man + ".tmp", man); os.replace(tmp, npz); n_shard += 1; el = time.time() - t0
            print(f"[{shard}] {len(rows):>5}턴 · 누적 파일 {done_sess:,} · 턴 {done_turn:,} · {done_s/3600:.2f} h · 실시간의 {done_s/max(el,1e-9):.0f}배 · {el/60:.1f}분 · 겹침 {n_ov} · 마스킹 {n_mask} · 짧아서 뺌 {n_short} · 오디오 밖 {n_oob} · 채널 뒤바뀜 파일 {n_swapped} · 분리 안 됨 파일 {n_mixed_sess}"
                  + (f" · 건너뜀 {dict(sk)}" if sk else ""), flush=True)
            if a.limit_shards and n_shard >= a.limit_shards: print("limit-shards 도달"); return
    print(f"끝. 파일 {done_sess:,} · 턴 {done_turn:,} · {done_s/3600:.2f} h · {(time.time()-t0)/60:.1f}분 → {a.out}")
    print(f"  겹침 플래그 {n_ov} · 마스킹 {n_mask} · 짧아서 뺌 {n_short} · 오디오 밖 {n_oob} · 채널 뒤바뀜 파일 {n_swapped} · 분리 안 됨 파일 {n_mixed_sess} · 피크 상한 {n_capped} · weak {n_weak} · 건너뜀 {dict(skipped)}")


if __name__ == "__main__":
    main()
