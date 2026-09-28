# -*- coding: utf-8 -*-
"""KsponSpeech → Mimi 32코드북 토큰. 집 PC(4060)에서 돈다. zip 을 풀지 않고 읽는다.

  16 kHz 16-bit raw `.pcm` → 24 kHz 리샘플(고정 windowed-sinc, 의존성 없음) → `kyutai/mimi` 로 **발화 하나씩** 인코딩
  → 폴더(1,000발화) 단위 샤드 `codes/KsponSpeech_0001.npz` + `manifest/KsponSpeech_0001.jsonl`

확인해 둔 것(스터디 레포, 2026-09-21):
  · `kyutai/mimi` 는 `sesame/csm-1b` 안의 codec_model 과 **비트 단위로 같다**(350텐서, 차이 0.0) → 게이트 모델을 받을 필요 없음
  · 발화를 하나씩 인코딩한 코드가 HF CSM 학습 경로(`_merge_input_ids_with_input_values`)가 내부에서 만드는 코드와 같다(README 의 검증 참조)
전사는 **버리지 않고 세 가지로 남긴다**: raw(원문) · spell(이중전사의 철자 쪽) · pron(발음 쪽). 어느 쪽으로 학습할지는 나중에 정한다(노트 09 §3).
  b/ → [숨] · l/ → [웃음] · o/ n/ u/ → 지우고 플래그 · 간투어 `어/` → `어` · `+` `*` → 지우고 플래그

사용:
  python tok_kspon.py --audio ~/aihub/10.한국어음성/*.zip --trn ~/aihub/10.한국어음성/KsponSpeech_scripts.zip --out ~/tok/kspon
  (--audio 는 zip 여러 개 또는 푼 폴더. 중단했다 다시 돌리면 끝난 샤드는 건너뛴다.)
"""
import argparse, glob, io, json, math, os, re, sys, time, zipfile
import numpy as np, torch, torch.nn.functional as F
from transformers import MimiModel

SR_IN, SR_OUT, UP, DOWN = 16000, 24000, 3, 2
HALF, CUTOFF_HZ, BETA = 192, 7650.0, 8.6            # 385탭 @48k · 통과대역 ~7.3 kHz · 저지대역 8.0 kHz 부터 ≈ -85 dB


def resample_kernel(device):
    n = torch.arange(-HALF, HALF + 1, dtype=torch.float64)
    fc = CUTOFF_HZ / (SR_IN * UP)                   # 48 kHz 격자에서의 정규화 차단 주파수
    h = 2 * fc * torch.sinc(2 * fc * n) * torch.kaiser_window(2 * HALF + 1, periodic=False, beta=BETA, dtype=torch.float64)
    return (h * UP).to(torch.float32).to(device)    # 0 채우기로 줄어든 이득(1/3)을 되돌린다


def resample_16k_to_24k(x, kernel):                 # x [N] float32 → [ceil(3N/2)]
    up = torch.zeros(x.numel() * UP, device=x.device, dtype=x.dtype); up[::UP] = x
    return F.conv1d(up[None, None], kernel[None, None], padding=HALF)[0, 0, ::DOWN]


DUAL = re.compile(r"\(([^()]*)\)/\(([^()]*)\)")
TAG = re.compile(r"^([blonu])/([.,?!]*)$")
FILL = re.compile(r"^(.+?)/([.,?!]*)$")


def parse_text(raw):
    """raw → (spell, pron, flags). 태그·간투어·반복·불명확 표기를 같은 규칙으로 양쪽에 적용한다."""
    flags = dict(breath=0, laugh=0, overlap=0, noise=0, unknown=0, filler=0, repeat=raw.count("+"), unclear=raw.count("*"), dual=len(DUAL.findall(raw)))
    out = []
    for side in (1, 2):
        s = DUAL.sub(lambda m: m.group(side), raw).replace("+", "").replace("*", "")
        toks = []
        for t in s.split():
            m = TAG.match(t)
            if m:
                k = m.group(1)
                if side == 1:
                    flags[{"b": "breath", "l": "laugh", "o": "overlap", "n": "noise", "u": "unknown"}[k]] += 1
                if k in "bl":
                    toks.append({"b": "[숨]", "l": "[웃음]"}[k] + m.group(2))
                continue
            m = FILL.match(t)
            if m:
                if side == 1: flags["filler"] += 1
                t = m.group(1) + m.group(2)
            toks.append(t)
        out.append(" ".join(toks))
    return out[0], out[1], flags


def load_trn(path):
    """`경로 :: 전사` 를 파일 이름(확장자 없는) → 전사 로. zip 이든 폴더든 .trn 을 전부 읽는다."""
    texts = {}
    def feed(raw):
        try: s = raw.decode("utf-8")
        except UnicodeDecodeError: s = raw.decode("cp949", "replace")
        for line in s.splitlines():
            if " :: " in line:
                p, t = line.split(" :: ", 1); texts[os.path.splitext(os.path.basename(p.strip()))[0]] = t.strip()
    if os.path.isdir(path):
        for f in glob.glob(os.path.join(path, "**", "*.trn"), recursive=True): feed(open(f, "rb").read())
    else:
        with zipfile.ZipFile(path) as z:
            for i in z.infolist():
                if i.filename.endswith(".trn"): feed(z.read(i))
    return texts


def iter_groups(src):
    """(샤드 이름, [(발화 id, bytes 를 돌려주는 함수)]) 를 폴더 단위로 낸다."""
    if os.path.isdir(src):
        groups = {}
        for f in sorted(glob.glob(os.path.join(src, "**", "*.pcm"), recursive=True)):
            groups.setdefault(os.path.basename(os.path.dirname(f)), []).append((os.path.splitext(os.path.basename(f))[0], (lambda f=f: open(f, "rb").read())))
        yield from groups.items()
    else:
        z = zipfile.ZipFile(src); groups = {}
        for i in z.infolist():
            if i.filename.lower().endswith(".pcm"):
                parts = i.filename.split("/")
                groups.setdefault(parts[-2] if len(parts) > 1 else "root", []).append((os.path.splitext(parts[-1])[0], (lambda i=i: z.read(i))))
        for k in sorted(groups): yield k, sorted(groups[k], key=lambda t: t[0])


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description="KsponSpeech → Mimi 토큰")
    ap.add_argument("--audio", nargs="+", required=True, help="오디오 zip 들 또는 푼 폴더")
    ap.add_argument("--trn", required=True, help="KsponSpeech_scripts.zip 또는 .trn 이 든 폴더")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mimi", default="kyutai/mimi")
    ap.add_argument("--limit-shards", type=int, default=0, help="시험용: 샤드 n개만")
    a = ap.parse_args()
    dev = torch.device(a.device)
    os.makedirs(os.path.join(a.out, "codes"), exist_ok=True); os.makedirs(os.path.join(a.out, "manifest"), exist_ok=True)
    texts = load_trn(a.trn); print(f"전사 {len(texts):,}개 읽음")
    mimi = MimiModel.from_pretrained(a.mimi).to(dev).eval(); kernel = resample_kernel(dev)
    json.dump(dict(mimi=a.mimi, codebooks=32, frame_hz=12.5, sr_src=SR_IN, sr_mimi=SR_OUT, dtype="int16",
                   resampler=f"kaiser-sinc {2*HALF+1}tap cutoff {CUTOFF_HZ:.0f}Hz beta {BETA}", layout="codes[32, sum(frames)] + offsets[N+1]"),
              open(os.path.join(a.out, "meta.json"), "w"), ensure_ascii=False, indent=1)
    done_s = done_n = n_shard = 0; t0 = time.time()
    for src in a.audio:
        for shard, items in iter_groups(src):
            npz, man = os.path.join(a.out, "codes", shard + ".npz"), os.path.join(a.out, "manifest", shard + ".jsonl")
            if os.path.exists(npz) and os.path.exists(man):
                continue
            codes, offsets, rows = [], [0], []; n_odd = 0
            for uid, read in items:
                b = read(); odd = len(b) % 2
                if odd:                                                 # KsponSpeech_eval 의 .pcm 6,000개는 전부 2N+1 바이트(2026-09-22 확인: 끝 바이트를 버려야 파형이 매끄럽다). 아니면 frombuffer 가 죽는다
                    n_odd += 1; b = b[:-1]
                pcm = np.frombuffer(b, dtype="<i2")
                if pcm.size < SR_IN // 10:                               # 0.1초 미만은 버린다
                    continue
                x = resample_16k_to_24k(torch.from_numpy(pcm.astype(np.float32) / 32768.0).to(dev), kernel)
                c = mimi.encode(x[None, None]).audio_codes[0].to(torch.int16).cpu().numpy()      # [32, T] — 발화 하나씩
                raw = texts.get(uid, ""); spell, pron, flags = parse_text(raw)
                codes.append(c); offsets.append(offsets[-1] + c.shape[1])
                rows.append(dict(id=uid, shard=shard, idx=len(rows), frames=int(c.shape[1]), dur_s=round(pcm.size / SR_IN, 3),
                                 raw=raw, spell=spell, pron=pron, flags=flags, has_text=bool(raw), odd_byte=bool(odd)))
                done_s += pcm.size / SR_IN; done_n += 1
            if not rows:
                continue
            tmp = npz + ".tmp.npz"
            np.savez(tmp, codes=np.concatenate(codes, 1), offsets=np.asarray(offsets, dtype=np.int64), ids=np.asarray([r["id"] for r in rows]))
            with open(man + ".tmp", "w", encoding="utf-8") as f:
                for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(man + ".tmp", man); os.replace(tmp, npz)          # npz 가 마지막 — 둘 다 있어야 끝난 샤드
            n_shard += 1; el = time.time() - t0
            print(f"[{shard}] {len(rows):>5}발화 · 누적 {done_n:,}발화 {done_s/3600:.2f} h · 실시간의 {done_s/el:.0f}배 · {el/60:.1f}분" + (f" · 홀수 바이트 {n_odd}개(끝 1 B 버림)" if n_odd else ""), flush=True)
            if a.limit_shards and n_shard >= a.limit_shards:
                return
    print(f"끝. {done_n:,}발화 · {done_s/3600:.2f} h · {(time.time()-t0)/60:.1f}분 → {a.out}")


if __name__ == "__main__":
    main()
