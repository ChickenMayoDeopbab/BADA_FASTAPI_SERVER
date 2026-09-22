# -*- coding: utf-8 -*-
"""G1 시험셋 만들기 — seed-tts-eval 형식(meta.lst)으로. 원본 zip 과 토큰이 있는 **집 PC**에서 돈다.

  학습에서 빠져 있는 `eval_clean`(3,000발화)에서, 숫자·영문·태그·간투어·겹침·잡음이 없는 3~10초 발화만 남기고
  고정 seed 로 **목표 n개 + 프롬프트 n개**를 짝짓는다(KsponSpeech 에는 화자 표시가 없어 프롬프트와 정답은 다른 사람이다).
출력(<out>/):
  meta.lst            `id|프롬프트 글|prompt/<id>.wav|합성할 글|human/<id>.wav`   ← seed-tts-eval 의 5칸
  set.jsonl           같은 내용 + 원본 발화 id·길이
  prompt/<id>.wav     프롬프트 원본(16 kHz)            human/<id>.wav        목표 문장을 사람이 말한 원본(16 kHz) = 대조군 (a)
  mimi/<id>.wav       목표 원본의 Mimi 재합성(24 kHz) = 대조군 (b)   mimi_prompt/<id>.wav  프롬프트의 Mimi 재합성 = SIM 천장
  prompt_codes.npz    프롬프트의 Mimi 코드 {id: int16 [32, T]} — 학교 서버의 g1_generate.py 가 문맥으로 쓴다

  python g1_pick.py --tok ~/tok/kspon --zip ~/aihub/10.한국어음성/KsponSpeech_eval.zip --out ~/g1/set
"""
import argparse, json, os, random, re, sys, wave, zipfile
import numpy as np

BAD_FLAGS = ("breath", "laugh", "overlap", "noise", "unknown", "filler", "repeat", "unclear", "dual")


def usable(r, lo=3.0, hi=10.0, min_chars=8):
    t = r["spell"].strip()
    return bool(r["has_text"] and lo <= r["dur_s"] <= hi and not any(r["flags"].get(k, 0) for k in BAD_FLAGS)
                and not re.search(r"[0-9A-Za-z\[\]|]", t) and sum("가" <= c <= "힣" for c in t) >= min_chars)


def pair(cand, n, seed):
    if len(cand) < 2 * n:
        sys.exit(f"후보가 모자란다: {len(cand)}개 < {2 * n}개 — 조건을 풀거나 n 을 줄여라")
    c = sorted(cand, key=lambda r: r["id"]); random.Random(seed).shuffle(c)
    return [dict(id=f"g1_{i:03d}", target=c[i], prompt=c[n + i]) for i in range(n)]


def meta_line(it):
    return f"{it['id']}|{it['prompt']['spell'].strip()}|prompt/{it['id']}.wav|{it['target']['spell'].strip()}|human/{it['id']}.wav"


def write_wav(path, x, sr):                                       # x: int16 또는 [-1,1] float
    if x.dtype != np.int16:
        x = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(x.astype("<i2").tobytes())


def main():
    import torch
    from transformers import MimiModel
    ap = argparse.ArgumentParser(description="G1 시험셋(seed-tts-eval 형식)")
    ap.add_argument("--tok", required=True, help="tok_kspon.py 의 출력 폴더"); ap.add_argument("--zip", required=True, help="KsponSpeech_eval.zip")
    ap.add_argument("--out", required=True); ap.add_argument("--shard", default="eval_clean")
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--mimi", default="kyutai/mimi")
    a = ap.parse_args()
    tok, out = os.path.expanduser(a.tok), os.path.expanduser(a.out)
    rows = [json.loads(l) for l in open(os.path.join(tok, "manifest", a.shard + ".jsonl"), encoding="utf-8")]
    cand = [r for r in rows if usable(r)]
    print(f"{a.shard}: {len(rows):,}발화 → 조건 통과 {len(cand):,}개 → 목표 {a.n} + 프롬프트 {a.n}")
    items = pair(cand, a.n, a.seed)
    for d in ("prompt", "human", "mimi", "mimi_prompt"):
        os.makedirs(os.path.join(out, d), exist_ok=True)

    z = np.load(os.path.join(tok, "codes", a.shard + ".npz")); codes, off = z["codes"], z["offsets"]
    zf = zipfile.ZipFile(os.path.expanduser(a.zip)); member = {os.path.splitext(os.path.basename(i.filename))[0]: i for i in zf.infolist() if i.filename.lower().endswith(".pcm")}
    mimi = MimiModel.from_pretrained(a.mimi).to(a.device).eval(); pc = {}
    with torch.inference_mode():
        for it in items:
            for role, d_orig, d_mimi in (("prompt", "prompt", "mimi_prompt"), ("target", "human", "mimi")):
                r = it[role]; b = zf.read(member[r["id"]]); pcm = np.frombuffer(b[: len(b) - len(b) % 2], dtype="<i2")   # eval .pcm 은 전부 2N+1 바이트 → 끝 1 B 버림(tok_kspon.py 와 같은 처리)
                assert abs(pcm.size - r["dur_s"] * 16000) <= 8, (r["id"], pcm.size, r["dur_s"])                       # 토큰화 때 읽은 길이와 같아야 한다(dur_s 는 소수 3자리 → ±8샘플)
                write_wav(os.path.join(out, d_orig, it["id"] + ".wav"), pcm, 16000)
                c = codes[:, off[r["idx"]]:off[r["idx"] + 1]]
                assert c.shape[1] == r["frames"], (r["id"], c.shape, r["frames"])
                y = mimi.decode(torch.from_numpy(c.astype(np.int64))[None].to(a.device)).audio_values[0, 0].float().cpu().numpy()
                write_wav(os.path.join(out, d_mimi, it["id"] + ".wav"), y[: int(round(r["dur_s"] * 24000))], 24000)
                if role == "prompt":
                    pc[it["id"]] = c.astype(np.int16)
    np.savez(os.path.join(out, "prompt_codes.npz"), **pc)
    with open(os.path.join(out, "meta.lst"), "w", encoding="utf-8") as f:
        f.write("\n".join(meta_line(it) for it in items) + "\n")
    with open(os.path.join(out, "set.jsonl"), "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(dict(id=it["id"], infer_text=it["target"]["spell"].strip(), prompt_text=it["prompt"]["spell"].strip(), target_uid=it["target"]["id"], prompt_uid=it["prompt"]["id"],
                                    target_s=it["target"]["dur_s"], prompt_s=it["prompt"]["dur_s"], prompt_frames=it["prompt"]["frames"]), ensure_ascii=False) + "\n")
    ts, ps = [it["target"]["dur_s"] for it in items], [it["prompt"]["dur_s"] for it in items]
    print(f"목표 {sum(ts):.0f}초(평균 {np.mean(ts):.1f}) · 프롬프트 {sum(ps):.0f}초(평균 {np.mean(ps):.1f}) · seed {a.seed}\n→ {out}  (meta.lst · set.jsonl · prompt/ human/ mimi/ mimi_prompt/ · prompt_codes.npz)")


if __name__ == "__main__":
    main()
