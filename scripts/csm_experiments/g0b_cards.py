# -*- coding: utf-8 -*-
"""G0 2차 — 기본 경로가 RTF 1.65 로 실패한 뒤(실측/대역폭 하한 = 13.6배, 오버헤드 지배) 카드를 잰다.

  --mode static   HF generate + 정적 KV 캐시. 이 조합이면 HF 가 forward 를
                  torch.compile(mode="reduce-overhead" = CUDA Graph)로 자동 컴파일한다.      [카드 ①②]
  --mode direct   generate() 를 쓰지 않고 백본 1스텝 + depth 31스텝을 직접 돈다(eager).       [카드 ③]
  --mode check    direct 가 HF generate 와 같은 토큰을 내는지 탐욕 디코딩으로 확인(CPU 가능).

같은 폴더에 g0_csm_speed.py 가 있어야 한다(로드·묶기·assert·문맥 생성을 거기서 가져온다).
서버 주의: torch.compile 은 런타임에 C 컴파일러를 부른다. 이 서버엔 시스템 gcc 가 없으므로
           Qwen 워커(boot.sh)와 같은 방식으로 CC/CXX 를 ~/gcc-env 로 지정할 것:
  CC=~/gcc-env/bin/x86_64-conda-linux-gnu-gcc CXX=~/gcc-env/bin/x86_64-conda-linux-gnu-g++ \
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python g0b_cards.py --mode static
--codebooks 16 은 속도만 본다(카드 ④). 나머지 코드북을 0 으로 채워 넣으므로 소리는 의미 없다.
"""
import argparse, json, math, statistics as st
import numpy as np, torch
from transformers import DynamicCache
import g0_csm_speed as g

FRAME_S, GATE_RTF = 0.08, 0.6


def build_inputs(proc, ctx, dtype):
    rng = np.random.default_rng(0); n = max(1, math.ceil(ctx / 5))
    conv = [g.turn(i, ctx / n, rng) for i in range(n)] + [{"role": str(n % 2), "content": [{"type": "text", "text": g.TARGET}]}]
    return proc.apply_chat_template(conv, tokenize=True, return_dict=True).to(g.DEV, dtype=dtype)


def sampler(greedy, temperature=0.9, top_k=50):
    if greedy:
        return lambda z: z.argmax(-1, keepdim=True)
    def f(z):
        v, i = torch.topk(z.float() / temperature, top_k)
        return i.gather(-1, torch.multinomial(torch.softmax(v, -1), 1))
    return f


@torch.no_grad()
def direct_generate(model, inp, frames, sample, n_cb=32):
    """HF generate 와 같은 계산을 GenerationMixin 없이 한다. 반환: (codes [1, frames, 32], 구간 시간 기록)"""
    dd, rec = model.depth_decoder, {"bb": [], "dd": [], "end": []}
    t0 = g.now()
    merged = model._merge_input_ids_with_input_values(inp["input_ids"], inp["input_values"], inp["input_values_cutoffs"], None)
    rec["enc"] = g.now() - t0
    t = g.now()
    out = model.backbone_model(inputs_embeds=merged["inputs_embeds"], attention_mask=inp.get("attention_mask"), use_cache=True)
    cache, h = out.past_key_values, out.last_hidden_state[:, -1, :]
    rec["prefill"] = g.now() - t
    codes = []
    for f in range(frames):
        ta = g.now()
        if f:                                                  # 직전 프레임의 32코드를 한 위치로 넣는다
            out = model.backbone_model(input_ids=codes[-1][:, None, :], past_key_values=cache, use_cache=True)
            h = out.last_hidden_state[:, -1, :]
        c0 = sample(model.lm_head(h))                          # 코드북 0
        tb = g.now()
        dcache = DynamicCache(config=dd.config)                # 프레임마다 새 캐시(최대 33위치)
        o = dd(input_ids=torch.cat([torch.zeros_like(c0), c0], 1), backbone_last_hidden_state=h,
               past_key_values=dcache, use_cache=True, logits_to_keep=1)
        frame = [c0]
        for k in range(1, n_cb):                               # 코드북 1..n_cb-1
            tok = sample(o.logits[:, -1, :]); frame.append(tok)
            if k + 1 < n_cb:
                o = dd(input_ids=tok, past_key_values=dcache, use_cache=True, logits_to_keep=1)
        fr = torch.cat(frame, 1)
        codes.append(torch.nn.functional.pad(fr, (0, 32 - n_cb)))
        tc = g.now(); rec["bb"].append(tb - ta); rec["dd"].append(tc - tb); rec["end"].append(tc)
    rec["t0"] = t0
    return torch.stack(codes, 1), rec


def summarize(rec_end, t0, dd_times, extra=None):
    n = len(rec_end); frame = st.median(np.diff(rec_end).tolist()) if n > 1 else float("nan")
    d = st.median(dd_times[1:] or dd_times)
    m = dict(frames=n, ttff_ms=(rec_end[0] - t0) * 1e3, frame_ms=frame * 1e3, depth_ms=d * 1e3, rest_ms=(frame - d) * 1e3, rtf=frame / FRAME_S)
    m.update(extra or {}); return m


def hf_frames(model, inp, frames, **kw):
    """HF generate 를 프레임 종료 시각만 재며 돌린다(컴파일 영역 안에는 훅을 넣지 않는다)."""
    dd, end, dur = model.depth_decoder, [], []
    gen0 = dd.generate
    def gen(*a, **k):
        t = g.now(); out = gen0(*a, **k); e = g.now(); dur.append(e - t); end.append(e); return out
    dd.generate = gen
    try:
        t0 = g.now(); out = model.generate(**inp, max_new_tokens=frames, output_audio=False, **kw)
    finally:
        del dd.generate
    return out, summarize(end, t0, dur)


def main():
    ap = argparse.ArgumentParser(description="CSM-1B G0 2차: 정적 캐시+자동 컴파일 / 직접 루프")
    ap.add_argument("repo", nargs="?", default="sesame/csm-1b")
    ap.add_argument("--mode", required=True, choices=["static", "direct", "check"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--ctx-seconds", type=float, default=30)
    ap.add_argument("--codebooks", type=int, default=32, help="direct 전용. 32 미만이면 속도만 의미 있다")
    a = ap.parse_args()
    g.DEV = torch.device(a.device)
    dtype = getattr(torch, a.dtype or ("bfloat16" if g.DEV.type == "cuda" else "float32"))
    gpu = torch.cuda.get_device_name(g.DEV) if g.DEV.type == "cuda" else "cpu"
    print("=" * 88); print(f"G0-2 · mode={a.mode} · {gpu} · {dtype} · torch {torch.__version__}"); print("=" * 88)
    proc, model = g.load(a.repo, dtype)
    inp = build_inputs(proc, a.ctx_seconds, dtype); T = inp["input_ids"].shape[1]
    print(f"문맥 {a.ctx_seconds:.0f}초 = {T}위치 · {a.frames}프레임")
    torch.manual_seed(0)

    if a.mode == "check":
        ref = model.generate(**inp, max_new_tokens=a.frames, do_sample=False, depth_decoder_do_sample=False, output_audio=False)
        mine, _ = direct_generate(model, inp, a.frames, sampler(True))
        n = min(ref.shape[1], mine.shape[1]); same = (ref[:, :n] == mine[:, :n])
        print(f"HF {tuple(ref.shape)} · direct {tuple(mine.shape)} · 비교 {n}프레임 × 32코드북")
        print(f"일치 {int(same.sum())}/{same.numel()}" + ("  → 완전 일치 ✓" if bool(same.all()) else
              f"  → 첫 불일치: 프레임 {int((~same).nonzero()[0][1])}, 코드북 {int((~same).nonzero()[0][2])}"))
        return

    if a.mode == "static":
        model.generation_config.cache_implementation = "static"
        model.depth_decoder.generation_config.cache_implementation = "static"
        for i in range(3):                                      # 1·2회차 = 컴파일 + CUDA Graph 캡처, 3회차를 기록
            t = g.now(); _, m = hf_frames(model, inp, a.frames)
            print(f"  {i+1}회차 {g.now()-t:6.1f} s · 프레임 {m['frame_ms']:.1f} ms · RTF {m['rtf']:.3f}" + ("   ← 컴파일 포함" if i < 2 else ""))
    else:
        sample = sampler(False)
        direct_generate(model, inp, min(3, a.frames), sample, a.codebooks)                       # 워밍업
        _, rec = direct_generate(model, inp, a.frames, sample, a.codebooks)
        m = summarize(rec["end"], rec["t0"], rec["dd"], dict(enc_ms=rec["enc"] * 1e3, prefill_ms=rec["prefill"] * 1e3,
                      backbone_ms=st.median(rec["bb"][1:] or rec["bb"]) * 1e3, codebooks=a.codebooks))

    m.update(mode=a.mode, positions=T, device=gpu)
    print(f"\n  첫 프레임 {m['ttff_ms']:.1f} ms · 프레임 {m['frame_ms']:.1f} ms = depth {m['depth_ms']:.1f} + 나머지 {m['rest_ms']:.1f}"
          + (f" (백본 {m['backbone_ms']:.1f})" if "backbone_ms" in m else ""))
    print(f"  RTF {m['rtf']:.3f}  (≤ {GATE_RTF})   {'통과 ✓' if m['rtf'] <= GATE_RTF else '실패 ✗'}   · 기본 경로 1.651 대비 {1.651 / m['rtf']:.1f}배")
    out = f"g0b_{a.mode}{'' if a.codebooks == 32 else '_cb' + str(a.codebooks)}.json"
    json.dump(m, open(out, "w"), ensure_ascii=False, indent=1); print(f"\n[저장] {out}")


if __name__ == "__main__":
    main()
