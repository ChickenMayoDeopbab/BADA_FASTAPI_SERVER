# -*- coding: utf-8 -*-
"""G1b 생성 — 다중 턴 접두어(문맥 120 / 60 / 0 초)로 목표 턴을 만든다. g1_generate.py 의 정적 루프·임베딩·wav 쓰기를 그대로 쓴다.
  접두어 = 문맥 턴마다 `<bos>[태그]글<eos>` + AUDIO×T + audio_eos(저장 코드) … + 목표 턴 `<bos>[태그]글<eos>` → 프레임 생성(EOS 또는 --max-seconds).
  TTFA 는 접두어 프리필을 포함한다(워커의 첫 프레임 지연과 같은 정의).

  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 python g1b_generate.py --set ~/CSM/g1b/set --weights ~/CSM/runs/b1/epoch_1 --ctx 120 --out ~/CSM/g1b/run/b1_ctx120
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1_generate as G


def build_prefix(tok, cfg, it, codes, ctx):
    """→ (ids, spans). ctx ∈ {120, 60, 0}."""
    turns = it["ctx120"] if ctx == 120 else it["ctx60"] if ctx == 60 else []
    ids, spans = [], []
    for t in turns:
        p = G.text_ids(tok, t["spell"], t["tag"]); c = torch.from_numpy(codes[f"{t['shard']}:{t['idx']}"].astype(np.int64)).T       # [T,32]
        spans.append((len(ids) + len(p), c)); ids += p + [cfg.audio_token_id] * c.shape[0] + [cfg.audio_eos_token_id]
    return ids + G.text_ids(tok, it["infer_text"], it["tag"]), spans


def main():
    ap = argparse.ArgumentParser(description="G1b 다중 턴 생성")
    ap.add_argument("--set", required=True); ap.add_argument("--out", required=True); ap.add_argument("--repo", default="sesame/csm-1b")
    ap.add_argument("--weights", required=True, help="a2/last · b1/epoch_1 …"); ap.add_argument("--ctx", type=int, default=120, choices=[120, 60, 0])
    ap.add_argument("--max-seconds", type=float, default=20.0); ap.add_argument("--first-frames", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.9); ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--dtype", default=None, choices=["bfloat16", "float32"])
    ap.add_argument("--compile", dest="compile_", action=argparse.BooleanOptionalAction, default=None); ap.add_argument("--backend", default="inductor")
    a = ap.parse_args()
    dev = torch.device(a.device); set_dir = os.path.expanduser(a.set)
    dtype = getattr(torch, a.dtype or ("bfloat16" if dev.type == "cuda" else "float32")); comp = a.compile_ if a.compile_ is not None else dev.type == "cuda"
    items = [json.loads(l) for l in open(os.path.join(set_dir, "set.jsonl"), encoding="utf-8")]; items = items[: a.limit] if a.limit else items
    codes = np.load(os.path.join(set_dir, "codes.npz")); max_frames = int(a.max_seconds / G.FRAME_S)
    print(f"G1b 생성 · {a.weights} · 문맥 {a.ctx} 초 · {dev} · {dtype} · 목표 {len(items)} · 최대 {max_frames}프레임")
    proc, model = G.load(a.repo, a.weights, dtype, dev); tok, cfg = proc.tokenizer, model.config
    out = os.path.expanduser(a.out); os.makedirs(out, exist_ok=True); log = os.path.join(out, "gen.jsonl")
    done = {json.loads(l)["id"] for l in open(log, encoding="utf-8")} if os.path.exists(log) else set()
    todo = [it for it in items if it["id"] not in done]
    if todo:
        from g0c_static_loop import StaticCsm, make_sampler
        sc = StaticCsm(model, False, comp, a.backend)
        if (a.temperature, a.top_k) != (0.9, 50): sc.bb.sample = sc.dd.sample = make_sampler(False, a.temperature, a.top_k)
        ids, spans = build_prefix(tok, cfg, todo[0], codes, a.ctx)
        for i in range(3 if comp else 1):
            t = time.perf_counter(); G.gen_static(sc, model, ids, spans, 6, a.first_frames); print(f"  예열 {i + 1}: {time.perf_counter() - t:.1f} s", flush=True)
    for i, it in enumerate(items):
        if it["id"] in done: continue
        torch.manual_seed(a.seed * 100003 + i)
        ids, spans = build_prefix(tok, cfg, it, codes, a.ctx)
        assert len(ids) + max_frames <= cfg.max_position_embeddings
        c, audio, eos, ttfa, total, T = G.gen_static(sc, model, ids, spans, max_frames, a.first_frames)
        G.write_wav(os.path.join(out, it["id"] + ".wav"), audio); n = int(c.shape[0]); sec = max(n * G.FRAME_S, G.FRAME_S)
        r = dict(id=it["id"], ctx=a.ctx, weights=a.weights, frames=n, eos=bool(eos), audio_s=round(n * G.FRAME_S, 2), ttfa_ms=None if ttfa is None else round(ttfa * 1e3, 1),
                 total_s=round(total, 3), rtf=round(total / sec, 4), ctx_positions=T, ctx_turns=len(spans), ctx_s=it["ctx120_s"] if a.ctx == 120 else it["ctx60_s"] if a.ctx == 60 else 0,
                 target_s=round(it["target"]["frames"] * G.FRAME_S, 2), first_frames=a.first_frames, seed=a.seed, temperature=a.temperature, top_k=a.top_k)
        with open(log, "a", encoding="utf-8") as f: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  [{i + 1:>3}/{len(items)}] {it['id']} {n:>3}프레임(원본 {it['target']['frames']}) {'끝남' if eos else '안끝남'} · 접두어 {T}위치 · TTFA {r['ttfa_ms']} ms · RTF {r['rtf']:.3f} · {it['infer_text'][:28]}", flush=True)
    rows = [json.loads(l) for l in open(log, encoding="utf-8")]; med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
    t = [r["ttfa_ms"] for r in rows if r["ttfa_ms"] is not None]
    print(f"\n{len(rows)}턴 · 문맥 {a.ctx} 초 · 접두어 p50 {med([r['ctx_positions'] for r in rows])}위치 · TTFA p50 {med(t):.0f} ms(프리필 포함) · RTF p50 {med([r['rtf'] for r in rows if r['frames']]):.3f} · 끝남 실패 {sum(not r['eos'] for r in rows)} · 길이 비 p50 {med([r['audio_s'] / max(r['target_s'], 0.08) for r in rows]):.2f} → {out}")


if __name__ == "__main__":
    main()
