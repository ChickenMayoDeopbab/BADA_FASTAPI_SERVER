# -*- coding: utf-8 -*-
"""E-D — 되먹임 드리프트: 문맥 뒤에 목표 턴을 만들고, 그 생성 코드를 자기 턴([태그]글 + 코드)으로 문맥에 붙여 같은 글을 다시 만들기를 --turns 번 반복한다.
  워커가 자기 턴을 코드로 되먹일 때 레벨이 턴마다 내려가는지(b1: 출력 ≈ 0.81 × 문맥 − 8.3 dB 외삽) 본다. 레벨은 맥에서 wav 로 잰다.
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 python g1b_drift.py --set ~/CSM/g1b/set_n26 --weights ~/CSM/runs/b1/epoch_1 --out ~/CSM/g1b/drift_n26 --items 10 --turns 5
"""
import argparse, json, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1_generate as G
from g1b_generate import build_prefix


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--set", required=True); ap.add_argument("--out", required=True); ap.add_argument("--repo", default="sesame/csm-1b"); ap.add_argument("--weights", required=True)
    ap.add_argument("--ctx", type=int, default=120, choices=[120, 60, 0]); ap.add_argument("--items", type=int, default=10); ap.add_argument("--turns", type=int, default=5); ap.add_argument("--max-seconds", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--dtype", default=None, choices=["bfloat16", "float32"])
    ap.add_argument("--compile", dest="compile_", action=argparse.BooleanOptionalAction, default=None); ap.add_argument("--backend", default="inductor")
    a = ap.parse_args(); dev = torch.device(a.device); set_dir = os.path.expanduser(a.set)
    dtype = getattr(torch, a.dtype or ("bfloat16" if dev.type == "cuda" else "float32")); comp = a.compile_ if a.compile_ is not None else dev.type == "cuda"
    items = [json.loads(l) for l in open(os.path.join(set_dir, "set.jsonl"), encoding="utf-8")][: a.items]; codes = np.load(os.path.join(set_dir, "codes.npz"))
    max_frames = int(a.max_seconds / G.FRAME_S); out = os.path.expanduser(a.out); os.makedirs(out, exist_ok=True); log = os.path.join(out, "gen.jsonl")
    proc, model = G.load(a.repo, a.weights, dtype, dev); tok, cfg = proc.tokenizer, model.config
    from g0c_static_loop import StaticCsm
    sc = StaticCsm(model, False, comp, a.backend); ids0, spans0 = build_prefix(tok, cfg, items[0], codes, a.ctx)
    for i in range(3 if comp else 1):
        t = time.perf_counter(); G.gen_static(sc, model, ids0, spans0, 6, 1); print(f"  예열 {i + 1}: {time.perf_counter() - t:.1f} s", flush=True)
    budget = 2048                                                                                     # 학습 창(2,048)을 넘기지 않는다
    def assemble(turns, target_ids):
        ids, spans = [], []
        for tids, c in turns: spans.append((len(ids) + len(tids), c)); ids += tids + [cfg.audio_token_id] * c.shape[0] + [cfg.audio_eos_token_id]
        return ids + target_ids, spans
    for n, it in enumerate(items):
        ctx_key = "ctx120" if a.ctx == 120 else "ctx60" if a.ctx == 60 else None
        turns = [(G.text_ids(tok, u["spell"], u["tag"]), torch.from_numpy(codes[f"{u['shard']}:{u['idx']}"].astype(np.int64)).T) for u in (it[ctx_key] if ctx_key else [])]
        target_ids = G.text_ids(tok, it["infer_text"], it["tag"])
        for k in range(1, a.turns + 1):
            torch.manual_seed(a.seed * 100003 + n * 101 + k)
            while len(assemble(turns, target_ids)[0]) + max_frames > budget and turns: turns.pop(0)         # 넘치면 가장 오래된 턴부터 버린다
            pre, sp = assemble(turns, target_ids)
            c, audio, eos, ttfa, total, T = G.gen_static(sc, model, pre, sp, max_frames, 1)
            G.write_wav(os.path.join(out, f"{it['id']}_k{k}.wav"), audio); turns.append((target_ids, c.cpu()))          # 자기 턴을 코드로 되먹인다
            r = dict(id=it["id"], k=k, frames=int(c.shape[0]), audio_s=round(int(c.shape[0]) * G.FRAME_S, 2), eos=bool(eos), ctx_positions=T, n_turns=len(turns) - 1, weights=a.weights, ctx=a.ctx)
            with open(log, "a", encoding="utf-8") as f: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  [{n + 1}/{len(items)}] {it['id']} k={k} {r['frames']}프레임 {'끝남' if eos else '안끝남'} · 접두어 {T}위치 · 문맥 턴 {r['n_turns']}", flush=True)
    print(f"→ {out} ({len(items)}항목 × {a.turns}턴). 레벨은 맥에서: python -c 'import level...' 또는 band/level 스크립트로 k 별 말소리 RMS 를 잰다")


if __name__ == "__main__":
    main()
