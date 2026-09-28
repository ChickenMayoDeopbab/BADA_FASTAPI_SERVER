# -*- coding: utf-8 -*-
"""W2 — G1b 세트(set_n26)를 워커 API 로 턴 단위 재생해 목표 턴 wav 를 만든다. 같은 코드(codes.npz)를 문맥으로 넣으므로 E-B(g1b_generate)와 같은 조건이다.
  python replay_g1b.py --url http://127.0.0.1:8020 --set ~/CSM/g1b/set_n26 --out ~/CSM/g1b/run_worker/b1_ctx120 [--ctx 120|60|0] [--limit N] [--drift 5]
  --drift K: 항목마다 같은 글을 K 번 이어 말하게 해(자기 턴 재인코딩 되먹임) k 별 wav 를 out/drift/ 에 둔다.
채점: 맥에서 g1_score.py --set ~/g1b/set_n26 --norm -26 --cond worker=<out> …  (출력은 16 kHz)
"""
import argparse, base64, json, os, time, wave
import httpx, numpy as np


def write_wav16(path, pcm_bytes):
    with wave.open(path, "wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(pcm_bytes)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", required=True); ap.add_argument("--set", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=int, default=120, choices=[120, 60, 0]); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--drift", type=int, default=0)
    a = ap.parse_args(); set_dir = os.path.expanduser(a.set); out = os.path.expanduser(a.out); os.makedirs(out, exist_ok=True)
    items = [json.loads(l) for l in open(os.path.join(set_dir, "set.jsonl"), encoding="utf-8")]; items = items[: a.limit] if a.limit else items
    codes = np.load(os.path.join(set_dir, "codes.npz")); c = httpx.Client(base_url=a.url, timeout=httpx.Timeout(120.0, connect=5.0))
    assert c.get("/health").json()["ready"], "워커 준비 안 됨"
    log = open(os.path.join(out, "gen.jsonl"), "a", encoding="utf-8"); done = set()
    if os.path.exists(os.path.join(out, "gen.jsonl")): done = {json.loads(l)["id"] for l in open(os.path.join(out, "gen.jsonl"), encoding="utf-8")}
    for i, it in enumerate(items):
        if it["id"] in done: continue
        sid = f"replay-{it['id']}"; assert c.post("/v1/session/open", json={"session_id": sid, "voice": "none"}).json()["ok"]
        turns = it["ctx120"] if a.ctx == 120 else it["ctx60"] if a.ctx == 60 else []
        if turns:
            payload = [dict(tag=t["tag"], text=t["spell"], frames=t["frames"], codes_b64=base64.b64encode(codes[f"{t['shard']}:{t['idx']}"].T.astype("<i2").tobytes()).decode()) for t in turns]
            r = c.post("/v1/session/context_codes", json={"session_id": sid, "turns": payload}).json(); assert r["ok"], r
        t0 = time.perf_counter(); first = None; buf = b""
        with c.stream("POST", "/v1/session/speak", json={"session_id": sid, "text": it["infer_text"]}) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_bytes():
                if chunk and first is None: first = (time.perf_counter() - t0) * 1e3
                buf += chunk
        total = time.perf_counter() - t0; sec = len(buf) / 2 / 16000; write_wav16(os.path.join(out, it["id"] + ".wav"), buf)
        rec = dict(id=it["id"], ctx=a.ctx, audio_s=round(sec, 2), ttfa_ms=None if first is None else round(first, 1), total_s=round(total, 3), rtf=round(total / max(sec, 0.08), 4), target_s=round(it["target"]["frames"] * 0.08, 2), frames=int(round(sec / 0.08)), eos=True)
        if a.drift:
            os.makedirs(os.path.join(out, "drift"), exist_ok=True); write_wav16(os.path.join(out, "drift", f"{it['id']}_k1.wav"), buf)
            for k in range(2, a.drift + 1):
                b2 = b""
                with c.stream("POST", "/v1/session/speak", json={"session_id": sid, "text": it["infer_text"]}) as resp:
                    for chunk in resp.iter_bytes(): b2 += chunk
                write_wav16(os.path.join(out, "drift", f"{it['id']}_k{k}.wav"), b2)
        c.post("/v1/session/close", json={"session_id": sid}); log.write(json.dumps(rec, ensure_ascii=False) + "\n"); log.flush()
        print(f"  [{i + 1:>3}/{len(items)}] {it['id']} {sec:.2f} s(원본 {rec['target_s']}) · 첫 청크 {rec['ttfa_ms']} ms · RTF {rec['rtf']:.3f} · {it['infer_text'][:28]}", flush=True)
    rows = [json.loads(l) for l in open(os.path.join(out, "gen.jsonl"), encoding="utf-8")]; med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
    print(f"\n{len(rows)}턴 · 첫 청크 p50 {med([r['ttfa_ms'] for r in rows if r['ttfa_ms'] is not None]):.0f} ms(문맥 프리필 별도) · RTF p50 {med([r['rtf'] for r in rows]):.3f} · 길이 비 p50 {med([r['audio_s'] / max(r['target_s'], 0.08) for r in rows]):.2f} → {out}")


if __name__ == "__main__":
    main()
