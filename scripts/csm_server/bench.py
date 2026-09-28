# -*- coding: utf-8 -*-
"""W3 — 워커 지연·동시성 벤치. 통화 하나를 흉내 낸다: open(voice) → 사용자 턴(8 s 합성 PCM) → 문장 3개 speak → close. --urls 에 워커 여러 개를 주면 동시에 각각 한 통화씩.
  python bench.py --urls http://127.0.0.1:8020,http://127.0.0.1:8021 --voice ai --turns 3
"""
import argparse, base64, json, statistics, threading, time
import httpx, numpy as np

SENT = ["네, 안녕하세요. 무엇을 도와드릴까요?", "주문하신 상품은 내일 오후에 도착할 예정입니다.", "네, 더 궁금하신 점 있으시면 말씀해 주세요."]


def one_call(url, voice, turns, out):
    c = httpx.Client(base_url=url, timeout=httpx.Timeout(60.0, connect=5.0)); sid = f"bench-{threading.get_ident()}"
    t = np.arange(8 * 16000) / 16000; pcm = (0.05 * np.sin(2 * np.pi * 200 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 2.5 * t)) * 32767).astype(np.int16)
    r = c.post("/v1/session/open", json={"session_id": sid, "voice": voice}).json(); rec = dict(url=url, open_prefill_ms=r["prefill_ms"], user_prefill_ms=[], ttfa_ms=[], rtf=[], audio_s=[])
    for k in range(turns):
        u = c.post("/v1/session/user", json={"session_id": sid, "text": "어 제가 어제 주문한 게 아직 안 왔는데요, 언제 오나요?", "pcm_b64": base64.b64encode(pcm.tobytes()).decode()}).json(); rec["user_prefill_ms"].append(u["prefill_ms"])
        t0 = time.perf_counter(); first = None; n = 0
        with c.stream("POST", "/v1/session/speak", json={"session_id": sid, "text": SENT[k % len(SENT)]}) as resp:
            for chunk in resp.iter_bytes():
                if chunk and first is None: first = (time.perf_counter() - t0) * 1e3
                n += len(chunk)
        total = time.perf_counter() - t0; sec = n / 2 / 16000; rec["ttfa_ms"].append(round(first or -1, 1)); rec["rtf"].append(round(total / max(sec, 0.08), 3)); rec["audio_s"].append(round(sec, 2))
    c.post("/v1/session/close", json={"session_id": sid}); out.append(rec)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--urls", required=True); ap.add_argument("--voice", default="ai"); ap.add_argument("--turns", type=int, default=3); a = ap.parse_args()
    urls = a.urls.split(","); out = []; th = [threading.Thread(target=one_call, args=(u, a.voice, a.turns, out)) for u in urls]
    t0 = time.perf_counter(); [t.start() for t in th]; [t.join() for t in th]; wall = time.perf_counter() - t0
    for r in out: print(json.dumps(r, ensure_ascii=False))
    allt = [x for r in out for x in r["ttfa_ms"]]; allr = [x for r in out for x in r["rtf"]]
    print(f"\n동시 {len(urls)}통화 × {a.turns}턴 · 벽시계 {wall:.1f} s · 첫 청크 p50 {statistics.median(allt):.0f} ms(최대 {max(allt):.0f}) · RTF p50 {statistics.median(allr):.3f}(최대 {max(allr):.3f}) · 사용자 턴 프리필 p50 {statistics.median([x for r in out for x in r['user_prefill_ms']]):.0f} ms")


if __name__ == "__main__":
    main()
