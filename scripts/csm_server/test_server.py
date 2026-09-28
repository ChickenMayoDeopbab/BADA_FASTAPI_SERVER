# -*- coding: utf-8 -*-
"""W1d — HTTP 층 시험(맥 CPU, 실 모델, 컴파일 없음). HF_HUB_OFFLINE=1 python test_server.py
  open → user → speak(청크·헤더) → 스트림 끊기(취소 경로) → close · 잘못된 session/voice 404"""
import json, os, sys, tempfile, time, wave
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)


def wav16(path, sec, hz, amp=0.05):
    t = np.arange(int(sec * 16000)) / 16000; x = (amp * np.sin(2 * np.pi * hz * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w: w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(x.tobytes())
    return x


def main():
    d = tempfile.mkdtemp(); ref = os.path.join(d, "ref.wav"); wav16(ref, 3.0, 180)
    json.dump({"ai": {"ref_audio": ref, "ref_text": "네, 안녕하세요. 무엇을 도와드릴까요?"}}, open(os.path.join(d, "voices.json"), "w", encoding="utf-8"), ensure_ascii=False)
    os.environ.update(CSM_WEIGHTS="sesame/csm-1b", CSM_DEVICE="cpu", CSM_COMPILE="0", VOICES_FILE=os.path.join(d, "voices.json"))
    from fastapi.testclient import TestClient
    import server
    t0 = time.time()
    with TestClient(server.app) as c:                                                    # startup: 모델 로드 + 워밍업
        h = c.get("/health").json(); assert h["ready"] and h["voices"] == ["ai"] and not h["busy"], h
        print(f"  ✓ health ready (기동 {time.time() - t0:.0f} s, CPU)")
        import base64
        assert c.post("/v1/session/user", json={"session_id": "nope", "text": "x", "pcm_b64": base64.b64encode(b"\x00\x00").decode()}).status_code == 404
        assert c.post("/v1/session/open", json={"session_id": "s1", "voice": "없는목소리"}).status_code == 404
        r = c.post("/v1/session/open", json={"session_id": "s1", "voice": "ai"}).json(); assert r["ok"] and r["positions"] > 30, r
        print(f"  ✓ open: 참조 프리필 {r['prefill_ms']:.0f} ms · 위치 {r['positions']}")
        pcm = wav16(os.path.join(d, "u.wav"), 2.0, 220, amp=0.01)                              # 조용한(−40 dBFS 대) 사용자 턴
        r = c.post("/v1/session/user", json={"session_id": "s1", "text": "어 제가 어제 주문한 게 아직 안 왔어요.", "pcm_b64": base64.b64encode(pcm.tobytes()).decode()}).json()
        assert r["ok"] and 23 <= r["frames"] <= 27 and r["gain_db"] > 5, r
        print(f"  ✓ user: {r['frames']}프레임 · 레벨 {r['level_db']:+.1f} → 이득 {r['gain_db']:+.1f} dB · 프리필 {r['prefill_ms']:.0f} ms · 위치 {r['positions']}")
        server.SEG_MAX_S = 1.2                                                                    # CPU 라 짧게(15프레임)
        with c.stream("POST", "/v1/session/speak", json={"session_id": "s1", "text": "네, 확인해 드리겠습니다."}) as resp:
            assert resp.status_code == 200 and resp.headers["x-sample-rate"] == "16000", resp.headers
            chunks = [b for b in resp.iter_bytes() if b]
        n = sum(len(b) for b in chunks) // 2; assert chunks and all(len(b) % 2 == 0 for b in chunks) and n >= 10 * 1280, (len(chunks), n)
        s = server.S["session"]; assert s.turns[-1]["codes"] is not None and not server.S["lock"].locked()
        print(f"  ✓ speak: 청크 {len(chunks)}개 · {n / 16000:.2f} s · 캐시 확정 위치 {s.pos} · 락 해제")
        with c.stream("POST", "/v1/session/speak", json={"session_id": "s1", "text": "다른 문의는 없으신가요?"}) as resp:
            first = next(resp.iter_bytes())                                                      # 첫 청크만 받고 끊는다
        for _ in range(50):
            if not server.S["lock"].locked(): break
            time.sleep(0.2)
        assert not server.S["lock"].locked() and s.turns[-1]["codes"] is not None, "끊은 뒤 락이 안 풀림"
        print(f"  ✓ 스트림 끊기: 첫 청크 {len(first)} B 뒤 종료 → 확정 {s.turns[-1]['codes'].shape[0]}프레임 · 위치 {s.pos} · 락 해제")
        assert c.post("/v1/session/close", json={"session_id": "s1"}).json()["ok"] and c.get("/health").json()["session"] is None
        print("  ✓ close")
    print("전부 통과 ✓")


if __name__ == "__main__":
    main()
