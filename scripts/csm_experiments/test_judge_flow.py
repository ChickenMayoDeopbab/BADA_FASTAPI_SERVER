# -*- coding: utf-8 -*-
"""심판(g1_score.gemini_transcribe)의 비동기 흐름을 **가짜 세션**으로 시험한다 — 네트워크·키 없이.  python test_judge_flow.py
보는 것: receive() 가 턴 단위로 끊겨도 이어 받는가 · 끝 신호 뒤 늦게 오는 FINAL 을 기다리는가 · 보내다 실패하면 멈추지 않고 예외로 올리는가.
(진짜 서버에 붙는지는 `g1_score.py --selftest judge --wav x.wav` 로 확인한다.)
"""
import asyncio, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1_score as S
from google.genai import types

S.EOS_IDLE_S, S.EOS_MAX_S = 0.4, 3.0
final = lambda t: types.LiveServerMessage(server_content=types.LiveServerContent(input_transcription=types.Transcription(text=t)))


class FakeSession:
    def __init__(self, fail_send=False):
        self.q, self.sent, self.closed, self.fail_send = asyncio.Queue(), 0, False, fail_send

    async def send_realtime_input(self, audio=None, audio_stream_end=None):
        if audio is not None:
            self.sent += len(audio.data)
            if self.fail_send and self.sent > 6400:
                raise RuntimeError("송신 실패(가짜)")
            if self.sent == 3200 * 5:                             # 말 중간에 FINAL 하나 + 턴 끝
                await self.q.put(final("안녕하세요")); await self.q.put("TURN_END")
        if audio_stream_end:
            async def late():                                     # 끝 신호 0.25초 뒤에 마지막 FINAL
                await asyncio.sleep(0.25); await self.q.put(final("반갑습니다"))
            self.late = asyncio.create_task(late())

    async def receive(self):
        while True:
            m = await self.q.get()
            if m == "TURN_END":
                return
            if m == "CLOSED":
                raise ConnectionError("closed(가짜)")
            yield m

    async def close(self):
        self.closed = True; await self.q.put("CLOSED")


class FakeClient:
    def __init__(self, s):
        class Conn:
            async def __aenter__(_): return s
            async def __aexit__(_, *a): return False
        self.aio = type("A", (), {"live": type("L", (), {"connect": staticmethod(lambda model, config: Conn())})()})()


ok = True
def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(f"  {'✓' if cond else '✗'} {name} {extra}")


run = lambda s: asyncio.run(asyncio.wait_for(S.gemini_transcribe(FakeClient(s), types, b"\0" * 3200 * 10, pace=0), 10))
s = FakeSession(); t = time.time(); out = run(s)
check("턴이 끊겨도 두 FINAL 을 다 모은다", out == "안녕하세요 반갑습니다", f"→ {out!r}")
check("끝 신호 뒤 조용해지면 닫는다", s.closed and time.time() - t < 2.0, f"({time.time() - t:.2f}s)")
check("보낸 바이트 = 전부", s.sent == 32000)
s = FakeSession(fail_send=True); t = time.time()
try:
    run(s); check("송신 실패는 예외로 올린다", False)
except RuntimeError as e:
    check("송신 실패는 예외로 올린다(멈추지 않는다)", "송신 실패" in str(e) and s.closed and time.time() - t < 2.0, f"({time.time() - t:.2f}s)")
except asyncio.TimeoutError:
    check("송신 실패는 예외로 올린다(멈추지 않는다)", False, "← 10초 동안 멈춰 있었다")
print("\n전부 통과 ✓" if ok else "\n실패한 항목이 있다 ✗"); sys.exit(0 if ok else 1)
