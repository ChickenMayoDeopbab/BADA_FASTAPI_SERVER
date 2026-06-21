import asyncio
import json
import logging
import sys
import threading
from contextlib import suppress

import numpy as np
import sounddevice as sd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pipeline_feedback")

from app.core.config import get_settings
from app.services.pipeline import VoicePipeline, _State
from app.services.spring_client import SpringInternalClient

SAMPLE_RATE = 16000
BLOCK = 1600

HALF_DUPLEX = True

_SESSION = {
    "userId": 1,
    "type": "SCENARIO",
    "aiPersonality": "NORMAL",
    "maxDurationSeconds": 180,
    "scenario": {
        "title": "병원 예약 변경",
        "aiRole": "병원 접수 직원",
        "script": [
            {"step": 1, "aiGoal": "전화를 받고 어느 병원인지 밝히며 용건을 묻는다"},
            {"step": 2, "aiGoal": "기존 예약자 본인 확인(이름, 생년월일)을 요청한다"},
            {"step": 3, "aiGoal": "변경을 원하는 날짜와 시간을 물어본다"},
            {"step": 4, "aiGoal": "변경 가능 여부를 안내하고 확정한다"},
            {"step": 5, "aiGoal": "마무리 인사를 한다"},
        ],
    },
}


class FakeWebSocket:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._inbound: asyncio.Queue[dict] = asyncio.Queue()
        self._player = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        self._player.start()
        self.end_reason: str | None = None
        self.end_feedback: dict | None = None
        self.error_code: str | None = None

    def feed_mic(self, pcm: bytes) -> None:
        self._loop.call_soon_threadsafe(self._inbound.put_nowait, {"bytes": pcm})

    def feed_end(self) -> None:
        # 클라가 정상적으로 보내는 종료 프레임 — ws_alive 유지되어 end+feedback 수신 가능
        self._loop.call_soon_threadsafe(
            self._inbound.put_nowait, {"text": json.dumps({"type": "end"})}
        )

    def disconnect(self) -> None:
        # 소켓 강제 단절 시뮬레이션 — 이 경로로는 end 프레임을 못 받음
        self._loop.call_soon_threadsafe(
            self._inbound.put_nowait, {"type": "websocket.disconnect"}
        )

    async def receive(self) -> dict:
        return await self._inbound.get()

    async def send_bytes(self, data: bytes) -> None:
        audio = np.frombuffer(data, dtype=np.int16)
        await asyncio.to_thread(self._player.write, audio)

    async def send_json(self, payload: dict) -> None:
        ftype = payload.get("type")
        if ftype == "emotion":
            logger.info("AI emotion=%s", payload.get("value"))
        elif ftype == "end":
            self.end_reason = payload.get("reason")
            self.end_feedback = payload.get("feedback")
            logger.info("=== 세션 종료 reason=%s ===", payload.get("reason"))
        elif ftype == "error":
            self.error_code = payload.get("code")
            logger.info("=== 에러 종료 code=%s ===", payload.get("code"))

    async def close(self, code: int = 1000) -> None:
        with suppress(Exception):
            self._player.stop()
            self._player.close()


def _print_verdict(ws: FakeWebSocket) -> None:
    fb = ws.end_feedback
    print("\n================ FEEDBACK ================")
    print(f"종료 사유 : {ws.end_reason}")
    if fb is None:
        print("feedback  : (없음)")
    else:
        print(json.dumps(fb, ensure_ascii=False, indent=2, default=str))

    required = ("shake_count", "silence_total", "good_segments")
    ok = (
        ws.error_code is None
        and ws.end_reason is not None
        and isinstance(fb, dict)
        and all(k in fb for k in required)
        and isinstance(fb["shake_count"], int)
        and isinstance(fb["silence_total"], (int, float))
        and isinstance(fb["good_segments"], list)
    )
    print("\n판정:", "PASS (end_frame 에 feedback 정상 동봉)" if ok else "CHECK (아래 사유)")
    if not ok:
        if ws.error_code is not None:
            print(f"  - 에러 종료(code={ws.error_code}) — feedback 생략이 정상")
        elif ws.end_reason is None:
            print("  - end 프레임을 못 받음(강제 종료)")
        elif not isinstance(fb, dict):
            print("  - end 프레임에 feedback 키가 없음")
        else:
            missing = [k for k in required if k not in fb]
            if missing:
                print(f"  - feedback 누락 키: {missing}")
            else:
                print("  - feedback 값 타입이 예상과 다름(위 출력 확인)")
    print("==========================================")


async def main() -> None:
    settings = get_settings()
    loop = asyncio.get_running_loop()
    ws = FakeWebSocket(loop)

    pipeline = VoicePipeline(
        ws,
        "manual-feedback-session",
        _SESSION,
        settings=settings,
        spring=SpringInternalClient(settings),
    )

    def on_mic(indata, frames, time_info, status_) -> None:
        if status_:
            logger.debug("mic status: %s", status_)
        buf = bytes(indata)
        if HALF_DUPLEX and pipeline._state is not _State.LISTENING:
            buf = b"\x00" * len(buf)
        ws.feed_mic(buf)

    mic = sd.RawInputStream(
        samplerate=SAMPLE_RATE, blocksize=BLOCK, channels=1, dtype="int16", callback=on_mic
    )

    def wait_enter() -> None:
        with suppress(EOFError):
            input()
        ws.feed_end()

    threading.Thread(target=wait_enter, daemon=True).start()

    print("=== 피드백 종료 하니스 — 시나리오대로 대화. Enter/Ctrl+C 로 종료 ===")
    try:
        with mic:
            await pipeline.run()
    finally:
        _print_verdict(ws)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n(Ctrl+C) 종료 중...")
