import asyncio
import logging
import sys
import threading
import time
from contextlib import suppress

import numpy as np
import sounddevice as sd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pipeline_manual")

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
        self._player = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16"
        )
        self._player.start()
        self._emotion_at: float | None = None
        self._first_audio_at: float | None = None
        self._turn_bytes = 0

    def feed_mic(self, pcm: bytes) -> None:
        self._loop.call_soon_threadsafe(self._inbound.put_nowait, {"bytes": pcm})

    def disconnect(self) -> None:
        self._loop.call_soon_threadsafe(
            self._inbound.put_nowait, {"type": "websocket.disconnect"}
        )

    async def receive(self) -> dict:
        return await self._inbound.get()

    async def send_bytes(self, data: bytes) -> None:
        now = time.perf_counter()
        if self._first_audio_at is None:
            self._first_audio_at = now
            if self._emotion_at is not None:
                logger.info("  TTFB(emotion->첫오디오) %.0fms", (now - self._emotion_at) * 1000)
        self._turn_bytes += len(data)
        audio = np.frombuffer(data, dtype=np.int16)
        await asyncio.to_thread(self._player.write, audio)

    async def send_json(self, payload: dict) -> None:
        ftype = payload.get("type")
        if ftype == "emotion":
            self._emotion_at = time.perf_counter()
            self._first_audio_at = None
            self._turn_bytes = 0
            logger.info("AI emotion=%s", payload.get("value"))
        elif ftype == "speaking_end":
            secs = self._turn_bytes / 2 / SAMPLE_RATE
            logger.info("  발화 끝 (오디오 %.2fs)", secs)
        elif ftype == "interrupt":
            logger.info(">>> barge-in (interrupt) <<<")
        elif ftype == "end":
            logger.info("=== 세션 종료 reason=%s ===", payload.get("reason"))
        elif ftype == "error":
            logger.info("=== 에러 종료 code=%s ===", payload.get("code"))
        else:
            logger.info("[frame] %s", payload)

    async def close(self, code: int = 1000) -> None:
        with suppress(Exception):
            self._player.stop()
            self._player.close()

async def _state_watcher(pipeline: VoicePipeline) -> None:
    prev = None
    while True:
        cur = pipeline._state
        if cur != prev:
            logger.info("[state] %s", cur.value)
            prev = cur
        await asyncio.sleep(0.05)


async def main() -> None:
    settings = get_settings()
    loop = asyncio.get_running_loop()
    ws = FakeWebSocket(loop)

    pipeline = VoicePipeline(
        ws,
        "manual-session",
        _SESSION,
        settings=settings,
        spring=SpringInternalClient(settings),
    )
    _orig_handle = pipeline._handle_stt_event

    async def _logged_handle(event):
        logger.info("[stt] %-12s %r", event.type.value, (event.text or "")[:40])
        await _orig_handle(event)

    pipeline._handle_stt_event = _logged_handle

    mic_peak = {"v": 0}

    def on_mic(indata, frames, time_info, status_) -> None:
        if status_:
            logger.debug("mic status: %s", status_)
        buf = bytes(indata)
        samples = np.frombuffer(buf, dtype=np.int16)
        if samples.size:
            mic_peak["v"] = max(mic_peak["v"], int(np.abs(samples).max()))
        if HALF_DUPLEX and pipeline._state is not _State.LISTENING:
            buf = b"\x00" * len(buf)
        ws.feed_mic(buf)

    async def _meter() -> None:
        while True:
            await asyncio.sleep(1.0)
            logger.info(
                "[mic] peak=%d  q=%d", mic_peak["v"], pipeline._audio_queue.qsize()
            )
            mic_peak["v"] = 0

    mic = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK,
        channels=1,
        dtype="int16",
        callback=on_mic,
    )

    def wait_enter() -> None:
        with suppress(EOFError):
            input()
        ws.disconnect()

    threading.Thread(target=wait_enter, daemon=True).start()
    watcher = asyncio.create_task(_state_watcher(pipeline))
    meter = asyncio.create_task(_meter())
    mode = "스피커로 ㄱㄱ" if HALF_DUPLEX else "이어폰 필수 ㅇㅇ"
    print(f"model: {mode}")
    with mic:
        try:
            await pipeline.run()
        finally:
            watcher.cancel()
            meter.cancel()
            with suppress(asyncio.CancelledError):
                await watcher
            with suppress(asyncio.CancelledError):
                await meter


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n종료.")
