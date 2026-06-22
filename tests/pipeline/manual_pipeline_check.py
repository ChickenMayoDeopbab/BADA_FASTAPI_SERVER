import asyncio
import logging
import sys
from contextlib import suppress

import numpy as np
import sounddevice as sd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pipeline_manual")

from app.core.config import get_settings
from app.services.pipeline import VoicePipeline
from app.services.spring_client import SpringInternalClient

SAMPLE_RATE = 16000
BLOCK = 1600

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

    def feed_mic(self, pcm: bytes) -> None:
        self._loop.call_soon_threadsafe(self._inbound.put_nowait, {"bytes": pcm})

    def disconnect(self) -> None:
        self._loop.call_soon_threadsafe(
            self._inbound.put_nowait, {"type": "websocket.disconnect"}
        )

    async def receive(self) -> dict:
        return await self._inbound.get()

    async def send_bytes(self, data: bytes) -> None:
        audio = np.frombuffer(data, dtype=np.int16)
        await asyncio.to_thread(self._player.write, audio)

    async def send_json(self, payload: dict) -> None:
        logger.info("[frame] %s", payload)

    async def close(self, code: int = 1000) -> None:
        self._player.stop()
        self._player.close()


async def main() -> None:
    settings = get_settings()
    loop = asyncio.get_running_loop()
    ws = FakeWebSocket(loop)

    def on_mic(indata, frames, time_info, status_):
        if status_:
            logger.debug("mic status: %s", status_)
        ws.feed_mic(bytes(indata))

    pipeline = VoicePipeline(
        ws,
        "manual-session",
        _SESSION,
        settings=settings,
        spring=SpringInternalClient(settings),
    )

    mic = sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCK,
        channels=1,
        dtype="int16",
        callback=on_mic,
    )

    print("테스트 시작")
    with mic, suppress(asyncio.CancelledError):
        await pipeline.run()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n종료.")
