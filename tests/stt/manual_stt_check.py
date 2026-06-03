import asyncio
import logging
import os
import sys
import time

import sounddevice as sd

from app.services.stt import AUDIO_EOS, GoogleSTTClient, STTEventType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("test_stt")

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
BLOCKSIZE = 1600


async def feed_microphone(
    audio_queue: "asyncio.Queue[bytes | None]",
    stop_event: asyncio.Event,
) -> None:
    """마이크 콜백에서 큐로"""
    loop = asyncio.get_running_loop()

    def callback(indata, frames, time_info, status) -> None:
        if status:
            logger.warning("마이크 상태 경고: %s", status)
        pcm_bytes = bytes(indata)
        loop.call_soon_threadsafe(audio_queue.put_nowait, pcm_bytes)

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=BLOCKSIZE,
        callback=callback,
    )
    with stream:
        logger.info("마이크 입력 시작. 말해보세요. (Ctrl+C 종료)")
        await stop_event.wait()

    audio_queue.put_nowait(AUDIO_EOS)
    logger.info("마이크 입력 종료")


async def consume_events(client: GoogleSTTClient, audio_queue) -> None:
    """STTEvent 소비 + 지연 측정."""
    speech_begin_at: float | None = None

    async for event in client.stream(audio_queue):
        now = time.monotonic()

        if event.type is STTEventType.SPEECH_BEGIN:
            speech_begin_at = now
            logger.info("[발화 시작]")

        elif event.type is STTEventType.INTERIM:
            logger.info("[중간] %s", event.text)

        elif event.type is STTEventType.FINAL:
            latency = ""
            if speech_begin_at is not None:
                latency = f" (발화시작->FINAL {now - speech_begin_at:.2f}s)"
                speech_begin_at = None
            logger.info("[확정] %s%s", event.text, latency)
            logger.info("       end_offset=%.2fs", event.end_offset_sec or 0.0)

        elif event.type is STTEventType.SPEECH_END:
            logger.info("[발화 종료]")


async def main() -> None:
    project_id = os.environ.get("GOOGLE_PROJECT_ID")
    if not project_id:
        logger.error("GOOGLE_PROJECT_ID 환경변수가 필요합니다.")
        sys.exit(1)

    client = GoogleSTTClient(
        project_id=project_id,
        location=os.environ.get("GOOGLE_STT_LOCATION", "us"),
        model=os.environ.get("GOOGLE_STT_MODEL", "chirp_3"),
        language=os.environ.get("GOOGLE_STT_LANGUAGE", "ko-KR"),
        sample_rate_hertz=SAMPLE_RATE,
    )

    audio_queue: "asyncio.Queue[bytes | None]" = asyncio.Queue()
    stop_event = asyncio.Event()

    feeder = asyncio.create_task(feed_microphone(audio_queue, stop_event))
    consumer = asyncio.create_task(consume_events(client, audio_queue))

    try:
        await consumer
    except asyncio.CancelledError:
        pass
    finally:
        stop_event.set()
        feeder.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("종료")