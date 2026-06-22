import asyncio
import logging
import os
import statistics
import sys
import time

import sounddevice as sd

from app.core.config import get_settings
from app.services.stt import AUDIO_EOS, GoogleSTTClient, STTEventType

settings = get_settings()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("latency_check")

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
BLOCKSIZE = 1600  # 약 100ms


async def feed_microphone(
    audio_queue: "asyncio.Queue[bytes | None]",
    stop_event: asyncio.Event,
) -> None:
    loop = asyncio.get_running_loop()

    def callback(indata, frames, time_info, status) -> None:
        if status:
            logger.warning("마이크 상태 경고: %s", status)
        loop.call_soon_threadsafe(audio_queue.put_nowait, bytes(indata))

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=BLOCKSIZE,
        callback=callback,
    )
    with stream:
        logger.info("마이크 입력 시작. 짧은 문장으로 6번 이상 말해보세요. (Ctrl+C 종료)")
        await stop_event.wait()

    audio_queue.put_nowait(AUDIO_EOS)


def _summary(label: str, samples: list[float]) -> None:
    """ms 단위 통계 출력. 첫 샘플(콜드 스타트)은 제외."""
    if not samples:
        logger.info("%s: 측정값 없음", label)
        return

    cold = samples[0]
    warm = samples[1:] if len(samples) > 1 else []

    logger.info("=" * 50)
    logger.info("%s (단위 ms)", label)
    logger.info("  콜드 스타트(첫 발화, 제외됨): %.0f", cold)
    if not warm:
        logger.info("  정상 상태 샘플 부족(2개 이상 필요)")
        return

    warm_sorted = sorted(warm)
    p90_idx = max(0, int(len(warm_sorted) * 0.9) - 1)
    logger.info("  정상 상태 n=%d", len(warm))
    logger.info("  평균   : %.0f", statistics.mean(warm))
    logger.info("  중앙값 : %.0f", statistics.median(warm))
    logger.info("  최소   : %.0f", min(warm))
    logger.info("  최대   : %.0f", max(warm))
    logger.info("  p90    : %.0f", warm_sorted[p90_idx])


async def consume_events(client: GoogleSTTClient, audio_queue) -> None:
    speech_begin_at: float | None = None
    speech_end_at: float | None = None
    first_interim_seen = False

    end_to_final: list[float] = []
    begin_to_interim: list[float] = []

    try:
        async for event in client.stream(audio_queue):
            now = time.monotonic()

            if event.type is STTEventType.SPEECH_BEGIN:
                speech_begin_at = now
                first_interim_seen = False
                logger.info("[발화 시작]")

            elif event.type is STTEventType.INTERIM:
                if not first_interim_seen and speech_begin_at is not None:
                    begin_to_interim.append((now - speech_begin_at) * 1000)
                    first_interim_seen = True
                logger.info("[중간] %s", event.text)

            elif event.type is STTEventType.SPEECH_END:
                speech_end_at = now
                logger.info("[발화 종료]")

            elif event.type is STTEventType.FINAL:
                if speech_end_at is not None:
                    delay_ms = (now - speech_end_at) * 1000
                    end_to_final.append(delay_ms)
                    logger.info(
                        "[확정] %s  <<< 발화종료->FINAL %.0f ms",
                        event.text, delay_ms,
                    )
                    speech_end_at = None
                else:
                    logger.info("[확정] %s  (SPEECH_END 누락, 측정 제외)", event.text)
    finally:
        _summary("발화종료 -> FINAL (진짜 STT 지연, NFR 150~400ms)", end_to_final)
        _summary("발화시작 -> 첫 INTERIM (반응성, 참고)", begin_to_interim)


async def main() -> None:
    project_id = settings.google_project_id
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

    logger.info("=" * 50)
    logger.info("STT 설정")
    logger.info("  region   : %s", settings.google_stt_location)
    logger.info("  model    : %s", settings.google_stt_model)
    logger.info("  language : %s", settings.google_stt_language)
    logger.info("  endpoint : %s-speech.googleapis.com", settings.google_stt_location)
    logger.info("=" * 50)

    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
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
