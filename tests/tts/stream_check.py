import asyncio
import logging
import wave
from collections.abc import AsyncIterator
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("tts_stream_check")

from app.core.config import get_settings
from app.services.tts import ElevenLabsTTSClient

OUT_DIR = Path(__file__).resolve().parent.parent / "output"
OUT_WAV = OUT_DIR / "tts_check_output.wav"
SAMPLE_RATE = 16000
TEXT = "안녕하세요, 무엇을 도와드릴까요?"


async def text_once(text: str) -> AsyncIterator[str]:
    """텍스트 한 줄을 흘려주는 최소 source. (오케스트레이터의 LLM 스트림 대역)"""
    yield text


def write_wav(pcm: bytes, path: str) -> None:
    """raw PCM 16k/16bit/mono -> WAV 헤더 씌워 저장."""
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16bit = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)


async def main() -> None:
    settings = get_settings()

    logger.info("=" * 50)
    logger.info("TTS 설정")
    logger.info("  model        : %s", settings.elevenlabs_model)
    logger.info("  voice_id     : %s", settings.elevenlabs_voice_id)
    logger.info("  output_format: %s", settings.elevenlabs_output_format)
    logger.info("  language     : %s", settings.elevenlabs_language_code)
    logger.info("  auto_mode    : %s", settings.elevenlabs_auto_mode)
    logger.info("=" * 50)

    client = ElevenLabsTTSClient(settings)

    chunks: list[bytes] = []
    chunk_count = 0
    total_bytes = 0
    first_byte_at: float | None = None
    start = asyncio.get_event_loop().time()

    logger.info("TTS 호출 시작: %r", TEXT)
    try:
        async for pcm in client.stream(text_once(TEXT)):
            now = asyncio.get_event_loop().time()
            if first_byte_at is None:
                first_byte_at = now
                logger.info("TTFB(첫 PCM까지) = %.0fms", (now - start) * 1000)
            chunk_count += 1
            total_bytes += len(pcm)
            chunks.append(pcm)
        logger.info("스트림 정상 종료")
    except Exception:
        logger.exception("TTS 에러 발생")
        return

    pcm_all = b"".join(chunks)
    duration_sec = total_bytes / (SAMPLE_RATE * 2)  # 16bit mono
    logger.info(
        "수신 완료: 청크 %d개, %d bytes, 약 %.2f초 오디오",
        chunk_count, total_bytes, duration_sec,
    )

    if pcm_all:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        write_wav(pcm_all, str(OUT_WAV))
        logger.info("저장: %s (이 파일 재생해서 들어보세요)", OUT_WAV)
    else:
        logger.warning("받은 PCM이 비어있음. 메시지 형식/쿼리 파라미터 확인 필요.")


if __name__ == "__main__":
    asyncio.run(main())
