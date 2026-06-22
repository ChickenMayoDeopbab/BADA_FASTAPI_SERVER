import asyncio
import logging
from collections.abc import AsyncIterator

import numpy as np
import sounddevice as sd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("tts_manual")

from app.core.config import get_settings
from app.schemas.llm import AiEmotion
from app.services.tts import ElevenLabsTTSClient

SAMPLE_RATE = 16000

TEST_TEXTS = [
    "안녕하세요, 무엇을 도와드릴까요?",
    "예약은 토요일 저녁 7시 30분으로 잡아드릴게요.",
    "확인 전화는 010-1234-5678로 드리겠습니다.",
    "총 3명이고, 금액은 45000원입니다.",
]

async def text_once(text: str) -> AsyncIterator[str]:
    yield text


async def synth_and_play(client: ElevenLabsTTSClient, text: str, emotion: AiEmotion.NEUTRAL) -> None:
    logger.info("입력: %r [%s]", text, emotion.value)
    chunks: list[bytes] = []
    start = asyncio.get_event_loop().time()
    first = None

    async for pcm in client.stream(text_once(text), emotion):
        if first is None:
            first = asyncio.get_event_loop().time()
            logger.info("  TTFB = %.0fms", (first - start) * 1000)
        chunks.append(pcm)

    pcm_all = b"".join(chunks)
    if not pcm_all:
        logger.warning("  PCM 비어있음")
        return

    audio = np.frombuffer(pcm_all, dtype=np.int16)
    pad = np.zeros(int(SAMPLE_RATE * 0.3), dtype=np.int16)  # 0.3초 무음
    audio = np.concatenate([audio, pad])
    sd.play(audio, SAMPLE_RATE)
    sd.wait()
    sd.stop()
    logger.info("  재생 완료 (%.2f초)", len(audio) / SAMPLE_RATE)


async def main() -> None:
    settings = get_settings()
    client = ElevenLabsTTSClient(settings)
    text = "예약은 토요일 저녁 7시 30분으로 잡아드릴게요."
    logger.info("voice_id=%s, model=%s", settings.elevenlabs_voice_id, settings.elevenlabs_model)

    # for emotion in AiEmotion:
    #     await synth_and_play(client, text, emotion)
    #     await asyncio.sleep(0.5)

    for text in TEST_TEXTS:
        try:
            await synth_and_play(client, text, AiEmotion.NEUTRAL)
        except Exception:
            logger.exception("합성/재생 실패: %r", text)
        await asyncio.sleep(0.3)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("종료")
