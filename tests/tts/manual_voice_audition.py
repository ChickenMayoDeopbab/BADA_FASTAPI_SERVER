import argparse
import asyncio
import pathlib
import struct
import time
from collections.abc import AsyncIterator

import numpy as np

from app.core.config import get_settings
from app.core.enums import Difficulty
from app.schemas.llm import AiEmotion
from app.services.tts import ElevenLabsTTSClient

SAMPLE_RATE = 16000
OUT_DIR = pathlib.Path("tests/output")

AUDITION_TEXT = (
    "손님, 지금 몇 번째 말씀드려요. 그 메뉴는 재고가 없다니까요? "
    "바쁜 시간에 자꾸 이러시면 저도 곤란해요."
)

_SILENCE_AMPLITUDE = 500
_MIN_SILENCE_MS = 120


async def _text_once(text: str) -> AsyncIterator[str]:
    yield text


def _write_wav(path: pathlib.Path, pcm: bytes) -> None:
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, SAMPLE_RATE, SAMPLE_RATE * 2, 2, 16)
    header += b"data" + struct.pack("<I", len(pcm))
    path.write_bytes(header + pcm)


def _count_internal_silences(pcm: bytes) -> int:
    audio = np.frombuffer(pcm, dtype=np.int16)
    if audio.size == 0:
        return 0
    loud = np.abs(audio) > _SILENCE_AMPLITUDE
    if not loud.any():
        return 0
    body = loud[np.argmax(loud) : len(loud) - np.argmax(loud[::-1])]

    min_samples = int(SAMPLE_RATE * _MIN_SILENCE_MS / 1000)
    count = 0
    run = 0
    for is_loud in body:
        if is_loud:
            if run >= min_samples:
                count += 1
            run = 0
        else:
            run += 1
    return count


async def _audition(voice_id: str, difficulty: Difficulty | None) -> dict:
    client = ElevenLabsTTSClient(get_settings(), difficulty)
    session = await client.open(voice_id)
    chunks: list[bytes] = []
    started = time.monotonic()
    first_pcm_at: float | None = None
    try:
        await session.begin(AiEmotion.ANGRY)
        async for pcm in session.stream(_text_once(AUDITION_TEXT)):
            if first_pcm_at is None:
                first_pcm_at = time.monotonic()
            chunks.append(pcm)
    finally:
        await session.aclose()

    pcm_all = b"".join(chunks)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{voice_id}.wav"
    _write_wav(out, pcm_all)

    return {
        "voice_id": voice_id,
        "ttfb_ms": None if first_pcm_at is None else (first_pcm_at - started) * 1000,
        "duration_s": len(pcm_all) / 2 / SAMPLE_RATE,
        "silences": _count_internal_silences(pcm_all),
        "path": out,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="보이스 오디션")
    parser.add_argument("voice_ids", nargs="+")
    parser.add_argument(
        "--difficulty",
        choices=[d.value for d in Difficulty],
        default=None,
        help="지정하면 그 난이도의 voice_settings 로 합성한다(하=완화 확인용)",
    )
    args = parser.parse_args()
    difficulty = Difficulty(args.difficulty) if args.difficulty else None

    print(f"대사: {AUDITION_TEXT}")
    print(f"난이도: {difficulty.value if difficulty else '(없음 — 현행 파라미터)'}\n")
    print(f"{'voice_id':<26} {'TTFB':>8} {'길이':>8} {'내부침묵':>8}  파일")
    print("-" * 78)

    for voice_id in args.voice_ids:
        try:
            r = await _audition(voice_id, difficulty)
        except Exception as exc:
            print(f"{voice_id:<26} {'실패':>8}  {type(exc).__name__}: {exc}")
            continue
        ttfb = f"{r['ttfb_ms']:.0f}ms" if r["ttfb_ms"] is not None else "-"
        print(
            f"{r['voice_id']:<26} {ttfb:>8} {r['duration_s']:>7.2f}s "
            f"{r['silences']:>8}  {r['path']}"
        )

    print("\n등록 전 확인: 한국어 발음이 어색하지 않은지, 길이가 유독 길지 않은지.")


if __name__ == "__main__":
    asyncio.run(main())
