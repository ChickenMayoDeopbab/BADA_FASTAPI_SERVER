"""ElevenLabs 운영 경로로 벤치 문장·턴을 합성한다 (맥, 레포 venv).

  .venv/bin/python scripts/tts_bench/gen_el.py --out ~/school/narsha/2026/qwen-tests/el_vs_qwen/out
운영 ElevenLabsTTSClient 와 레포 .env 를 그대로 쓴다(감정 NEUTRAL). 문장마다 세션 하나 → el/<id>.wav,
턴마다 세션 하나(문장을 차례로 흘려 넣음) → el_turns/<tid>.wav. 이미 있는 파일은 건너뛴다.
"""

import argparse
import asyncio
import json
import sys
import wave
from collections.abc import AsyncIterator, Iterable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from websockets.exceptions import WebSocketException  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.schemas.llm import AiEmotion  # noqa: E402
from app.services.tts import ElevenLabsTTSClient  # noqa: E402

HERE = Path(__file__).resolve().parent
SAMPLE_RATE = 16000
RETRIES = 2
RETRY_WAIT_S = 2.0


def load_tsv(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            key, value = line.split("\t", 1)
            rows[key] = value
    return rows


def write_wav(path: Path, pcm: bytes, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm[: len(pcm) // 2 * 2])
    tmp.replace(path)


def turn_parts(sentence_ids: Iterable[str], sentences: dict[str, str]) -> list[str]:
    """LLM 스트림처럼 문장 사이에 공백을 둔다 — 운영 분할기가 문장마다 한 번씩 보낸다"""
    return [sentences[sid] + " " for sid in sentence_ids]


async def _feed(parts: list[str]) -> AsyncIterator[str]:
    for part in parts:
        yield part


async def synth(client: ElevenLabsTTSClient, parts: list[str], voice_id: str | None) -> tuple[bytes, int]:
    """파이프라인 _open_tts → begin → stream 과 같은 순서"""
    session = await client.open(voice_id)
    try:
        await session.begin(AiEmotion.NEUTRAL)
        pcm = b"".join([chunk async for chunk in session.stream(_feed(parts))])
        return pcm, session.chars_sent
    finally:
        await session.aclose()


async def run(args: argparse.Namespace, client: ElevenLabsTTSClient) -> int:
    sentences = load_tsv(args.sentences)
    turns = {} if args.no_turns else {k: v.split() for k, v in load_tsv(args.turns).items()}
    only = set(args.only.split(",")) if args.only else None
    jobs = [("el", sid, [text]) for sid, text in sentences.items() if not only or sid in only]
    jobs += [("el_turns", tid, turn_parts(ids, sentences)) for tid, ids in turns.items() if not only or tid in only]
    failed = 0
    for folder, key, parts in jobs:
        path = args.out / folder / f"{key}.wav"
        if path.exists() and not args.force:
            continue
        for attempt in range(RETRIES + 1):
            try:
                pcm, chars = await synth(client, parts, args.voice_id)
                break
            except (WebSocketException, OSError) as e:
                print(f"[{folder}/{key}] 실패 {attempt + 1}회: {type(e).__name__}: {e}", flush=True)
                await asyncio.sleep(RETRY_WAIT_S * (attempt + 1))
        else:
            failed += 1
            continue
        if len(pcm) < 2:
            print(f"[{folder}/{key}] 빈 소리", flush=True)
            failed += 1
            continue
        write_wav(path, pcm)
        row = {"id": key, "audio_s": round(len(pcm) / 2 / SAMPLE_RATE, 3), "chars": chars}
        with open(args.out / folder / "gen.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{folder}/{key}] {row['audio_s']:5.2f}s · {chars}자 · {''.join(parts)[:30]}", flush=True)
    print(f"끝 — 작업 {len(jobs)} · 실패 {failed} → {args.out}", flush=True)
    return 1 if failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="ElevenLabs 운영 경로로 벤치 문장·턴 합성")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sentences", type=Path, default=HERE / "sentences.tsv")
    ap.add_argument("--turns", type=Path, default=HERE / "turns.tsv")
    ap.add_argument("--only", help="이 id 들만(쉼표, 문장·턴 공통)")
    ap.add_argument("--no-turns", action="store_true")
    ap.add_argument("--voice-id", help="기본은 .env ELEVENLABS_VOICE_ID(프리셋 시나리오)")
    ap.add_argument("--force", action="store_true", help="있는 파일도 다시")
    args = ap.parse_args()
    args.out = args.out.expanduser()
    settings = get_settings()
    assert settings.elevenlabs_output_format == "pcm_16000", "wav 헤더가 16kHz PCM 이라 운영 형식이 바뀌면 고칠 것"
    print(f"모델 {settings.elevenlabs_model} · {settings.elevenlabs_output_format} · "
          f"보이스 {args.voice_id or settings.elevenlabs_voice_id}", flush=True)
    return asyncio.run(run(args, ElevenLabsTTSClient(settings)))


if __name__ == "__main__":
    sys.exit(main())
