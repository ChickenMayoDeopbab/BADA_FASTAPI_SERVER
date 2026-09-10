"""STT 비교 측정용 발화 오디오 생성 (계획 0052 F76).

ElevenLabs 로 '사용자 역할' 문장을 16kHz/mono/int16 wav 로 만들고 manifest.json(파일 → 정답 텍스트)을 쓴다.
학습자 실발화가 아니라 TTS 낭독이라는 한계는 리포트에 적는다. 두 엔진에 같은 오디오를 넣는 것이 목적.

    .venv/bin/python -m tests.perf.make_stt_utterances .harness/stt-compare-2026-09/audio
"""
import asyncio
import json
import sys
import wave
from collections.abc import AsyncIterator
from pathlib import Path

from app.core.config import get_settings
from app.schemas.llm import AiEmotion
from app.services.tts import ElevenLabsTTSClient

SAMPLE_RATE = 16000

# 시드 세션(병원 예약 변경)의 사용자 쪽 발화. 이름·날짜·전화번호·되묻기·군말 포함.
UTTERANCES: dict[str, str] = {
    "u01_greeting": "안녕하세요, 예약 변경하려고 전화드렸는데요.",
    "u02_identity": "이름은 김민준이고요, 생년월일은 1998년 3월 15일이에요.",
    "u03_reschedule": "다음 주 화요일 오후 세 시로 바꿀 수 있을까요?",
    "u04_repeat": "죄송한데 다시 한번 말씀해 주시겠어요?",
    "u05_alternative": "음, 그러면 수요일 오전 열 시 반은 어떠세요?",
    "u06_phone": "전화번호는 010-2345-6789예요.",
    "u07_question": "혹시 진료 전에 준비해야 할 게 있나요?",
    "u08_closing": "네, 알겠습니다. 감사합니다. 안녕히 계세요.",
}


async def _once(text: str) -> AsyncIterator[str]:
    yield text


async def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    client = ElevenLabsTTSClient(get_settings())
    manifest: dict[str, str] = {}
    for name, text in UTTERANCES.items():
        chunks: list[bytes] = []
        async for pcm in client.stream(_once(text), AiEmotion.NEUTRAL):
            chunks.append(pcm)
        pcm_all = b"".join(chunks)
        if not pcm_all:
            raise SystemExit(f"PCM 비어있음: {name}")
        path = out_dir / f"{name}.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm_all)
        manifest[path.name] = text
        print(f"{path.name}  {len(pcm_all) / (SAMPLE_RATE * 2):.2f}s  {text}")
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    asyncio.run(main(Path(sys.argv[1])))
