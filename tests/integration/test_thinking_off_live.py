import os
from pathlib import Path

import pytest
from google import genai
from google.genai import types

from app.services.llm import _THINKING_OFF

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_LLM_TESTS") != "1",
    reason="실서버 호출이라 옵트인 (RUN_LIVE_LLM_TESTS=1)",
)


def _real_api_key() -> str:
    env = Path(__file__).resolve().parents[2] / ".env"
    for line in env.read_text(encoding="utf-8").splitlines():
        name, _, value = line.partition("=")
        if name.strip() == "GEMINI_API_KEY" and value.strip():
            return value.strip()
    pytest.skip(".env 에 GEMINI_API_KEY 가 없다")


@pytest.mark.parametrize("model", sorted(_THINKING_OFF))
async def test_table_entry_is_still_accepted(model: str) -> None:
    client = genai.Client(api_key=_real_api_key())
    config = types.GenerateContentConfig(
        max_output_tokens=16, thinking_config=_THINKING_OFF[model]
    )
    await client.aio.models.generate_content(
        model=model, contents="안녕", config=config
    )
