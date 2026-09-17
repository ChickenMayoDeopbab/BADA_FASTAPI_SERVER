import json
import logging
from types import SimpleNamespace

import pytest

from app.core import tts_voices
from app.core import usage as usage_mod
from app.core.enums import SpeakerAge, SpeakerGender, SpeakerTone
from app.core.tts_voices import VoiceProfile
from app.core.usage import PCM_BYTES_PER_SECOND, llm_usage_fields
from app.services import example_service, scenario_image_service, scenario_service
from app.services.example_service import get_example_conversation
from app.services.scenario_image_service import generate_scenario_thumbnail
from app.services.scenario_service import ScenarioGenInvalidError, create_custom_scenario
from tests.unit.test_example_service import (
    _DIALOGUE,
    _custom_row,
    _FakeDB,
    _FakeQwenClient,
    _preset_row,
    _wire,
    _wire_qwen,
)
from tests.unit.test_scenario_safety_guard import _FakeWriteDB, _ok_payload, _request, _script
from tests.unit.test_scenario_thumbnail import _IMAGE_BYTES, _FakeS3, _FakeSession, _scenario_row


def _metrics(caplog, name: str):
    return [r for r in caplog.records
            if r.name == "app.metrics" and getattr(r, "metric", None) == name]


def _anthropic_usage(input_tokens: int, output_tokens: int, **extra) -> SimpleNamespace:
    base = {"input_tokens": input_tokens, "output_tokens": output_tokens,
            "cache_read_input_tokens": None, "cache_creation_input_tokens": None}
    base.update(extra)
    return SimpleNamespace(**base)


def _anthropic_with_usage(*items: tuple[object, object]) -> type:
    recorded: list[dict] = []

    class _Messages:
        async def create(self, **kw):
            recorded.append(kw)
            payload, usage = items[min(len(recorded) - 1, len(items) - 1)]
            text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
            return SimpleNamespace(content=[SimpleNamespace(text=text)], usage=usage)

    class _Client:
        calls = recorded

        def __init__(self, api_key=None) -> None:
            self.messages = _Messages()

    return _Client


def test_llm_usage_fields_normalizes_anthropic_and_gemini() -> None:
    anthropic = SimpleNamespace(input_tokens=100, output_tokens=20,
                                cache_read_input_tokens=None, cache_creation_input_tokens=5)
    assert llm_usage_fields("anthropic", anthropic) == {
        "input_tokens": 100, "output_tokens": 20, "cache_read_tokens": 0,
        "cache_write_tokens": 5, "thought_tokens": 0,
    }
    gemini = SimpleNamespace(prompt_token_count=30, candidates_token_count=1290,
                             cached_content_token_count=None, thoughts_token_count=None)
    assert llm_usage_fields("gemini", gemini) == {
        "input_tokens": 30, "output_tokens": 1290, "cache_read_tokens": 0,
        "cache_write_tokens": 0, "thought_tokens": 0,
    }
    assert set(llm_usage_fields("anthropic", None).values()) == {0}


def _gen_env(monkeypatch) -> None:
    monkeypatch.setattr(
        scenario_service, "get_settings",
        lambda: SimpleNamespace(anthropic_api_key="k", llm_analysis_model="claude-test"),
    )
    monkeypatch.setattr(
        tts_voices, "VOICE_REGISTRY",
        [VoiceProfile("v-test", SpeakerGender.MALE, SpeakerAge.MIDDLE, SpeakerTone.NEUTRAL, "테스트")],
    )


_BAD = {"content": "c", "script": _script(2)}


@pytest.mark.asyncio
async def test_scenario_gen_logs_every_attempt_with_saved_scenario_id(caplog, monkeypatch) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    _gen_env(monkeypatch)
    monkeypatch.setattr(scenario_service, "AsyncAnthropic", _anthropic_with_usage(
        (_BAD, _anthropic_usage(700, 50)),
        (_ok_payload(), _anthropic_usage(710, 220)),
    ))

    response = await create_custom_scenario(_FakeWriteDB(), _request(), user_id=1)

    assert response.scenario.scenario_id == 123
    recs = _metrics(caplog, "llm_usage")
    assert [(r.attempt, r.ok) for r in recs] == [(1, False), (2, True)]
    assert {r.scenario_id for r in recs} == {123}
    assert {r.user_id for r in recs} == {1}
    assert {r.provider for r in recs} == {"anthropic"} and {r.model for r in recs} == {"claude-test"}
    assert {r.purpose for r in recs} == {"scenario_gen"}
    assert [r.input_tokens for r in recs] == [700, 710]
    assert [r.output_tokens for r in recs] == [50, 220]


@pytest.mark.asyncio
async def test_scenario_gen_logs_attempts_even_when_generation_fails(caplog, monkeypatch) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    _gen_env(monkeypatch)
    monkeypatch.setattr(scenario_service, "AsyncAnthropic", _anthropic_with_usage(
        (_BAD, _anthropic_usage(700, 50)),
    ))

    with pytest.raises(ScenarioGenInvalidError):
        await create_custom_scenario(_FakeWriteDB(), _request(), user_id=1)

    recs = _metrics(caplog, "llm_usage")
    assert [(r.attempt, r.ok) for r in recs] == [(1, False), (2, False)]
    assert {r.scenario_id for r in recs} == {None}, "저장 전 실패 → scenario_id 없음"


_CUSTOM_DIALOGUE = [
    {"speaker": "ai", "text": "여보세요."},
    {"speaker": "user", "text": "안녕하세요, 302호 세입자입니다."},
]


@pytest.mark.asyncio
async def test_example_dialogue_and_elevenlabs_audio_usage(caplog, monkeypatch) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    _tts, storage = _wire(monkeypatch)
    monkeypatch.setattr(example_service, "AsyncAnthropic", _anthropic_with_usage(
        (_CUSTOM_DIALOGUE, _anthropic_usage(400, 90)),
    ))
    row = _custom_row()

    resp = await get_example_conversation(_FakeDB(row), 42, user_id=7)
    assert resp.dialogue

    [llm] = _metrics(caplog, "llm_usage")
    assert llm.purpose == "example_dialogue" and llm.provider == "anthropic"
    assert llm.user_id == 7 and llm.scenario_id == 42
    assert llm.input_tokens == 400 and llm.output_tokens == 90

    [tts] = _metrics(caplog, "tts_usage")
    assert tts.engine == "eleven" and tts.purpose == "example_audio"
    assert tts.user_id == 7 and tts.scenario_id == 42 and tts.trigger == "request"
    assert tts.turns == 2
    assert tts.chars == sum(len(t["text"]) for t in _CUSTOM_DIALOGUE)
    assert tts.audio_sec == round(len(storage.uploads[0][1]) / PCM_BYTES_PER_SECOND, 3)


@pytest.mark.asyncio
async def test_example_audio_on_qwen_counts_chars_with_zero_cost_engine(caplog, monkeypatch) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    _wire(monkeypatch)
    _wire_qwen(monkeypatch, _FakeQwenClient())

    await get_example_conversation(_FakeDB(_preset_row()), 1, user_id=7)

    [tts] = _metrics(caplog, "tts_usage")
    assert tts.engine == "qwen" and tts.model == "qwen"
    assert tts.chars == sum(len(t["text"]) for t in _DIALOGUE)
    assert tts.user_id is None, "프리셋은 만든 사용자가 없다"
    assert tts.scenario_id == 1
    assert _metrics(caplog, "llm_usage") == [], "프리셋 대화는 LLM 을 안 부른다"


def _thumb_env(monkeypatch):
    from app.core.config import get_settings

    settings = get_settings().model_copy(update={"s3_bucket": "test-bucket"})
    monkeypatch.setattr(scenario_image_service, "get_settings", lambda: settings)
    monkeypatch.setattr(scenario_image_service, "_s3_client", lambda _s: _FakeS3())
    return settings


def _genai_with_usage(usage: object, *, with_image: bool = True):
    parts = [SimpleNamespace(inline_data=None, text="Here is your image")]
    if with_image:
        parts.append(SimpleNamespace(inline_data=SimpleNamespace(data=_IMAGE_BYTES, mime_type="image/png")))
    response = SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))],
        usage_metadata=usage,
    )

    class _Models:
        async def generate_content(self, **kwargs):
            return response

    class _Client:
        def __init__(self, api_key=None) -> None:
            self.aio = SimpleNamespace(models=_Models())

    return SimpleNamespace(Client=_Client)


_IMAGE_USAGE = SimpleNamespace(prompt_token_count=30, candidates_token_count=1290,
                               cached_content_token_count=None, thoughts_token_count=None)


@pytest.mark.asyncio
async def test_thumbnail_logs_prompt_and_image_usage(caplog, monkeypatch) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    settings = _thumb_env(monkeypatch)
    row = _scenario_row()
    monkeypatch.setattr(scenario_image_service, "AsyncSessionLocal", lambda: _FakeSession(row))
    monkeypatch.setattr(scenario_image_service, "AsyncAnthropic", _anthropic_with_usage(
        ("A friendly cafe owner behind a counter", _anthropic_usage(250, 40)),
    ))
    monkeypatch.setattr(scenario_image_service, "genai", _genai_with_usage(_IMAGE_USAGE))

    await generate_scenario_thumbnail(row.scenario_id)

    by_purpose = {r.purpose: r for r in _metrics(caplog, "llm_usage")}
    assert set(by_purpose) == {"thumbnail_prompt", "thumbnail_image"}
    prompt, image = by_purpose["thumbnail_prompt"], by_purpose["thumbnail_image"]
    assert prompt.provider == "anthropic" and prompt.input_tokens == 250 and prompt.output_tokens == 40
    assert prompt.user_id == 1 and prompt.scenario_id == 101
    assert image.provider == "gemini" and image.model == settings.gemini_image_model
    assert image.images == 1 and image.ok is True
    assert image.input_tokens == 30 and image.output_tokens == 1290
    assert image.user_id == 1 and image.scenario_id == 101


@pytest.mark.asyncio
async def test_thumbnail_image_without_image_part_is_logged_as_failed(caplog, monkeypatch) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    _thumb_env(monkeypatch)
    row = _scenario_row()
    monkeypatch.setattr(scenario_image_service, "AsyncSessionLocal", lambda: _FakeSession(row))
    monkeypatch.setattr(scenario_image_service, "AsyncAnthropic", _anthropic_with_usage(
        ("scene", _anthropic_usage(1, 1)),
    ))
    monkeypatch.setattr(
        scenario_image_service, "genai", _genai_with_usage(_IMAGE_USAGE, with_image=False)
    )

    await generate_scenario_thumbnail(row.scenario_id)  # 내부 예외는 삼켜진다

    [image] = [r for r in _metrics(caplog, "llm_usage") if r.purpose == "thumbnail_image"]
    assert image.ok is False and image.images == 0
    assert image.output_tokens == 1290, "실패해도 토큰은 썼다"


@pytest.mark.asyncio
async def test_usage_logging_failure_does_not_change_the_response(caplog, monkeypatch) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")

    def _boom(*_a, **_k):
        raise RuntimeError("정규화 버그")

    monkeypatch.setattr(usage_mod, "llm_usage_fields", _boom)
    _wire(monkeypatch)
    monkeypatch.setattr(example_service, "AsyncAnthropic", _anthropic_with_usage(
        (_CUSTOM_DIALOGUE, _anthropic_usage(400, 90)),
    ))

    resp = await get_example_conversation(_FakeDB(_custom_row()), 42, user_id=7)

    assert [t.text for t in resp.dialogue] == [t["text"] for t in _CUSTOM_DIALOGUE]
    assert _metrics(caplog, "llm_usage") == []
    assert len(_metrics(caplog, "tts_usage")) == 1, "TTS 기록은 정규화와 무관하게 남는다"
