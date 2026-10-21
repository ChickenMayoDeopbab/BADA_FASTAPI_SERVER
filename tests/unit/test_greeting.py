import pytest

from app.core.preset_scenarios import PRESET_MAP
from app.services import greeting as greeting_mod
from app.services.greeting import GREETING_FALLBACK, GREETING_MAX_CHARS, greeting_enabled, pick_greeting
from tests.unit.test_qwen_realtime_tts import _real_settings


class _Settings:
    def __init__(self, on: bool, ids: str = "") -> None:
        self.ai_greeting = on
        self.ai_greeting_user_ids = ids


@pytest.mark.parametrize(
    "on, ids, user_id, expected",
    [
        (False, "", 81, False),
        (True, "", 81, True),
        (True, " , ", 81, True),
        (True, "81, 90", 90, True),
        (True, "90", 81, False),
        (True, "81", None, False),
    ],
)
def test_switch_and_allowlist(on, ids, user_id, expected) -> None:
    assert greeting_enabled(_Settings(on, ids), user_id) is expected


def test_missing_settings_means_off() -> None:
    assert greeting_enabled(None, 81) is False
    assert greeting_enabled(object(), 81) is False


def test_env_reaches_settings(monkeypatch) -> None:
    for name in ("AI_GREETING", "AI_GREETING_USER_IDS"):
        monkeypatch.delenv(name, raising=False)
    default = _real_settings()
    assert default.ai_greeting is False and default.ai_greeting_user_ids == ""
    monkeypatch.setenv("AI_GREETING", "true")
    monkeypatch.setenv("AI_GREETING_USER_IDS", "81, 90")
    settings = _real_settings()
    assert greeting_enabled(settings, 90) and not greeting_enabled(settings, 7)


@pytest.mark.parametrize("scenario_id", sorted(PRESET_MAP))
async def test_preset_greets_with_its_example_first_line(scenario_id) -> None:
    line = PRESET_MAP[scenario_id]["example_dialogue"][0]["text"]
    assert await pick_greeting(scenario_id) == (line, "preset")
    assert line.endswith("?"), "프리셋 첫마디는 사용자에게 묻는 말이다"


async def test_custom_greets_with_its_example_first_line(monkeypatch) -> None:
    async def _dialogue(scenario_id):
        assert scenario_id == 501
        return [{"speaker": "AI", "text": " 네, 김 부장입니다. "}, {"speaker": "user", "text": "부장님, 저예요."}]

    monkeypatch.setattr(greeting_mod, "_custom_dialogue", _dialogue)
    assert await pick_greeting(501) == ("네, 김 부장입니다.", "custom")


@pytest.mark.parametrize(
    "dialogue",
    [
        None,
        [],
        "네, 여보세요",
        [{"speaker": "user", "text": "여보세요?"}, {"speaker": "ai", "text": "네."}],
        [{"speaker": "ai", "text": "   "}],
        [{"speaker": "ai", "text": "가" * (GREETING_MAX_CHARS + 1)}],
        ["네, 여보세요"],
    ],
)
async def test_custom_without_a_usable_first_ai_line_uses_fallback(monkeypatch, dialogue) -> None:
    async def _dialogue(_scenario_id):
        return dialogue

    monkeypatch.setattr(greeting_mod, "_custom_dialogue", _dialogue)
    assert await pick_greeting(501) == (GREETING_FALLBACK, "fallback")


async def test_lookup_failure_uses_fallback(monkeypatch) -> None:
    async def _boom(_scenario_id):
        raise RuntimeError("db down")

    monkeypatch.setattr(greeting_mod, "_custom_dialogue", _boom)
    assert await pick_greeting(501) == (GREETING_FALLBACK, "fallback")


async def test_slow_lookup_uses_fallback(monkeypatch) -> None:
    import asyncio

    async def _slow(_scenario_id):
        await asyncio.Event().wait()

    monkeypatch.setattr(greeting_mod, "_LOOKUP_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(greeting_mod, "_custom_dialogue", _slow)
    assert await pick_greeting(501) == (GREETING_FALLBACK, "fallback")


async def test_unknown_scenario_uses_fallback_without_lookup(monkeypatch) -> None:
    calls: list = []

    async def _record(scenario_id):
        calls.append(scenario_id)

    monkeypatch.setattr(greeting_mod, "_custom_dialogue", _record)
    assert await pick_greeting(None) == (GREETING_FALLBACK, "fallback")
    assert calls == [], "scenarioId 가 없으면 조회하지 않는다"


def test_fallback_is_a_short_question() -> None:
    assert GREETING_FALLBACK == "네, 여보세요?"
