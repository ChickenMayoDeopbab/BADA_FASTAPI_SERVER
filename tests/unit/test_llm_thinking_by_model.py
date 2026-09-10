import pytest
from google.genai import types

from app.core.config import Settings, get_settings
from app.services.llm import LLMClient, _minimal_thinking


def _llm(model: str, budget: int | None = None) -> LLMClient:
    c = LLMClient.__new__(LLMClient)
    c._model = model
    c._thinking_budget = budget
    return c


@pytest.mark.parametrize("model", ["gemini-2.5-flash", "gemini-2.0-flash"])
def test_two_x_uses_zero_budget(model: str) -> None:
    cfg = _minimal_thinking(model)
    assert cfg is not None
    assert cfg.thinking_budget == 0


@pytest.mark.parametrize(
    "model", ["gemini-3.5-flash-lite", "gemini-3-flash-preview", "gemini-3.8-flash"]
)
def test_three_x_uses_minimal_level_not_zero_budget(model: str) -> None:
    cfg = _minimal_thinking(model)
    assert cfg is not None
    assert cfg.thinking_budget is None
    assert cfg.thinking_level == types.ThinkingLevel.MINIMAL


@pytest.mark.parametrize("model", ["gemini-flash-lite-latest", "custom-model", ""])
def test_unknown_generation_sends_nothing(model: str) -> None:
    assert _minimal_thinking(model) is None


def test_env_budget_zero_is_translated_per_model() -> None:
    assert _llm("gemini-2.5-flash", 0)._thinking_config().thinking_budget == 0
    three = _llm("gemini-3.5-flash-lite", 0)._thinking_config()
    assert three.thinking_budget is None
    assert three.thinking_level == types.ThinkingLevel.MINIMAL


def test_positive_budget_passes_through_unchanged() -> None:
    cfg = _llm("gemini-3.5-flash-lite", 128)._thinking_config()
    assert cfg.thinking_budget == 128


def test_budget_unset_still_omits_thinking_config() -> None:
    assert _llm("gemini-3.5-flash-lite", None)._thinking_config() is None


class _CapturingModels:
    def __init__(self) -> None:
        self.kwargs: dict | None = None

    async def generate_content(self, **kwargs):
        self.kwargs = kwargs
        return type("R", (), {"text": "1. 잘했어요 | 또박또박 말했어요."})()


def _with_capture(model: str) -> tuple[LLMClient, _CapturingModels]:
    llm = _llm(model)
    models = _CapturingModels()
    llm._client = type("C", (), {"aio": type("A", (), {"models": models})()})()
    return llm, models


@pytest.mark.asyncio
async def test_segment_feedback_uses_the_model_aware_thinking_config() -> None:
    llm, models = _with_capture("gemini-3.5-flash-lite")
    pairs = await llm.segment_feedback([{"type": "GOOD", "utterance": "안녕하세요", "avti": None}])
    assert pairs
    tc = models.kwargs["config"].thinking_config
    assert tc.thinking_budget is None
    assert tc.thinking_level == types.ThinkingLevel.MINIMAL


@pytest.mark.asyncio
async def test_segment_feedback_still_disables_thinking_on_two_x() -> None:
    llm, models = _with_capture("gemini-2.5-flash")
    await llm.segment_feedback([{"type": "GOOD", "utterance": "안녕하세요", "avti": None}])
    assert models.kwargs["config"].thinking_config.thinking_budget == 0


def test_default_realtime_model_is_not_a_retired_one() -> None:
    retired = {"gemini-2.5-flash-lite", "gemini-2.5-pro"}
    assert Settings.model_fields["llm_realtime_model"].default not in retired
    assert get_settings().llm_realtime_model not in retired
