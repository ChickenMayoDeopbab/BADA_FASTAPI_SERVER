import pytest
from google.genai import errors, types

from app.core.config import Settings, get_settings
from app.services.llm import (
    _THINKING_OFF,
    LLMClient,
    _off_candidates,
    _thinking_off,
)


def _llm(model: str, budget: int | None = None) -> LLMClient:
    c = LLMClient.__new__(LLMClient)
    c._model = model
    c._thinking_budget = budget
    return c


@pytest.mark.parametrize(
    "model",
    ["gemini-2.5-flash", "gemini-3.7-flash", "gemini-3.8-flash", "gemini-flash-latest"],
)
def test_table_maps_these_to_zero_budget(model: str) -> None:
    cfg = _thinking_off(model)
    assert cfg is not None
    assert cfg.thinking_budget == 0


@pytest.mark.parametrize(
    "model",
    [
        "gemini-3.5-flash-lite",
        "gemini-3.1-flash-lite",
        "gemini-3.6-flash",
        "gemini-flash-lite-latest",
    ],
)
def test_table_maps_these_to_minimal_level(model: str) -> None:
    cfg = _thinking_off(model)
    assert cfg is not None
    assert cfg.thinking_budget is None
    assert cfg.thinking_level == types.ThinkingLevel.MINIMAL


def test_generation_alone_does_not_decide() -> None:
    assert _thinking_off("gemini-3.7-flash").thinking_budget == 0
    assert _thinking_off("gemini-3.5-flash-lite").thinking_budget is None


def test_non_lite_does_not_imply_zero_budget() -> None:
    assert _thinking_off("gemini-3.5-flash").thinking_budget == 0
    assert _thinking_off("gemini-3.6-flash").thinking_budget is None
    assert _thinking_off("gemini-3.7-flash").thinking_budget == 0


@pytest.mark.parametrize("model", ["gemini-3.1-pro-preview", "custom-model", ""])
def test_unmeasured_model_tries_candidates_instead_of_giving_up(model: str, caplog) -> None:
    import logging

    from app.services.llm import _MINIMAL_LEVEL, _ZERO_BUDGET, _off_candidates

    with caplog.at_level(logging.WARNING, logger="app.services.llm"):
        candidates = _off_candidates(model)
    assert candidates == (_ZERO_BUDGET, _MINIMAL_LEVEL, None)
    assert _thinking_off(model) is _ZERO_BUDGET
    assert any("thinking" in r.getMessage() for r in caplog.records)


def test_known_model_costs_no_extra_round_trip() -> None:
    assert _off_candidates("gemini-3.5-flash-lite")[0] is _thinking_off("gemini-3.5-flash-lite")
    assert len(_off_candidates("gemini-3.5-flash-lite")) == 2


def test_every_candidate_chain_ends_by_sending_nothing() -> None:
    for model in (*_THINKING_OFF, "모르는-모델"):
        assert _off_candidates(model)[-1] is None


def test_default_realtime_model_is_in_the_table() -> None:
    assert get_settings().llm_realtime_model in _THINKING_OFF


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


class _RejectingModels(_CapturingModels):
    def __init__(self) -> None:
        super().__init__()
        self.seen: list[object] = []

    async def generate_content(self, **kwargs):
        tc = kwargs["config"].thinking_config
        self.seen.append(tc)
        if tc is not None:
            raise errors.ClientError(
                400, {"error": {"message": "Budget", "status": "INVALID_ARGUMENT"}}
            )
        return await super().generate_content(**kwargs)


@pytest.mark.asyncio
async def test_segment_feedback_retries_without_thinking_when_rejected(caplog) -> None:
    import logging

    llm = _llm("gemini-2.5-flash")
    models = _RejectingModels()
    llm._client = type("C", (), {"aio": type("A", (), {"models": models})()})()
    with caplog.at_level(logging.WARNING, logger="app.services.llm"):
        pairs = await llm.segment_feedback([{"type": "GOOD", "utterance": "안녕하세요", "avti": None}])
    assert pairs, "400 후 재시도로 살아나야 한다"
    assert models.seen[0] is not None and models.seen[1] is None


@pytest.mark.asyncio
async def test_segment_feedback_warns_when_output_is_truncated(caplog) -> None:
    import logging

    class _Truncating(_CapturingModels):
        async def generate_content(self, **kwargs):
            self.kwargs = kwargs
            return type("R", (), {
                "text": "1. 잘했어요 |",
                "candidates": [
                    type("C", (), {"finish_reason": types.FinishReason.MAX_TOKENS})()
                ],
            })()

    llm = _llm("gemini-2.5-flash")
    models = _Truncating()
    llm._client = type("C", (), {"aio": type("A", (), {"models": models})()})()
    with caplog.at_level(logging.WARNING, logger="app.services.llm"):
        await llm.segment_feedback([{"type": "GOOD", "utterance": "안녕하세요", "avti": None}])
    assert any("잘렸" in r.getMessage() for r in caplog.records)


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


@pytest.mark.asyncio
async def test_feedback_cap_leaves_room_for_unmapped_models() -> None:
    llm, models = _with_capture("gemini-3.5-flash-lite")
    await llm.segment_feedback([{"type": "GOOD", "utterance": "안녕하세요", "avti": None}])
    assert models.kwargs["config"].max_output_tokens >= 1536
