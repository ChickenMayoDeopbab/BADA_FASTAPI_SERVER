import pytest

from app.services.llm import LLMClient


def _make_llm(budget: int | None) -> LLMClient:
    c = LLMClient.__new__(LLMClient)
    c._model = "test-model"
    c._thinking_budget = budget
    return c


def test_budget_none_omits_thinking_config() -> None:
    cfg = _make_llm(None)._build_gen_config("sys")
    assert cfg.thinking_config is None


def test_budget_zero_sets_explicit_disable() -> None:
    cfg = _make_llm(0)._build_gen_config("sys")
    assert cfg.thinking_config is not None
    assert cfg.thinking_config.thinking_budget == 0


def test_budget_positive_passes_through() -> None:
    cfg = _make_llm(128)._build_gen_config("sys")
    assert cfg.thinking_config is not None
    assert cfg.thinking_config.thinking_budget == 128


class _FakeModels:
    def __init__(self) -> None:
        self.last_kwargs: dict | None = None

    async def generate_content_stream(self, **kwargs):
        self.last_kwargs = kwargs

        async def _gen():
            yield object()

        return _gen()


class _FakeAio:
    def __init__(self, models: _FakeModels) -> None:
        self.models = models


class _FakeClient:
    def __init__(self, models: _FakeModels) -> None:
        self.aio = _FakeAio(models)


@pytest.mark.asyncio
async def test_warmup_respects_budget_setting() -> None:
    for budget, expect_none in ((None, True), (0, False)):
        models = _FakeModels()
        llm = _make_llm(budget)
        llm._client = _FakeClient(models)
        await llm.warmup()
        assert models.last_kwargs is not None
        tc = models.last_kwargs["config"].thinking_config
        assert (tc is None) is expect_none
