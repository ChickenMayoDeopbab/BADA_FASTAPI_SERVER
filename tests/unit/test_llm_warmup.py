import asyncio

import pytest

from app.services.llm import LLMClient


class _FakeModels:
    def __init__(self, exc: BaseException | None = None) -> None:
        self.calls = 0
        self.last_kwargs: dict | None = None
        self._exc = exc

    async def generate_content_stream(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if self._exc is not None:
            raise self._exc

        async def _gen():
            yield object()
            yield object()

        return _gen()


class _FakeAio:
    def __init__(self, models: _FakeModels) -> None:
        self.models = models


class _FakeClient:
    def __init__(self, models: _FakeModels) -> None:
        self.aio = _FakeAio(models)


def _make_llm(models: _FakeModels) -> LLMClient:
    c = LLMClient.__new__(LLMClient)
    c._client = _FakeClient(models)  # type: ignore[attr-defined]
    c._model = "test-model"  # type: ignore[attr-defined]
    return c


@pytest.mark.asyncio
async def test_warmup_calls_stream_once_with_model() -> None:
    """워밍업은 throwaway 스트림을 정확히 1회 연다."""
    models = _FakeModels()
    await _make_llm(models).warmup()
    assert models.calls == 1
    assert models.last_kwargs is not None
    assert models.last_kwargs["model"] == "test-model"


@pytest.mark.asyncio
async def test_warmup_swallows_api_errors() -> None:
    """워밍업 실패는 무해해야 한다(첫 턴 경로에 영향 없음)."""
    models = _FakeModels(exc=RuntimeError("boom"))
    await _make_llm(models).warmup()  # 예외가 새어나오면 실패
    assert models.calls == 1


@pytest.mark.asyncio
async def test_warmup_propagates_cancellation() -> None:
    """teardown 취소는 정상 전파되어야 한다(삼켜서 매달리면 안 됨)."""
    models = _FakeModels(exc=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await _make_llm(models).warmup()
