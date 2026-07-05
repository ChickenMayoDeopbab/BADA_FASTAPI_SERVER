"""LLMClient.praise_segments / _parse_numbered 테스트 (더미 응답 사용)."""
from __future__ import annotations

import asyncio

import pytest

from app.services.llm import LLMClient


class _Resp:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeModels:
    def __init__(self, text: str = "", exc: BaseException | None = None) -> None:
        self.calls = 0
        self.last_kwargs: dict | None = None
        self._text = text
        self._exc = exc

    async def generate_content(self, **kwargs):
        self.calls += 1
        self.last_kwargs = kwargs
        if self._exc is not None:
            raise self._exc
        return _Resp(self._text)


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


# ----------------------------- _parse_numbered -----------------------------

def test_parse_numbered_strips_numbering_and_quotes():
    raw = '1. "인사를 잘했어요"\n2) 상황 설명을 잘했어요'
    assert LLMClient._parse_numbered(raw, 2) == [
        "인사를 잘했어요",
        "상황 설명을 잘했어요",
    ]


def test_parse_numbered_pads_when_too_few():
    # 발화 3개인데 응답이 1줄뿐 → 길이 3으로 빈 문자열 채움
    assert LLMClient._parse_numbered("1. 좋았어요", 3) == ["좋았어요", "", ""]


def test_parse_numbered_truncates_when_too_many():
    raw = "1. a\n2. b\n3. c\n4. d"
    assert LLMClient._parse_numbered(raw, 2) == ["a", "b"]


def test_parse_numbered_skips_blank_and_bullet_lines():
    raw = "- 1. a\n\n• 2. b\n"
    assert LLMClient._parse_numbered(raw, 2) == ["a", "b"]


# ----------------------------- praise_segments -----------------------------

@pytest.mark.asyncio
async def test_praise_segments_empty_skips_llm():
    """발화가 없으면 LLM을 호출하지 않고 빈 리스트."""
    models = _FakeModels(text="unused")
    out = await _make_llm(models).praise_segments([])
    assert out == []
    assert models.calls == 0


@pytest.mark.asyncio
async def test_praise_segments_returns_one_per_utterance():
    """구간 전체를 LLM 1회 호출로 처리하고 발화당 칭찬 1개를 순서대로 반환."""
    models = _FakeModels(text="1. 인사를 잘했어요\n2. 설명을 잘했어요")
    out = await _make_llm(models).praise_segments(["안녕하세요", "저는 학생입니다"])
    assert out == ["인사를 잘했어요", "설명을 잘했어요"]
    assert models.calls == 1
    assert models.last_kwargs is not None
    assert models.last_kwargs["model"] == "test-model"
    # 발화가 번호 매겨져 그대로 전달됐는지
    assert models.last_kwargs["contents"] == "1. 안녕하세요\n2. 저는 학생입니다"


@pytest.mark.asyncio
async def test_praise_segments_error_returns_blank_list_same_length():
    """LLM 실패 시 세션 종료를 막지 않도록 길이 보존한 빈 문자열 리스트."""
    models = _FakeModels(exc=RuntimeError("boom"))
    out = await _make_llm(models).praise_segments(["a", "b", "c"])
    assert out == ["", "", ""]


@pytest.mark.asyncio
async def test_praise_segments_propagates_cancellation():
    """teardown 취소는 정상 전파되어야 한다(삼켜서 매달리면 안 됨)."""
    models = _FakeModels(exc=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await _make_llm(models).praise_segments(["a"])
