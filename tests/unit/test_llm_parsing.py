"""LLMClient 파싱 헬퍼 테스트."""
from __future__ import annotations

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
