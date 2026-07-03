import pytest

from app.schemas.llm import (
    AiPersonality,
    LLMEventType,
    ScenarioTurn,
    TurnContext,
)
from app.services.llm import LLMClient
from app.services.llm_prompt import build_contents, build_system_prompt


def _ctx(step: int = 1, history: list[dict] | None = None) -> TurnContext:
    return TurnContext(
        personality=AiPersonality.NORMAL,
        scenario_title="음식점 예약",
        scenario_role="사장님",
        script=[
            ScenarioTurn(step=1, ai_goal="전화를 받는다"),
            ScenarioTurn(step=2, ai_goal="예약 정보를 묻는다"),
            ScenarioTurn(step=3, ai_goal="예약을 확정한다"),
        ],
        current_step=step,
        history=history or [],
        user_utterance="여보세요",
    )

def test_system_prompt_is_invariant_across_steps() -> None:
    """step 이 바뀌어도 시스템 프롬프트가 동일해야 implicit 캐시 프리픽스가 산다."""
    prompts = {build_system_prompt(_ctx(step=s)) for s in (1, 2, 3)}
    assert len(prompts) == 1


def test_system_prompt_has_no_current_step_marker() -> None:
    assert "지금 이 단계" not in build_system_prompt(_ctx(step=2))


def test_system_prompt_explains_step_annotation() -> None:
    """마커 대신 '지금 단계' 주석 규약을 시스템 프롬프트가 설명해야 한다."""
    assert "지금 단계" in build_system_prompt(_ctx())


def test_system_prompt_still_lists_script_steps() -> None:
    prompt = build_system_prompt(_ctx())
    assert "1. 전화를 받는다" in prompt
    assert "3. 예약을 확정한다" in prompt


def test_contents_tail_carries_current_step() -> None:
    contents = build_contents(_ctx(step=2))
    tail = contents[-1]
    assert tail["role"] == "user"
    text = tail["parts"][0]["text"]
    assert "(지금 단계: 2)" in text
    assert "여보세요" in text


def test_contents_history_stays_raw() -> None:
    history = [
        {"role": "user", "text": "여보세요"},
        {"role": "assistant", "text": "네, 사장입니다"},
    ]
    contents = build_contents(_ctx(step=2, history=history))
    assert contents[0]["parts"][0]["text"] == "여보세요"
    assert contents[1]["parts"][0]["text"] == "네, 사장입니다"
    assert contents[1]["role"] == "model"

class _Usage:
    def __init__(self, prompt: int, cached: int | None) -> None:
        self.prompt_token_count = prompt
        self.cached_content_token_count = cached


class _Chunk:
    def __init__(self, text: str, usage: _Usage | None = None) -> None:
        self.text = text
        self.usage_metadata = usage
        self.prompt_feedback = None
        self.candidates = []


class _FakeModels:
    def __init__(self, chunks: list[_Chunk]) -> None:
        self._chunks = chunks

    async def generate_content_stream(self, **kwargs):
        async def _gen():
            for chunk in self._chunks:
                yield chunk

        return _gen()


def _make_llm(chunks: list[_Chunk]) -> LLMClient:
    class _Aio:
        models = _FakeModels(chunks)

    class _Client:
        aio = _Aio()

    c = LLMClient.__new__(LLMClient)
    c._client = _Client()
    c._model = "test-model"
    return c


@pytest.mark.asyncio
async def test_turn_end_carries_usage_tokens() -> None:
    chunks = [
        _Chunk("[EMOTION:NEUTRAL]\n안녕"),
        _Chunk("하세요", usage=_Usage(prompt=1500, cached=1200)),
    ]
    events = [ev async for ev in _make_llm(chunks).stream(_ctx())]
    turn_end = next(ev for ev in events if ev.type == LLMEventType.TURN_END)
    assert turn_end.prompt_tokens == 1500
    assert turn_end.cached_tokens == 1200


@pytest.mark.asyncio
async def test_turn_end_usage_defaults_to_zero_cached() -> None:
    chunks = [_Chunk("[EMOTION:NEUTRAL]\n네", usage=_Usage(prompt=900, cached=None))]
    events = [ev async for ev in _make_llm(chunks).stream(_ctx())]
    turn_end = next(ev for ev in events if ev.type == LLMEventType.TURN_END)
    assert turn_end.prompt_tokens == 900
    assert turn_end.cached_tokens == 0


@pytest.mark.asyncio
async def test_turn_end_usage_none_when_absent() -> None:
    chunks = [_Chunk("[EMOTION:NEUTRAL]\n네")]
    events = [ev async for ev in _make_llm(chunks).stream(_ctx())]
    turn_end = next(ev for ev in events if ev.type == LLMEventType.TURN_END)
    assert turn_end.prompt_tokens is None
    assert turn_end.cached_tokens is None
