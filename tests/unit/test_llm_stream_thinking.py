import logging

import pytest
from google.genai import errors, types

from app.schemas.llm import (
    AiPersonality,
    LLMEventType,
    ScenarioTurn,
    TurnContext,
)
from app.services.llm import LLMClient


def _ctx() -> TurnContext:
    return TurnContext(
        personality=list(AiPersonality)[0],
        scenario_title="치킨집 주문",
        scenario_role="치킨집 사장님",
        script=[ScenarioTurn(step=1, ai_goal="주문을 받는다")],
        current_step=1,
        history=[],
        user_utterance="후라이드 한 마리 주세요",
    )


def _chunk(text: str, *, finish: types.FinishReason | None = None):
    cand = type("Cand", (), {"finish_reason": finish})()
    return type("Chunk", (), {
        "text": text,
        "candidates": [cand],
        "usage_metadata": None,
        "prompt_feedback": None,
    })()


class _FakeStream:
    """청크를 흘리다가, 지정한 위치에서 예외를 던진다."""

    def __init__(self, chunks, raise_at: int | None = None, exc: Exception | None = None):
        self._chunks = chunks
        self._raise_at = raise_at
        self._exc = exc
        self._i = 0
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._raise_at is not None and self._i == self._raise_at:
            self._raise_at = None
            raise self._exc
        if self._i >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._i]
        self._i += 1
        return chunk


class _FakeModels:
    def __init__(self, streams):
        self._streams = list(streams)
        self.sent_thinking: list[object] = []

    async def generate_content_stream(self, **kwargs):
        self.sent_thinking.append(kwargs["config"].thinking_config)
        return self._streams.pop(0)


def _llm(model: str, budget: int | None, streams) -> tuple[LLMClient, _FakeModels]:
    client = LLMClient.__new__(LLMClient)
    client._model = model
    client._thinking_budget = budget
    models = _FakeModels(streams)
    client._client = type("C", (), {"aio": type("A", (), {"models": models})()})()
    return client, models


async def _drain(client: LLMClient) -> tuple[list[str], str]:
    events, text = [], []
    async for ev in client.stream(_ctx()):
        events.append(ev.type.name)
        if ev.type == LLMEventType.TEXT_DELTA:
            text.append(ev.text)
    return events, "".join(text)


def _rejection() -> errors.ClientError:
    return errors.ClientError(
        400, {"error": {"message": "Request contains an invalid argument.",
                        "status": "INVALID_ARGUMENT"}}
    )


@pytest.mark.asyncio
async def test_truncated_turn_is_logged(caplog) -> None:
    """잘리면 경고가 남아야 한다. 지금은 grep 할 로그가 없어 운영 발생률을 못 잰다."""
    chunks = [
        _chunk("[EMOTION: NEUTRAL]네, 후라이드"),
        _chunk(" 한 마리요", finish=types.FinishReason.MAX_TOKENS),
    ]
    client, _ = _llm("gemini-3.5-flash-lite", None, [_FakeStream(chunks)])

    with caplog.at_level(logging.WARNING, logger="app.services.llm"):
        events, text = await _drain(client)

    assert "TURN_END" in events, "잘려도 턴은 정상 종료돼야 한다"
    assert text.strip() == "네, 후라이드 한 마리요"
    assert any("잘렸" in r.getMessage() for r in caplog.records), (
        "MAX_TOKENS 인데 로그가 없다 — 잘림이 무탐지로 지나간다"
    )


@pytest.mark.asyncio
async def test_normal_turn_does_not_warn_about_truncation(caplog) -> None:
    chunks = [_chunk("[EMOTION: NEUTRAL]네, 주문 받았습니다.",
                     finish=types.FinishReason.STOP)]
    client, _ = _llm("gemini-3.5-flash-lite", None, [_FakeStream(chunks)])

    with caplog.at_level(logging.WARNING, logger="app.services.llm"):
        await _drain(client)

    assert not [r for r in caplog.records if "잘렸" in r.getMessage()]


@pytest.mark.asyncio
async def test_rejected_thinking_config_retries_without_it(caplog) -> None:
    first = _FakeStream([], raise_at=0, exc=_rejection())
    second = _FakeStream([_chunk("[EMOTION: NEUTRAL]네, 주문 받았습니다.")])
    client, models = _llm("gemini-3.5-flash-lite", 0, [first, second])

    with caplog.at_level(logging.WARNING, logger="app.services.llm"):
        events, text = await _drain(client)

    assert LLMEventType.ERROR.name not in events, "재시도로 살아나야 한다"
    assert text.strip() == "네, 주문 받았습니다."
    assert models.sent_thinking[0] is not None
    assert models.sent_thinking[1] is None, "재시도는 thinking 을 빼고 보내야 한다"


@pytest.mark.asyncio
async def test_no_retry_once_text_was_already_emitted() -> None:
    first = _FakeStream(
        [_chunk("[EMOTION: NEUTRAL]네, 후라이드")], raise_at=1, exc=_rejection()
    )
    never = _FakeStream([_chunk("두 번째로 열리면 안 된다")])
    client, models = _llm("gemini-3.5-flash-lite", 0, [first, never])

    events, text = await _drain(client)

    assert events[-1] == LLMEventType.ERROR.name
    assert "두 번째" not in text
    assert len(models.sent_thinking) == 1, "첫 청크 뒤에는 다시 열면 안 된다"


@pytest.mark.asyncio
async def test_non_400_errors_are_not_retried() -> None:
    rate_limited = errors.ClientError(
        429, {"error": {"message": "Resource exhausted", "status": "RESOURCE_EXHAUSTED"}}
    )
    first = _FakeStream([], raise_at=0, exc=rate_limited)
    never = _FakeStream([_chunk("열리면 안 된다")])
    client, models = _llm("gemini-3.5-flash-lite", 0, [first, never])

    events, _ = await _drain(client)

    assert events == [LLMEventType.ERROR.name]
    assert len(models.sent_thinking) == 1


@pytest.mark.asyncio
async def test_no_retry_when_no_thinking_config_was_sent() -> None:
    first = _FakeStream([], raise_at=0, exc=_rejection())
    never = _FakeStream([_chunk("열리면 안 된다")])
    client, models = _llm("gemini-3.5-flash-lite", None, [first, never])

    events, _ = await _drain(client)

    assert events == [LLMEventType.ERROR.name]
    assert models.sent_thinking == [None]


@pytest.mark.asyncio
async def test_warmup_finds_a_working_config_so_turn_one_does_not_pay(caplog) -> None:
    client, models = _llm("gemini-3.5-flash-lite", 0, [
        _FakeStream([], raise_at=0, exc=_rejection()),
        _FakeStream([_chunk("네")]),
    ])

    with caplog.at_level(logging.DEBUG, logger="app.services.llm"):
        await client.warmup()

    assert len(models.sent_thinking) == 2, "400 이면 다음 후보로 내려가야 한다"
    stepped = [r for r in caplog.records if "400 을 줬다" in r.getMessage()]
    assert stepped and stepped[0].levelno == logging.WARNING, "400 은 DEBUG 로 묻으면 안 된다"


@pytest.mark.asyncio
async def test_step_down_log_carries_the_api_message(caplog) -> None:
    oversized = errors.ClientError(
        400, {"error": {"message": "The input token count exceeds the maximum "
                                   "number of tokens allowed (1048576).",
                        "status": "INVALID_ARGUMENT"}}
    )
    client, _ = _llm("gemini-3.5-flash-lite", 0, [
        _FakeStream([], raise_at=0, exc=oversized),
        _FakeStream([_chunk("[EMOTION: NEUTRAL]네.")]),
    ])

    with caplog.at_level(logging.WARNING, logger="app.services.llm"):
        await _drain(client)

    stepped = [r for r in caplog.records if "400 을 줬다" in r.getMessage()]
    assert stepped, "물러설 때 로그가 있어야 한다"
    assert "input token count" in stepped[0].getMessage(), (
        "API 메시지를 안 실으면 thinking 탓이 아닌 400 의 진짜 원인을 못 찾는다"
    )
    assert "거절했다" not in stepped[0].getMessage(), "원인을 단정하면 안 된다"


@pytest.mark.asyncio
async def test_abandoned_stream_is_closed() -> None:
    first = _FakeStream([], raise_at=0, exc=_rejection())
    client, _ = _llm("gemini-3.5-flash-lite", 0,
                     [first, _FakeStream([_chunk("[EMOTION: NEUTRAL]네.")])])

    await _drain(client)

    assert first.closed, "거절당한 스트림은 닫아야 한다"


@pytest.mark.asyncio
async def test_warmup_still_swallows_transient_failures(caplog) -> None:
    client, _ = _llm("gemini-3.5-flash-lite", 0,
                     [_FakeStream([], raise_at=0, exc=RuntimeError("network"))])

    with caplog.at_level(logging.WARNING, logger="app.services.llm"):
        await client.warmup()

    assert not caplog.records


@pytest.mark.asyncio
async def test_unmapped_model_warns_once_per_turn(caplog) -> None:
    chunks = [_chunk("[EMOTION: NEUTRAL]네.", finish=types.FinishReason.STOP)]
    client, _ = _llm("gemini-9.9-unknown", 0, [_FakeStream(chunks)])

    with caplog.at_level(logging.WARNING, logger="app.services.llm"):
        await _drain(client)

    unmapped = [r for r in caplog.records if "실측하지 못한" in r.getMessage()]
    assert len(unmapped) == 1, f"턴당 한 번이어야 한다 (실제 {len(unmapped)}번)"
