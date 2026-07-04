import asyncio

import pytest

from app.schemas.llm import (
    AiEmotion,
    AiPersonality,
    LLMEvent,
    LLMEventType,
    ScenarioTurn,
    TurnContext,
)
from app.services import pipeline as pipeline_mod
from app.services.llm import LLMClient
from app.services.llm_prompt import build_system_prompt
from app.services.pipeline import VoicePipeline, _State, _TurnTimings
from app.services.session import build_turn_context


def _ctx(
    script_level: int | None = None,
    step: int = 1,
    utterance: str = "여보세요",
    history: list[dict] | None = None,
) -> TurnContext:
    return TurnContext(
        personality=AiPersonality.NORMAL,
        scenario_title="음식점 예약",
        scenario_role="사장님",
        script=[
            ScenarioTurn(step=1, ai_goal="전화를 받는다", hint="예약하고 싶다고 말해보세요"),
            ScenarioTurn(step=2, ai_goal="예약 정보를 묻는다", hint="원하는 날짜를 말하세요"),
        ],
        current_step=step,
        history=history or [],
        user_utterance=utterance,
        script_level=script_level,
    )


class _Chunk:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage_metadata = None
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


async def _events(chunks: list[str]) -> list:
    llm = _make_llm([_Chunk(t) for t in chunks])
    return [ev async for ev in llm.stream(_ctx(script_level=1))]


def _spoken(events) -> str:
    return "".join(e.text for e in events if e.type == LLMEventType.TEXT_DELTA)


def _suggestions(events) -> list[str]:
    return [e.text for e in events if e.type == LLMEventType.SUGGESTION]


@pytest.mark.asyncio
async def test_suggest_text_not_streamed_to_tts() -> None:
    events = await _events(
        ["[EMOTION:NEUTRAL]\n네, 알겠습니다.\n[SUGGEST] 그러면 불고기 피자로 주문하겠습니다."]
    )

    assert _spoken(events).strip() == "네, 알겠습니다."
    assert "불고기" not in _spoken(events)
    assert _suggestions(events) == ["그러면 불고기 피자로 주문하겠습니다."]


@pytest.mark.asyncio
async def test_suggest_absent_no_event() -> None:
    events = await _events(["[EMOTION:NEUTRAL]\n네, 알겠습니다."])

    assert _spoken(events).strip() == "네, 알겠습니다."
    assert _suggestions(events) == []


@pytest.mark.asyncio
async def test_suggest_split_across_chunks() -> None:
    events = await _events(
        ["[EMOTION:NEUTRAL]\n네, 알겠습니다.", "\n[SUG", "GEST] 원하는 날짜를 말해보세요."]
    )

    assert _spoken(events).strip() == "네, 알겠습니다."
    assert "[SUG" not in _spoken(events)
    assert _suggestions(events) == ["원하는 날짜를 말해보세요."]


@pytest.mark.asyncio
async def test_control_tags_after_suggest_stripped() -> None:
    events = await _events(
        ["[EMOTION:NEUTRAL]\n안녕히 계세요.\n[SUGGEST] 감사합니다, 라고 인사해보세요.\n[END_CALL]"]
    )

    types = [e.type for e in events]
    assert LLMEventType.END_CALL in types
    assert _suggestions(events) == ["감사합니다, 라고 인사해보세요."]


def test_prompt_level_blocks() -> None:
    base = build_system_prompt(_ctx(script_level=None))
    l1 = build_system_prompt(_ctx(script_level=1))
    l2 = build_system_prompt(_ctx(script_level=2))
    l3 = build_system_prompt(_ctx(script_level=3))

    assert "[SUGGEST]" in l1
    assert "[SUGGEST]" in l2
    assert l1 != l2
    assert "[SUGGEST]" not in base
    assert l3 == base


def test_prompt_byte_stable_within_level() -> None:
    for level in (None, 1, 2, 3):
        prompts = {
            build_system_prompt(_ctx(script_level=level, step=s, utterance=u))
            for s, u in ((1, "여보세요"), (2, "예약이요"), (2, "3시요"))
        }
        assert len(prompts) == 1, f"level={level} 프롬프트가 턴에 따라 변함"


def test_build_turn_context_parses_script_level() -> None:
    cases = [
        (1, 1),
        (2, 2),
        ("2", 2),
        (3, 3),
        (None, None),
        ("x", None),
        (7, None),
    ]
    for raw, expected in cases:
        session = {"scenario": {}}
        if raw is not None:
            session["scriptLevel"] = raw
        ctx = build_turn_context(
            session, current_step=1, history=[], user_utterance="여보세요"
        )
        assert ctx.script_level == expected, f"raw={raw!r}"


class _FakeWS:
    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.pcm: list[bytes] = []

    async def send_json(self, payload: dict) -> None:
        self.frames.append(payload)

    async def send_bytes(self, data: bytes) -> None:
        self.pcm.append(data)


class _FakeTTSSession:
    async def begin(self, emotion) -> None:
        pass

    async def stream(self, text_source):
        async for _ in text_source:
            pass
        yield b"\x00\x01"

    async def aclose(self) -> None:
        pass


class _FakeTTSClient:
    async def open(self) -> _FakeTTSSession:
        return _FakeTTSSession()


class _SuggestLLM:

    async def stream(self, ctx):
        yield LLMEvent(type=LLMEventType.EMOTION_RESOLVED, emotion=AiEmotion.NEUTRAL)
        yield LLMEvent(type=LLMEventType.TEXT_DELTA, text="네")
        yield LLMEvent(
            type=LLMEventType.SUGGESTION, text="그러면 불고기 피자로 주문하겠습니다."
        )
        yield LLMEvent(type=LLMEventType.TURN_END)


class _NoSuggestLLM:
    async def stream(self, ctx):
        yield LLMEvent(type=LLMEventType.EMOTION_RESOLVED, emotion=AiEmotion.NEUTRAL)
        yield LLMEvent(type=LLMEventType.TEXT_DELTA, text="네")
        yield LLMEvent(type=LLMEventType.TURN_END)


class _EndCallSuggestLLM:
    async def stream(self, ctx):
        yield LLMEvent(type=LLMEventType.EMOTION_RESOLVED, emotion=AiEmotion.NEUTRAL)
        yield LLMEvent(type=LLMEventType.TEXT_DELTA, text="안녕히 계세요")
        yield LLMEvent(type=LLMEventType.END_CALL)
        yield LLMEvent(type=LLMEventType.SUGGESTION, text="쓸모없는 힌트")
        yield LLMEvent(type=LLMEventType.TURN_END)


class _StallLLM:
    async def stream(self, ctx):
        yield LLMEvent(type=LLMEventType.EMOTION_RESOLVED, emotion=AiEmotion.NEUTRAL)
        yield LLMEvent(type=LLMEventType.TEXT_DELTA, text="잠시")
        await asyncio.Event().wait()


def _session_dict(level: int | None, with_script: bool = True) -> dict:
    scenario: dict = {"title": "음식점 예약", "aiRole": "사장님"}
    if with_script:
        scenario["script"] = [
            {"step": 1, "aiGoal": "전화를 받는다", "hint": "예약하고 싶다고 말해보세요"},
            {"step": 2, "aiGoal": "정보를 묻는다", "hint": "원하는 날짜를 말하세요"},
        ]
    session: dict = {"scenario": scenario}
    if level is not None:
        session["scriptLevel"] = level
    return session


def _make_pipeline(llm, session: dict) -> VoicePipeline:
    p = VoicePipeline.__new__(VoicePipeline)
    p._ws = _FakeWS()
    p._session_id = "sess-hint"
    p._session = session
    p._llm = llm
    p._tts = _FakeTTSClient()
    p._state = _State.THINKING
    p._history = []
    p._current_step = 1
    p._ws_alive = True
    p._time_up = False
    p._closing = asyncio.Event()
    p._turn_task = None
    p._listening_since = None
    p._silence_total = 0.0
    p._tremor_buf = bytearray()
    p._user_turn_intervals = []
    p._turn_open_at = None
    p._script_len = len(session.get("scenario", {}).get("script", []) or [])
    return p


def _hints(ws: _FakeWS) -> list[dict]:
    return [f for f in ws.frames if f.get("type") == "script_hint"]


async def _run(p: VoicePipeline) -> None:
    await asyncio.wait_for(
        p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0
    )


@pytest.mark.asyncio
async def test_normal_turn_sends_script_hint_after_ai_transcript() -> None:
    p = _make_pipeline(_SuggestLLM(), _session_dict(level=1))

    await _run(p)

    frame_types = [f.get("type") for f in p._ws.frames]
    assert frame_types.index("transcript") < frame_types.index("script_hint")
    assert _hints(p._ws) == [
        {
            "type": "script_hint",
            "level": 1,
            "text": "그러면 불고기 피자로 주문하겠습니다.",
        }
    ]


@pytest.mark.asyncio
async def test_no_hint_frame_for_level3_or_absent() -> None:
    for level in (3, None):
        p = _make_pipeline(_SuggestLLM(), _session_dict(level=level))
        await _run(p)
        assert _hints(p._ws) == [], f"level={level}"


@pytest.mark.asyncio
async def test_fallback_to_step_hint_when_llm_omits_tag() -> None:
    p = _make_pipeline(_NoSuggestLLM(), _session_dict(level=2))
    await _run(p)
    assert _hints(p._ws) == [
        {"type": "script_hint", "level": 2, "text": "예약하고 싶다고 말해보세요"}
    ]

    p2 = _make_pipeline(_NoSuggestLLM(), _session_dict(level=2, with_script=False))
    await _run(p2)
    assert _hints(p2._ws) == []


@pytest.mark.asyncio
async def test_no_hint_on_watchdog_or_endcall_turn(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_mod, "_TURN_WATCHDOG_SECONDS", 0.05)
    p = _make_pipeline(_StallLLM(), _session_dict(level=1))
    await _run(p)
    assert _hints(p._ws) == []

    monkeypatch.setattr(pipeline_mod, "_TURN_WATCHDOG_SECONDS", 30.0)
    p2 = _make_pipeline(_EndCallSuggestLLM(), _session_dict(level=1))
    await _run(p2)
    assert _hints(p2._ws) == []
    assert p2._closing.is_set()
