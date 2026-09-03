import asyncio

import pytest

from app.schemas.llm import AiEmotion, LLMEvent, LLMEventType
from app.services import pipeline as pipeline_mod
from app.services.pipeline import VoicePipeline, _State, _TurnTimings

_FALLBACK_MARKER = "다시 한번"


class _LogWS:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []

    async def send_json(self, payload: dict) -> None:
        self.events.append(("json", payload))

    async def send_bytes(self, data: bytes) -> None:
        self.events.append(("pcm", data))

    @property
    def frames(self) -> list[dict]:
        return [p for kind, p in self.events if kind == "json"]

    @property
    def pcm(self) -> list[bytes]:
        return [p for kind, p in self.events if kind == "pcm"]

    def frames_of(self, ftype: str) -> list[dict]:
        return [f for f in self.frames if f.get("type") == ftype]


class _FakeTTSSession:
    def __init__(self) -> None:
        self.closed = False

    async def begin(self, emotion) -> None:
        pass

    async def stream(self, text_source):
        async for _ in text_source:
            pass
        yield b"\x00\x01"

    async def aclose(self) -> None:
        self.closed = True


class _FakeTTSClient:
    def __init__(self) -> None:
        self.session = _FakeTTSSession()

    async def open(self) -> _FakeTTSSession:
        return self.session


class _BoomOpenTTSClient:
    async def open(self):
        raise RuntimeError("tts open boom")


class _StallTTSSession(_FakeTTSSession):
    async def stream(self, text_source):
        await asyncio.Event().wait()
        yield b""


class _StallTTSClient(_FakeTTSClient):
    def __init__(self) -> None:
        self.session = _StallTTSSession()


class _StalledLLM:
    async def stream(self, ctx):
        await asyncio.Event().wait()
        yield


class _EmptyLLM:
    async def stream(self, ctx):
        yield LLMEvent(type=LLMEventType.TURN_END)


class _HappyLLM:
    async def stream(self, ctx):
        yield LLMEvent(type=LLMEventType.EMOTION_RESOLVED, emotion=AiEmotion.NEUTRAL)
        yield LLMEvent(type=LLMEventType.TEXT_DELTA, text="네")
        yield LLMEvent(type=LLMEventType.TURN_END)


class _EndCallOnlyLLM:
    async def stream(self, ctx):
        yield LLMEvent(type=LLMEventType.END_CALL)


class _PartialThenStallLLM:
    async def stream(self, ctx):
        yield LLMEvent(type=LLMEventType.EMOTION_RESOLVED, emotion=AiEmotion.NEUTRAL)
        yield LLMEvent(type=LLMEventType.TEXT_DELTA, text="잠시만요")
        await asyncio.Event().wait()


def _make_pipeline(llm, tts) -> VoicePipeline:
    p = VoicePipeline.__new__(VoicePipeline)
    p._ws = _LogWS()
    p._session_id = "sess-fallback"
    p._session = {}
    p._llm = llm
    p._tts = tts
    p._state = _State.THINKING
    p._history = []
    p._current_step = 1
    p._ws_alive = True
    p._time_up = False
    p._closing = asyncio.Event()
    p._turn_task = None
    p._listening_since = None
    p._tremor_buf = bytearray()
    p._user_turn_intervals = []
    p._turn_open_at = None
    p._script_len = 0
    p._ai_pcm_bytes = 0
    p._server_wait_duration_ms = 0
    p._completed_script_steps = 0
    return p


async def _run(p: VoicePipeline) -> None:
    await asyncio.wait_for(
        p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0
    )


def _fallback_notices(ws: _LogWS) -> list[dict]:
    return [f for f in ws.frames_of("notice") if f.get("code") == "TURN_FALLBACK"]


@pytest.mark.asyncio
async def test_watchdog_turn_plays_fallback(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_mod, "_TURN_WATCHDOG_SECONDS", 0.05)
    p = _make_pipeline(_StalledLLM(), _FakeTTSClient())

    await _run(p)

    ws: _LogWS = p._ws
    assert p._state == _State.LISTENING
    assert not p._closing.is_set()

    notices = _fallback_notices(ws)
    assert len(notices) == 1
    assert _FALLBACK_MARKER in notices[0].get("text", "")
    assert ws.pcm, "폴백 멘트 PCM 이 송신되어야 한다"

    kinds = [
        (kind, p.get("type") if kind == "json" else None)
        for kind, p in ws.events
    ]
    notice_idx = kinds.index(("json", "notice"))
    last_pcm_idx = max(i for i, (kind, _) in enumerate(kinds) if kind == "pcm")
    end_idx = max(
        i for i, item in enumerate(kinds) if item == ("json", "speaking_end")
    )
    assert notice_idx < last_pcm_idx < end_idx


@pytest.mark.asyncio
async def test_watchdog_fallback_recorded_in_history_and_transcript(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_mod, "_TURN_WATCHDOG_SECONDS", 0.05)
    p = _make_pipeline(_StalledLLM(), _FakeTTSClient())

    await _run(p)

    assert p._history[-1]["role"] == "assistant"
    assert _FALLBACK_MARKER in p._history[-1]["text"]
    ai_lines = [
        f for f in p._ws.frames_of("transcript") if f.get("role") == "ai"
    ]
    assert any(_FALLBACK_MARKER in f.get("text", "") for f in ai_lines)


@pytest.mark.asyncio
async def test_empty_turn_plays_fallback_and_sends_speaking_end() -> None:
    p = _make_pipeline(_EmptyLLM(), _FakeTTSClient())

    await _run(p)

    ws: _LogWS = p._ws
    assert p._state == _State.LISTENING
    assert not p._closing.is_set()
    assert len(_fallback_notices(ws)) == 1
    assert ws.pcm, "폴백 멘트 PCM 이 송신되어야 한다"
    assert ws.frames_of("speaking_end"), "빈 턴에도 speaking_end 는 반드시 나가야 한다"


@pytest.mark.asyncio
async def test_partial_watchdog_merges_fallback_into_last_assistant(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_mod, "_TURN_WATCHDOG_SECONDS", 0.05)
    p = _make_pipeline(_PartialThenStallLLM(), _FakeTTSClient())

    await _run(p)

    assistants = [m for m in p._history if m["role"] == "assistant"]
    assert len(assistants) == 1
    assert "잠시만요" in assistants[0]["text"]
    assert _FALLBACK_MARKER in assistants[0]["text"]


@pytest.mark.asyncio
async def test_fallback_tts_failure_still_sends_notice_and_speaking_end(
    monkeypatch,
) -> None:
    monkeypatch.setattr(pipeline_mod, "_TURN_WATCHDOG_SECONDS", 0.05)
    p = _make_pipeline(_StalledLLM(), _BoomOpenTTSClient())

    await _run(p)

    ws: _LogWS = p._ws
    assert p._state == _State.LISTENING
    assert not p._closing.is_set()
    assert len(_fallback_notices(ws)) == 1
    assert ws.frames_of("speaking_end")
    assert ws.pcm == []


@pytest.mark.asyncio
async def test_fallback_tts_stall_gives_up_within_guard(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_mod, "_TURN_WATCHDOG_SECONDS", 0.05)
    monkeypatch.setattr(pipeline_mod, "_FALLBACK_TTS_TIMEOUT", 0.05, raising=False)
    p = _make_pipeline(_StalledLLM(), _StallTTSClient())

    await _run(p)

    ws: _LogWS = p._ws
    assert p._state == _State.LISTENING
    assert len(_fallback_notices(ws)) == 1
    assert ws.frames_of("speaking_end")


@pytest.mark.asyncio
async def test_normal_turn_no_fallback() -> None:
    p = _make_pipeline(_HappyLLM(), _FakeTTSClient())

    await _run(p)

    ws: _LogWS = p._ws
    assert ws.frames_of("notice") == []
    assert len(ws.frames_of("speaking_end")) == 1
    assert p._history[-1]["text"] == "네"


@pytest.mark.asyncio
async def test_end_call_empty_turn_no_fallback() -> None:
    p = _make_pipeline(_EndCallOnlyLLM(), _FakeTTSClient())

    await _run(p)

    ws: _LogWS = p._ws
    assert p._closing.is_set()
    assert ws.frames_of("notice") == []
    assert not any(
        _FALLBACK_MARKER in m["text"] for m in p._history if m["role"] == "assistant"
    )


@pytest.mark.asyncio
async def test_voice_turn_metric_has_fallback_flag(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_mod, "_TURN_WATCHDOG_SECONDS", 0.05)
    captured: list[dict] = []

    def fake_metric(event: str, **fields) -> None:
        captured.append({"event": event, **fields})

    monkeypatch.setattr(pipeline_mod, "log_metric", fake_metric)

    p = _make_pipeline(_StalledLLM(), _FakeTTSClient())
    await _run(p)
    assert captured[-1]["fallback"] is True

    monkeypatch.setattr(pipeline_mod, "_TURN_WATCHDOG_SECONDS", 30.0)
    p2 = _make_pipeline(_HappyLLM(), _FakeTTSClient())
    await _run(p2)
    assert captured[-1]["fallback"] is False
