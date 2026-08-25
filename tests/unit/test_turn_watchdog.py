import asyncio

import pytest

from app.schemas.llm import AiEmotion, LLMEvent, LLMEventType
from app.services import pipeline as pipeline_mod
from app.services.pipeline import VoicePipeline, _State, _TurnTimings


class _FakeWS:
    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.pcm: list[bytes] = []

    async def send_json(self, payload: dict) -> None:
        self.frames.append(payload)

    async def send_bytes(self, data: bytes) -> None:
        self.pcm.append(data)


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


class _StalledLLM:
    async def stream(self, ctx):
        await asyncio.Event().wait()
        yield


class _HappyLLM:
    async def stream(self, ctx):
        yield LLMEvent(type=LLMEventType.EMOTION_RESOLVED, emotion=AiEmotion.NEUTRAL)
        yield LLMEvent(type=LLMEventType.TEXT_DELTA, text="네")
        yield LLMEvent(type=LLMEventType.TURN_END)


class _EndlessLLM:
    """감정 확정 후 무한히 토큰을 내는 LLM — 취소 여부를 기록."""

    def __init__(self) -> None:
        self.cancelled = False

    async def stream(self, ctx):
        yield LLMEvent(type=LLMEventType.EMOTION_RESOLVED, emotion=AiEmotion.NEUTRAL)
        try:
            while True:
                await asyncio.sleep(0.005)
                yield LLMEvent(type=LLMEventType.TEXT_DELTA, text="a")
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class _BoomTTSSession(_FakeTTSSession):
    async def stream(self, text_source):
        raise RuntimeError("tts boom")
        yield  # pragma: no cover — async generator 형식 유지


class _BoomTTSClient(_FakeTTSClient):
    def __init__(self) -> None:
        self.session = _BoomTTSSession()


def _make_pipeline(llm, tts) -> VoicePipeline:
    p = VoicePipeline.__new__(VoicePipeline)
    p._ws = _FakeWS()
    p._session_id = "sess-watchdog"
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


@pytest.mark.asyncio
async def test_watchdog_recovers_stalled_turn(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_mod, "_TURN_WATCHDOG_SECONDS", 0.05)
    tts = _FakeTTSClient()
    p = _make_pipeline(_StalledLLM(), tts)

    await asyncio.wait_for(
        p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0
    )

    assert p._state == _State.LISTENING
    assert not p._closing.is_set()
    assert tts.session.closed
    frame_types = [f.get("type") for f in p._ws.frames]
    assert "speaking_end" in frame_types


@pytest.mark.asyncio
async def test_consume_failure_cancels_producer() -> None:
    """consume 예외 시 produce 가 좀비로 남지 않고 취소되어야 한다(gather 는 형제를 안 죽임)."""
    llm = _EndlessLLM()
    p = _make_pipeline(llm, _BoomTTSClient())

    await asyncio.wait_for(
        p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0
    )

    assert llm.cancelled is True          # 좀비 produce 없음
    assert p._closing.is_set()            # 기존 에러 종료 경로 유지


@pytest.mark.asyncio
async def test_fast_turn_unaffected_by_watchdog() -> None:
    tts = _FakeTTSClient()
    p = _make_pipeline(_HappyLLM(), tts)

    await asyncio.wait_for(
        p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0
    )

    assert p._state == _State.LISTENING
    assert not p._closing.is_set()
    assert p._ws.pcm == [b"\x00\x01"]
    assert p._history[-1]["text"] == "네"
