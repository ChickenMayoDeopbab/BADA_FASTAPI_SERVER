import asyncio

import pytest

from app.schemas.frames import TranscriptRole, transcript_frame
from app.schemas.llm import AiEmotion, LLMEvent, LLMEventType
from app.services import pipeline as pipeline_mod
from app.services.pipeline import VoicePipeline, _State, _TurnTimings
from app.services.stt import STTEvent, STTEventType


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


class _HappyLLM:
    """대사 '네' 한 마디를 내는 정상 턴."""

    async def stream(self, ctx):
        yield LLMEvent(type=LLMEventType.EMOTION_RESOLVED, emotion=AiEmotion.NEUTRAL)
        yield LLMEvent(type=LLMEventType.TEXT_DELTA, text="네")
        yield LLMEvent(type=LLMEventType.TURN_END)


class _SilentLLM:
    """감정만 확정하고 대사 없이 끝나는 턴."""

    async def stream(self, ctx):
        yield LLMEvent(type=LLMEventType.EMOTION_RESOLVED, emotion=AiEmotion.NEUTRAL)
        yield LLMEvent(type=LLMEventType.TURN_END)


class _StallAfterTextLLM:
    """부분 대사만 내고 스톨 → 워치독 경로."""

    async def stream(self, ctx):
        yield LLMEvent(type=LLMEventType.EMOTION_RESOLVED, emotion=AiEmotion.NEUTRAL)
        yield LLMEvent(type=LLMEventType.TEXT_DELTA, text="잠시")
        await asyncio.Event().wait()


def _make_pipeline(llm=None, tts=None) -> VoicePipeline:
    p = VoicePipeline.__new__(VoicePipeline)
    p._ws = _FakeWS()
    p._session_id = "sess-transcript"
    p._session = {}
    p._llm = llm
    p._tts = tts
    p._state = _State.LISTENING
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
    p._script_len = 0
    p._ai_pcm_bytes = 0
    p._server_wait_duration_ms = 0
    p._completed_script_steps = 0
    return p


def _transcripts(ws: _FakeWS) -> list[dict]:
    return [f for f in ws.frames if f.get("type") == "transcript"]


def test_transcript_frame_shape() -> None:
    frame = transcript_frame(TranscriptRole.USER, "여보세요")
    assert frame == {"type": "transcript", "role": "user", "text": "여보세요"}

    frame = transcript_frame(TranscriptRole.AI, "네, 말씀하세요.")
    assert frame == {"type": "transcript", "role": "ai", "text": "네, 말씀하세요."}


@pytest.mark.asyncio
async def test_accepted_final_sends_user_transcript(monkeypatch) -> None:
    """채택된 STT FINAL → user transcript 송신 후 턴 시작(순서 보장)."""
    p = _make_pipeline()
    frames_at_start: list[int] = []

    def _fake_start_turn(text: str, *, final_at: float) -> None:
        frames_at_start.append(len(p._ws.frames))

    p._start_turn = _fake_start_turn

    await p._handle_stt_event(STTEvent(type=STTEventType.FINAL, text="여보세요"))

    assert _transcripts(p._ws) == [
        {"type": "transcript", "role": "user", "text": "여보세요"}
    ]
    # transcript 프레임이 _start_turn 호출 전에 이미 송신됨
    assert frames_at_start == [1]


@pytest.mark.asyncio
async def test_rejected_final_sends_nothing(monkeypatch) -> None:
    """미채택 FINAL(비-LISTENING/빈 텍스트/time_up)은 프레임 미송신."""
    started: list[str] = []

    def _make(state: _State, time_up: bool) -> VoicePipeline:
        p = _make_pipeline()
        p._state = state
        p._time_up = time_up
        p._start_turn = lambda text, *, final_at: started.append(text)
        return p

    p = _make(_State.SPEAKING, time_up=False)
    await p._handle_stt_event(STTEvent(type=STTEventType.FINAL, text="에코 발화"))
    assert _transcripts(p._ws) == []

    p = _make(_State.LISTENING, time_up=False)
    await p._handle_stt_event(STTEvent(type=STTEventType.FINAL, text="   "))
    assert _transcripts(p._ws) == []

    p = _make(_State.LISTENING, time_up=True)
    await p._handle_stt_event(STTEvent(type=STTEventType.FINAL, text="늦은 발화"))
    assert _transcripts(p._ws) == []

    assert started == []


@pytest.mark.asyncio
async def test_turn_sends_ai_transcript_after_speaking_end() -> None:
    """정상 턴 종료 시 ai transcript 가 speaking_end 이후에 송신된다."""
    p = _make_pipeline(_HappyLLM(), _FakeTTSClient())
    p._state = _State.THINKING

    await asyncio.wait_for(
        p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0
    )

    frame_types = [f.get("type") for f in p._ws.frames]
    assert frame_types.index("speaking_end") < frame_types.index("transcript")
    assert _transcripts(p._ws) == [
        {"type": "transcript", "role": "ai", "text": "네"}
    ]
    assert p._history[-1] == {"role": "assistant", "text": "네"}


@pytest.mark.asyncio
async def test_empty_ai_text_sends_fallback_transcript_only() -> None:
    p = _make_pipeline(_SilentLLM(), _FakeTTSClient())
    p._state = _State.THINKING

    await asyncio.wait_for(
        p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0
    )

    fallback = pipeline_mod._TURN_FALLBACK_TEXT
    assert _transcripts(p._ws) == [
        {"type": "transcript", "role": "ai", "text": fallback}
    ]
    assert p._history == [
        {"role": "user", "text": "여보세요"},
        {"role": "assistant", "text": fallback},
    ]


@pytest.mark.asyncio
async def test_watchdog_turn_still_sends_partial_ai_transcript(monkeypatch) -> None:
    """워치독 턴도 부분 대사를 송신하고, 폴백 멘트가 뒤따른다(F28). speaking_end 는 마지막."""
    monkeypatch.setattr(pipeline_mod, "_TURN_WATCHDOG_SECONDS", 0.05)
    p = _make_pipeline(_StallAfterTextLLM(), _FakeTTSClient())
    p._state = _State.THINKING

    await asyncio.wait_for(
        p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0
    )

    fallback = pipeline_mod._TURN_FALLBACK_TEXT
    frame_types = [f.get("type") for f in p._ws.frames]
    assert frame_types.index("transcript") < frame_types.index("speaking_end")
    assert _transcripts(p._ws) == [
        {"type": "transcript", "role": "ai", "text": "잠시"},
        {"type": "transcript", "role": "ai", "text": fallback},
    ]
    assert p._history[-1] == {"role": "assistant", "text": f"잠시 {fallback}"}


@pytest.mark.asyncio
async def test_cancelled_turn_salvages_history_and_partial_transcript() -> None:
    """턴 취소 시에도 나눈 대화는 히스토리에 남기기"""
    p = _make_pipeline(_StallAfterTextLLM(), _FakeTTSClient())
    p._state = _State.THINKING

    task = asyncio.create_task(p._run_turn("여보세요", _TurnTimings(final_at=0.0)))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert p._history == [
        {"role": "user", "text": "여보세요"},
        {"role": "assistant", "text": "잠시"},
    ]
    assert _transcripts(p._ws) == [
        {"type": "transcript", "role": "ai", "text": "잠시"}
    ]
    assert p._state != _State.LISTENING
