import asyncio
import json
from types import SimpleNamespace

import pytest

from app.core.preset_scenarios import PRESET_MAP
from app.schemas.llm import AiEmotion
from app.services import pipeline as pipeline_mod
from app.services.llm_prompt import build_contents
from app.services.pipeline import VoicePipeline, _State
from app.services.qwen_tts import QwenTTSRunawayError, QwenTTSUnavailableError
from app.services.session import build_turn_context
from app.services.stt import STTEvent, STTEventType

GREETING = PRESET_MAP[1]["example_dialogue"][0]["text"]
_MIC = b"\x00" * 640  # 앱이 연결 직후 보내는 20ms 무음 프레임


class _FakeWS:
    def __init__(self) -> None:
        self.events: list[tuple[str, object]] = []
        self.inbox: asyncio.Queue[dict] = asyncio.Queue()

    async def receive(self) -> dict:
        return await self.inbox.get()

    async def send_json(self, payload: dict) -> None:
        self.events.append(("json", payload))

    async def send_bytes(self, data: bytes) -> None:
        self.events.append(("pcm", data))

    def frames(self, ftype: str) -> list[dict]:
        return [p for kind, p in self.events if kind == "json" and p.get("type") == ftype]

    def kinds(self) -> list[str]:
        return [p.get("type") if kind == "json" else "pcm" for kind, p in self.events]

    def pcm(self) -> list[bytes]:
        return [p for kind, p in self.events if kind == "pcm"]


class _FakeSession:
    def __init__(self, pcm: bytes = b"\x01\x02" * 160, fail: BaseException | None = None, stall: bool = False) -> None:
        self.pcm = pcm
        self.fail = fail
        self.stall = stall
        self.texts: list[str] = []
        self.chars_sent = 0
        self.closed = False

    async def begin(self, emotion=AiEmotion.NEUTRAL) -> None:
        pass

    async def stream(self, text_source):
        async for text in text_source:
            self.texts.append(text)
            self.chars_sent += len(text)
        if self.stall:
            await asyncio.Event().wait()
        if self.fail is not None:
            raise self.fail
        yield self.pcm

    async def aclose(self) -> None:
        self.closed = True


class _FakeEL:
    def __init__(self, session: _FakeSession | None = None) -> None:
        self._session = session
        self.sessions: list[_FakeSession] = []

    async def open(self, voice_id=None) -> _FakeSession:
        session = self._session or _FakeSession(pcm=b"\xee\xee" * 160)
        self.sessions.append(session)
        return session


class _FakeQwen:
    def __init__(self, fail: BaseException | None = None, pcm: bytes = b"\x0a\x0b" * 160) -> None:
        self.fail = fail
        self.pcm = pcm
        self.released = False
        self.sessions: list[_FakeSession] = []

    async def open(self, voice_id=None) -> _FakeSession:
        session = _FakeSession(pcm=self.pcm, fail=self.fail)
        self.sessions.append(session)
        return session

    def release_slot(self) -> None:
        self.released = True


class _PartialThenRunaway(_FakeSession):
    async def stream(self, text_source):
        async for text in text_source:
            self.texts.append(text)
        yield b"\x0c\x0d" * 3200
        raise QwenTTSRunawayError(worker="w1", chars=30, pcm_bytes=6400, cap_ms=100, wall_ms=50.0)


class _RunawayQwen(_FakeQwen):
    async def open(self, voice_id=None) -> _FakeSession:
        session = _PartialThenRunaway()
        self.sessions.append(session)
        return session


def _make_pipeline(*, tts=None, qwen=None, settings=None, session=None) -> VoicePipeline:
    p = VoicePipeline.__new__(VoicePipeline)
    p._ws = _FakeWS()
    p._session_id = "sess-greeting"
    p._session = session or {}
    p._settings = settings
    p._tts = tts or _FakeEL()
    p._qwen_tts = qwen
    p._state = _State.LISTENING
    p._history = []
    p._current_step = 1
    p._muted = False
    p._ws_alive = True
    p._time_up = False
    p._closing = asyncio.Event()
    p._turn_task = None
    p._audio_queue = asyncio.Queue()
    p._last_audio_at = None
    p._listening_since = None
    p._silence_total = 0.0
    p._tremor_buf = bytearray()
    p._user_turn_intervals = []
    p._user_turn_texts = []
    p._user_turn_prompts = []
    p._heard_at_by_turn = {}
    p._turn_open_at = 0.0
    p._script_len = 0
    p._ai_pcm_bytes = 0
    p._server_wait_duration_ms = 0
    p._completed_script_steps = 0
    return p


def _capture(monkeypatch) -> list[tuple[str, dict]]:
    records: list[tuple[str, dict]] = []
    monkeypatch.setattr(pipeline_mod, "log_metric", lambda name, **kw: records.append((name, kw)))
    return records


def _metric(records, name: str) -> list[dict]:
    return [kw for n, kw in records if n == name]


async def _until(predicate, timeout: float = 1.0) -> None:
    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(_poll(), timeout)


def _text(payload: dict) -> dict:
    return {"type": "websocket.receive", "text": json.dumps(payload)}


async def _stop(task: asyncio.Task) -> None:
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


async def test_ai_greets_on_first_mic_audio_before_the_user_speaks(monkeypatch) -> None:
    records = _capture(monkeypatch)
    p = _make_pipeline()
    p._pending_greeting = GREETING
    p._greeting_source = "preset"
    recv = asyncio.create_task(p._recv_loop())
    try:
        await p._ws.inbox.put(_text({"type": "playback_capabilities", "completion_ack": True}))
        await asyncio.sleep(0.01)
        assert p._ws.events == [], "마이크 오디오가 오기 전엔 말하지 않는다"

        await p._ws.inbox.put({"type": "websocket.receive", "bytes": _MIC})
        await _until(lambda: p._ws.frames("speaking_end"))
        assert p._ws.kinds() == ["emotion", "pcm", "transcript", "speaking_end"]
        assert p._ws.frames("transcript") == [{"type": "transcript", "role": "ai", "text": GREETING}]
        assert p._history == [{"role": "assistant", "text": GREETING}]
        assert p._user_turn_intervals == [], "인사는 가짜 사용자 턴을 만들지 않는다"
        assert p._state == _State.GREETING, "기기에서 다 들릴 때까지 듣지 않는다"
        assert p._audio_queue.qsize() == 1, "첫 마이크 오디오도 STT 로 간다"

        turn_id = p._ws.frames("speaking_end")[0]["turn_id"]
        await p._ws.inbox.put(_text({"type": "playback_done", "turn_id": turn_id}))
        await _until(lambda: p._state == _State.LISTENING)
        assert p._turn_task is None
        assert p._turn_open_at == p._buf_sec() == 0.02, "첫 사용자 턴은 인사가 들린 뒤에 열린다"
        assert p._ai_heard_at == 0.02
        assert p._listening_since is not None
        summary = _metric(records, "greeting")
        assert len(summary) == 1
        assert summary[0]["source"] == "preset" and summary[0]["spoken"] is True and summary[0]["heard"] is True
        assert summary[0]["chars"] == len(GREETING) and summary[0]["dropped_finals"] == 0
    finally:
        await _stop(recv)


async def test_greeting_starts_only_once(monkeypatch) -> None:
    _capture(monkeypatch)
    monkeypatch.setattr(pipeline_mod, "_PLAYBACK_GRACE_MS", 0.0)
    p = _make_pipeline()
    p._pending_greeting = GREETING
    recv = asyncio.create_task(p._recv_loop())
    try:
        for _ in range(3):
            await p._ws.inbox.put({"type": "websocket.receive", "bytes": _MIC})
        await _until(lambda: p._state == _State.LISTENING and p._ws.frames("speaking_end"))
        await asyncio.sleep(0.01)
        assert len(p._ws.frames("transcript")) == 1
        assert p._audio_queue.qsize() == 3
    finally:
        await _stop(recv)


async def test_final_while_tts_is_still_opening_cannot_open_a_user_turn(monkeypatch) -> None:
    records = _capture(monkeypatch)
    session = _FakeSession(stall=True)
    p = _make_pipeline(tts=_FakeEL(session=session))
    p._pending_greeting = GREETING
    p._start_greeting()
    try:
        assert p._state == _State.GREETING, "TTS 를 열기 전부터 듣지 않는다"
        await p._handle_stt_event(STTEvent(type=STTEventType.FINAL, text="여보세요"))
        assert p._ws.frames("transcript") == []
        assert [kw["reason"] for kw in _metric(records, "final_dropped")] == ["greeting"]
    finally:
        await _stop(p._turn_task)


async def test_fallback_line_is_spoken_in_speaking_state() -> None:
    session = _FakeSession(stall=True)
    p = _make_pipeline(tts=_FakeEL(session=session))
    task = asyncio.create_task(p._speak_fallback())
    try:
        await _until(lambda: session.texts)
        assert p._state == _State.SPEAKING
        assert session.texts == [pipeline_mod._TURN_FALLBACK_TEXT]
    finally:
        await _stop(task)
    assert session.closed is True


@pytest.mark.parametrize("state", [_State.CLOSING, _State.THINKING])
async def test_no_greeting_unless_listening(state) -> None:
    p = _make_pipeline()
    p._state = state
    p._pending_greeting = GREETING
    p._start_greeting()
    assert p._turn_task is None and p._state == state and p._pending_greeting is None


async def test_no_pending_greeting_keeps_todays_behavior() -> None:
    p = _make_pipeline()
    recv = asyncio.create_task(p._recv_loop())
    try:
        await p._ws.inbox.put({"type": "websocket.receive", "bytes": _MIC})
        await _until(lambda: not p._audio_queue.empty())
        await asyncio.sleep(0.01)
        assert p._ws.events == [] and p._state == _State.LISTENING and p._turn_task is None
        assert p._history == []
    finally:
        await _stop(recv)


async def test_final_during_greeting_is_dropped_and_counted(monkeypatch) -> None:
    records = _capture(monkeypatch)
    p = _make_pipeline()
    p._state = _State.GREETING

    def _no_turn(*_args, **_kwargs):
        raise AssertionError("인사 중 FINAL 이 턴을 열면 안 된다")

    p._start_turn = _no_turn
    await p._handle_stt_event(STTEvent(type=STTEventType.FINAL, text="여보세요"))
    await p._handle_stt_event(STTEvent(type=STTEventType.FINAL, text=""))
    assert p._ws.events == []
    assert [kw["reason"] for kw in _metric(records, "final_dropped")] == ["greeting", "empty"]
    assert p._greeting_dropped == 1


async def test_qwen_failure_switches_to_eleven_and_still_greets(monkeypatch) -> None:
    records = _capture(monkeypatch)
    monkeypatch.setattr(pipeline_mod, "_PLAYBACK_GRACE_MS", 0.0)
    qwen = _FakeQwen(fail=QwenTTSUnavailableError("boom"))
    el = _FakeEL()
    p = _make_pipeline(tts=el, qwen=qwen)
    p._state = _State.GREETING
    await asyncio.wait_for(p._greet(GREETING), 2.0)
    assert qwen.released is True and p._qwen_tts is None
    assert {"session_id": "sess-greeting", "reason": "greeting_failed"} in _metric(records, "realtime_tts_switch")
    assert qwen.sessions[0].closed is True
    assert p._ws.frames("error") == [] and p._ws.frames("interrupt") == []
    assert p._ws.pcm() == [b"\xee\xee" * 160]
    assert p._ws.frames("transcript") == [{"type": "transcript", "role": "ai", "text": GREETING}]
    assert p._history == [{"role": "assistant", "text": GREETING}]
    assert p._state == _State.LISTENING and not p._closing.is_set()


async def test_runaway_after_partial_audio_drops_it_before_eleven_retry(monkeypatch) -> None:
    records = _capture(monkeypatch)
    monkeypatch.setattr(pipeline_mod, "_PLAYBACK_GRACE_MS", 0.0)
    qwen = _RunawayQwen()
    p = _make_pipeline(tts=_FakeEL(), qwen=qwen)
    p._state = _State.GREETING
    await asyncio.wait_for(p._greet(GREETING), 2.0)
    assert {"session_id": "sess-greeting", "reason": "greeting_runaway"} in _metric(records, "realtime_tts_switch")
    kinds = p._ws.kinds()
    assert kinds.index("interrupt") < len(kinds) - 1 - kinds[::-1].index("pcm"), "버린 소리 뒤에 다시 말한다"
    assert p._ws.pcm()[-1] == b"\xee\xee" * 160
    assert p._ws.frames("transcript") == [{"type": "transcript", "role": "ai", "text": GREETING}]


async def test_failed_retry_still_drops_the_partial_qwen_audio(monkeypatch) -> None:
    _capture(monkeypatch)

    class _DownEL:
        async def open(self, voice_id=None):
            raise RuntimeError("el down")

    p = _make_pipeline(tts=_DownEL(), qwen=_RunawayQwen())
    p._state = _State.GREETING
    await asyncio.wait_for(p._greet(GREETING), 2.0)
    assert p._ws.frames("interrupt"), "다음 턴 소리가 잘리지 않게 지금 버린다"
    assert p._drop_pending is False
    assert p._history == [] and p._state == _State.LISTENING and p._ws.frames("error") == []


async def test_greeting_failure_on_every_engine_goes_quietly_to_listening(monkeypatch) -> None:
    records = _capture(monkeypatch)
    el = _FakeEL(session=_FakeSession(fail=RuntimeError("el down")))
    p = _make_pipeline(tts=el)
    p._state = _State.GREETING
    await asyncio.wait_for(p._greet(GREETING), 2.0)
    assert p._ws.frames("error") == [] and p._ws.frames("transcript") == [] and p._ws.frames("notice") == []
    assert p._ws.frames("speaking_end"), "열어 둔 재생 턴은 닫아 준다"
    assert p._history == [] and p._state == _State.LISTENING and p._turn_task is None
    assert not p._closing.is_set()
    assert _metric(records, "greeting")[0]["spoken"] is False


async def test_greeting_tts_timeout_skips_without_switching_engine(monkeypatch) -> None:
    records = _capture(monkeypatch)
    monkeypatch.setattr(pipeline_mod, "_GREETING_TTS_TIMEOUT", 0.05)
    qwen = _FakeQwen()
    p = _make_pipeline(qwen=qwen)
    stalled = _FakeSession(stall=True)

    async def _open(voice_id=None):
        qwen.sessions.append(stalled)
        return stalled

    qwen.open = _open
    p._state = _State.GREETING
    await asyncio.wait_for(p._greet(GREETING), 2.0)
    assert p._qwen_tts is qwen and _metric(records, "realtime_tts_switch") == []
    assert stalled.closed is True
    assert p._history == [] and p._state == _State.LISTENING


@pytest.mark.parametrize("ack, expected", [(True, 2.0), (False, 1.0)])
def test_greeting_wait_uses_ack_grace_or_playback_estimate(monkeypatch, ack, expected) -> None:
    clock = [0.0]
    monkeypatch.setattr(pipeline_mod, "now_ms", lambda: clock[0])
    p = _make_pipeline()
    p._playback_ack_supported = ack
    p._begin_playback()
    p._playback.record(32000)  # 1초 분량
    clock[0] = 1000.0
    assert p._greeting_heard_timeout() == pytest.approx(expected)


async def test_greeting_does_not_reopen_listening_once_the_call_is_closing(monkeypatch) -> None:
    _capture(monkeypatch)
    p = _make_pipeline()
    p._playback_ack_supported = True
    p._state = _State.GREETING
    task = asyncio.create_task(p._greet(GREETING))
    try:
        await _until(lambda: p._ws.frames("speaking_end"))
        p._state = _State.CLOSING
        turn_id = p._ws.frames("speaking_end")[0]["turn_id"]
        await p._handle_client_text(json.dumps({"type": "playback_done", "turn_id": turn_id}))
        await asyncio.wait_for(task, 1.0)
        assert p._state == _State.CLOSING
    finally:
        await _stop(task)


async def test_cancelled_greeting_closes_tts(monkeypatch) -> None:
    _capture(monkeypatch)
    session = _FakeSession(stall=True)
    p = _make_pipeline(tts=_FakeEL(session=session))
    p._state = _State.GREETING
    task = asyncio.create_task(p._greet(GREETING))
    await _until(lambda: session.texts)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert session.closed is True
    assert p._state == _State.GREETING


def test_first_user_turn_after_greeting_has_greeting_as_previous_line() -> None:
    p = _make_pipeline()
    p._history = [{"role": "assistant", "text": GREETING}]
    p._close_user_turn("예약하고 싶어서요")
    assert p._user_turn_prompts == [GREETING]
    ctx = build_turn_context({}, current_step=1, history=p._history, user_utterance="예약하고 싶어서요")
    assert build_contents(ctx)[0] == {"role": "model", "parts": [{"text": GREETING}]}


@pytest.mark.parametrize(
    "settings, user_id, expected",
    [
        (SimpleNamespace(ai_greeting=True, ai_greeting_user_ids="81"), 81, GREETING),
        (SimpleNamespace(ai_greeting=True, ai_greeting_user_ids="81"), 90, None),
        (SimpleNamespace(ai_greeting=False, ai_greeting_user_ids=""), 81, None),
        (SimpleNamespace(), 81, None),
    ],
)
async def test_prepare_greeting_follows_switch_allowlist_and_scenario(settings, user_id, expected) -> None:
    p = _make_pipeline(settings=settings, session={"userId": user_id, "scenarioId": 1})
    await p._prepare_greeting()
    assert p._pending_greeting == expected


async def test_greeting_preparation_failure_only_skips_the_greeting(monkeypatch) -> None:
    async def _boom(_scenario_id):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(pipeline_mod, "pick_greeting", _boom)
    p = _make_pipeline(
        settings=SimpleNamespace(ai_greeting=True, ai_greeting_user_ids=""),
        session={"userId": 81, "scenarioId": 1},
    )
    await p._prepare_greeting()
    assert p._pending_greeting is None


async def test_run_prepares_the_greeting(monkeypatch) -> None:
    async def _acquire(_settings):
        return None, "disabled"

    async def _noop():
        return None

    async def _teardown(*tasks):
        for task in tasks:
            if task is not None:
                task.cancel()

    monkeypatch.setattr(pipeline_mod, "try_acquire_realtime_tts", _acquire)
    p = _make_pipeline(
        settings=SimpleNamespace(ai_greeting=True, ai_greeting_user_ids=""),
        session={"userId": 81, "scenarioId": 1},
    )
    p._max_duration = None
    p._llm = SimpleNamespace(warmup=_noop)
    p._recv_loop = _noop
    p._stt_consumer = _noop
    p._teardown = _teardown
    p._closing.set()
    await asyncio.wait_for(p.run(), 2.0)
    assert p._pending_greeting == GREETING
