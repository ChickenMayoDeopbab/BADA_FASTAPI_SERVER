import asyncio
from contextlib import suppress

import pytest
from google.genai import errors, types
from websockets.exceptions import InvalidHandshake

import app.services.pipeline as pipeline_module
import app.services.stt as stt_module
from app.schemas.frames import EndReason
from app.services.pipeline import VoicePipeline, _State
from app.services.stt import (
    STTError,
    STTEvent,
    STTEventType,
    STTStreamAbortedError,
)


def _msg(**tr) -> types.LiveServerMessage:
    return types.LiveServerMessage(
        server_content=types.LiveServerContent(input_transcription=types.Transcription(**tr))
    )


def test_real_server_shape_finished_none_still_emits_final() -> None:
    events = stt_module.GeminiLiveSTTClient._parse_message(_msg(text="안녕하세요"))
    assert [e.type for e in events] == [STTEventType.FINAL]


def test_finished_true_emits_final() -> None:
    events = stt_module.GeminiLiveSTTClient._parse_message(_msg(text="안녕하세요", finished=True))
    assert [e.type for e in events] == [STTEventType.FINAL]


def test_finished_false_is_not_a_final() -> None:
    assert stt_module.GeminiLiveSTTClient._parse_message(_msg(text="안녕", finished=False)) == []


class _FailingConnect:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.aio = type("Aio", (), {})()
        self.aio.live = self

    def connect(self, *, model, config):
        exc = self._exc

        class _CM:
            async def __aenter__(self):
                raise exc

            async def __aexit__(self, *a):
                return False

        return _CM()


def _client_failing_with(exc: BaseException) -> stt_module.GeminiLiveSTTClient:
    client = stt_module.GeminiLiveSTTClient(api_key="k", model="m", language="ko-KR")
    client._client = _FailingConnect(exc)
    return client


@pytest.mark.parametrize(
    "exc",
    [
        errors.APIError(1006, {"message": "Abnormal closure."}),
        errors.APIError(1011, {"message": "internal error"}),
        errors.APIError(1013, {"message": "try again later"}),
        InvalidHandshake("429 too many requests"),
        OSError("dns failure"),
    ],
)
async def test_transient_connect_failures_are_reopenable(exc) -> None:
    client = _client_failing_with(exc)
    with pytest.raises(STTStreamAbortedError):
        async for _ in client.stream(asyncio.Queue()):
            pass


async def test_http_error_at_connect_is_not_reopenable() -> None:
    client = _client_failing_with(errors.ClientError(403, {"message": "denied"}))
    with pytest.raises(STTError) as info:
        async for _ in client.stream(asyncio.Queue()):
            pass
    assert not isinstance(info.value, STTStreamAbortedError)


def _turn_pipeline() -> tuple[VoicePipeline, list[str], asyncio.Event]:
    p = VoicePipeline.__new__(VoicePipeline)
    p._session_id = "sess-test"
    p._state = _State.LISTENING
    p._time_up = False
    p._listening_since = None
    started: list[str] = []
    sending = asyncio.Event()

    async def slow_send(frame) -> None:
        sending.set()
        await asyncio.sleep(3600)

    def start_turn(text, *, final_at) -> None:
        started.append(text)

    p._send_json = slow_send
    p._start_turn = start_turn
    return p, started, sending


async def test_turn_starts_even_if_transcript_send_is_cancelled() -> None:
    p, started, sending = _turn_pipeline()
    task = asyncio.create_task(
        p._handle_stt_event(STTEvent(type=STTEventType.FINAL, text="다음 주로 바꿔주세요"))
    )
    await asyncio.wait_for(sending.wait(), timeout=1.0)
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
    assert started == ["다음 주로 바꿔주세요"]

class _FlushingSession:
    def __init__(self, head: list | None = None) -> None:
        self.sent: list[dict] = []
        self._head = head if head is not None else [
            types.LiveServerMessage(
                server_content=types.LiveServerContent(
                    interim_input_transcription=types.Transcription(text="다음 주")
                )
            )
        ]
        self._eos = asyncio.Event()
        self._closed = asyncio.Event()

    async def send_realtime_input(self, **kwargs) -> None:
        self.sent.append(kwargs)
        if kwargs.get("audio_stream_end"):
            self._eos.set()

    async def close(self) -> None:
        self._closed.set()

    @property
    def closed_by_client(self) -> bool:
        return self._closed.is_set()

    @property
    def eos_received(self) -> bool:
        return self._eos.is_set()

    async def receive(self):
        for msg in self._head:
            yield msg
        await self._eos.wait()
        await asyncio.sleep(0.02)
        yield _msg(text="다음 주 화요일로 바꿔주세요")
        raise errors.APIError(1000, {"message": "OK"})


class _FlushingClient(stt_module.GeminiLiveSTTClient):
    def __init__(self, session: _FlushingSession) -> None:
        super().__init__(api_key="k", model="m", language="ko-KR")
        self.session = session
        client = type("C", (), {})()
        aio = type("Aio", (), {})()
        live = type("L", (), {})()

        def connect(*, model, config):
            sess = session

            class _CM:
                async def __aenter__(self):
                    return sess

                async def __aexit__(self, *a):
                    await sess.close()
                    return False

            return _CM()

        live.connect = connect
        aio.live = live
        client.aio = aio
        self._client = client


def _flush_pipeline(stt) -> tuple[VoicePipeline, list[STTEvent]]:
    p = VoicePipeline.__new__(VoicePipeline)
    p._session_id = "sess-test"
    p._closing = asyncio.Event()
    p._audio_queue = asyncio.Queue()
    p._state = _State.LISTENING
    p._stt = stt
    p._last_audio_at = None
    p._stream_opened_at = None
    p._stream_had_event = False
    handled: list[STTEvent] = []

    async def record(event) -> None:
        handled.append(event)

    async def send_json(frame) -> None:
        return None

    p._handle_stt_event = record
    p._send_json = send_json
    return p, handled


async def test_hard_deadline_flushes_tail_transcript(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "_STT_RECYCLE_SECONDS", 0.05)
    monkeypatch.setattr(pipeline_module, "_STT_RECYCLE_HARD_SECONDS", 0.1)
    monkeypatch.setattr(pipeline_module, "_STT_NO_AUDIO_TIMEOUT", 30.0)
    session = _FlushingSession()
    p, handled = _flush_pipeline(_FlushingClient(session))
    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait(b"pcm")
    await asyncio.wait_for(p._consume_one_stream(queue), timeout=5.0)
    assert session.eos_received, "half-close 가 서버에 도달하지 않았다"
    assert any(e.type == STTEventType.FINAL for e in handled), "꼬리 FINAL 이 유실됐다"


async def test_flush_swaps_audio_queue_before_half_close(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "_STT_RECYCLE_SECONDS", 0.05)
    monkeypatch.setattr(pipeline_module, "_STT_RECYCLE_HARD_SECONDS", 0.1)
    monkeypatch.setattr(pipeline_module, "_STT_NO_AUDIO_TIMEOUT", 30.0)
    session = _FlushingSession()
    p, _ = _flush_pipeline(_FlushingClient(session))
    queue = p._audio_queue
    queue.put_nowait(b"pcm")
    await asyncio.wait_for(p._consume_one_stream(queue), timeout=5.0)
    assert p._audio_queue is not queue

def _consumer_pipeline() -> tuple[VoicePipeline, list[EndReason]]:
    p = VoicePipeline.__new__(VoicePipeline)
    p._session_id = "sess-test"
    p._closing = asyncio.Event()
    p._audio_queue = asyncio.Queue()
    p._stream_opened_at = None
    p._stream_had_event = False
    closed: list[EndReason] = []

    async def fake_close(reason: EndReason) -> None:
        closed.append(reason)
        p._closing.set()

    p._close = fake_close
    return p, closed


async def test_idle_wait_does_not_count_as_a_healthy_stream(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "_STT_REOPEN_BACKOFF_SECONDS", (0.0,))
    monkeypatch.setattr(pipeline_module, "_STT_REOPEN_MAX_CONSECUTIVE", 3)
    monkeypatch.setattr(pipeline_module, "_STT_REOPEN_HEALTHY_SECONDS", 0.05)
    p, closed = _consumer_pipeline()
    attempts = 0

    async def idle_then_abort(queue) -> None:
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(0.08)
        p._stream_opened_at = pipeline_module.time.monotonic()
        raise STTStreamAbortedError

    p._consume_one_stream = idle_then_abort
    await asyncio.wait_for(p._stt_consumer(), timeout=5.0)
    assert closed == [EndReason.ERROR]
    assert attempts == 4


async def test_long_lived_stream_still_resets_the_counter(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "_STT_REOPEN_BACKOFF_SECONDS", (0.0,))
    monkeypatch.setattr(pipeline_module, "_STT_REOPEN_MAX_CONSECUTIVE", 2)
    monkeypatch.setattr(pipeline_module, "_STT_REOPEN_HEALTHY_SECONDS", 0.05)
    p, closed = _consumer_pipeline()
    attempts = 0

    async def healthy_then_abort(queue) -> None:
        nonlocal attempts
        attempts += 1
        if attempts >= 6:
            p._closing.set()
            return
        p._stream_opened_at = pipeline_module.time.monotonic()
        await asyncio.sleep(0.08)
        raise STTStreamAbortedError

    p._consume_one_stream = healthy_then_abort
    await asyncio.wait_for(p._stt_consumer(), timeout=5.0)
    assert closed == []
    assert attempts == 6


async def test_event_free_instant_return_is_also_backed_off(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "_STT_REOPEN_BACKOFF_SECONDS", (0.0,))
    monkeypatch.setattr(pipeline_module, "_STT_REOPEN_MAX_CONSECUTIVE", 3)
    monkeypatch.setattr(pipeline_module, "_STT_REOPEN_HEALTHY_SECONDS", 30.0)
    p, closed = _consumer_pipeline()
    attempts = 0

    async def open_and_return(queue) -> None:
        nonlocal attempts
        attempts += 1
        if attempts > 20:
            p._closing.set()
            return
        await asyncio.sleep(0)
        p._stream_opened_at = pipeline_module.time.monotonic()
        return

    p._consume_one_stream = open_and_return
    await asyncio.wait_for(p._stt_consumer(), timeout=5.0)
    assert closed == [EndReason.ERROR]
    assert attempts == 4


async def test_stream_that_delivered_events_resets_the_counter(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "_STT_REOPEN_BACKOFF_SECONDS", (0.0,))
    monkeypatch.setattr(pipeline_module, "_STT_REOPEN_MAX_CONSECUTIVE", 2)
    monkeypatch.setattr(pipeline_module, "_STT_REOPEN_HEALTHY_SECONDS", 30.0)
    p, closed = _consumer_pipeline()
    attempts = 0

    async def short_but_productive(queue) -> None:
        nonlocal attempts
        attempts += 1
        p._stream_opened_at = pipeline_module.time.monotonic()
        p._stream_had_event = True
        if attempts >= 6:
            p._closing.set()

    p._consume_one_stream = short_but_productive
    await asyncio.wait_for(p._stt_consumer(), timeout=5.0)
    assert closed == []
    assert attempts == 6


async def test_boundary_recycle_also_half_closes_and_drains(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "_STT_RECYCLE_SECONDS", 0.0)
    monkeypatch.setattr(pipeline_module, "_STT_RECYCLE_HARD_SECONDS", 30.0)
    monkeypatch.setattr(pipeline_module, "_STT_NO_AUDIO_TIMEOUT", 30.0)
    session = _FlushingSession(head=[_msg(text="다음 주 화요일이요")])
    p, handled = _flush_pipeline(_FlushingClient(session))
    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait(b"pcm")
    await asyncio.wait_for(p._consume_one_stream(queue), timeout=5.0)
    assert session.eos_received, "경계 재활용에서 half-close 가 안 나갔다"
    finals = [e.text for e in handled if e.type == STTEventType.FINAL]
    assert finals == ["다음 주 화요일이요", "다음 주 화요일로 바꿔주세요"]


class _FailingTailSTT:
    multi_utterance = True

    def stream(self, queue, first_chunk=None):
        async def _gen():
            yield STTEvent(type=STTEventType.INTERIM, text="다음 주")
            await asyncio.sleep(0.05)
            raise STTError("플러시 중 장애")
            yield  # pragma: no cover - async generator 마커

        return _gen()


async def test_failure_during_hard_deadline_flush_is_not_swallowed(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "_STT_RECYCLE_SECONDS", 0.01)
    monkeypatch.setattr(pipeline_module, "_STT_RECYCLE_HARD_SECONDS", 0.02)
    monkeypatch.setattr(pipeline_module, "_STT_NO_AUDIO_TIMEOUT", 30.0)
    p, _ = _flush_pipeline(_FailingTailSTT())
    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait(b"pcm")
    with pytest.raises(STTError):
        await asyncio.wait_for(p._consume_one_stream(queue), timeout=5.0)


def test_incremental_chunk_is_dropped_but_logged(caplog) -> None:
    import logging

    stt_module._incremental_warned = False
    with caplog.at_level(logging.WARNING, logger="app.services.stt"):
        events = stt_module.GeminiLiveSTTClient._parse_message(_msg(text="안녕", finished=False))
    assert events == []
    assert any("finished=False" in r.getMessage() for r in caplog.records)


def test_observed_server_shape_interims_accumulate_then_one_complete_final() -> None:
    interims = ["이름은", "이름은 김민준이고", "이름은 김민준이고요. 생년월일은 1998년 3월 15일이에요."]
    seen: list[STTEvent] = []
    for text in interims:
        seen += stt_module.GeminiLiveSTTClient._parse_message(
            types.LiveServerMessage(
                server_content=types.LiveServerContent(
                    interim_input_transcription=types.Transcription(text=text)
                )
            )
        )
    seen += stt_module.GeminiLiveSTTClient._parse_message(
        _msg(text="이름은 김민준이고요, 생년월일은 1998년 3월 15일이에요.")
    )
    assert [e.type for e in seen] == [STTEventType.INTERIM] * 3 + [STTEventType.FINAL]
    assert seen[2].text.startswith(interims[0])
    assert seen[-1].text.startswith(interims[0])


def test_hard_deadline_is_derived_from_the_engine_stream_limit() -> None:
    assert pipeline_module._STT_RECYCLE_HARD_SECONDS < stt_module.STREAM_LIMIT_SECONDS
    assert pipeline_module._STT_RECYCLE_SECONDS < pipeline_module._STT_RECYCLE_HARD_SECONDS
    assert (
        pipeline_module._STT_RECYCLE_HARD_SECONDS
        == stt_module.STREAM_LIMIT_SECONDS - pipeline_module._STT_HARD_RECYCLE_LEAD
    )


def test_flush_window_outlives_the_client_eos_grace() -> None:
    assert pipeline_module._STT_FLUSH_SECONDS > stt_module.EOS_GRACE_SECONDS


def test_hard_deadline_is_before_every_engine_stream_limit() -> None:
    limits = [
        stt_module.GoogleSTTClient.stream_limit_seconds,
        stt_module.GeminiLiveSTTClient.stream_limit_seconds,
    ]
    assert min(limits) > pipeline_module._STT_RECYCLE_HARD_SECONDS


class _BoundaryThenFailingSTT:
    multi_utterance = True
    stream_limit_seconds = 600

    def __init__(self) -> None:
        self.closed = False

    def stream(self, queue, first_chunk=None):
        fake = self

        async def _gen():
            try:
                yield STTEvent(type=STTEventType.FINAL, text="다음 주 화요일이요")
                raise STTError("드레인 중 장애")
            finally:
                fake.closed = True

        return _gen()


async def test_boundary_drain_failure_is_logged_not_silent(caplog, monkeypatch) -> None:
    import logging

    monkeypatch.setattr(pipeline_module, "_STT_RECYCLE_SECONDS", 0.0)
    p, handled = _flush_pipeline(_BoundaryThenFailingSTT())
    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait(b"pcm")
    with caplog.at_level(logging.WARNING, logger="app.services.pipeline"):
        await asyncio.wait_for(p._consume_one_stream(queue), timeout=5.0)
    assert [e.type for e in handled] == [STTEventType.FINAL]
    assert any("꼬리" in r.getMessage() for r in caplog.records)


async def test_boundary_recycle_closes_the_stream(monkeypatch) -> None:
    monkeypatch.setattr(pipeline_module, "_STT_RECYCLE_SECONDS", 0.0)
    monkeypatch.setattr(pipeline_module, "_STT_RECYCLE_HARD_SECONDS", 30.0)
    monkeypatch.setattr(pipeline_module, "_STT_NO_AUDIO_TIMEOUT", 30.0)
    session = _FlushingSession(head=[_msg(text="다음 주 화요일이요")])
    client = _FlushingClient(session)
    p, _ = _flush_pipeline(client)
    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait(b"pcm")
    await asyncio.wait_for(p._consume_one_stream(queue), timeout=5.0)
    assert session.closed_by_client, "재활용했는데 세션이 안 닫혔다(웹소켓 누수)"


def test_incremental_chunk_warns_once_then_drops_to_debug(caplog) -> None:
    import logging

    stt_module._incremental_warned = False
    with caplog.at_level(logging.DEBUG, logger="app.services.stt"):
        for _ in range(3):
            stt_module.GeminiLiveSTTClient._parse_message(_msg(text="안녕", finished=False))
    levels = [r.levelno for r in caplog.records if "finished=False" in r.getMessage()]
    assert levels.count(logging.WARNING) == 1
    assert levels.count(logging.DEBUG) == 2
