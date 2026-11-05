import asyncio
import json
import logging

import pytest

from app.core.config import get_settings
from app.services import pipeline as pipeline_mod
from app.services.biosignal_service import STATUS_AWAITING_DATA, save_call_timeline
from app.services.call_timeline import CallTimeline
from app.services.pipeline import VoicePipeline, _PlaybackTurn, _State, _TurnTimings
from tests.unit.test_avti_save import _factory
from tests.unit.test_avti_save import _pipeline as _closing_pipeline
from tests.unit.test_call_timeline import T0, Clock
from tests.unit.test_pipeline_coalesce import _make_pipeline as _turn_pipeline

_CHUNK = b"\x00\x00" * 1600


class _WS:
    def __init__(self, clock: Clock | None = None, script: list | None = None) -> None:
        self.frames: list[dict] = []
        self._clock = clock
        self._script = list(script or [])

    async def send_json(self, payload: dict) -> None:
        self.frames.append(payload)

    async def receive(self) -> dict:
        advance, msg = self._script.pop(0)
        self._clock.advance(advance)
        return msg


def _frame_pipeline(timeline: CallTimeline | None, ws: _WS | None = None) -> VoicePipeline:
    p = VoicePipeline.__new__(VoicePipeline)
    p._ws, p._ws_alive = ws or _WS(), True
    p._session_id = "sess-bio"
    p._timeline = timeline
    p._muted = False
    p._tremor_buf = bytearray()
    p._audio_queue = asyncio.Queue()
    p._state = _State.THINKING
    p._closing = asyncio.Event()
    p._playback = None
    return p


def _text(payload: dict) -> str:
    return json.dumps(payload)


def _constructed(monkeypatch, *, enabled: bool) -> VoicePipeline:
    monkeypatch.setattr(pipeline_mod, "build_stt_client", lambda settings: object())
    monkeypatch.setattr(pipeline_mod, "LLMClient", lambda: object())
    monkeypatch.setattr(pipeline_mod, "RecordingStorageService", lambda s: object())
    settings = get_settings().model_copy(update={"biosignal_enabled": enabled})
    return VoicePipeline(ws=object(), session_id="s-1", session={}, settings=settings, spring=object())


def test_timeline_exists_only_when_the_flag_is_on(monkeypatch) -> None:
    assert _constructed(monkeypatch, enabled=False)._timeline is None
    assert isinstance(_constructed(monkeypatch, enabled=True)._timeline, CallTimeline)


async def test_client_frames_fill_the_timeline() -> None:
    clock = Clock()
    tl = CallTimeline(clock)
    p = _frame_pipeline(tl)
    p._playback = _PlaybackTurn(turn_id=3, sent_end=True)

    intent = {"type": "biosignal_intent", "v": 1, "source": "samsung_sensor_sdk", "phone_ms": T0 - 40}
    await p._handle_client_text(_text(intent))
    clock.advance(1000)
    await p._handle_client_text(_text({"type": "ping", "phone_ms": T0 + 960}))
    await p._handle_client_text(_text({"type": "mute", "muted": True}))
    await p._handle_client_text(_text({"type": "mute", "muted": True}))
    await p._handle_client_text(_text({"type": "mute", "muted": False}))
    await p._handle_client_text(_text({"type": "playback_done", "turn_id": 3}))
    await p._handle_client_text(_text({"type": "playback_done", "turn_id": 3}))
    clock.advance(500)
    await p._handle_client_text(_text({"type": "end", "phone_ms": T0 + 1460}))

    assert tl.intent == {"v": 1, "source": "samsung_sensor_sdk", "at_ms": T0}
    assert tl.clock_pairs == [[T0 - 40, T0], [T0 + 960, T0 + 1000], [T0 + 1460, T0 + 1500]]
    assert [(e["kind"], e["at_ms"], e.get("playback"), e.get("reason")) for e in tl.events] == [
        ("mute_on", T0 + 1000, None, None),
        ("mute_off", T0 + 1000, None, None),
        ("ai_playback_done", T0 + 1000, 3, None),
        ("call_close", T0 + 1500, None, "USER_END"),
    ]
    assert {"type": "pong"} in p._ws.frames


async def test_frames_without_a_timeline_behave_as_before() -> None:
    p = _frame_pipeline(None)
    p._playback = _PlaybackTurn(turn_id=1, sent_end=True)

    await p._handle_client_text(_text({"type": "biosignal_intent", "v": 1, "phone_ms": T0}))
    await p._handle_client_text(_text({"type": "ping", "phone_ms": T0}))
    await p._handle_client_text(_text({"type": "mute", "muted": True}))
    await p._handle_client_text(_text({"type": "playback_done", "turn_id": 1}))
    await p._handle_client_text(_text({"type": "end", "phone_ms": T0}))

    assert p._timeline is None
    assert p._ws.frames == [{"type": "pong"}]
    assert p._muted is True
    assert p._playback.done.is_set()
    assert p._closing.is_set()


async def test_mic_audio_adds_anchors_and_muted_audio_is_left_out() -> None:
    clock = Clock()
    tl = CallTimeline(clock)
    ws = _WS(clock, [
        (0, {"type": "websocket.receive", "bytes": _CHUNK}),
        (100, {"type": "websocket.receive", "bytes": _CHUNK}),
        (100, {"type": "websocket.receive", "bytes": _CHUNK}),
        (0, {"type": "websocket.receive", "text": _text({"type": "mute", "muted": True})}),
        (1000, {"type": "websocket.receive", "bytes": _CHUNK}),
        (0, {"type": "websocket.receive", "text": _text({"type": "mute", "muted": False})}),
        (100, {"type": "websocket.receive", "bytes": _CHUNK}),
        (0, {"type": "websocket.disconnect", "code": 1000}),
    ])
    p = _frame_pipeline(tl, ws)

    await asyncio.wait_for(p._recv_loop(), timeout=2.0)

    assert tl.anchors == [[T0, 0.0], [T0 + 1300, 0.3]]
    assert [e["kind"] for e in tl.events] == ["audio_first", "mute_on", "mute_off", "call_close"]
    assert len(p._tremor_buf) == 4 * len(_CHUNK)


async def test_turn_marks_ai_audio_speaking_end_and_the_next_user_turn() -> None:
    p = _turn_pipeline(coalesce_ms=320)
    tl = CallTimeline(Clock())
    p._timeline = tl
    p._user_turn_texts, p._user_turn_prompts = [], []
    p._turn_open_at = 0.0

    p._close_user_turn("여보세요")
    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)
    await asyncio.wait_for(p._speak_fallback(), timeout=2.0)

    assert [(e["kind"], e.get("turn"), e.get("playback")) for e in tl.events] == [
        ("user_turn_close", 1, None),
        ("ai_audio_start", None, 1),
        ("ai_speaking_end", None, 1),
        ("user_turn_open", 2, None),
        ("ai_audio_start", None, 2),
    ]


async def _no_avti(**kwargs):
    return {}


async def _no_feedback(**kwargs):
    return True


def _closing_call(monkeypatch, *, timeline=True, intent=True, session=None, fail=False):
    monkeypatch.setattr(pipeline_mod, "run_avti", _no_avti)
    monkeypatch.setattr(pipeline_mod, "save_feedback", _no_feedback)
    log: list[str] = []
    calls: list[dict] = []

    async def _save(**kwargs):
        log.append("biosignal")
        calls.append(kwargs)
        if fail:
            raise RuntimeError("DB 연결 끊김")
        return True

    monkeypatch.setattr(pipeline_mod, "save_call_timeline", _save)
    p = _closing_pipeline(seconds=5, turns=[(1.0, 4.0)], spans=[(0.0, 4.0)], log=log)
    p._session = session if session is not None else {"type": "SCENARIO", "userId": 7, "scenarioId": 3}
    if timeline:
        p._timeline = CallTimeline(Clock())
        if intent:
            p._timeline.intend({"v": 1, "phone_ms": T0})
    return p, log, calls


def _metrics(caplog, name: str) -> list:
    return [r for r in caplog.records if r.name == "app.metrics" and getattr(r, "metric", None) == name]


async def test_timeline_is_saved_after_the_end_frame(monkeypatch, caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    p, log, calls = _closing_call(monkeypatch)

    await p._teardown()

    assert log[-2:] == ["end_frame", "biosignal"]
    assert len(calls) == 1
    saved = calls[0]
    assert {k: saved[k] for k in ("session_id", "user_id", "scenario_id", "session_type", "difficulty")} == {
        "session_id": "sess-1", "user_id": 7, "scenario_id": 3, "session_type": "SCENARIO", "difficulty": None,
    }
    assert saved["timeline"]["intent"] == {"v": 1, "source": None, "at_ms": T0}
    assert saved["timeline"]["clock_pairs"] == [[T0, T0]]
    (metric,) = _metrics(caplog, "biosignal_timeline")
    assert metric.saved is True
    assert metric.clock_pairs == 1


@pytest.mark.parametrize("timeline, intent", [(False, False), (True, False)])
async def test_nothing_is_saved_without_a_timeline_or_intent(monkeypatch, caplog, timeline, intent) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    p, log, calls = _closing_call(monkeypatch, timeline=timeline, intent=intent)

    await p._teardown()

    assert calls == []
    assert log[-1] == "end_frame"
    assert _metrics(caplog, "biosignal_timeline") == []


async def test_save_failure_is_logged_and_the_close_still_finishes(monkeypatch, caplog) -> None:
    caplog.set_level(logging.INFO)
    p, log, calls = _closing_call(monkeypatch, fail=True)

    await p._teardown()

    assert "end_frame" in log
    assert any("생체신호 사건표 저장 실패" in r.getMessage() for r in caplog.records)
    (metric,) = _metrics(caplog, "biosignal_timeline")
    assert metric.saved is False


async def test_missing_user_id_skips_the_save(monkeypatch, caplog) -> None:
    caplog.set_level(logging.INFO)
    p, log, calls = _closing_call(monkeypatch, session={"type": "SCENARIO", "scenarioId": 3})

    await p._teardown()

    assert calls == []
    assert any("세션에 userId 없음" in r.getMessage() for r in caplog.records)
    (metric,) = _metrics(caplog, "biosignal_timeline")
    assert metric.saved is False


async def test_save_call_timeline_inserts_an_awaiting_row_once() -> None:
    store: dict = {}
    timeline = {"v": 1, "events": []}

    saved = await save_call_timeline(
        session_id="sess-1", user_id=7, scenario_id=3, session_type="SCENARIO", difficulty="easy",
        timeline=timeline, session_factory=_factory(store),
    )
    again = await save_call_timeline(
        session_id="sess-1", user_id=7, scenario_id=3, session_type="SCENARIO", difficulty="easy",
        timeline=timeline, session_factory=_factory({}, rowcount=0),
    )

    values = set(map(str, store["params"].values()))
    assert saved is True and again is False
    assert {"sess-1", "7", "3", "SCENARIO", "easy", STATUS_AWAITING_DATA} <= values
    assert store["committed"] is True
