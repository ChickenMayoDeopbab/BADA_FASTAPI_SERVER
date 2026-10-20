import json
import logging

import pytest
from starlette.websockets import WebSocketDisconnect

from app.schemas.frames import EndReason
from app.services.pipeline import _sanitize_playback_stats, _State
from tests.unit.test_transcript_frames import _make_pipeline

_TRIGGER = "파이프라인 종료 트리거"


class _RecvWS:
    """receive() 가 정해 둔 메시지를 돌려주거나 예외를 던진다."""

    def __init__(self, result) -> None:
        self._result = result
        self.frames: list[dict] = []

    async def receive(self) -> dict:
        if isinstance(self._result, Exception):
            raise self._result
        return self._result

    async def send_json(self, payload: dict) -> None:
        self.frames.append(payload)


def _pipeline(result=None):
    p = _make_pipeline()
    p._ws = _RecvWS(result)
    return p


def _fields(caplog) -> list[dict[str, str]]:
    """트리거 줄의 메시지 본문(텍스트 로그에 보이는 부분)을 key=value 로 푼다."""
    out = []
    for record in caplog.records:
        message = record.getMessage()
        if message.startswith(_TRIGGER):
            pairs = message[len(_TRIGGER):].split()
            out.append(dict(pair.split("=", 1) for pair in pairs))
    return out


@pytest.fixture(autouse=True)
def _capture(caplog):
    caplog.set_level(logging.INFO, logger="app")


async def test_ws_disconnect_shows_close_code(caplog) -> None:
    p = _pipeline({"type": "websocket.disconnect", "code": 1006})
    await p._recv_loop()

    [line] = _fields(caplog)
    assert line["session_id"] == "sess-transcript"
    assert line["reason"] == "USER_END"
    assert line["source"] == "ws_disconnect"
    assert line["code"] == "1006"
    assert line["state"] == "LISTENING"
    assert p._end_reason == EndReason.USER_END


async def test_end_frame_is_its_own_source(caplog) -> None:
    p = _pipeline()
    await p._handle_client_text(json.dumps({"type": "end"}))

    [line] = _fields(caplog)
    assert line["reason"] == "USER_END"
    assert line["source"] == "end_frame"
    assert "code" not in line


@pytest.mark.parametrize("exc", [RuntimeError("receive after disconnect"), WebSocketDisconnect(1001)])
async def test_receive_error_names_the_exception(caplog, exc) -> None:
    p = _pipeline(exc)
    await p._recv_loop()

    [line] = _fields(caplog)
    assert line["reason"] == "USER_END"
    assert line["source"] == "recv_error"
    assert line["error"] == type(exc).__name__
    assert p._ws_alive is False


async def test_hangup_while_ai_audio_still_playing(caplog) -> None:
    p = _pipeline()
    p._state = _State.SPEAKING
    p._begin_playback()
    p._playback.record(320000)  # 10초 분량을 막 다 보낸 상황
    await p._handle_client_text(json.dumps({"type": "end"}))

    [line] = _fields(caplog)
    assert line["state"] == "SPEAKING"
    assert 9000 < int(line["playback_left_ms"]) <= 10000


async def test_acked_playback_leaves_nothing(caplog) -> None:
    p = _pipeline()
    p._begin_playback()
    p._playback.record(320000)
    p._playback.done.set()
    await p._handle_client_text(json.dumps({"type": "end"}))

    [line] = _fields(caplog)
    assert line["playback_left_ms"] == "0"


async def test_no_playback_yet_is_zero(caplog) -> None:
    p = _pipeline()
    await p._handle_client_text(json.dumps({"type": "end"}))

    [line] = _fields(caplog)
    assert line["playback_left_ms"] == "0"


async def test_only_first_trigger_is_logged(caplog) -> None:
    p = _pipeline()
    await p._handle_client_text(json.dumps({"type": "end"}))
    await p._close(EndReason.ERROR)

    [line] = _fields(caplog)
    assert line["reason"] == "USER_END"
    assert line["source"] == "end_frame"


async def test_extra_matches_message(caplog) -> None:
    p = _pipeline({"type": "websocket.disconnect", "code": 1006})
    await p._recv_loop()

    [record] = [r for r in caplog.records if r.getMessage().startswith(_TRIGGER)]
    [line] = _fields(caplog)
    assert set(line) == {"session_id", "reason", "source", "code", "state", "playback_left_ms"}
    for key, value in line.items():
        assert str(getattr(record, key)) == value


class _Unformattable:
    def __format__(self, spec: str) -> str:
        raise ValueError("포맷 실패")


@pytest.mark.parametrize("detail", [{"source": _Unformattable()}, {"name": "LogRecord 예약 이름"}])
async def test_log_failure_does_not_escape_close(detail) -> None:
    p = _pipeline()
    await p._close(EndReason.USER_END, **detail)

    assert p._closing.is_set()
    assert p._end_reason == EndReason.USER_END


@pytest.mark.parametrize("value", ["call_end", "playback_error", "interrupt", "disposed", "completed"])
def test_playback_stats_keep_app_end_reason(value) -> None:
    out = _sanitize_playback_stats({"type": "playback_stats", "turn": 1, "end_reason": value})
    assert out == {"turn": 1, "end_reason": value}


@pytest.mark.parametrize("value", ["x" * 33, "call_end\nmetric=fake", "", 3])
def test_playback_stats_drop_bad_end_reason(value) -> None:
    out = _sanitize_playback_stats({"turn": 1, "end_reason": value})
    assert out == {"turn": 1}
