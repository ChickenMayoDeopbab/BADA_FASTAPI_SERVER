import asyncio
import json
import time

import pytest

from tests.perf.ws_bench import _recv_loop


class _FakeWS:

    def __init__(self, script: list[tuple[float, object]]) -> None:
        self._script = list(script)

    async def recv(self):
        if not self._script:
            await asyncio.sleep(3600)
        delay, msg = self._script.pop(0)
        await asyncio.sleep(delay)
        return json.dumps(msg) if isinstance(msg, dict) else msg


def _user(text: str) -> dict:
    return {"type": "transcript", "role": "user", "text": text}


def _ai(text: str) -> dict:
    return {"type": "transcript", "role": "ai", "text": text}


async def test_stt_final_measured_from_speech_end_and_transcript_kept() -> None:
    ws = _FakeWS([
        (0.05, _user("안녕하세요")), (0.01, _ai("네")), (0.01, b"\x00\x00"), (0.01, {"type": "speaking_end"}),
    ])
    speech_end = time.perf_counter()
    res = await _recv_loop(ws, speech_end=speech_end, timeout=2.0)
    assert res["terminal"] is None
    assert res["transcript"] == "안녕하세요"
    assert 40 <= res["stt_final_ms"] <= 400
    assert res["client_response_ms"] is not None and res["client_turn_ms"] is not None


async def test_no_user_transcript_leaves_stt_final_none() -> None:
    ws = _FakeWS([(0.01, b"\x00\x00"), (0.01, {"type": "speaking_end"})])
    res = await _recv_loop(ws, speech_end=time.perf_counter(), timeout=2.0)
    assert res["stt_final_ms"] is None
    assert res["transcript"] is None


async def test_timeout_without_final_is_reported() -> None:
    ws = _FakeWS([])
    res = await _recv_loop(ws, speech_end=time.perf_counter(), timeout=0.05)
    assert res["terminal"] == "TIMEOUT"
    assert res["stt_final_ms"] is None


async def test_end_frame_terminates_with_reason() -> None:
    ws = _FakeWS([(0.01, {"type": "end", "reason": "ERROR"})])
    res = await _recv_loop(ws, speech_end=time.perf_counter(), timeout=1.0)
    assert res["terminal"] == "end"


@pytest.mark.parametrize("bad", [b"", "not json"])
async def test_unparseable_text_is_ignored(bad) -> None:
    ws = _FakeWS([(0.0, "{}"), (0.01, {"type": "speaking_end"})])
    res = await _recv_loop(ws, speech_end=time.perf_counter(), timeout=1.0)
    assert res["terminal"] is None


async def test_ai_transcript_before_user_is_not_counted_as_final() -> None:
    ws = _FakeWS([(0.0, _ai("여보세요")), (0.06, _user("안녕하세요")), (0.01, {"type": "speaking_end"})])
    res = await _recv_loop(ws, speech_end=time.perf_counter(), timeout=1.0)
    assert res["transcript"] == "안녕하세요"
    assert res["stt_final_ms"] >= 50


async def test_run_turn_reports_closed_when_send_fails() -> None:
    from websockets.exceptions import ConnectionClosedOK

    from tests.perf.ws_bench import _run_turn

    class _ClosedWS:
        async def send(self, data):
            raise ConnectionClosedOK(None, None)

        async def recv(self):
            raise ConnectionClosedOK(None, None)

    res = await _run_turn(_ClosedWS(), audio=b"\x00" * 3200, silence=b"", chunk_bytes=3200, chunk_s=0.0, timeout=1.0)
    assert res["terminal"] == "CLOSED"
    assert res["stt_final_ms"] is None


async def test_turn_gap_silence_is_sent_after_response() -> None:
    from tests.perf.ws_bench import _run_turn

    class _WS:
        def __init__(self) -> None:
            self.sent: list[bytes] = []
            self._replied = False

        async def send(self, data):
            self.sent.append(data)

        async def recv(self):
            if not self._replied:
                self._replied = True
                return json.dumps({"type": "speaking_end"})
            await asyncio.sleep(3600)

    ws = _WS()
    gap = b"\x00" * 6400
    res = await _run_turn(
        ws, audio=b"\x01" * 3200, silence=b"", chunk_bytes=3200, chunk_s=0.0, timeout=1.0, turn_gap=gap
    )
    assert res["terminal"] is None
    assert sum(len(b) for b in ws.sent if b and b[0] == 0) == 6400
