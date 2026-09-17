import asyncio
import json

import pytest

from app.schemas.frames import EndReason
from app.services import pipeline as pipeline_mod
from app.services.pipeline import _State
from tests.unit.test_transcript_frames import _make_pipeline


async def _pending_playback():
    p = _make_pipeline()
    p._state = _State.SPEAKING
    p._begin_playback()
    p._playback.record(320000)  # 재생 10초 분량을 즉시 송출한 상황
    await p._send_speaking_end()
    return p


async def _ack(p, turn_id):
    await p._handle_client_text(json.dumps({"type": "playback_done", "turn_id": turn_id}))


@pytest.mark.parametrize("reason", [EndReason.END_CALL, EndReason.SCENARIO_DONE, EndReason.TIMEOUT])
async def test_normal_close_keeps_receiver_alive_until_playback_ack(reason):
    p = await _pending_playback()
    close = asyncio.create_task(p._close(reason))
    try:
        await asyncio.sleep(0)
        assert p._state == _State.CLOSING
        assert not p._closing.is_set(), "수신 루프가 ACK 전에 종료되면 안 된다"
        assert not close.done()
        turn_id = p._ws.frames[-1]["turn_id"]
        await _ack(p, turn_id)
        await asyncio.wait_for(close, 0.5)
        assert p._closing.is_set()
        assert p._end_reason == reason
    finally:
        close.cancel()
        await asyncio.gather(close, return_exceptions=True)


@pytest.mark.parametrize("reason", [EndReason.USER_END, EndReason.ERROR])
async def test_manual_end_and_error_interrupt_normal_close_without_overwriting_reason(reason):
    p = await _pending_playback()
    close = asyncio.create_task(p._close(EndReason.END_CALL))
    try:
        await asyncio.sleep(0)
        await asyncio.wait_for(p._close(reason), 0.5)
        await asyncio.wait_for(close, 0.5)
        assert p._end_reason == reason
    finally:
        close.cancel()
        await asyncio.gather(close, return_exceptions=True)


@pytest.mark.parametrize("turn_id", [0, -1, 2, True, "1", 1.0, None])
async def test_wrong_or_malformed_ack_does_not_complete_current_playback(turn_id):
    p = await _pending_playback()
    await _ack(p, turn_id)
    assert not p._playback.done.is_set()


async def test_premature_duplicate_and_previous_turn_ack():
    p = _make_pipeline()
    p._begin_playback()
    await _ack(p, 1)
    assert not p._playback.done.is_set()
    await p._send_speaking_end()
    await _ack(p, 1)
    await _ack(p, 1)
    assert p._playback.done.is_set()
    p._begin_playback()
    await p._send_speaking_end()
    await _ack(p, 1)
    assert not p._playback.done.is_set()


@pytest.mark.parametrize("supports_ack,expected_wait", [(False, 8.3306), (True, 22.3306)])
async def test_missing_ack_uses_bounded_deadline_from_actual_pcm(monkeypatch, supports_ack, expected_wait):
    clock = [0.0]
    monkeypatch.setattr(pipeline_mod, "now_ms", lambda: clock[0])
    p = _make_pipeline()
    await p._handle_client_text(json.dumps({"type": "playback_capabilities", "completion_ack": supports_ack}))
    p._begin_playback()
    p._playback.record(409600)  # 로그의 마지막 발화 12.8초
    clock[0] = 5469.4
    await p._send_speaking_end()
    timeouts = []

    async def expire(tasks, *, timeout, return_when):
        timeouts.append(timeout)
        return set(), tasks

    monkeypatch.setattr(pipeline_mod.asyncio, "wait", expire)
    await p._close(EndReason.END_CALL)
    assert timeouts == pytest.approx([expected_wait])
    assert p._closing.is_set()


async def test_ack_timeout_has_hard_upper_bound(monkeypatch):
    p = await _pending_playback()
    p._playback.record(32000000)
    timeouts = []

    async def expire(tasks, *, timeout, return_when):
        timeouts.append(timeout)
        return set(), tasks

    monkeypatch.setattr(pipeline_mod.asyncio, "wait", expire)
    await p._close(EndReason.END_CALL)
    assert timeouts == [60.0]


def test_fallback_keeps_unplayed_audio_and_accounts_for_queue_starvation(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(pipeline_mod, "now_ms", lambda: clock[0])
    p = _make_pipeline()
    p._begin_playback()
    p._playback.record(32000)
    clock[0] = 200.0
    p._begin_playback()
    p._playback.record(32000)
    assert p._playback.expected_end_ms == 2000.0
    clock[0] = 3000.0
    p._playback.record(16000)
    assert p._playback.expected_end_ms == 3500.0


@pytest.mark.parametrize("disconnected", [False, True])
async def test_no_audio_or_disconnected_socket_closes_immediately(disconnected):
    p = _make_pipeline()
    if disconnected:
        p = await _pending_playback()
        p._ws_alive = False
    await asyncio.wait_for(p._close(EndReason.END_CALL), 0.5)
    assert p._closing.is_set()


async def test_cancelled_playback_wait_reaps_its_event_tasks():
    p = await _pending_playback()
    before = asyncio.all_tasks()
    close = asyncio.create_task(p._close(EndReason.END_CALL))
    await asyncio.sleep(0)
    close.cancel()
    with pytest.raises(asyncio.CancelledError):
        await close
    assert not (asyncio.all_tasks() - before)
    assert not p._closing.is_set()
