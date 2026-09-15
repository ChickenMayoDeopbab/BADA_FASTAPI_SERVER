import asyncio
import logging

import pytest

from app.schemas.frames import EndReason
from app.services.pipeline import _State, _TurnTimings
from tests.unit.test_turn_watchdog import _FakeTTSClient, _HappyLLM, _make_pipeline


async def _run(p) -> None:
    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)


@pytest.mark.asyncio
async def test_finalize_failure_returns_to_listening(caplog) -> None:
    p = _make_pipeline(_HappyLLM(), _FakeTTSClient())

    async def boom(ctx, suggestion):
        raise RuntimeError("hint boom")

    p._maybe_send_script_hint = boom

    with caplog.at_level(logging.ERROR, logger="app.services.pipeline"):
        await _run(p)

    assert p._state == _State.LISTENING, "고착되면 이후 FINAL 이 전부 버려진다"
    assert p._turn_task is None
    assert p._listening_since is not None
    assert p._turn_open_at is not None, "다음 사용자 발화 구간이 열려야 한다"
    assert not p._closing.is_set(), "마무리 실패로 통화를 끊지는 않는다"
    assert any(r.levelno >= logging.ERROR for r in caplog.records), "조용히 넘어가면 안 된다"


@pytest.mark.asyncio
async def test_finalize_failure_after_close_does_not_reopen_listening() -> None:
    p = _make_pipeline(_HappyLLM(), _FakeTTSClient())

    async def close_then_boom(ctx, suggestion):
        await p._close(EndReason.ERROR)
        raise RuntimeError("boom after close")

    p._maybe_send_script_hint = close_then_boom

    await _run(p)

    assert p._state == _State.CLOSING, "끝나가는 통화를 다시 듣기 상태로 되살리면 안 된다"


@pytest.mark.asyncio
async def test_finalize_cancellation_still_propagates() -> None:
    p = _make_pipeline(_HappyLLM(), _FakeTTSClient())

    async def cancelled(ctx, suggestion):
        raise asyncio.CancelledError

    p._maybe_send_script_hint = cancelled

    with pytest.raises(asyncio.CancelledError):
        await _run(p)


@pytest.mark.asyncio
async def test_normal_turn_still_returns_to_listening_once() -> None:
    p = _make_pipeline(_HappyLLM(), _FakeTTSClient())

    await _run(p)

    assert p._state == _State.LISTENING
    assert p._turn_open_at is not None
    assert not p._closing.is_set()
