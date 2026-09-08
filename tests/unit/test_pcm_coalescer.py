
import asyncio
from contextlib import suppress

import pytest

from app.core.audio_stats import TurnAudioStats
from app.services.pcm_coalescer import coalesce_pcm
from app.services.qwen_tts import QwenTTSUnavailableError

_100MS = 3200
_TARGET = 10240


async def _gen(*chunks: bytes):
    for c in chunks:
        yield c
        await asyncio.sleep(0)


async def _burst(*chunks: bytes):
    for c in chunks:
        yield c


async def _collect(agen) -> list[bytes]:
    out = []
    async for c in agen:
        out.append(c)
    return out


def _clock(values):
    it = iter(values)
    last = {"v": 0.0}

    def _now():
        with suppress(StopIteration):
            last["v"] = next(it)
        return last["v"]

    return _now


@pytest.mark.asyncio
async def test_disabled_is_pure_passthrough_even_for_odd_chunks() -> None:
    src = _gen(b"\x01" * 3201, b"\x02" * 7, b"\x03" * 3200)
    out = await _collect(coalesce_pcm(src, target_bytes=0))
    assert [len(c) for c in out] == [3201, 7, 3200]


@pytest.mark.asyncio
async def test_empty_source_yields_nothing() -> None:
    assert await _collect(coalesce_pcm(_gen(), target_bytes=_TARGET)) == []


@pytest.mark.asyncio
async def test_first_chunk_goes_out_immediately_even_aligned() -> None:
    src = _gen(b"\x01" * 1601, b"\x02" * 1599)
    out = await _collect(coalesce_pcm(src, target_bytes=_TARGET, clock=_clock([0.0])))
    assert [len(c) for c in out] == [1600, 1600]
    assert out[1][:1] == b"\x01" and out[1][1:] == b"\x02" * 1599


@pytest.mark.asyncio
async def test_batches_to_target_size_when_lead_is_ample() -> None:
    src = _gen(*([b"\x00" * 1600] * 20))
    out = await _collect(coalesce_pcm(src, target_bytes=_TARGET, clock=_clock([0.0, -1e9])))
    assert [len(c) for c in out] == [1600, _TARGET, _TARGET, 30400 - 2 * _TARGET]


@pytest.mark.asyncio
async def test_burst_already_buffered_is_cut_into_target_blocks() -> None:
    src = _burst(*([b"\x00" * 1600] * 20))
    out = await _collect(coalesce_pcm(src, target_bytes=_TARGET, clock=_clock([0.0, -1e9])))
    assert [len(c) for c in out] == [_TARGET, _TARGET, _TARGET, 32000 - 3 * _TARGET]


@pytest.mark.asyncio
async def test_deadline_flush_when_source_stalls() -> None:
    gate = asyncio.Event()

    async def src():
        yield b"\x01" * _100MS
        await asyncio.sleep(0)
        yield b"\x02" * 1600
        await gate.wait()
        yield b"\x03" * _100MS

    agen = coalesce_pcm(src(), target_bytes=_TARGET, clock=_clock([0.0, 100.0]), lead_margin_ms=60.0)
    first = await asyncio.wait_for(anext(agen), 1.0)
    second = await asyncio.wait_for(anext(agen), 1.0)
    assert (len(first), len(second)) == (_100MS, 1600)
    gate.set()
    rest = await asyncio.wait_for(_collect(agen), 1.0)
    assert [len(c) for c in rest] == [_100MS]


@pytest.mark.asyncio
async def test_holds_partial_buffer_while_lead_remains() -> None:
    gate = asyncio.Event()

    async def src():
        yield b"\x01" * _100MS
        await asyncio.sleep(0)
        yield b"\x02" * 1600
        await gate.wait()
        yield b"\x03" * 1600

    agen = coalesce_pcm(src(), target_bytes=_TARGET, clock=_clock([0.0, -1e9]))
    first = await asyncio.wait_for(anext(agen), 1.0)
    assert len(first) == _100MS
    nxt = asyncio.ensure_future(anext(agen))
    await asyncio.sleep(0.05)
    assert not nxt.done()
    gate.set()
    assert len(await asyncio.wait_for(nxt, 1.0)) == 3200
    assert await _collect(agen) == []


@pytest.mark.asyncio
async def test_odd_bytes_carry_over_and_output_is_always_even() -> None:
    src = _gen(b"\x00" * 3201, b"\x00" * 3199, b"\x00" * 5, b"\x00" * 4001)
    out = await _collect(coalesce_pcm(src, target_bytes=_TARGET, clock=_clock([0.0, -1e9])))
    assert all(len(c) % 2 == 0 for c in out)
    assert sum(len(c) for c in out) == 3201 + 3199 + 5 + 4001 - 0


@pytest.mark.asyncio
async def test_end_flush_with_odd_buffer_emits_even_part_only() -> None:
    src = _gen(b"\x00" * 3200, b"\x00" * 1601)
    out = await _collect(coalesce_pcm(src, target_bytes=_TARGET, clock=_clock([0.0, -1e9])))
    assert [len(c) for c in out] == [3200, 1600]


@pytest.mark.asyncio
async def test_deadline_flush_with_odd_buffer_keeps_the_odd_byte_for_later() -> None:
    gate = asyncio.Event()

    async def src():
        yield b"\x01" * _100MS
        await asyncio.sleep(0)
        yield b"\x02" * 1601
        await gate.wait()
        yield b"\x03" * 1599

    agen = coalesce_pcm(src(), target_bytes=_TARGET, clock=_clock([0.0, 100.0]), lead_margin_ms=60.0)
    first = await asyncio.wait_for(anext(agen), 1.0)
    second = await asyncio.wait_for(anext(agen), 1.0)
    assert (len(first), len(second)) == (_100MS, 1600)
    gate.set()
    rest = await asyncio.wait_for(_collect(agen), 1.0)
    assert [len(c) for c in rest] == [1600]
    assert rest[0][:1] == b"\x02" and rest[0][1:] == b"\x03" * 1599


@pytest.mark.asyncio
async def test_trailing_single_odd_byte_is_dropped() -> None:
    src = _gen(b"\x00" * 3200, b"\x00" * 1)
    out = await _collect(coalesce_pcm(src, target_bytes=_TARGET, clock=_clock([0.0, -1e9])))
    assert [len(c) for c in out] == [3200]


@pytest.mark.asyncio
async def test_engine_error_propagates_after_partial_output() -> None:
    async def src():
        yield b"\x00" * _100MS
        raise QwenTTSUnavailableError("worker busy")

    agen = coalesce_pcm(src(), target_bytes=_TARGET, clock=_clock([0.0, -1e9]))
    assert len(await anext(agen)) == _100MS
    with pytest.raises(QwenTTSUnavailableError):
        await anext(agen)


@pytest.mark.asyncio
async def test_consumer_cancel_closes_source() -> None:
    closed = asyncio.Event()

    async def src():
        try:
            yield b"\x00" * _100MS
            await asyncio.Event().wait()
            yield b""
        finally:
            closed.set()

    async def consume():
        async for _ in coalesce_pcm(src(), target_bytes=_TARGET, clock=_clock([0.0, -1e9])):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=1.0)
    assert task in done, "취소된 소비자가 1초 안에 안 끝났다 — 펌프를 안 닫고 기다리는 것"
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.wait_for(closed.wait(), 1.0)


@pytest.mark.asyncio
async def test_records_engine_chunks_into_stats() -> None:
    stats = TurnAudioStats(clock=_clock([0.0]))
    src = _gen(b"\x00" * 1601, b"\x00" * 1599, b"\x00" * 3200)
    out = await _collect(coalesce_pcm(src, target_bytes=_TARGET, clock=_clock([0.0, -1e9]), stats=stats))
    for c in out:
        stats.record(c)
    m = stats.as_metrics()
    assert m["engine_chunks"] == 3
    assert m["pcm_chunks"] == len(out)
    assert m["odd_chunks"] == 0


@pytest.mark.asyncio
async def test_bounded_queue_applies_backpressure_to_engine() -> None:
    pulled = {"n": 0}

    async def src():
        for _ in range(200):
            pulled["n"] += 1
            yield b"\x00" * 1600
            await asyncio.sleep(0)

    agen = coalesce_pcm(src(), target_bytes=_TARGET, clock=_clock([0.0, -1e9]), max_queue=4)
    await anext(agen)
    await asyncio.sleep(0.05)
    assert pulled["n"] <= 4 + 2
    await agen.aclose()
