import asyncio

from app.core.concurrency import loop_semaphore

_shared = loop_semaphore(1)


async def _contend(sem_factory, n: int) -> int:
    inflight = 0
    peak = 0

    async def one() -> None:
        nonlocal inflight, peak
        async with sem_factory():
            inflight += 1
            peak = max(peak, inflight)
            await asyncio.sleep(0.005)
            inflight -= 1

    await asyncio.gather(*(one() for _ in range(n)))
    return peak


async def test_limits_concurrency() -> None:
    assert await _contend(loop_semaphore(1), 4) == 1


async def test_respects_configured_value() -> None:
    assert await _contend(loop_semaphore(2), 6) == 2


async def test_same_loop_shares_one_semaphore() -> None:
    factory = loop_semaphore(1)
    assert factory() is factory()


async def test_survives_a_new_event_loop_first() -> None:
    assert await _contend(lambda: _shared(), 3) == 1


async def test_survives_a_new_event_loop_second() -> None:
    assert await _contend(lambda: _shared(), 3) == 1
