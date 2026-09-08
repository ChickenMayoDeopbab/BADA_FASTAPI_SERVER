from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import suppress

from app.core.audio_stats import BYTES_PER_MS, TurnAudioStats
from app.core.metrics import now_ms

DEFAULT_TARGET_MS = 320
DEFAULT_LEAD_MARGIN_MS = 60.0


def target_bytes_for(coalesce_ms: int | None) -> int:
    """설정값에서 송출 단위 바이트"""
    if not coalesce_ms or coalesce_ms <= 0:
        return 0
    return int(coalesce_ms) * BYTES_PER_MS


async def coalesce_pcm(
    source: AsyncIterator[bytes],
    *,
    target_bytes: int,
    lead_margin_ms: float = DEFAULT_LEAD_MARGIN_MS,
    clock: Callable[[], float] | None = None,
    stats: TurnAudioStats | None = None,
) -> AsyncIterator[bytes]:
    clock = clock or (lambda: now_ms())

    if target_bytes <= 0:
        async for pcm in source:
            if stats is not None:
                stats.record_engine(pcm)
            yield pcm
        return

    queue: asyncio.Queue[bytes | BaseException | None] = asyncio.Queue()

    async def pump() -> None:
        try:
            async for pcm in source:
                if stats is not None:
                    stats.record_engine(pcm)
                if pcm:
                    await queue.put(pcm)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            await queue.put(exc)
            return
        await queue.put(None)

    pump_task = asyncio.create_task(pump())
    buf = bytearray()
    first_sent_at: float | None = None
    sent_ms = 0.0
    done = False

    def _emit(n: int) -> bytes:
        nonlocal first_sent_at, sent_ms
        out = bytes(buf[:n])
        del buf[:n]
        if first_sent_at is None:
            first_sent_at = clock()
        sent_ms += n / BYTES_PER_MS
        return out

    try:
        while not done:
            timeout: float | None = None
            if first_sent_at is not None and len(buf) >= 2:
                deadline = first_sent_at + sent_ms - lead_margin_ms
                timeout = (deadline - clock()) / 1000.0

            items: list[bytes | BaseException | None] = []
            if timeout is None:
                items.append(await queue.get())
            elif timeout > 0:
                with suppress(TimeoutError):
                    items.append(await asyncio.wait_for(queue.get(), timeout))
            while True:
                try:
                    items.append(queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            deadline_hit = not items

            pending_exc: BaseException | None = None
            for item in items:
                if item is None:
                    done = True
                    break
                if isinstance(item, BaseException):
                    pending_exc = item
                    break
                buf += item

            if first_sent_at is None and len(buf) >= 2:
                yield _emit(min(len(buf) - len(buf) % 2, target_bytes))
            while len(buf) >= target_bytes:
                yield _emit(target_bytes)
            if (deadline_hit or done or pending_exc is not None) and len(buf) >= 2:
                yield _emit(len(buf) - len(buf) % 2)
            if pending_exc is not None:
                raise pending_exc
    finally:
        if not pump_task.done():
            pump_task.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await pump_task
