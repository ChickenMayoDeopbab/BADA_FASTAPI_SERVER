import asyncio
from types import SimpleNamespace

from fastapi import FastAPI

from app import main


async def _none(*_a, **_k):
    return None


def _wire(monkeypatch, *, gate: bool, prebake, cancelled: list) -> None:
    async def _cancel_jobs():
        cancelled.append(True)

    monkeypatch.setattr(main, "setup_logging", lambda: None)
    monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(redis_url="redis://x"))
    for name in ("init_redis", "init_db", "close_redis", "drain_avti", "drain_morphs"):
        monkeypatch.setattr(main, name, _none)
    monkeypatch.setattr(main, "usage_service", SimpleNamespace(install=lambda: None, drain=_none))
    monkeypatch.setattr(main, "prebake_presets", prebake)
    monkeypatch.setattr(main, "_prebake_gate", lambda _s: gate)
    monkeypatch.setattr(main, "cancel_example_jobs", _cancel_jobs)
    monkeypatch.setattr(main, "warm_tremor", lambda: None)


async def test_lifespan_starts_prebake_when_gate_open_and_cancels_on_shutdown(monkeypatch) -> None:
    started, stopped = asyncio.Event(), asyncio.Event()
    jobs_cancelled: list = []

    async def _prebake():
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            stopped.set()
            raise

    _wire(monkeypatch, gate=True, prebake=_prebake, cancelled=jobs_cancelled)
    async with main.lifespan(FastAPI()):
        await asyncio.wait_for(started.wait(), timeout=1.0)
        assert not stopped.is_set()
    assert stopped.is_set() and jobs_cancelled == [True]


async def test_lifespan_skips_prebake_when_gate_closed(monkeypatch) -> None:
    calls: list = []
    jobs_cancelled: list = []

    async def _prebake():
        calls.append(1)

    _wire(monkeypatch, gate=False, prebake=_prebake, cancelled=jobs_cancelled)
    async with main.lifespan(FastAPI()):
        await asyncio.sleep(0)
    assert calls == [] and jobs_cancelled == [True]


async def test_lifespan_shutdown_survives_a_failed_prebake(monkeypatch) -> None:
    async def _boom():
        raise RuntimeError("미리 굽기 실패")

    _wire(monkeypatch, gate=True, prebake=_boom, cancelled=[])
    async with main.lifespan(FastAPI()):
        await asyncio.sleep(0)
