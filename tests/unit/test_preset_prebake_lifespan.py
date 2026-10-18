import asyncio
from types import SimpleNamespace

from fastapi import FastAPI

from app import main


async def test_lifespan_starts_preset_prebake_and_cancels_it_on_shutdown(monkeypatch) -> None:
    started, cancelled = asyncio.Event(), asyncio.Event()

    async def _prebake():
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    async def _none(*_a, **_k):
        return None

    monkeypatch.setattr(main, "setup_logging", lambda: None)
    monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(redis_url="redis://x"))
    monkeypatch.setattr(main, "init_redis", _none)
    monkeypatch.setattr(main, "init_db", _none)
    monkeypatch.setattr(main, "close_redis", _none)
    monkeypatch.setattr(main, "drain_avti", _none)
    monkeypatch.setattr(main, "drain_morphs", _none)
    monkeypatch.setattr(main, "usage_service", SimpleNamespace(install=lambda: None, drain=_none))
    monkeypatch.setattr(main, "prebake_presets", _prebake)

    async with main.lifespan(FastAPI()):
        await asyncio.wait_for(started.wait(), timeout=1.0)
        assert not cancelled.is_set()
    assert cancelled.is_set()


async def test_lifespan_shutdown_survives_a_failed_prebake(monkeypatch) -> None:
    async def _none(*_a, **_k):
        return None

    async def _boom():
        raise RuntimeError("미리 굽기 실패")

    monkeypatch.setattr(main, "setup_logging", lambda: None)
    monkeypatch.setattr(main, "get_settings", lambda: SimpleNamespace(redis_url="redis://x"))
    for name in ("init_redis", "init_db", "close_redis", "drain_avti", "drain_morphs"):
        monkeypatch.setattr(main, name, _none)
    monkeypatch.setattr(main, "usage_service", SimpleNamespace(install=lambda: None, drain=_none))
    monkeypatch.setattr(main, "prebake_presets", _boom)

    async with main.lifespan(FastAPI()):
        await asyncio.sleep(0)
