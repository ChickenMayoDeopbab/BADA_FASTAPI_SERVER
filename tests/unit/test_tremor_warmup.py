import asyncio
import threading
import time

import librosa.core.pitch
import librosa.feature.spectral
from fastapi import FastAPI

from app import main
from app.services import tremor
from tests.unit.test_preset_prebake_lifespan import _none, _wire


def _metrics(monkeypatch) -> list:
    seen: list = []
    monkeypatch.setattr(tremor, "log_metric", lambda name, **fields: seen.append((name, fields)))
    return seen


def test_warm_up_runs_the_real_yin_and_rms_path(monkeypatch) -> None:
    calls: list = []
    real_yin, real_rms = librosa.core.pitch.yin, librosa.feature.spectral.rms

    def _yin(*a, **k):
        calls.append("yin")
        return real_yin(*a, **k)

    def _rms(*a, **k):
        calls.append("rms")
        return real_rms(*a, **k)

    monkeypatch.setattr(librosa.core.pitch, "yin", _yin)
    monkeypatch.setattr(librosa.feature.spectral, "rms", _rms)
    seen = _metrics(monkeypatch)

    tremor.warm_up()

    assert "yin" in calls and "rms" in calls
    assert [name for name, _ in seen] == ["tremor_warmup"]
    assert seen[0][1]["ok"] is True and seen[0][1]["duration_ms"] >= 0


def test_warm_up_swallows_analysis_errors(monkeypatch) -> None:
    def _boom(self, pcm):
        raise RuntimeError("분석 실패")

    monkeypatch.setattr(tremor.TremorAnalyzer, "analyze", _boom)
    seen = _metrics(monkeypatch)

    tremor.warm_up()

    assert [name for name, _ in seen] == ["tremor_warmup"]
    assert seen[0][1]["ok"] is False


def test_startup_and_shutdown_do_not_wait_for_warmup(monkeypatch) -> None:
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def _slow_warm():
        started.set()
        release.wait(30)
        finished.set()

    _wire(monkeypatch, gate=False, prebake=_none, cancelled=[])
    monkeypatch.setattr(main, "warm_tremor", _slow_warm)

    async def _serve():
        async with main.lifespan(FastAPI()):
            assert await asyncio.to_thread(started.wait, 5.0)
            assert not finished.is_set()

    safety = threading.Timer(10, release.set)
    safety.start()
    try:
        t0 = time.monotonic()
        asyncio.run(_serve())
        elapsed = time.monotonic() - t0
        assert not finished.is_set() and elapsed < 5
        warm = [t for t in threading.enumerate() if t.name == "tremor-warmup"]
        assert warm and all(t.daemon for t in warm)
    finally:
        release.set()
        safety.cancel()
    assert finished.wait(5)
