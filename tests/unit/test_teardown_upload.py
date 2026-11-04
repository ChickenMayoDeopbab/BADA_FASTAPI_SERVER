import asyncio
import gc
import logging
import threading
import time
from contextlib import suppress

from app.services import pipeline as pipeline_mod
from tests.unit.test_avti_save import _FakeTremor, _pipeline

_WAIT_S = 2.0
_SPANS = [(0.0, 4.0)]


async def _no_avti(**kwargs):
    return {}


def _five_second_call(monkeypatch, storage, *, tremor=None, log=None):
    monkeypatch.setattr("app.services.pipeline.run_avti", _no_avti)
    p = _pipeline(seconds=5, turns=[(1.0, 4.0)], spans=_SPANS, log=log)
    p._recording_storage = storage
    if tremor is not None:
        p._tremor = tremor
    return p


def _capture_callback(p, captured: dict, order: list | None = None) -> None:
    async def _capture(*args, **kwargs):
        if order is not None:
            order.append("spring")
        captured.update(kwargs)
        return True

    p._spring.notify_session_closed = _capture


async def test_upload_runs_while_the_tremor_analysis_runs(monkeypatch) -> None:
    uploading, analyzing = threading.Event(), threading.Event()
    seen: dict = {}

    class _Storage:
        def upload_pcm(self, session_id, pcm):
            uploading.set()
            seen["analysis_running"] = analyzing.wait(_WAIT_S)
            return "rec.wav"

    class _Tremor(_FakeTremor):
        def analyze(self, pcm):
            analyzing.set()
            uploading.wait(_WAIT_S)
            return super().analyze(pcm)

    p = _five_second_call(monkeypatch, _Storage(), tremor=_Tremor(_SPANS))
    await p._teardown()

    assert seen["analysis_running"] is True


async def test_callback_waits_for_an_upload_that_finishes_last(monkeypatch) -> None:
    order: list[str] = []
    captured: dict = {}

    class _Storage:
        def upload_pcm(self, session_id, pcm):
            time.sleep(0.2)
            order.append("upload")
            return "rec.wav"

    p = _five_second_call(monkeypatch, _Storage(), log=order)
    _capture_callback(p, captured, order)
    await p._teardown()

    assert captured["recording_key"] == "rec.wav"
    assert order == ["upload", "spring", "end_frame"]


async def test_upload_failure_keeps_the_warning_and_closes(monkeypatch, caplog) -> None:
    order: list[str] = []
    captured: dict = {}

    class _Storage:
        def upload_pcm(self, session_id, pcm):
            raise RuntimeError("S3 끊김")

    p = _five_second_call(monkeypatch, _Storage(), log=order)
    _capture_callback(p, captured, order)
    with caplog.at_level(logging.WARNING, logger="app.services.pipeline"):
        await p._teardown()

    assert captured["recording_key"] is None
    assert order == ["spring", "end_frame"]
    assert "녹음본 업로드 실패" in caplog.text


async def test_upload_metric_separates_upload_time_from_the_wait(monkeypatch) -> None:
    metrics: list = []
    monkeypatch.setattr(pipeline_mod, "log_metric", lambda name, **kw: metrics.append((name, kw)))

    class _Storage:
        def upload_pcm(self, session_id, pcm):
            time.sleep(0.1)
            return "rec.wav"

    class _Tremor(_FakeTremor):
        def analyze(self, pcm):
            time.sleep(0.4)
            return super().analyze(pcm)

    p = _five_second_call(monkeypatch, _Storage(), tremor=_Tremor(_SPANS))
    await p._teardown()

    [fields] = [kw for name, kw in metrics if name == "recording_upload"]
    assert fields["session_id"] == "sess-1"
    assert fields["stored"] is True
    assert fields["pcm_bytes"] == 16000 * 2 * 5
    assert fields["upload_ms"] >= 100
    assert 0 <= fields["waited_ms"] < fields["upload_ms"]


async def test_cancelled_close_still_reports_the_upload(monkeypatch, caplog) -> None:
    analyzing, release, uploaded = threading.Event(), threading.Event(), threading.Event()
    errors: list = []
    loop = asyncio.get_running_loop()
    loop.set_exception_handler(lambda _loop, context: errors.append(context))

    class _Storage:
        def upload_pcm(self, session_id, pcm):
            try:
                analyzing.wait(_WAIT_S)
                raise RuntimeError("S3 끊김")
            finally:
                uploaded.set()

    class _Tremor(_FakeTremor):
        def analyze(self, pcm):
            analyzing.set()
            release.wait(_WAIT_S)
            return super().analyze(pcm)

    p = _five_second_call(monkeypatch, _Storage(), tremor=_Tremor(_SPANS))
    try:
        with caplog.at_level(logging.WARNING, logger="app.services.pipeline"):
            teardown = asyncio.create_task(p._teardown())
            await asyncio.to_thread(analyzing.wait, _WAIT_S)
            teardown.cancel()
            with suppress(asyncio.CancelledError):
                await teardown
            release.set()
            await asyncio.to_thread(uploaded.wait, _WAIT_S)
            others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            if others:
                await asyncio.wait(others, timeout=_WAIT_S)
        del teardown, others
        gc.collect()
    finally:
        release.set()
        loop.set_exception_handler(None)

    assert uploaded.is_set()
    assert "녹음본 업로드 실패" in caplog.text
    assert not errors
