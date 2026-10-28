import asyncio
import inspect
import multiprocessing
import os
import signal
import threading
import time
from concurrent.futures.process import BrokenProcessPool
from types import SimpleNamespace

import numpy as np
import pytest

from app.services.morphed_recording import ensure_morphed, morphed_key
from app.services.voice_morph import DEFAULT_SEMITONES, morph_pcm
from app.workers import voice_morph_worker as worker
from tests.unit import _spawn_probes as probes

SR = 16_000


def _pcm(seconds: float = 1.0, f0: float = 180.0) -> bytes:
    t = np.arange(int(seconds * SR)) / SR
    y = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 15))
    return (y / np.abs(y).max() * 0.4 * 32767).astype(np.int16).tobytes()


class _Storage:

    def __init__(self, pcm: bytes) -> None:
        self.pcm = pcm
        self.uploaded: dict[str, bytes] = {}

    def exists(self, key: str) -> bool:
        return key in self.uploaded

    def download_pcm(self, key: str) -> bytes:
        return self.pcm

    def upload_wav(self, key: str, pcm: bytes) -> str:
        self.uploaded[key] = pcm
        return key


def _wire(monkeypatch, storage: _Storage) -> None:
    monkeypatch.setattr(worker, "build_storage", lambda settings: storage)
    monkeypatch.setattr(worker, "_gate", asyncio.Semaphore(1))


def test_run_in_child_runs_in_a_spawned_not_forked_process(monkeypatch) -> None:
    monkeypatch.setattr(probes, "INHERITED", True)

    pid, inherited = worker.run_in_child(probes.probe)

    assert pid != os.getpid()
    assert inherited is False, "fork 처럼 부모 상태를 물려받았다"


def test_each_call_gets_a_new_child_that_is_gone_afterwards() -> None:
    pid1, _ = worker.run_in_child(probes.probe)
    pid2, _ = worker.run_in_child(probes.probe)

    assert pid1 != pid2
    assert multiprocessing.active_children() == [], "끝난 뒤에도 자식이 남아 있다"


def test_child_ignores_sigint_so_ctrl_c_reaches_only_the_server() -> None:
    assert worker.run_in_child(signal.getsignal, signal.SIGINT) == signal.SIG_IGN


def test_child_errors_surface_and_a_dead_child_does_not_poison_the_next_call() -> None:
    with pytest.raises(ValueError, match="자식 예외"):
        worker.run_in_child(probes.boom)
    with pytest.raises(BrokenProcessPool):
        worker.run_in_child(probes.die)

    pid, _ = worker.run_in_child(probes.probe)

    assert pid != os.getpid()


def test_morph_in_child_matches_in_process_morph_byte_for_byte() -> None:
    pcm = _pcm(1.0)

    assert worker.morph_in_child(pcm, SR, DEFAULT_SEMITONES) == morph_pcm(pcm, SR, DEFAULT_SEMITONES)


async def test_worker_morphs_through_the_child_not_in_the_server(monkeypatch) -> None:
    calls: list = []

    def _fake_child(fn, *args):
        calls.append(fn)
        return fn(*args)

    storage = _Storage(_pcm(0.5))
    _wire(monkeypatch, storage)
    monkeypatch.setattr(worker, "run_in_child", _fake_child)

    await worker._run(settings=SimpleNamespace(), recording_key="rec/a.wav", semitones=5.5)

    assert calls == [morph_pcm]
    assert storage.uploaded == {morphed_key("rec/a.wav", 5.5): morph_pcm(storage.pcm, SR, 5.5)}


async def test_morph_does_not_stall_the_event_loop(monkeypatch) -> None:
    storage = _Storage(b"\x01\x00" * 100)
    _wire(monkeypatch, storage)
    monkeypatch.setattr(worker, "morph_pcm", probes.hold_gil)
    gaps: list[float] = []
    done = asyncio.Event()

    async def _tick() -> None:
        last = time.monotonic()
        while not done.is_set():
            await asyncio.sleep(0.005)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    ticker = asyncio.create_task(_tick())
    await asyncio.sleep(0.05)
    await worker._run(settings=SimpleNamespace(), recording_key="rec/a.wav", semitones=5.5)
    done.set()
    await ticker

    assert morphed_key("rec/a.wav", 5.5) in storage.uploaded
    assert max(gaps) < 0.5, f"루프가 {max(gaps):.2f}초 멈췄다"


async def test_overlapping_jobs_run_one_at_a_time_and_share_the_result(monkeypatch) -> None:
    active, peak = 0, 0
    morphed: list = []
    lock = threading.Lock()

    def _slow_child(fn, *args):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.2)
        with lock:
            active -= 1
        morphed.append(args)
        return b"morphed"

    storage = _Storage(b"\x01\x00" * 100)
    _wire(monkeypatch, storage)
    monkeypatch.setattr(worker, "run_in_child", _slow_child)
    settings = SimpleNamespace()

    tasks = [
        worker.schedule_morph(settings=settings, recording_key=key)
        for key in ("rec/a.wav", "rec/a.wav", "rec/b.wav")
    ]
    await asyncio.wait_for(asyncio.gather(*tasks), 10)

    assert peak == 1
    assert len(morphed) == 2, "같은 녹음을 두 번 변조했다"
    assert set(storage.uploaded) == {morphed_key("rec/a.wav"), morphed_key("rec/b.wav")}


async def test_child_failure_is_logged_and_releases_the_gate(monkeypatch, caplog) -> None:
    failed: list = []

    def _flaky_child(fn, *args):
        if not failed:
            failed.append(True)
            raise BrokenProcessPool("자식이 죽음")
        return b"morphed"

    storage = _Storage(b"\x01\x00" * 100)
    _wire(monkeypatch, storage)
    monkeypatch.setattr(worker, "run_in_child", _flaky_child)
    settings = SimpleNamespace()

    await worker._run(settings=settings, recording_key="rec/a.wav", semitones=5.5)
    await asyncio.wait_for(
        worker._run(settings=settings, recording_key="rec/b.wav", semitones=5.5), 5
    )

    assert "녹음 변조 실패" in caplog.text
    assert storage.uploaded == {morphed_key("rec/b.wav", 5.5): b"morphed"}


async def test_real_gate_admits_one_job() -> None:
    await worker._gate.acquire()
    try:
        assert worker._gate.locked(), "자식이 한 번에 둘 이상 뜰 수 있다"
    finally:
        worker._gate.release()


def test_ensure_morphed_uses_the_given_morph_and_defaults_to_in_process() -> None:
    storage = _Storage(b"\x01\x00" * 100)

    key = ensure_morphed(storage, "rec/a.wav", 5.5, morph=lambda pcm, sr, semitones: b"X")

    assert storage.uploaded == {key: b"X"}
    assert inspect.signature(ensure_morphed).parameters["morph"].default is morph_pcm
