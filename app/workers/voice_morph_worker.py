from __future__ import annotations

import asyncio
import logging
import multiprocessing
import signal
from concurrent.futures import ProcessPoolExecutor

from app.core.config import Settings
from app.services.morphed_recording import build_storage, ensure_morphed
from app.services.voice_morph import DEFAULT_SEMITONES, morph_pcm

logger = logging.getLogger(__name__)

_running: set[asyncio.Task] = set()
_gate = asyncio.Semaphore(1)
_SPAWN = multiprocessing.get_context("spawn")


def schedule_morph(
    *, settings: Settings, recording_key: str | None, semitones: float = DEFAULT_SEMITONES
) -> asyncio.Task | None:
    """변조본 생성 과정 백그라운드로 띄우기"""
    if not recording_key:
        return None

    task = asyncio.create_task(
        _run(settings=settings, recording_key=recording_key, semitones=semitones)
    )
    _running.add(task)
    task.add_done_callback(_running.discard)
    return task


def run_in_child(fn, *args):
    """작업마다 새 spawn 프로세스에서 실행, 워커 스레드에서만 부를 것"""
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=_SPAWN,
        initializer=signal.signal,
        initargs=(signal.SIGINT, signal.SIG_IGN),
    ) as pool:
        return pool.submit(fn, *args).result()


def morph_in_child(pcm: bytes, sr: int, semitones: float) -> bytes:
    """pyin 이 GIL 을 쥐므로 서버 프로세스 밖에서 변조"""
    return run_in_child(morph_pcm, pcm, sr, semitones)


async def _run(*, settings: Settings, recording_key: str, semitones: float) -> None:
    try:
        async with _gate:
            key = await asyncio.to_thread(
                ensure_morphed, build_storage(settings), recording_key, semitones, morph_in_child
            )
    except Exception:
        logger.exception("녹음 변조 실패", extra={"recording_key": recording_key})
        return

    if key is None:
        logger.warning("변조본을 만들지 못함", extra={"recording_key": recording_key})


async def drain(timeout: float = 30.0) -> None:
    """종료 시 진행 중인 변조를 기다림 남기면 절반 올라간 파일이 생김"""
    if not _running:
        return
    await asyncio.wait(set(_running), timeout=timeout)
