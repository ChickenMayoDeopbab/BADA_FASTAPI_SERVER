"""세션 종료 후 떼어놓고 도는 AVTI 측정 작업.

end 프레임보다 뒤일 뿐 아니라 Spring 콜백에서도 분리돼 있다. 1단계는 이 값이
쓸 만한지 아직 모르는 섀도 수집이라, 여기서 무슨 일이 나도 세션 종료 흐름은
아무 영향을 받지 않아야 한다.
"""
from __future__ import annotations

import asyncio
import logging

from app.core.config import Settings
from app.services.avti import AvtiAnalyzer, AvtiConfig
from app.services.avti_service import save_avti_metrics

logger = logging.getLogger(__name__)

# 참조를 놓으면 실행 중인 태스크가 GC 에 조용히 취소된다.
_running: set[asyncio.Task] = set()


def build_analyzer(settings: Settings) -> AvtiAnalyzer:
    return AvtiAnalyzer(
        praat_bin=settings.praat_bin,
        script_path=settings.avti_script_path,
        config=AvtiConfig(
            window_sec=settings.avti_window_sec,
            min_sustained_sec=settings.avti_min_sustained_sec,
            timeout_sec=settings.avti_timeout_sec,
        ),
    )


def schedule_avti(
    *,
    settings: Settings,
    session_id: str,
    pcm: bytes,
    parts: list[tuple[float, float]],
    sustained_spans: list,
) -> asyncio.Task | None:
    """측정을 백그라운드로 띄운다. 끄져 있거나 재료가 없으면 아무것도 안 한다."""
    if not settings.avti_enabled or not pcm or not parts:
        return None

    task = asyncio.create_task(
        _run(
            settings=settings,
            session_id=session_id,
            pcm=pcm,
            parts=parts,
            sustained_spans=sustained_spans,
        )
    )
    _running.add(task)
    task.add_done_callback(_running.discard)
    return task


async def run_avti(
    *,
    settings: Settings,
    session_id: str,
    pcm: bytes,
    parts: list[tuple[float, float]],
    sustained_spans: list,
) -> dict[int, float]:
    """측정 후 저장하고 **파트별 AVTI** 를 돌려준다.

    {0: 세션 전체, 1..n: 사용자 턴} — 못 잰 파트는 키 자체가 없다.
    구간별 피드백이 턴 번호로 찾아 쓴다.
    """
    if not settings.avti_enabled or not pcm or not parts:
        return {}
    results = await _run(
        settings=settings,
        session_id=session_id,
        pcm=pcm,
        parts=parts,
        sustained_spans=sustained_spans,
    )
    return {r.part_index: r.avti for r in results if r.avti is not None}


async def _run(
    *,
    settings: Settings,
    session_id: str,
    pcm: bytes,
    parts: list[tuple[float, float]],
    sustained_spans: list,
) -> list:
    try:
        analyzer = build_analyzer(settings)
        if not analyzer.available:
            # 그래도 행은 남긴다. 안 남기면 '기능이 꺼짐'과 '설치가 깨짐'을
            # 테이블만 보고 구분할 수 없다 — 1단계에서 알아야 할 게 바로 그거다.
            logger.warning(
                "AVTI 실행 환경 없음 — NO_SCRIPT 로 기록",
                extra={
                    "session_id": session_id,
                    "script": settings.avti_script_path,
                    "praat": settings.praat_bin,
                },
            )

        results = await asyncio.to_thread(analyzer.analyze, pcm, parts, sustained_spans)
        await save_avti_metrics(session_id=session_id, parts=results)
        return results
    except asyncio.CancelledError:
        logger.warning("AVTI 측정 취소됨", extra={"session_id": session_id})
        raise
    except Exception:
        logger.warning(
            "AVTI 측정 실패 — 세션에는 영향 없음",
            exc_info=True,
            extra={"session_id": session_id},
        )
        return []


async def drain(timeout: float = 30.0) -> None:
    """앱 종료 시 남은 측정을 잠깐 기다린다."""
    if not _running:
        return
    await asyncio.wait(set(_running), timeout=timeout)
