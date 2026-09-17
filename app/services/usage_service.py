from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping

from sqlalchemy.exc import IntegrityError

from app.core import usage as usage_core
from app.core.concurrency import loop_semaphore
from app.core.timeutil import now_utc
from app.db.base import AsyncSessionLocal
from app.db.models import UsageEventORM

logger = logging.getLogger(__name__)

_KIND = {"session_usage": "session", "llm_usage": "llm", "tts_usage": "tts"}

_running: set[asyncio.Task] = set()

# 사용량 기록이 DB 커넥션 풀(pool_size 10)을 잠식하지 않게 동시 쓰기를 제한한다(리뷰 지적).
_WRITE_MAX_CONCURRENCY = 2
_write_semaphore = loop_semaphore(_WRITE_MAX_CONCURRENCY)


def _int_or_none(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _str_or_none(value: object, limit: int) -> str | None:
    if value is None:
        return None
    return str(value)[:limit]


def build_row(kind: str, fields: Mapping[str, object]) -> dict[str, object]:
    """지표 dict → 컬럼"""
    short = _KIND.get(kind)
    if short is None:
        raise ValueError(f"알 수 없는 사용량 종류: {kind}")
    if short == "llm":
        provider = fields.get("provider")
    elif short == "tts":
        provider = fields.get("engine")
    else:
        provider = None
    return {
        "kind": short,
        "session_id": _str_or_none(fields.get("session_id"), 64),
        "user_id": _int_or_none(fields.get("user_id")),
        "scenario_id": _int_or_none(fields.get("scenario_id")),
        "provider": _str_or_none(provider, 24),
        "model": _str_or_none(fields.get("model"), 64),
        "purpose": _str_or_none(fields.get("purpose"), 32),
        "payload": dict(fields),
        "created_at": now_utc(),
    }


async def record_event(
    kind: str,
    fields: Mapping[str, object],
    *,
    session_factory=None,
) -> bool:
    """행 하나 저장, 중복은 건너뜀"""
    try:
        row = build_row(kind, fields)
    except Exception:
        logger.warning("사용량 행 변환 실패(무시)", exc_info=True, extra={"kind": kind})
        return False
    factory = session_factory or AsyncSessionLocal
    try:
        async with factory() as db:
            db.add(UsageEventORM(**row))
            await db.commit()
    except IntegrityError:
        logger.info(
            "사용량 행 중복 — 건너뜀",
            extra={"kind": kind, "session_id": row["session_id"]},
        )
        return False
    except Exception:
        logger.warning("사용량 행 저장 실패(무시)", exc_info=True, extra={"kind": kind})
        return False
    return True


async def _record_limited(kind: str, fields: Mapping[str, object]) -> bool:
    async with _write_semaphore():
        return await record_event(kind, fields)


def db_sink(kind: str, fields: dict[str, object]) -> None:
    """emit() 싱크"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = loop.create_task(_record_limited(kind, fields))
    _running.add(task)
    task.add_done_callback(_running.discard)


def install() -> None:
    usage_core.register_sink(db_sink)


def uninstall() -> None:
    usage_core.unregister_sink(db_sink)


async def drain(timeout: float = 10.0) -> None:
    """앱 종료 시 남은 저장을 잠깐 기다림"""
    if not _running:
        return
    await asyncio.wait(set(_running), timeout=timeout)
