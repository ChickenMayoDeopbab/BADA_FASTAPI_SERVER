import asyncio
import gzip
import json
import logging
from datetime import datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.timeutil import ensure_utc, now_utc
from app.db.base import AsyncSessionLocal
from app.db.models import CallBiosignalORM, FeedbackORM
from app.services.biosignal.cardiac import analyze_cardiac
from app.services.biosignal.clock import number
from app.services.biosignal.eda import analyze_eda
from app.services.biosignal.fusion import fuse_turns
from app.services.biosignal.skin_temp import analyze_skin_temp

logger = logging.getLogger(__name__)

STATUS_AWAITING_DATA = "AWAITING_DATA"
STATUS_ANALYZED = "ANALYZED"
STATUS_EXPIRED = "EXPIRED"
STATUS_REJECTED = "REJECTED"
AWAITING_TTL = timedelta(hours=24)
PREVIOUS_BASELINES = 5
RAW_SWEEP_LIMIT = 10
TREND_SESSIONS = 200
_CONFLICTS = {STATUS_ANALYZED: "ALREADY_ANALYZED", STATUS_EXPIRED: "EXPIRED", STATUS_REJECTED: "REJECTED"}


class BiosignalError(Exception):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


async def save_call_timeline(
    *,
    session_id: str,
    user_id: int,
    scenario_id: int | None,
    session_type: str | None,
    difficulty: str | None,
    timeline: dict,
    ai_personality: str | None = None,
    session_factory=AsyncSessionLocal,
) -> bool:
    """통화 사건표를 생체신호 업로드 대기 행으로 저장"""
    async with session_factory() as db:
        stmt = (
            pg_insert(CallBiosignalORM)
            .values(
                session_id=session_id,
                user_id=user_id,
                scenario_id=scenario_id,
                session_type=session_type,
                difficulty=difficulty,
                ai_personality=ai_personality,
                status=STATUS_AWAITING_DATA,
                timeline=timeline,
                created_at=now_utc(),
            )
            .on_conflict_do_nothing(index_elements=["session_id"])
        )
        result = await db.execute(stmt)
        await db.commit()
    return bool(result.rowcount)


def raw_key_for(session_id: str) -> str:
    return f"biosignal-raw/{session_id}.json.gz"


def raw_allowed(allowlist: str, user_id: int) -> bool:
    return str(user_id) in {item.strip() for item in allowlist.split(",") if item.strip()}


def pack_raw(payload: dict) -> bytes:
    return gzip.compress(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode())


async def expire_stale(db: AsyncSession, now: datetime) -> None:
    """24시간 넘게 업로드가 없던 사건표를 만료하고 파기한다"""
    await db.execute(
        update(CallBiosignalORM)
        .where(CallBiosignalORM.status == STATUS_AWAITING_DATA, CallBiosignalORM.created_at < now - AWAITING_TTL)
        .values(status=STATUS_EXPIRED, timeline={})
    )


async def find_owned(db: AsyncSession, session_id: str, user_id: int) -> CallBiosignalORM | None:
    stmt = select(CallBiosignalORM).where(CallBiosignalORM.session_id == session_id)
    row = (await db.execute(stmt)).scalar_one_or_none()
    if row is not None and row.user_id != user_id:
        raise BiosignalError("FORBIDDEN")
    return row


def check_uploadable(row: CallBiosignalORM | None) -> CallBiosignalORM:
    if row is None:
        raise BiosignalError("NO_TIMELINE")
    if row.status != STATUS_AWAITING_DATA:
        raise BiosignalError(_CONFLICTS.get(row.status, row.status))
    return row


def reject(row: CallBiosignalORM) -> None:
    row.status = STATUS_REJECTED
    row.timeline = {}


async def previous_baselines(db: AsyncSession, user_id: int, before: datetime) -> list[float]:
    """이전 회차들의 휴식 기준선 심박"""
    stmt = (
        select(CallBiosignalORM.metrics)
        .where(
            CallBiosignalORM.user_id == user_id,
            CallBiosignalORM.status == STATUS_ANALYZED,
            CallBiosignalORM.created_at < before,
        )
        .order_by(CallBiosignalORM.created_at.desc())
        .limit(PREVIOUS_BASELINES)
    )
    values = []
    for metrics in (await db.execute(stmt)).scalars():
        baseline = ((metrics or {}).get("cardiac") or {}).get("baseline") or {}
        if baseline.get("source") == "rest" and baseline.get("status") == "MEASURED":
            value = number(baseline.get("value"))
            if value is not None:
                values.append(value)
    return list(reversed(values))


async def turn_metrics_for(db: AsyncSession, session_id: str) -> dict | None:
    stmt = select(FeedbackORM.turn_metrics).where(FeedbackORM.session_id == session_id)
    stored = (await db.execute(stmt)).scalar_one_or_none()
    if not stored:
        return None
    try:
        data = json.loads(stored)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def analyze_upload(timeline: dict, payload: dict, previous: list[float], turn_metrics: dict | None) -> dict:
    """업로드 하나를 심박·EDA·피부온도·음성 결합까지 분석한다"""
    common = {"timeline": timeline, "clock": payload["clock"], "markers": payload["markers"]}
    cardiac = analyze_cardiac(
        hr=payload["hr"], ibi=payload["ibi"], acc=payload["acc"], previous_baselines=previous, **common,
    )
    eda = analyze_eda(eda=payload["eda"], acc=payload["acc"], **common)
    return {
        "cardiac": cardiac,
        "eda": eda,
        "skin_temp": analyze_skin_temp(skin_temp=payload["skin_temp"], **common),
        "fusion": fuse_turns(cardiac=cardiac, eda=eda, turn_metrics=turn_metrics),
    }


def mark_analyzed(
    row: CallBiosignalORM,
    payload: dict,
    metrics: dict,
    now: datetime,
    raw_key: str | None,
    raw_expires_at: datetime | None,
) -> None:
    row.status = STATUS_ANALYZED
    row.source = payload["source"]
    row.device = payload.get("device")
    row.metrics = metrics
    row.analyzed_at = now
    row.raw_key = raw_key
    row.raw_expires_at = raw_expires_at


async def sweep_expired_raw(db: AsyncSession, storage, now: datetime) -> None:
    """보관 기한이 지난 원시 신호를 지운다(한 번에 몇 개씩)"""
    stmt = (
        select(CallBiosignalORM)
        .where(CallBiosignalORM.raw_key.is_not(None), CallBiosignalORM.raw_expires_at < now)
        .limit(RAW_SWEEP_LIMIT)
    )
    for row in (await db.execute(stmt)).scalars():
        try:
            await asyncio.to_thread(storage.delete, row.raw_key)
        except Exception:
            logger.warning("만료 원시 생체신호 삭제 실패", exc_info=True, extra={"session_id": row.session_id})
            continue
        row.raw_key = None
        row.raw_expires_at = None


async def trend_rows(db: AsyncSession, user_id: int) -> list[dict]:
    """추세에 쓸 최근 분석 회차(날짜 오름차 순)"""
    stmt = (
        select(CallBiosignalORM)
        .where(CallBiosignalORM.user_id == user_id, CallBiosignalORM.status == STATUS_ANALYZED)
        .order_by(CallBiosignalORM.created_at.desc())
        .limit(TREND_SESSIONS)
    )
    rows = (await db.execute(stmt)).scalars().all()
    return [
        {
            "session_id": row.session_id,
            "at": ensure_utc(row.created_at).isoformat(),
            "scenario_id": row.scenario_id,
            "difficulty": row.difficulty,
            "ai_personality": row.ai_personality,
            "session_type": row.session_type,
            "metrics": row.metrics,
        }
        for row in reversed(rows)
    ]


def view(row: CallBiosignalORM) -> dict:
    if row.status != STATUS_ANALYZED:
        return {"session_id": row.session_id, "status": row.status}
    return {
        "session_id": row.session_id,
        "status": row.status,
        "source": row.source,
        "analyzed_at": ensure_utc(row.analyzed_at).isoformat() if row.analyzed_at else None,
        "metrics": row.metrics,
    }
