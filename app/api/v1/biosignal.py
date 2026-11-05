import asyncio
import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.metrics import log_metric, now_ms
from app.core.timeutil import now_utc
from app.deps.auth import ensure_biosignal_consent, get_current_user_id
from app.deps.db import get_db
from app.schemas.biosignal import MAX_BODY_BYTES, BiosignalUpload
from app.services import biosignal_service as svc
from app.services.biosignal.trend import summarize_trend
from app.services.recording_storage import RecordingStorageService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/biosignals", tags=["biosignal"])


def get_biosignal_storage(settings: Settings = Depends(get_settings)) -> RecordingStorageService:
    return RecordingStorageService(settings)


def _require_enabled(settings: Settings) -> None:
    if not settings.biosignal_enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="NOT_FOUND")


def _http_error(error: svc.BiosignalError) -> HTTPException:
    code = status.HTTP_403_FORBIDDEN if error.code == "FORBIDDEN" else status.HTTP_409_CONFLICT
    return HTTPException(status_code=code, detail=error.code)


@router.post("/{session_id}", summary="통화 생체신호 업로드와 분석")
async def upload_biosignal(
    session_id: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
    settings: Settings = Depends(get_settings),
    storage: RecordingStorageService = Depends(get_biosignal_storage),
) -> dict:
    _require_enabled(settings)
    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="PAYLOAD_TOO_LARGE")

    now = now_utc()
    await svc.expire_stale(db, now)
    try:
        row = svc.check_uploadable(await svc.find_owned(db, session_id, user_id, for_update=True))
    except svc.BiosignalError as error:
        await db.commit()
        raise _http_error(error) from error
    try:
        await ensure_biosignal_consent(db, user_id)
    except HTTPException:
        svc.reject(row)
        await db.commit()
        raise
    try:
        payload = BiosignalUpload.model_validate_json(body).model_dump(mode="json")
    except ValidationError as exc:
        await db.commit()
        detail = exc.errors(include_url=False, include_context=False, include_input=False)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail) from exc

    previous = await svc.previous_baselines(db, user_id, row.created_at)
    turn_metrics = await svc.turn_metrics_for(db, session_id)
    started = now_ms()
    metrics = await asyncio.to_thread(svc.analyze_upload, row.timeline, payload, previous, turn_metrics)
    analyze_ms = round(now_ms() - started, 1)

    raw_key = raw_expires_at = None
    if svc.raw_allowed(settings.biosignal_raw_user_ids, user_id):
        try:
            packed = svc.pack_raw(payload)
            raw_key = await asyncio.to_thread(storage.upload_encrypted, svc.raw_key_for(session_id), packed)
        except Exception:
            logger.warning("원시 생체신호 보관 실패", exc_info=True, extra={"session_id": session_id})
        if raw_key:
            raw_expires_at = now + timedelta(days=settings.biosignal_raw_retention_days)

    svc.mark_analyzed(row, payload, metrics, now, raw_key, raw_expires_at)
    await svc.sweep_expired_raw(db, storage, now)
    await db.commit()
    log_metric(
        "biosignal_upload",
        session_id=session_id,
        source=payload["source"],
        hr=len(payload["hr"]),
        ibi=len(payload["ibi"]),
        eda=len(payload["eda"]),
        skin_temp=len(payload["skin_temp"]),
        acc=len(payload["acc"]),
        body_bytes=len(body),
        analyze_ms=analyze_ms,
        raw=raw_key is not None,
    )
    return svc.view(row)


@router.get("/trend", summary="생체신호 회차 추세 조회")
async def get_biosignal_trend(
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
    settings: Settings = Depends(get_settings),
) -> dict:
    _require_enabled(settings)
    return summarize_trend(await svc.trend_rows(db, user_id))


@router.get("/{session_id}", summary="통화 생체신호 분석 결과 조회")
async def get_biosignal(
    session_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
    settings: Settings = Depends(get_settings),
) -> dict:
    _require_enabled(settings)
    try:
        row = await svc.find_owned(db, session_id, user_id)
    except svc.BiosignalError as error:
        raise _http_error(error) from error
    if row is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="NOT_FOUND")
    return svc.view(row)
