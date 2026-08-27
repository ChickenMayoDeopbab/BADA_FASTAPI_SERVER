from datetime import UTC, datetime
from typing import Annotated
from zoneinfo import ZoneInfo

from pydantic import PlainSerializer

KST = ZoneInfo("Asia/Seoul")


def now_utc() -> datetime:
    """현재 시각 저장용"""
    return datetime.now(UTC)


def ensure_utc(value: datetime) -> datetime:
    """utc로 돌려주는거"""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def as_kst(value: datetime) -> datetime:
    """응답용 KST 변환"""
    return ensure_utc(value).astimezone(KST)


KstDatetime = Annotated[
    datetime,
    PlainSerializer(lambda v: as_kst(v).isoformat(), return_type=str, when_used="json"),
]
