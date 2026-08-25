"""시간 헬퍼"""

from datetime import UTC, datetime


def utc_naive_now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)
