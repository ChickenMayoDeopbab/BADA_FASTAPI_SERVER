import logging
import time
from contextlib import suppress

_metric_logger = logging.getLogger("app.metrics")


def now_ms() -> float:
    """단조 증가 시각(ms). 구간 측정 전용"""
    return time.monotonic() * 1000.0


def _ms_between(start: float | None, end: float | None) -> float | None:
    """두 monotonic-ms 시점의 차이. 한쪽이라도 없으면 None(부분 턴)."""
    if start is None or end is None:
        return None
    return round(end - start, 1)


class Stopwatch:
    """단조 시각 기반 다지점 측정기.

    >>> sw = Stopwatch()
    >>> sw.mark("first_token")   # 임의 라벨 시점 기록
    >>> sw.since("first_token")  # 라벨 이후 경과 ms
    >>> sw.elapsed_ms            # 생성 이후 경과 ms
    """

    def __init__(self) -> None:
        self._start = now_ms()
        self._marks: dict[str, float] = {}

    def mark(self, label: str) -> float:
        """현재 시각을 라벨로 기록하고 그 시각(ms)을 반환함"""
        t = now_ms()
        self._marks[label] = t
        return t

    def at(self, label: str) -> float | None:
        """라벨로 기록된 시각(ms), 없으면 None."""
        return self._marks.get(label)

    def since(self, label: str) -> float | None:
        """라벨 시점 이후 현재까지 경과 ms, 라벨 없으면 None"""
        return _ms_between(self._marks.get(label), now_ms())

    def between(self, start_label: str, end_label: str) -> float | None:
        """두 라벨 사이 경과 ms, 한쪽이라도 없으면 None"""
        return _ms_between(self._marks.get(start_label), self._marks.get(end_label))

    @property
    def elapsed_ms(self) -> float:
        """생성 이후 경과 ms"""
        return round(now_ms() - self._start, 1)


def log_metric(name: str, **fields: object) -> None:
    """메트릭 한 줄을 구조적 로그로 남김"""
    with suppress(Exception):
        summary = " ".join(f"{k}={v}" for k, v in fields.items())
        _metric_logger.info(
            "metric=%s %s", name, summary, extra={"metric": name, **fields}
        )
