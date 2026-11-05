from dataclasses import dataclass

from app.services.biosignal.clock import number, watch_to_server
from app.services.biosignal.segments import Interval

MOTION_SD_G = 0.15


@dataclass(frozen=True)
class Motion:
    seconds: dict[int, float]

    def moving(self, t_ms: float) -> bool:
        second = int(t_ms // 1000)
        return any(self.seconds.get(s, 0.0) > MOTION_SD_G for s in (second - 1, second, second + 1))

    def level(self, intervals: list[Interval]) -> float | None:
        inside = [sd for second, sd in self.seconds.items() if any(s <= second * 1000 < e for s, e in intervals)]
        if not inside:
            return None
        return round(sum(sd > MOTION_SD_G for sd in inside) / len(inside) * 100, 1)


def prepare_motion(rows: list, clock: dict) -> Motion:
    """워치가 1초로 줄인 가속도 [t, 크기 평균, 크기 표준편차] 를 서버 시각 초로 모은다"""
    seconds: dict[int, float] = {}
    for row in rows:
        t, sd = number(row[0]), number(row[2]) if len(row) > 2 else None
        if t is not None and sd is not None:
            second = int(watch_to_server(t, clock) // 1000)
            seconds[second] = max(sd, seconds.get(second, 0.0))
    return Motion(seconds=seconds)
