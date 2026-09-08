from __future__ import annotations

from collections.abc import Callable, Sequence

from app.core.metrics import now_ms

SAMPLE_RATE = 16000
SAMPLE_BYTES = 2
BYTES_PER_MS = SAMPLE_RATE * SAMPLE_BYTES // 1000
GAP_THRESHOLD_MS = 80.0
PREBUFFER_NONE_MS = 0.0
PREBUFFER_S1_MS = 300.0

Send = tuple[float, int]


def starvation(sends: Sequence[Send], *, prebuffer_ms: float) -> list[float]:
    """청크별로 재생 헤드가 못 나온 시간"""
    if not sends:
        return []
    t0 = sends[0][0]
    played = 0.0
    out: list[float] = []
    for at, nbytes in sends:
        starve = (at - t0) - (prebuffer_ms + played)
        if starve > 0:
            played += starve
            out.append(starve)
        else:
            out.append(0.0)
        played += nbytes / BYTES_PER_MS
    return out


def starvation_gaps(
    sends: Sequence[Send],
    *,
    prebuffer_ms: float,
    threshold_ms: float = GAP_THRESHOLD_MS,
) -> list[tuple[float, float]]:
    """임계 이상으로 못 나온 구간만"""
    gaps: list[tuple[float, float]] = []
    played = 0.0
    for (_, nbytes), starve in zip(sends, starvation(sends, prebuffer_ms=prebuffer_ms), strict=True):
        played += starve
        if starve > 0 and starve >= threshold_ms:
            gaps.append((played, starve))
        played += nbytes / BYTES_PER_MS
    return gaps


class TurnAudioStats:
    """한 턴 동안 클라로 보낸 PCM 청크를 지표로 만든다"""

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or (lambda: now_ms())
        self._sends: list[Send] = []
        self._odd = 0
        self._engine_chunks = 0

    def record_engine(self, pcm: bytes) -> None:
        """엔진이 준 청크 송출 청크 별개로 계산"""
        if pcm:
            self._engine_chunks += 1

    def record(self, pcm: bytes) -> None:
        n = len(pcm)
        if n == 0:
            return
        self._sends.append((self._clock(), n))
        if n % 2:
            self._odd += 1

    @property
    def sends(self) -> list[Send]:
        return list(self._sends)

    def as_metrics(self) -> dict[str, float | int | None]:
        """voice_turn 에 붙는 지표"""
        chunks = len(self._sends)
        nbytes = sum(n for _, n in self._sends)
        audio_ms = round(nbytes / BYTES_PER_MS, 1)
        if chunks == 0:
            return {
                "pcm_chunks": 0,
                "pcm_bytes": 0,
                "audio_ms": 0.0,
                "odd_chunks": 0,
                "send_wall_ms": None,
                "arrival_rtf": None,
                "max_gap_ms": None,
                "gap0_count": 0,
                "gap300_count": 0,
                "engine_chunks": self._engine_chunks,
            }
        times = [t for t, _ in self._sends]
        wall = round(times[-1] - times[0], 1)
        intervals = [b - a for a, b in zip(times, times[1:], strict=False)]
        return {
            "pcm_chunks": chunks,
            "pcm_bytes": nbytes,
            "audio_ms": audio_ms,
            "odd_chunks": self._odd,
            "send_wall_ms": wall,
            "arrival_rtf": round(wall / (nbytes / BYTES_PER_MS), 3) if nbytes > 0 else None,
            "max_gap_ms": round(max(intervals), 1) if intervals else None,
            "gap0_count": len(starvation_gaps(self._sends, prebuffer_ms=PREBUFFER_NONE_MS)),
            "gap300_count": len(starvation_gaps(self._sends, prebuffer_ms=PREBUFFER_S1_MS)),
            "engine_chunks": self._engine_chunks,
        }
