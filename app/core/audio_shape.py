from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

_BLOCK = 32  # 2ms @16kHz
_BLOCK_MS = 2.0
_LOUD_DB, _QUIET_DB, _MIN_SIL_BLOCKS = -25.0, -45.0, 75  # scripts/tts_bench/post.py 끊김 판정과 같은 값
ABRUPT_MS, HARD_MS = 10.0, 4.0


class RequestMeter:
    """요청 PCM 의 2ms 블록 최대값만 모아 끝 모양을 낸다"""

    def __init__(self) -> None:
        self._carry = b""
        self._rest = np.zeros(0, dtype=np.int16)
        self._env: list[np.ndarray] = []
        self.samples = 0
        self.broken = False

    def push(self, data: bytes) -> None:
        if self.broken:
            return
        try:
            data = self._carry + data
            even = len(data) - len(data) % 2
            self._carry = data[even:]
            if not even:
                return
            x = np.frombuffer(data[:even], dtype="<i2")
            self.samples += len(x)
            x = np.concatenate([self._rest, x]) if len(self._rest) else x
            n = len(x) // _BLOCK
            if n:
                blocks = np.abs(x[: n * _BLOCK].astype(np.int32)).reshape(n, _BLOCK).max(1)
                self._env.append(blocks.astype(np.float32))
            self._rest = x[n * _BLOCK :].copy()
        except Exception:
            self.broken = True
            logger.info("요청 계측 실패(무시)", exc_info=True)

    def result(self) -> dict:
        """decay_ms, cut, hard, trail_ms, lead_ms, inner_ends, inner_cuts, pause_max_ms"""
        if self.broken:
            return {"broken": True}
        try:
            return self._result()
        except Exception:
            self.broken = True
            logger.info("요청 계측 실패(무시)", exc_info=True)
            return {"broken": True}

    def _result(self) -> dict:
        env = np.concatenate(self._env) if self._env else np.zeros(0, dtype=np.float32)
        n = len(env)
        if n == 0 or env.max() <= 0:
            return {"decay_ms": None, "cut": False, "hard": False, "trail_ms": 0.0, "lead_ms": 0.0,
                    "inner_ends": 0, "inner_cuts": 0, "pause_max_ms": 0.0}
        db = 20 * np.log10(env / env.max() + 1e-12)
        quiet = db < _QUIET_DB
        loud_idx = np.nonzero(db >= _LOUD_DB)[0]
        edges = np.diff(np.concatenate([[0], quiet.astype(np.int8), [0]]))
        starts, stops = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
        ends: list[float] = []
        pauses: list[float] = []
        for i, j in zip(starts, stops, strict=True):
            if i == 0 or not (j - i >= _MIN_SIL_BLOCKS or j == n):
                continue
            pos = int(np.searchsorted(loud_idx, i))
            if pos == 0:
                continue
            ends.append(float((int(i) - 1 - int(loud_idx[pos - 1])) * _BLOCK_MS))
            if j < n:
                pauses.append(float((int(j) - int(i)) * _BLOCK_MS))
        if not bool(quiet[-1]):
            ends.append(0.0)
        lead = float(stops[0] * _BLOCK_MS) if len(starts) and starts[0] == 0 else 0.0
        ends_quiet = len(starts) and stops[-1] == n and starts[-1] > 0
        trail = float((stops[-1] - starts[-1]) * _BLOCK_MS) if ends_quiet else 0.0
        last = ends[-1] if ends else None
        return {
            "decay_ms": last,
            "cut": last is not None and last <= ABRUPT_MS,
            "hard": last is not None and last <= HARD_MS,
            "trail_ms": trail,
            "lead_ms": lead,
            "inner_ends": max(0, len(ends) - 1),
            "inner_cuts": int(sum(e <= ABRUPT_MS for e in ends[:-1])),
            "pause_max_ms": max(pauses, default=0.0),
        }
