from __future__ import annotations

import math

import numpy as np

SAMPLE_RATE = 16000
_FRAME_MS = 30
_TOLERANCE_MS = 10
_MIN_SPEED = 0.8
_MAX_SPEED = 1.25


def clamp_speed(speed: float) -> float:
    """허용 범위로 자른 속도, 이상한 값이면 1.0"""
    speed = float(speed)
    if not math.isfinite(speed) or speed <= 0:
        return 1.0
    return min(max(speed, _MIN_SPEED), _MAX_SPEED)


class PcmStretcher:
    """16kHz PCM을 피치 유지한 채 speed배 빠르게(스트리밍 WSOLA), 앞 warmup_ms 는 원래 속도임"""

    def __init__(self, speed: float, sample_rate: int = SAMPLE_RATE, warmup_ms: int = 0) -> None:
        self.speed = clamp_speed(speed)
        n = sample_rate * _FRAME_MS // 1000
        n -= n % 2
        self._n = n
        self._hs = n // 2
        self._ha = self._hs * self.speed
        self._k0 = round(sample_rate * warmup_ms / 1000 / self._hs)
        self._tol = sample_rate * _TOLERANCE_MS // 1000
        self._win = np.hanning(n + 1)[:n]
        self._x = np.zeros(n // 2 + self._tol)
        self._x0 = 0
        self._k = 0
        self._prev: int | None = None
        self._acc = np.zeros(n)
        self._o0 = 0
        self._drop = n // 2
        self._carry = b""
        self._n_in = 0
        self._n_out = 0

    @property
    def bypass(self) -> bool:
        return self.speed == 1.0

    def push(self, data: bytes) -> bytes:
        if self.bypass:
            return data
        data = self._carry + data
        even = len(data) - len(data) % 2
        self._carry = data[even:]
        if not even:
            return b""
        samples = np.frombuffer(data[:even], dtype="<i2").astype(np.float64)
        self._n_in += len(samples)
        self._x = np.concatenate([self._x, samples])
        return self._emit(self._run())

    def flush(self) -> bytes:
        if self.bypass or not self._n_in:
            return b""
        self._x = np.concatenate([self._x, np.zeros(2 * (self._n + self._tol))])
        y = self._run()
        warm = min(self._n_in, self._k0 * self._hs)
        target = round(warm + (self._n_in - warm) / self.speed) + self._tol
        return self._emit(y[: max(0, target - self._n_out)])

    def _a(self, k: int) -> int:
        k0 = min(k, self._k0)
        return self._tol + int(round(k0 * self._hs + (k - k0) * self._ha))

    def _run(self) -> np.ndarray:
        n, hs, tol = self._n, self._hs, self._tol
        out = []
        while True:
            a = self._a(self._k)
            need = a + tol + n
            if self._prev is not None:
                need = max(need, self._prev + hs + n)
            if need > self._x0 + len(self._x):
                break
            if self._prev is None:
                p = a
            else:
                t0 = self._prev + hs - self._x0
                template = self._x[t0 : t0 + n]
                r0 = a - tol - self._x0
                region = self._x[r0 : r0 + n + 2 * tol]
                corr = np.correlate(region, template, mode="valid")
                energy = np.cumsum(np.concatenate([[0.0], region * region]))
                norm = np.sqrt(energy[n:] - energy[:-n] + 1e-9)
                p = a - tol + int(np.argmax(corr / norm))
            s = p - self._x0
            pos = self._k * hs - self._o0
            if pos + n > len(self._acc):
                self._acc = np.concatenate([self._acc, np.zeros(pos + n - len(self._acc))])
            self._acc[pos : pos + n] += self._x[s : s + n] * self._win
            done = pos + hs
            out.append(self._acc[:done].copy())
            self._acc = self._acc[done:]
            self._o0 += done
            self._prev = p
            self._k += 1
            cut = min(self._a(self._k) - tol, self._prev + hs) - self._x0
            if cut > 0:
                self._x = self._x[cut:]
                self._x0 += cut
        y = np.concatenate(out) if out else np.zeros(0)
        if self._drop:
            d = min(self._drop, len(y))
            y = y[d:]
            self._drop -= d
        return y

    def _emit(self, y: np.ndarray) -> bytes:
        self._n_out += len(y)
        return np.clip(np.rint(y), -32768, 32767).astype("<i2").tobytes()
