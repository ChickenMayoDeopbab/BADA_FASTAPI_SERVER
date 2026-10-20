"""Qwen N 응답의 말소리 뒤 쉼을 줄이는 압축기와 앱 재생 버퍼 흉내"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

SAMPLE_RATE = 16000
FRAME = SAMPLE_RATE // 100  # 10ms
FRAME_BYTES = FRAME * 2
BYTES_PER_MS = 32
_FLOOR_IGNORE_DB = -100.0


@dataclass
class PauseStats:
    in_ms: float = 0.0
    out_ms: float = 0.0
    end_pause_ms: float = 0.0
    end_dropped_ms: float = 0.0
    capped_ms: float = 0.0
    extra_ms: float = 0.0
    pauses: int = 0
    pauses_capped: int = 0
    max_held_ms: float = 0.0
    end_island_ms: float = 0.0
    thr_db: float | None = None
    outcome: str = "open"  # trimmed_end / ended_in_speech / no_speech / aborted


class PauseCompressor:
    """말소리 뒤 쉼을 keep 만 남기고 줄인다(원속도 16k s16le, 출력 = 입력 10ms 프레임의 부분열)"""

    PRE, SPEECH, PAUSE = 0, 1, 2

    def __init__(
        self,
        *,
        keep_ms: float = 160,
        cap_over_ms: float = 350,
        cap_to_ms: float = 250,
        sil_min_db: float = -60.0,
        sil_max_db: float = -45.0,
        margin_db: float = 12.0,
        speech_delta_db: float = 15.0,
        on_frames: int = 3,
        breath_max_ms: float = 150,
        island_gap_ms: float = 150,
        island_max_ms: float = 100,
    ) -> None:
        self.keep = max(0, round(keep_ms / 10))
        self.cap_over = max(self.keep, round(cap_over_ms / 10)) if cap_over_ms > 0 else 10**9
        self.cap_total = max(self.keep, round(cap_to_ms / 10))
        self.sil_min, self.sil_max, self.margin = sil_min_db, sil_max_db, margin_db
        self.delta = speech_delta_db
        self.on_frames = max(1, on_frames)
        self.breath_max = max(self.on_frames, round(breath_max_ms / 10))
        self.island_gap = round(island_gap_ms / 10)
        self.island_max = round(island_max_ms / 10)
        self.stats = PauseStats()
        self._carry = b""
        self._part = np.zeros(0, dtype=np.int16)
        self._pw: list[float] = []
        self._floor = math.inf
        self._state = self.PRE
        self._out: list[bytes] = []
        self._probe: list[tuple[bytes | None, float]] = []
        self._probe_counted = 0
        self._probe_peak = -math.inf
        self._plen = 0
        self._head = 0
        self._held: deque[bytes] = deque()
        self._pdrop = 0
        self._spare = 0
        self._seg = 0  # 지금 말소리 덩어리 프레임 수
        self._last_seg = 0
        self._seg_after_gap = False  # 그 덩어리가 island_gap 이상 쉼 뒤에 시작했나

    def push(self, data: bytes, spare_ms: float = 0.0) -> bytes:
        self._set_spare(spare_ms)
        if data:
            data = self._carry + data
            even = len(data) - len(data) % 2
            self._carry = data[even:]
            if even:
                x = np.frombuffer(data[:even], dtype="<i2")
                self._part = np.concatenate([self._part, x]) if len(self._part) else x.copy()
                n = len(self._part) // FRAME
                if n:
                    frames = self._part[: n * FRAME].reshape(n, FRAME)
                    energies = self._energies(frames)
                    for i in range(n):
                        self._frame(frames[i].tobytes(), energies[i], FRAME)
                    self._part = self._part[n * FRAME :]
        self._release_held()
        return self._flush()

    def finish(self, eos: bool = True, spare_ms: float = 0.0) -> bytes:
        self._carry = b""
        st = self.stats
        if not eos:
            self._held.clear()
            self._probe.clear()
            self._out.clear()
            self._part = np.zeros(0, dtype=np.int16)
            st.outcome = "aborted"
            return b""
        self._set_spare(spare_ms)
        if len(self._part):
            x = self._part
            self._part = np.zeros(0, dtype=np.int16)
            self._frame(x.tobytes(), self._energies(x.reshape(1, -1))[0], len(x))
        if self._probe:
            if self._probe_peak >= self._thr() + self.delta:
                self._onset()
            else:
                self._probe_to_pause()
        if self._state == self.PAUSE:
            self._release_held()
            st.end_dropped_ms = (len(self._held) + self._pdrop) * 10.0
            st.end_pause_ms = self._plen * 10.0
            self._held.clear()
            st.outcome = "trimmed_end"
            if self._seg_after_gap and self._last_seg <= self.island_max:
                st.end_island_ms = self._last_seg * 10.0
        elif self._state == self.SPEECH:
            st.outcome = "ended_in_speech"
        else:
            st.outcome = "no_speech"
        st.thr_db = round(self._thr(), 1)
        return self._flush()

    def _set_spare(self, spare_ms: float) -> None:
        self._spare = max(0, int(spare_ms // 10)) if spare_ms and spare_ms > 0 else 0

    def _energies(self, frames: np.ndarray) -> np.ndarray:
        a = frames.astype(np.float64) / 32768.0
        p = (a * a).mean(axis=1)
        prev = self._pw[-2:]
        allp = np.concatenate([np.asarray(prev, dtype=np.float64), p])
        csum = np.concatenate([[0.0], np.cumsum(allp)])
        j = len(prev) + np.arange(len(p))
        lo = np.maximum(0, j - 2)
        self._pw = list(allp[-2:])
        return 10 * np.log10((csum[j + 1] - csum[lo]) / (j + 1 - lo) + 1e-12)

    def _thr(self) -> float:
        if math.isinf(self._floor):
            return self.sil_max
        return min(self.sil_max, max(self.sil_min, self._floor + self.margin))

    def _flush(self) -> bytes:
        if not self._out:
            return b""
        b = b"".join(self._out)
        self._out.clear()
        self.stats.out_ms += len(b) / BYTES_PER_MS
        return b

    def _release_held(self) -> None:
        """버퍼가 얇으면 붙잡은 쉼을 앞에서부터 내보낸다"""
        while self._spare > 0 and self._held and self._state == self.PAUSE:
            self._out.append(self._held.popleft())
            self._head += 1
            self._spare -= 1
            self.stats.extra_ms += 10.0

    def _pause_frame(self, b: bytes) -> None:
        if self._state == self.SPEECH:
            self._state = self.PAUSE
            self._plen = self._head = self._pdrop = 0
            self._held.clear()
            self._last_seg = self._seg
        self._plen += 1
        self._release_held()
        if not self._held and (self._head < self.keep or self._spare > 0):
            if self._head >= self.keep:
                self._spare -= 1
                self.stats.extra_ms += 10.0
            self._head += 1
            self._out.append(b)
            return
        self._held.append(b)
        if self._plen > self.cap_over:
            room = max(0, self.cap_total - self._head)
            while len(self._held) > room:
                self._held.popleft()
                self._pdrop += 1
        self.stats.max_held_ms = max(self.stats.max_held_ms, len(self._held) * 10.0)

    def _probe_to_pause(self) -> None:
        probe, self._probe = self._probe, []
        self._probe_peak = -math.inf
        self._probe_counted = 0
        for b, _ in probe:
            if b is None:
                continue
            self._pause_frame(b)

    def _onset(self) -> None:
        probe = [b for b, _ in self._probe if b is not None]
        counted = self._probe_counted
        n_probe = len(self._probe)
        self._probe.clear()
        self._probe_peak = -math.inf
        self._probe_counted = 0
        st = self.stats
        if self._state == self.PAUSE:
            p = self._plen - counted
            if p > self.cap_over:
                room = max(0, self.cap_total - self._head)
                while len(self._held) > room:
                    self._held.popleft()
                    self._pdrop += 1
                st.pauses_capped += 1
                st.capped_ms += self._pdrop * 10.0
            st.pauses += 1
            self._seg_after_gap = p >= self.island_gap
            self._out.extend(self._held)
            self._held.clear()
            self._plen = self._head = self._pdrop = 0
        else:
            self._seg_after_gap = False
        self._seg = n_probe
        self._out.extend(probe)
        self._state = self.SPEECH

    def _frame(self, b: bytes, e: float, nsamp: int) -> None:
        self.stats.in_ms += nsamp / 16.0
        if e > _FLOOR_IGNORE_DB:
            self._floor = min(self._floor, e)
        thr = self._thr()
        active = e >= thr

        if self._state == self.PRE:
            self._out.append(b)
            if active:
                self._probe.append((None, e))
                self._probe_peak = max(self._probe_peak, e)
                if len(self._probe) >= self.on_frames and self._probe_peak >= thr + self.delta:
                    self._onset()
            else:
                self._probe.clear()
                self._probe_peak = -math.inf
            return

        if not active:
            if self._probe:
                self._probe_to_pause()
            self._pause_frame(b)
            return

        if self._state == self.SPEECH and not self._probe:
            self._seg += 1
            self._out.append(b)
            return
        if not self._held and self._head < self.keep:
            self._head += 1
            self._plen += 1
            self._out.append(b)
            self._probe.append((None, e))
            self._probe_counted += 1
        else:
            self._probe.append((b, e))
        self._probe_peak = max(self._probe_peak, e)
        if len(self._probe) >= self.on_frames and self._probe_peak >= thr + self.delta:
            self._onset()
        elif len(self._probe) >= self.breath_max:
            self._probe_to_pause()


class ClientBuffer:
    """앱 재생 흉내: prebuffer_ms 쌓이면 재생, 재충전 없음, 굶으면 그만큼 밀림"""

    def __init__(self, prebuffer_ms: float = 250.0) -> None:
        self.prebuffer_ms = prebuffer_ms
        self.sent_ms = 0.0
        self.t0: float | None = None
        self.stall_ms = 0.0
        self.starves = 0
        self.min_ahead_ms: float | None = None

    def on_yield(self, nbytes: int, now_ms: float) -> None:
        ms = nbytes / BYTES_PER_MS
        if self.t0 is None:
            self.sent_ms += ms
            if self.sent_ms >= self.prebuffer_ms:
                self.t0 = now_ms
            return
        ahead = self.t0 + self.stall_ms + self.sent_ms - now_ms
        if ahead < 0:
            self.stall_ms -= ahead
            self.starves += 1
        self.sent_ms += ms

    def ahead_ms(self, now_ms: float) -> float:
        if self.t0 is None:
            return self.sent_ms
        a = max(0.0, self.t0 + self.stall_ms + self.sent_ms - now_ms)
        self.min_ahead_ms = a if self.min_ahead_ms is None else min(self.min_ahead_ms, a)
        return a

    def raw_ahead_ms(self, now_ms: float) -> float:
        """앱 버퍼에 남은 소리, 음수면 이미 굶는 중"""
        if self.t0 is None:
            return self.sent_ms
        return self.t0 + self.stall_ms + self.sent_ms - now_ms
