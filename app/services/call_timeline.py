import time
from bisect import bisect_right
from collections.abc import Callable

_MAX_EVENTS = 5000
_MAX_ANCHORS = 5000
_MAX_CLOCK_PAIRS = 200
_ANCHOR_DRIFT_MS = 300
_PHONE_SKEW_LIMIT_MS = 86_400_000
_SOURCE_MAX_CHARS = 32


def _epoch_clock() -> Callable[[], float]:
    epoch0 = time.time() * 1000.0
    mono0 = time.monotonic()
    return lambda: epoch0 + (time.monotonic() - mono0) * 1000.0


def recording_to_epoch_ms(anchors: list[list[float]], recording_sec: float) -> int | None:
    """실제 녹음 끝나는 시간(초)를 ms로 바꾼다"""
    if not anchors:
        return None
    index = max(bisect_right([rec for _, rec in anchors], recording_sec) - 1, 0)
    wall, rec = anchors[index]
    estimate = wall + (recording_sec - rec) * 1000.0
    if index + 1 < len(anchors):
        estimate = min(estimate, anchors[index + 1][0])
    return round(estimate)


class CallTimeline:
    """실제 녹음 끝나는 시간 실제 시각으로 바꾸는 기준점"""

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or _epoch_clock()
        self.events: list[dict] = []
        self.anchors: list[list[float]] = []
        self.clock_pairs: list[list[int]] = []
        self.intent: dict | None = None
        self._ai_playbacks: set[int] = set()
        self._muted = False

    def now_ms(self) -> int:
        return round(self._clock())

    def mark(self, kind: str, *, at_ms: int | None = None, **fields: object) -> None:
        if len(self.events) < _MAX_EVENTS:
            self.events.append({"kind": kind, "at_ms": self.now_ms() if at_ms is None else at_ms, **fields})

    def ai_audio(self, playback: int | None) -> None:
        if playback is None or playback in self._ai_playbacks:
            return
        self._ai_playbacks.add(playback)
        self.mark("ai_audio_start", playback=playback)

    def mute(self, muted: bool) -> None:
        if muted != self._muted:
            self._muted = muted
            self.mark("mute_on" if muted else "mute_off")

    def audio(self, recording_sec: float) -> None:
        now = self.now_ms()
        if not self.anchors:
            self.mark("audio_first", at_ms=now)
        else:
            wall, rec = self.anchors[-1]
            if abs(now - (wall + (recording_sec - rec) * 1000.0)) <= _ANCHOR_DRIFT_MS:
                return
        if len(self.anchors) < _MAX_ANCHORS:
            self.anchors.append([now, recording_sec])

    def clock_pair(self, phone_ms: object) -> None:
        if type(phone_ms) is not int or len(self.clock_pairs) >= _MAX_CLOCK_PAIRS:
            return
        server_ms = self.now_ms()
        if abs(phone_ms - server_ms) <= _PHONE_SKEW_LIMIT_MS:
            self.clock_pairs.append([phone_ms, server_ms])

    def intend(self, data: dict) -> None:
        if self.intent is None:
            version, source = data.get("v"), data.get("source")
            self.intent = {
                "v": version if type(version) is int else None,
                "source": source[:_SOURCE_MAX_CHARS] if isinstance(source, str) else None,
                "at_ms": self.now_ms(),
            }
        self.clock_pair(data.get("phone_ms"))

    def recording_to_epoch_ms(self, recording_sec: float) -> int | None:
        return recording_to_epoch_ms(self.anchors, recording_sec)

    def as_dict(self) -> dict:
        return {
            "v": 1,
            "intent": self.intent,
            "events": self.events,
            "anchors": self.anchors,
            "clock_pairs": self.clock_pairs,
        }
