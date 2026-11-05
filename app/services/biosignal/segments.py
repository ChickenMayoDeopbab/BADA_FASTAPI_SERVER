from dataclasses import dataclass

from app.services.biosignal.clock import number, phone_to_server

BASELINE_STABILIZE_MS = 60_000
BASELINE_MIN_MS = 60_000
ANTICIPATION_DEFAULT_MS = 30_000
ANTICIPATION_MIN_MS = 5_000
RECOVERY_MS = 60_000

Interval = tuple[float, float]


@dataclass(frozen=True)
class Segments:
    call: Interval
    speak: list[tuple[int, float, float]]
    listen: list[Interval]
    baseline: Interval | None
    anticipation: Interval | None
    meditation: Interval | None
    recovery: Interval

    def windows(self) -> dict[str, list[Interval]]:
        return {
            "baseline": [self.baseline] if self.baseline else [],
            "anticipation": [self.anticipation] if self.anticipation else [],
            "call": [self.call],
            "listen": list(self.listen),
            "speak": [(s, e) for _, s, e in self.speak],
            "recovery": [self.recovery],
        }

    def phase_at(self, t: float) -> tuple[str | None, int | None]:
        for turn, s, e in self.speak:
            if s <= t < e:
                return "speak", turn
        if self.call[0] <= t < self.call[1]:
            return "listen", next((turn for turn, s, _ in self.speak if s >= t), None)
        for name, window in (("baseline", self.baseline), ("anticipation", self.anticipation)):
            if window and window[0] <= t < window[1]:
                return name, None
        if self.recovery[0] <= t < self.recovery[1]:
            return "recovery", None
        return None, None


def _events(timeline: dict) -> list[dict]:
    events = [e for e in timeline.get("events") or [] if number(e.get("at_ms")) is not None]
    return sorted(events, key=lambda e: e["at_ms"])


def _marker_interval(markers: dict, key: str, clock: dict) -> Interval | None:
    raw = markers.get(key)
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None
    start, end = number(raw[0]), number(raw[1])
    if start is None or end is None or end <= start:
        return None
    return phone_to_server(start, clock), phone_to_server(end, clock)


def _speak_intervals(events: list[dict], start: float, end: float) -> list[tuple[int, float, float]]:
    audio_first = next((e["at_ms"] for e in events if e["kind"] == "audio_first"), start)
    speak: list[tuple[int, float, float]] = []
    previous_close = start
    for event in events:
        if event["kind"] != "user_turn_close":
            continue
        turn, close = event.get("turn"), event["at_ms"]
        heard = [
            e["at_ms"] for e in events if e["kind"] == "ai_playback_done" and previous_close <= e["at_ms"] <= close
        ]
        opened = [e["at_ms"] for e in events if e["kind"] == "user_turn_open" and e.get("turn") == turn]
        if heard:
            begin = max(heard)
        elif opened:
            begin = opened[-1]
        elif turn == 1:
            begin = audio_first
        else:
            begin = previous_close
        begin, finish = max(begin, start), min(close, end)
        if isinstance(turn, int) and finish > begin:
            speak.append((turn, begin, finish))
        previous_close = close
    return speak


def _listen_intervals(speak: list[tuple[int, float, float]], start: float, end: float) -> list[Interval]:
    listen: list[Interval] = []
    cursor = start
    for _, s, e in sorted(speak, key=lambda item: item[1]):
        if s > cursor:
            listen.append((cursor, s))
        cursor = max(cursor, e)
    if end > cursor:
        listen.append((cursor, end))
    return listen


def build_segments(timeline: dict, markers: dict | None, clock: dict) -> Segments:
    """사건표와 앱 표시로 통화 전·중·후 구간을 서버 시각으로 나눈다"""
    markers = markers or {}
    events = _events(timeline)
    kinds = [e["kind"] for e in events]
    if "call_open" in kinds:
        start = events[kinds.index("call_open")]["at_ms"]
    elif "audio_first" in kinds:
        start = events[kinds.index("audio_first")]["at_ms"]
    else:
        start = events[0]["at_ms"] if events else 0.0
    closes = [e["at_ms"] for e in events if e["kind"] == "call_close"]
    end = max(closes[-1] if closes else (events[-1]["at_ms"] if events else start), start)

    speak = _speak_intervals(events, start, end)

    rest = _marker_interval(markers, "rest_baseline", clock)
    baseline = None
    if rest:
        b0, b1 = rest[0] + BASELINE_STABILIZE_MS, min(rest[1], start)
        if b1 - b0 >= BASELINE_MIN_MS:
            baseline = (b0, b1)

    meditation = _marker_interval(markers, "meditation", clock)
    ringing = number(markers.get("ringing_ms"))
    a0 = phone_to_server(ringing, clock) if ringing is not None else start - ANTICIPATION_DEFAULT_MS
    for other in (rest, meditation):
        if other and other[0] < start and other[1] > a0:
            a0 = max(a0, other[1])
    anticipation = (a0, start) if start - a0 >= ANTICIPATION_MIN_MS else None

    r1 = end + RECOVERY_MS
    recovery_end = number(markers.get("recovery_end_ms"))
    if recovery_end is not None and phone_to_server(recovery_end, clock) > end:
        r1 = min(r1, phone_to_server(recovery_end, clock))

    return Segments(
        call=(start, end),
        speak=speak,
        listen=_listen_intervals(speak, start, end),
        baseline=baseline,
        anticipation=anticipation,
        meditation=meditation,
        recovery=(end, r1),
    )
