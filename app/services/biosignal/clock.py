from statistics import median

SUSPICIOUS_OFFSET_MS = 5000


def number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _first_event_ms(timeline: dict, kind: str) -> float | None:
    for event in timeline.get("events") or []:
        if event.get("kind") == kind:
            return number(event.get("at_ms"))
    return None


def estimate_clock(timeline: dict, clock: dict | None) -> dict:
    """폰·워치 시각을 서버 시각으로 옮길 차이를 고른다"""
    clock = clock or {}
    diffs: list[float] = []
    for pair in timeline.get("clock_pairs") or []:
        phone, server = number(pair[0]), number(pair[1])
        if phone is not None and server is not None:
            diffs.append(server - phone)
    source = "pairs"
    if not diffs:
        source = "ws_events"
        for phone_key, kind in (("phone_ws_open_ms", "call_open"), ("phone_call_end_ms", "call_close")):
            phone, server = number(clock.get(phone_key)), _first_event_ms(timeline, kind)
            if phone is not None and server is not None:
                diffs.append(server - phone)
    if not diffs:
        source = "none"
    offset = median(diffs) if diffs else 0.0
    return {
        "phone_to_server_ms": offset,
        "watch_minus_phone_ms": number(clock.get("watch_minus_phone_ms")) or 0.0,
        "source": source,
        "pairs": len(diffs),
        "suspicious": abs(offset) > SUSPICIOUS_OFFSET_MS,
    }


def watch_to_server(t_ms: float, clock: dict) -> float:
    return t_ms - clock["watch_minus_phone_ms"] + clock["phone_to_server_ms"]


def phone_to_server(t_ms: float, clock: dict) -> float:
    return t_ms + clock["phone_to_server_ms"]
