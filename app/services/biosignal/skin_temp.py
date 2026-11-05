import numpy as np

from app.services.biosignal.clock import estimate_clock, number, watch_to_server
from app.services.biosignal.eda import valid_status
from app.services.biosignal.segments import build_segments
from app.services.biosignal.stats import UNAVAILABLE, difference, slope_per_min, window_mean

VERSION = "BIOSIGNAL_SKIN_TEMP_V1"
TEMP_MIN_C = 20.0
TEMP_MAX_C = 45.0
MIN_TEMP_SAMPLES = 3
TREND_MIN_SPAN_MS = 60_000
WINDOWS = ("baseline", "anticipation", "call", "recovery")


def _series(rows: list, clock: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    t, values = [], []
    for row in rows:
        at, value = number(row[0]), number(row[1])
        status = row[2] if len(row) > 2 else None
        if at is None or value is None or not valid_status(status) or not TEMP_MIN_C <= value <= TEMP_MAX_C:
            continue
        t.append(watch_to_server(at, clock))
        values.append(value)
    order = np.argsort(t, kind="stable")
    counts = {"temp_rows": len(rows), "temp_valid": len(t)}
    return np.asarray(t, dtype=float)[order], np.asarray(values, dtype=float)[order], counts


def analyze_skin_temp(*, timeline: dict, skin_temp: list, clock: dict | None, markers: dict | None) -> dict:
    """손목 피부온도의 구간 평균, 기준선 대비 변화, 통화 중 기울기"""
    aligned = estimate_clock(timeline, clock)
    segments = build_segments(timeline, markers, aligned)
    t, values, counts = _series(skin_temp or [], aligned)
    windows = segments.windows()
    means = {
        name: window_mean(t, values, windows[name], min_samples=MIN_TEMP_SAMPLES, min_coverage=0.0, digits=2)
        if windows[name] else dict(UNAVAILABLE)
        for name in WINDOWS
    }
    start, end = segments.call
    return {
        "version": VERSION,
        "status": "MEASURED" if t.size else "UNAVAILABLE",
        "means": means,
        "delta_vs_baseline_c": {
            name: difference(means[name], means["baseline"], digits=2) for name in ("anticipation", "call", "recovery")
        },
        "call_trend_c_per_min": slope_per_min(
            t, values, start, end, min_samples=MIN_TEMP_SAMPLES, min_span_ms=TREND_MIN_SPAN_MS, digits=3,
        ),
        "quality": counts,
    }
