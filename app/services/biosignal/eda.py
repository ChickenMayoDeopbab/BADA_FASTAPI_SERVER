import numpy as np

from app.services.biosignal.clock import estimate_clock, number, watch_to_server
from app.services.biosignal.motion import prepare_motion
from app.services.biosignal.segments import build_segments
from app.services.biosignal.stats import (
    UNAVAILABLE,
    difference,
    inside,
    percent_change,
    report_bins,
    rounded,
    slope_per_min,
    window_mean,
)

VERSION = "BIOSIGNAL_EDA_V1"
EDA_MAX = 100.0
MIN_EDA_SAMPLES = 10
TREND_SKIP_MS = 30_000
TREND_MIN_SPAN_MS = 60_000
TREND_MIN_SAMPLES = 30
RISE_RATIO = 0.03
RISE_LOOKBACK_MS = 5000
RISE_REFRACTORY_MS = 5000
TURN_LAG_MS = 3000
TURN_MIN_SAMPLES = 3
CHANGE_WINDOWS = ("anticipation", "call", "listen", "speak", "recovery")


def valid_status(status: object) -> bool:
    value = number(status)
    return status is None or (value is not None and value >= 0)


def _series(rows: list, clock: dict, motion) -> tuple[np.ndarray, np.ndarray, dict]:
    t, values = [], []
    valid = moving = 0
    for row in rows:
        at, value = number(row[0]), number(row[1])
        status = row[2] if len(row) > 2 else None
        if at is None or value is None or not valid_status(status) or not 0 < value <= EDA_MAX:
            continue
        valid += 1
        server = watch_to_server(at, clock)
        if motion.moving(server):
            moving += 1
            continue
        t.append(server)
        values.append(value)
    order = np.argsort(t, kind="stable")
    counts = {"eda_rows": len(rows), "eda_valid": valid, "eda_motion_excluded": moving}
    return np.asarray(t, dtype=float)[order], np.asarray(values, dtype=float)[order], counts


def rise_times(t: np.ndarray, values: np.ndarray) -> np.ndarray:
    """직전 상승의 불응기가 끝난 뒤 최저점에서 5초 안에 3% 넘게 오른 시각"""
    found: list[float] = []
    last = -np.inf
    lo = 0
    for i in range(t.size):
        while lo < i and t[lo] < max(t[i] - RISE_LOOKBACK_MS, last + RISE_REFRACTORY_MS):
            lo += 1
        if lo == i:
            continue
        low = values[lo:i].min()
        if values[i] - low >= RISE_RATIO * low:
            found.append(t[i])
            last = t[i]
    return np.asarray(found, dtype=float)


def _rises(times: np.ndarray, windows: dict, means: dict) -> dict:
    rates = {}
    for name, intervals in windows.items():
        if means[name]["status"] != "MEASURED":
            rates[name] = {"status": means[name]["status"]}
            continue
        count = int(inside(times, intervals).sum())
        minutes = sum(e - s for s, e in intervals) / 60_000
        rates[name] = {"status": "MEASURED", "count": count, "per_min": rounded(count / minutes, 2)}
    return rates


def _relative(change: dict, reference: dict) -> dict:
    if change["status"] != "MEASURED" or reference["status"] != "MEASURED":
        return {"status": change["status"] if change["status"] != "MEASURED" else reference["status"]}
    if reference["value"] == 0:
        return {"status": "INSUFFICIENT_DATA", "reason": "zero_reference"}
    return {"status": "MEASURED", "value": rounded(change["value"] / reference["value"] * 100, 2)}


def analyze_eda(*, timeline: dict, eda: list, acc: list, clock: dict | None, markers: dict | None) -> dict:
    """피부전도 수준의 구간별 변화율, 비특이 상승, 턴별 반응"""
    aligned = estimate_clock(timeline, clock)
    segments = build_segments(timeline, markers, aligned)
    motion = prepare_motion(acc or [], aligned)
    t, values, counts = _series(eda or [], aligned, motion)
    windows = segments.windows()
    means = {
        name: window_mean(t, values, ivs, min_samples=MIN_EDA_SAMPLES, digits=3) if ivs else dict(UNAVAILABLE)
        for name, ivs in windows.items()
    }
    baseline, call = means["baseline"], means["call"]

    start, end = segments.call
    slope = slope_per_min(
        t, values, start + TREND_SKIP_MS, end, min_samples=TREND_MIN_SAMPLES, min_span_ms=TREND_MIN_SPAN_MS, digits=4,
    )

    turns = {}
    for turn, s, e in segments.speak:
        stat = window_mean(t, values, [(s + TURN_LAG_MS, e + TURN_LAG_MS)], min_samples=TURN_MIN_SAMPLES, digits=3)
        entry = {"status": stat["status"], "mean": stat.get("value")}
        if stat["status"] == "MEASURED":
            entry["change_vs_baseline_pct"] = percent_change(stat, baseline).get("value")
            entry["change_vs_call_pct"] = percent_change(stat, call).get("value")
        turns[str(turn)] = entry

    return {
        "version": VERSION,
        "status": "MEASURED" if t.size else "UNAVAILABLE",
        "means": means,
        "change_vs_baseline_pct": {name: percent_change(means[name], baseline) for name in CHANGE_WINDOWS},
        "listen_minus_speak_pct": _relative(difference(means["listen"], means["speak"], digits=6), call),
        "trend": {"per_min": slope, "pct_of_baseline_per_min": _relative(slope, baseline)},
        "rises": _rises(rise_times(t, values), windows, means),
        "turns": turns,
        "quality": counts,
        "report_timeline": report_bins(t, values, segments, key="eda", digits=3),
    }
