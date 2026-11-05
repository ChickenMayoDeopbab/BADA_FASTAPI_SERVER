import numpy as np

from app.services.biosignal.segments import Interval, Segments

UNAVAILABLE = {"status": "UNAVAILABLE"}
MIN_COVERAGE = 0.5
REPORT_BIN_MS = 5000
REPORT_MIN_SAMPLES = 3


def rounded(value: float | None, digits: int = 1) -> float | None:
    return None if value is None else round(float(value), digits)


def seconds(window: Interval | None) -> float | None:
    return rounded((window[1] - window[0]) / 1000) if window else None


def inside(t: np.ndarray, intervals: list[Interval]) -> np.ndarray:
    mask = np.zeros(t.size, dtype=bool)
    for s, e in intervals:
        mask |= (t >= s) & (t < e)
    return mask


def window_mean(
    t: np.ndarray,
    values: np.ndarray,
    intervals: list[Interval],
    *,
    min_samples: int,
    min_coverage: float = MIN_COVERAGE,
    digits: int = 1,
) -> dict:
    expected = sum(e - s for s, e in intervals) / 1000
    mask = inside(t, intervals)
    samples = int(mask.sum())
    if samples == 0 or expected <= 0:
        return dict(UNAVAILABLE)
    coverage = samples / expected
    quality = {"samples": samples, "coverage_pct": rounded(min(coverage, 1.0) * 100)}
    if samples < min_samples or coverage < min_coverage:
        return {"status": "INSUFFICIENT_DATA", **quality}
    return {"status": "MEASURED", "value": rounded(values[mask].mean(), digits), **quality}


def difference(a: dict, b: dict, digits: int = 1) -> dict:
    if a["status"] == "MEASURED" and b["status"] == "MEASURED":
        return {"status": "MEASURED", "value": rounded(a["value"] - b["value"], digits)}
    if "UNAVAILABLE" in (a["status"], b["status"]):
        return dict(UNAVAILABLE)
    return {"status": "INSUFFICIENT_DATA"}


def percent_change(a: dict, b: dict) -> dict:
    change = difference(a, b, digits=6)
    if change["status"] != "MEASURED":
        return change
    if b["value"] == 0:
        return {"status": "INSUFFICIENT_DATA", "reason": "zero_reference"}
    return {"status": "MEASURED", "value": rounded(change["value"] / b["value"] * 100)}


def slope_per_min(
    t: np.ndarray,
    values: np.ndarray,
    start: float,
    end: float,
    *,
    min_samples: int,
    min_span_ms: float,
    digits: int = 2,
) -> dict:
    mask = (t >= start) & (t < end)
    if not mask.any():
        return dict(UNAVAILABLE)
    tt, vv = t[mask], values[mask]
    if tt.size < min_samples or tt[-1] - tt[0] < min_span_ms:
        return {"status": "INSUFFICIENT_DATA", "samples": int(tt.size)}
    slope = np.polyfit((tt - tt[0]) / 60_000, vv, 1)[0]
    return {"status": "MEASURED", "value": rounded(slope, digits), "samples": int(tt.size)}


def report_bins(t: np.ndarray, values: np.ndarray, segments: Segments, *, key: str, digits: int = 1) -> dict:
    """기준선 시작부터 회복 끝까지 5초 구간 중앙값과 그 구간의 단계"""
    origin = segments.call[0]
    starts = [w[0] for w in (segments.baseline, segments.anticipation) if w] + [origin]
    first = origin + np.floor((min(starts) - origin) / REPORT_BIN_MS) * REPORT_BIN_MS
    medians, phases = [], []
    edge = first
    while edge < segments.recovery[1]:
        mask = (t >= edge) & (t < edge + REPORT_BIN_MS)
        medians.append(rounded(float(np.median(values[mask])), digits) if mask.sum() >= REPORT_MIN_SAMPLES else None)
        phases.append(segments.phase_at(edge + REPORT_BIN_MS / 2)[0])
        edge += REPORT_BIN_MS
    return {
        "bin_sec": REPORT_BIN_MS // 1000,
        "start_sec": rounded((first - origin) / 1000),
        key: medians,
        "phase": phases,
    }
