from collections.abc import Sequence
from statistics import median

import numpy as np

from app.services.biosignal.clock import estimate_clock, number, watch_to_server
from app.services.biosignal.ibi import hrv_metrics, prepare_beats
from app.services.biosignal.motion import MOTION_SD_G, prepare_motion
from app.services.biosignal.segments import Interval, Segments, build_segments

VERSION = "BIOSIGNAL_CARDIAC_V1"
HR_MIN_BPM = 30
HR_MAX_BPM = 220
HR_VALID_STATUS = 1
MIN_COVERAGE = 0.5
MIN_HR_SAMPLES = 10
PEAK_HALF_WINDOW_MS = 5000
PEAK_MIN_SAMPLES = 6
TREND_SKIP_MS = 30_000
TREND_MIN_SPAN_MS = 60_000
TREND_MIN_SAMPLES = 30
EDGE_WINDOW_MS = 60_000
TURN_LAG_MS = 3000
TURN_MIN_SAMPLES = 3
RECOVERY_TAIL_MS = 30_000
RECOVERY_MIN_RISE_BPM = 2.0
PREVIOUS_BASELINES_MAX = 5
REPORT_BIN_MS = 5000
REPORT_MIN_SAMPLES = 3
HRV_WINDOWS = ("baseline", "anticipation", "listen", "speak", "recovery")

UNAVAILABLE = {"status": "UNAVAILABLE"}


def _r(value: float | None, digits: int = 1) -> float | None:
    return None if value is None else round(float(value), digits)


def _seconds(window: Interval | None) -> float | None:
    return _r((window[1] - window[0]) / 1000) if window else None


def _inside(t: np.ndarray, intervals: list[Interval]) -> np.ndarray:
    mask = np.zeros(t.size, dtype=bool)
    for s, e in intervals:
        mask |= (t >= s) & (t < e)
    return mask


def _window_mean(t: np.ndarray, bpm: np.ndarray, intervals: list[Interval], min_samples: int = MIN_HR_SAMPLES) -> dict:
    expected = sum(e - s for s, e in intervals) / 1000
    mask = _inside(t, intervals)
    samples = int(mask.sum())
    if samples == 0 or expected <= 0:
        return dict(UNAVAILABLE)
    coverage = samples / expected
    quality = {"samples": samples, "coverage_pct": _r(min(coverage, 1.0) * 100)}
    if samples < min_samples or coverage < MIN_COVERAGE:
        return {"status": "INSUFFICIENT_DATA", **quality}
    return {"status": "MEASURED", "value": _r(bpm[mask].mean()), **quality}


def _difference(a: dict, b: dict) -> dict:
    if a["status"] == "MEASURED" and b["status"] == "MEASURED":
        return {"status": "MEASURED", "value": _r(a["value"] - b["value"])}
    if "UNAVAILABLE" in (a["status"], b["status"]):
        return dict(UNAVAILABLE)
    return {"status": "INSUFFICIENT_DATA"}


def _heart_rate(rows: list, clock: dict, motion) -> tuple[np.ndarray, np.ndarray, dict]:
    t, bpm = [], []
    valid = moving = 0
    for row in rows:
        at, value = number(row[0]), number(row[1])
        status = row[2] if len(row) > 2 else None
        if at is None or value is None or status != HR_VALID_STATUS or not HR_MIN_BPM <= value <= HR_MAX_BPM:
            continue
        valid += 1
        server = watch_to_server(at, clock)
        if motion.moving(server):
            moving += 1
            continue
        t.append(server)
        bpm.append(value)
    order = np.argsort(t, kind="stable")
    counts = {"hr_rows": len(rows), "hr_valid": valid, "hr_motion_excluded": moving}
    return np.asarray(t, dtype=float)[order], np.asarray(bpm, dtype=float)[order], counts


def _baseline(t, bpm, segments: Segments, previous: Sequence[float]) -> dict:
    rest = _window_mean(t, bpm, [segments.baseline]) if segments.baseline else None
    if rest and rest["status"] == "MEASURED":
        return {**rest, "source": "rest", "window_sec": _r((segments.baseline[1] - segments.baseline[0]) / 1000)}
    history = [v for v in (number(x) for x in previous) if v is not None and HR_MIN_BPM <= v <= HR_MAX_BPM]
    if history:
        recent = history[-PREVIOUS_BASELINES_MAX:]
        return {
            "status": "MEASURED", "value": _r(median(recent)), "source": "previous_sessions", "sessions": len(recent),
        }
    if rest:
        return {**rest, "source": "none"}
    return {"status": "UNAVAILABLE", "source": "none"}


def _peak(t, bpm, segments: Segments, baseline: dict) -> dict:
    mask = _inside(t, [segments.call])
    tc, vc = t[mask], bpm[mask]
    if tc.size == 0:
        return dict(UNAVAILABLE)
    left = np.searchsorted(tc, tc - PEAK_HALF_WINDOW_MS, side="left")
    right = np.searchsorted(tc, tc + PEAK_HALF_WINDOW_MS, side="right")
    sums = np.concatenate(([0.0], np.cumsum(vc)))
    counts = right - left
    means = np.where(counts >= PEAK_MIN_SAMPLES, (sums[right] - sums[left]) / np.maximum(counts, 1), -np.inf)
    if not np.isfinite(means).any():
        return {"status": "INSUFFICIENT_DATA"}
    best = int(np.argmax(means))
    phase, turn = segments.phase_at(tc[best])
    delta = means[best] - baseline["value"] if baseline["status"] == "MEASURED" else None
    return {
        "status": "MEASURED",
        "bpm": _r(means[best]),
        "delta": _r(delta),
        "at_sec": _r((tc[best] - segments.call[0]) / 1000),
        "phase": phase,
        "turn": turn,
    }


def _trend(t, bpm, segments: Segments) -> dict:
    start, end = segments.call
    mask = (t >= start + TREND_SKIP_MS) & (t < end)
    if not mask.any():
        return dict(UNAVAILABLE)
    tt, vv = t[mask], bpm[mask]
    if tt.size < TREND_MIN_SAMPLES or tt[-1] - tt[0] < TREND_MIN_SPAN_MS:
        return {"status": "INSUFFICIENT_DATA", "samples": int(tt.size)}
    slope = np.polyfit((tt - tt[0]) / 60_000, vv, 1)[0]
    return {"status": "MEASURED", "value": _r(slope, 2), "samples": int(tt.size)}


def _first_vs_last(t, bpm, segments: Segments) -> dict:
    start, end = segments.call
    first = (start + TREND_SKIP_MS, start + TREND_SKIP_MS + EDGE_WINDOW_MS)
    last = (end - EDGE_WINDOW_MS, end)
    if last[0] < first[1]:
        return {"status": "INSUFFICIENT_DATA", "reason": "call_too_short"}
    return _difference(_window_mean(t, bpm, [last]), _window_mean(t, bpm, [first]))


def _recovery(t, bpm, segments: Segments, baseline: dict, call: dict) -> dict:
    r0, r1 = segments.recovery
    tail = _window_mean(t, bpm, [(max(r0, r1 - RECOVERY_TAIL_MS), r1)], min_samples=TURN_MIN_SAMPLES)
    if "UNAVAILABLE" in (tail["status"], baseline["status"], call["status"]):
        return dict(UNAVAILABLE)
    if "INSUFFICIENT_DATA" in (tail["status"], baseline["status"], call["status"]):
        return {"status": "INSUFFICIENT_DATA"}
    rise = call["value"] - baseline["value"]
    if rise < RECOVERY_MIN_RISE_BPM:
        return {"status": "INSUFFICIENT_DATA", "reason": "no_rise"}
    return {"status": "MEASURED", "value": _r((call["value"] - tail["value"]) / rise * 100), "tail_bpm": tail["value"]}


def _turns(t, bpm, segments: Segments, baseline: dict, call: dict) -> dict[str, dict]:
    turns: dict[str, dict] = {}
    for turn, s, e in segments.speak:
        stat = _window_mean(t, bpm, [(s + TURN_LAG_MS, e + TURN_LAG_MS)], min_samples=TURN_MIN_SAMPLES)
        entry = {"status": stat["status"], "mean_bpm": stat.get("value")}
        if stat["status"] == "MEASURED":
            entry["delta_vs_baseline"] = _difference(stat, baseline).get("value")
            entry["delta_vs_call"] = _difference(stat, call).get("value")
        turns[str(turn)] = entry
    return turns


def _report_timeline(t, bpm, segments: Segments) -> dict:
    origin = segments.call[0]
    starts = [w[0] for w in (segments.baseline, segments.anticipation) if w] + [origin]
    first = origin + np.floor((min(starts) - origin) / REPORT_BIN_MS) * REPORT_BIN_MS
    values, phases = [], []
    edge = first
    while edge < segments.recovery[1]:
        mask = (t >= edge) & (t < edge + REPORT_BIN_MS)
        values.append(_r(float(np.median(bpm[mask]))) if mask.sum() >= REPORT_MIN_SAMPLES else None)
        phases.append(segments.phase_at(edge + REPORT_BIN_MS / 2)[0])
        edge += REPORT_BIN_MS
    return {"bin_sec": REPORT_BIN_MS // 1000, "start_sec": _r((first - origin) / 1000), "hr": values, "phase": phases}


def analyze_cardiac(
    *,
    timeline: dict,
    hr: list,
    ibi: list,
    acc: list,
    clock: dict | None,
    markers: dict | None,
    previous_baselines: Sequence[float] = (),
) -> dict:
    """심박, IBI, 가속도와 통화 사건표로 구간별 심박 반응과 HRV 전해줌"""
    aligned = estimate_clock(timeline, clock)
    segments = build_segments(timeline, markers, aligned)
    motion = prepare_motion(acc or [], aligned)
    t, bpm, hr_counts = _heart_rate(hr or [], aligned, motion)
    windows = segments.windows()

    baseline = _baseline(t, bpm, segments, previous_baselines)
    means = {name: _window_mean(t, bpm, ivs) if ivs else dict(UNAVAILABLE) for name, ivs in windows.items()}
    call = means["call"]

    beats = prepare_beats(ibi or [], aligned)
    hrv = {name: hrv_metrics(beats, windows[name]) if windows[name] else dict(UNAVAILABLE) for name in HRV_WINDOWS}

    motion_levels = {name: motion.level(ivs) for name, ivs in windows.items() if ivs}
    return {
        "version": VERSION,
        "clock": {k: (_r(v) if isinstance(v, float) else v) for k, v in aligned.items()},
        "segments": {
            "call_sec": _seconds(segments.call),
            "speak_turns": len(segments.speak),
            "baseline_sec": _seconds(segments.baseline),
            "anticipation_sec": _seconds(segments.anticipation),
        },
        "baseline": baseline,
        "hr": {
            "means": means,
            "anticipation_delta": _difference(means["anticipation"], baseline),
            "call_delta": _difference(call, baseline),
            "listen_minus_speak": _difference(means["listen"], means["speak"]),
            "peak": _peak(t, bpm, segments, baseline),
            "trend_bpm_per_min": _trend(t, bpm, segments),
            "first_vs_last_min": _first_vs_last(t, bpm, segments),
            "recovery_pct": _recovery(t, bpm, segments, baseline, call),
        },
        "turns": _turns(t, bpm, segments, baseline, call),
        "hrv": hrv,
        "motion": {
            "status": "MEASURED" if motion.seconds else "UNAVAILABLE",
            "threshold_g": MOTION_SD_G,
            "moving_pct": motion_levels if motion.seconds else {},
        },
        "quality": {
            **hr_counts,
            "ibi_rows": len(ibi or []),
            "ibi_checked": beats.checked,
            "ibi_removed": beats.removed,
            "ibi_removed_pct": _r(beats.removed / beats.checked * 100) if beats.checked else None,
            "artifacts": beats.artifacts,
        },
        "report_timeline": _report_timeline(t, bpm, segments),
    }
