import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from app.services.biosignal.clock import estimate_clock, number, watch_to_server
from app.services.biosignal.motion import prepare_motion
from app.services.biosignal.segments import Interval, build_segments
from app.services.biosignal.stats import UNAVAILABLE, inside, percent_change, rounded

VERSION = "BIOSIGNAL_PPG_V1"
BAND_HZ = (0.5, 4.0)
FILTER_ORDER = 2
MIN_CHUNK_SEC = 4.0
MIN_BEAT_GAP_SEC = 0.33
PROMINENCE_RATIO = 0.3
OUTLIER_RATIO = 3.0
MIN_PPG_BEATS = 15
CHANGE_WINDOWS = ("anticipation", "call", "listen", "speak", "recovery")


def chunk_beats(chunk: dict, clock: dict) -> tuple[np.ndarray, np.ndarray, int] | None:
    """원파형 한 묶음의 박동 시각, 맥파 진폭 튀어서 뺀 박동 수. 짧거나 이상하면 None"""
    t0, hz = number(chunk.get("t0")), number(chunk.get("hz"))
    values = np.asarray([v for v in (chunk.get("green") or []) if number(v) is not None], dtype=float)
    if t0 is None or not hz or values.size < MIN_CHUNK_SEC * hz:
        return None
    b, a = butter(FILTER_ORDER, [BAND_HZ[0] / (hz / 2), BAND_HZ[1] / (hz / 2)], btype="band")
    filtered = filtfilt(b, a, values)
    spread = float(np.percentile(filtered, 95) - np.percentile(filtered, 5))
    gap = max(1, int(MIN_BEAT_GAP_SEC * hz))
    peaks, _ = find_peaks(filtered, distance=gap, prominence=PROMINENCE_RATIO * spread / 2)
    if peaks.size < 2:
        return np.empty(0), np.empty(0), 0
    amplitudes = np.array([filtered[p] - filtered[q:p + 1].min() for q, p in zip(peaks[:-1], peaks[1:], strict=True)])
    times = np.array([watch_to_server(t0 + p * 1000.0 / hz, clock) for p in peaks[1:]])
    median = float(np.median(amplitudes))
    keep = (amplitudes <= OUTLIER_RATIO * median) & (amplitudes >= median / OUTLIER_RATIO)
    return times[keep], amplitudes[keep], int((~keep).sum())


def _median_amplitude(t: np.ndarray, amplitudes: np.ndarray, intervals: list[Interval]) -> dict:
    mask = inside(t, intervals)
    beats = int(mask.sum())
    if beats == 0:
        return dict(UNAVAILABLE)
    if beats < MIN_PPG_BEATS:
        return {"status": "INSUFFICIENT_DATA", "beats": beats}
    return {"status": "MEASURED", "value": rounded(float(np.median(amplitudes[mask])), 2), "beats": beats}


def analyze_ppg(*, timeline: dict, ppg: list, acc: list, clock: dict | None, markers: dict | None) -> dict:
    """PPG 원파형의 박동별 맥파 진폭으로 구간별 기준선 대비 변화를 낸다(내부 지표)"""
    aligned = estimate_clock(timeline, clock)
    segments = build_segments(timeline, markers, aligned)
    motion = prepare_motion(acc or [], aligned)
    times: list[float] = []
    amplitudes: list[float] = []
    used = outliers = moving = 0
    for chunk in ppg or []:
        found = chunk_beats(chunk, aligned)
        if found is None:
            continue
        used += 1
        chunk_times, chunk_amplitudes, dropped = found
        outliers += dropped
        for t, amplitude in zip(chunk_times, chunk_amplitudes, strict=True):
            if motion.moving(t):
                moving += 1
                continue
            times.append(float(t))
            amplitudes.append(float(amplitude))
    order = np.argsort(times, kind="stable")
    t = np.asarray(times, dtype=float)[order]
    values = np.asarray(amplitudes, dtype=float)[order]
    windows = segments.windows()
    medians = {name: _median_amplitude(t, values, ivs) if ivs else dict(UNAVAILABLE) for name, ivs in windows.items()}
    return {
        "version": VERSION,
        "status": "MEASURED" if values.size else "UNAVAILABLE",
        "pulse_amplitude": medians,
        "change_vs_baseline_pct": {name: percent_change(medians[name], medians["baseline"]) for name in CHANGE_WINDOWS},
        "quality": {
            "chunks": len(ppg or []),
            "chunks_used": used,
            "beats": int(values.size),
            "beats_outlier": outliers,
            "beats_motion_excluded": moving,
        },
    }
