import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from app.services.biosignal.clock import number, watch_to_server
from app.services.biosignal.segments import Interval

IBI_MIN_MS = 250
IBI_MAX_MS = 2000
IBI_VALID_STATUS = 0
RUN_GAP_MS = 2000
LT_MIN_RUN = 5
HRV_MIN_WINDOW_MS = 60_000
HRV_MIN_COVERAGE = 0.5
HRV_MAX_REMOVED = 0.05

_ALPHA = 5.2
_THRESHOLD_WINDOW = 91
_MEDIAN_WINDOW = 11
_C1 = 0.13
_C2 = 0.17
_THRESHOLD_FLOOR_MS = 20.0


def _rolling(values: np.ndarray, width: int, fn: Callable[[np.ndarray], float]) -> np.ndarray:
    half = width // 2
    return np.array([fn(values[max(0, i - half): i + half + 1]) for i in range(values.size)])


def _threshold(values: np.ndarray) -> np.ndarray:
    magnitude = np.abs(values)
    q1 = _rolling(magnitude, _THRESHOLD_WINDOW, lambda w: float(np.quantile(w, 0.25)))
    q3 = _rolling(magnitude, _THRESHOLD_WINDOW, lambda w: float(np.quantile(w, 0.75)))
    threshold = _ALPHA * (q3 - q1) / 2
    return np.maximum(threshold, _THRESHOLD_FLOOR_MS)


def classify_artifacts(rr: np.ndarray) -> dict[str, list[int]]:
    """Lipponen & Tarvainen(2019) 박동 분류. 흐름은 NeuroKit2 _find_artifacts(MIT)를 따름"""
    found: dict[str, list[int]] = {"ectopic": [], "missed": [], "extra": [], "longshort": []}
    rr = np.asarray(rr, dtype=float)
    n = rr.size
    if n < 3:
        return found

    drrs = np.ediff1d(rr, to_begin=0.0)
    drrs[0] = np.mean(drrs[1:])
    drrs = drrs / _threshold(drrs)
    padded = np.pad(drrs, 2, mode="reflect")
    s12 = np.zeros(n)
    s22 = np.zeros(n)
    for i in range(n):
        d = padded[i + 2]
        if d > 0:
            s12[i] = max(padded[i + 1], padded[i + 3])
        elif d < 0:
            s12[i] = min(padded[i + 1], padded[i + 3])
        if d >= 0:
            s22[i] = min(padded[i + 3], padded[i + 4])
        else:
            s22[i] = max(padded[i + 3], padded[i + 4])

    medrr = _rolling(rr, _MEDIAN_WINDOW, lambda w: float(np.median(w)))
    mrrs = rr - medrr
    mrrs[mrrs < 0] *= 2
    th2 = _threshold(mrrs)
    mrrs = mrrs / th2

    i = 0
    while i < n - 2:
        if abs(drrs[i]) <= 1:
            i += 1
            continue
        if (drrs[i] > 1 and s12[i] < -_C1 * drrs[i] - _C2) or (drrs[i] < -1 and s12[i] > -_C1 * drrs[i] + _C2):
            found["ectopic"].append(i)
            i += 1
            continue
        candidates = [i]
        if abs(drrs[i + 1]) < abs(drrs[i + 2]):
            candidates.append(i + 1)
        for j in candidates:
            long_beat = drrs[j] > 1 and s22[j] < -1
            far_from_median = abs(mrrs[j]) > 3
            short_beat = drrs[j] < -1 and s22[j] > 1
            if not (long_beat or far_from_median or short_beat):
                i += 1
                continue
            missed = abs(rr[j] / 2 - medrr[j]) < th2[j]
            extra = j + 1 < n and abs(rr[j] + rr[j + 1] - medrr[j]) < th2[j]
            if short_beat and extra:
                found["extra"].append(j)
            elif long_beat and missed:
                found["missed"].append(j)
            else:
                found["longshort"].append(j)
            i += 1
    return found


def removal_mask(n: int, found: dict[str, list[int]]) -> np.ndarray:
    remove = np.zeros(n, dtype=bool)
    for index in found["missed"] + found["longshort"]:
        remove[index] = True
    for index in found["ectopic"] + found["extra"]:
        remove[max(index - 1, 0): index + 2] = True
    return remove


@dataclass(frozen=True)
class Beats:
    t: np.ndarray
    rr: np.ndarray
    ok: np.ndarray
    keep: np.ndarray
    linked: np.ndarray
    artifacts: dict[str, int]

    @property
    def checked(self) -> int:
        return int(self.ok.sum())

    @property
    def removed(self) -> int:
        return int(self.ok.sum() - self.keep.sum())


def prepare_beats(rows: list, clock: dict) -> Beats:
    """IBI 를 서버 시각으로 옮기고 상태, 범위, 박동 분류로 걸러냄"""
    parsed = []
    for row in rows:
        t, value = number(row[0]), number(row[1])
        if t is not None and value is not None:
            parsed.append((watch_to_server(t, clock), value, row[2] if len(row) > 2 else None))
    parsed.sort(key=lambda item: item[0])
    n = len(parsed)
    t = np.array([p[0] for p in parsed], dtype=float)
    rr = np.array([p[1] for p in parsed], dtype=float)
    ok = np.array([p[2] == IBI_VALID_STATUS and IBI_MIN_MS <= p[1] <= IBI_MAX_MS for p in parsed], dtype=bool)
    keep = np.zeros(n, dtype=bool)
    linked = np.zeros(n, dtype=bool)
    counts = {"ectopic": 0, "missed": 0, "extra": 0, "longshort": 0}

    runs: list[list[int]] = []
    for i in range(n):
        if not ok[i]:
            continue
        joined = runs and runs[-1][-1] == i - 1 and t[i] - t[i - 1] <= rr[i] + RUN_GAP_MS
        if joined:
            runs[-1].append(i)
        else:
            runs.append([i])
    for run in runs:
        if len(run) < LT_MIN_RUN:
            continue
        index = np.array(run)
        found = classify_artifacts(rr[index])
        for kind, hits in found.items():
            counts[kind] += len(hits)
        keep[index[~removal_mask(index.size, found)]] = True
        linked[index[:-1]] = True
    return Beats(t=t, rr=rr, ok=ok, keep=keep, linked=linked, artifacts=counts)


def hrv_metrics(beats: Beats, intervals: list[Interval]) -> dict:
    """구간 안 정상 박동으로 mean NN, RMSSD, SDNN, pNN50 을 낸다. LF/HF 는 내지 않는다"""
    span_ms = sum(e - s for s, e in intervals)
    if beats.t.size == 0 or span_ms <= 0:
        return {"status": "UNAVAILABLE"}
    slot = np.full(beats.t.size, -1)
    for k, (s, e) in enumerate(intervals):
        slot[(beats.t >= s) & (beats.t < e)] = k
    inside = slot >= 0
    checked = int((inside & beats.ok).sum())
    if checked == 0:
        return {"status": "UNAVAILABLE"}
    kept = inside & beats.keep
    nn = beats.rr[kept]
    pairs = kept[:-1] & kept[1:] & beats.linked[:-1] & (slot[:-1] == slot[1:])
    diffs = np.abs(np.diff(beats.rr))[pairs]
    removed_ratio = 1 - kept.sum() / checked
    quality = {
        "beats": int(kept.sum()),
        "nn_sec": round(float(nn.sum()) / 1000, 1),
        "coverage_pct": round(float(nn.sum()) / span_ms * 100, 1),
        "removed_pct": round(float(removed_ratio) * 100, 1),
    }
    enough = span_ms >= HRV_MIN_WINDOW_MS and nn.sum() / span_ms >= HRV_MIN_COVERAGE and diffs.size >= 2
    if not enough or removed_ratio > HRV_MAX_REMOVED:
        return {"status": "INSUFFICIENT_DATA", **quality}
    rmssd = math.sqrt(float(np.mean(diffs**2)))
    return {
        "status": "MEASURED",
        "mean_nn_ms": round(float(nn.mean()), 1),
        "sdnn_ms": round(float(nn.std(ddof=1)), 1),
        "rmssd_ms": round(rmssd, 1),
        "ln_rmssd": round(math.log(rmssd), 3) if rmssd > 0 else None,
        "pnn50_pct": round(float(np.mean(diffs > 50)) * 100, 1),
        **quality,
    }
