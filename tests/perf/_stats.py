import statistics


def percentile(samples: list[float], p: float) -> float | None:
    """선형 보간 백분위. samples 비어있으면 None."""
    if not samples:
        return None
    s = sorted(samples)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(samples: list[float | None]) -> dict | None:
    """None을 제외하고 n/min/p50/p95/p99/max/mean(ms) 산출."""
    vals = [x for x in samples if x is not None]
    if not vals:
        return None
    return {
        "n": len(vals),
        "min": round(min(vals), 1),
        "p50": round(percentile(vals, 50), 1),
        "p95": round(percentile(vals, 95), 1),
        "p99": round(percentile(vals, 99), 1),
        "max": round(max(vals), 1),
        "mean": round(statistics.fmean(vals), 1),
    }


_COLS = ("n", "min", "p50", "p95", "p99", "max", "mean")


def print_table(rows: list[tuple[str, dict | None]], *, unit: str = "ms") -> None:
    """정렬된 표로 출력."""
    label_w = max((len(label) for label, _ in rows), default=10)
    label_w = max(label_w, len("metric"))
    header = f"{'metric':<{label_w}}  " + "  ".join(f"{c:>8}" for c in _COLS)
    print(header)
    print("-" * len(header))
    for label, summ in rows:
        if summ is None:
            print(f"{label:<{label_w}}  {'(측정값 없음)':>8}")
            continue
        cells = "  ".join(f"{summ[c]:>8}" for c in _COLS)
        print(f"{label:<{label_w}}  {cells}")
    print(f"\n(단위: {unit}, 백분위는 선형 보간)")
