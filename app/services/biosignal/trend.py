import numpy as np

from app.services.biosignal.clock import number
from app.services.biosignal.stats import rounded

VERSION = "BIOSIGNAL_TREND_V1"
MIN_TREND_SESSIONS = 3
EXCLUDED_TYPES = frozenset({"WARMUP"})
METRICS = {
    "hr_call_delta": ("cardiac", "hr", "call_delta", "value"),
    "hr_anticipation_delta": ("cardiac", "hr", "anticipation_delta", "value"),
    "hr_peak_delta": ("cardiac", "hr", "peak", "delta"),
    "hr_listen_minus_speak": ("cardiac", "hr", "listen_minus_speak", "value"),
    "hr_recovery_pct": ("cardiac", "hr", "recovery_pct", "value"),
    "eda_call_change_pct": ("eda", "change_vs_baseline_pct", "call", "value"),
    "eda_rises_per_min": ("eda", "rises", "call", "per_min"),
}


def _dig(data: object, path: tuple[str, ...]) -> float | None:
    for key in path:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return number(data)


def session_summary(row: dict) -> dict:
    metrics = row.get("metrics") or {}
    values = {name: _dig(metrics, path) for name, path in METRICS.items()}
    return {"session_id": row["session_id"], "at": row["at"], **values}


def trend_of(values: list[float | None]) -> dict:
    points = [(i, v) for i, v in enumerate(values) if v is not None]
    if len(points) < MIN_TREND_SESSIONS:
        return {"status": "INSUFFICIENT_DATA", "sessions": len(points)}
    xs = np.array([i for i, _ in points], dtype=float)
    ys = np.array([v for _, v in points], dtype=float)
    return {
        "status": "MEASURED",
        "sessions": len(points),
        "first": rounded(ys[0]),
        "last": rounded(ys[-1]),
        "change": rounded(ys[-1] - ys[0]),
        "slope_per_session": rounded(np.polyfit(xs, ys, 1)[0], 2),
    }


def _trends(summaries: list[dict]) -> dict:
    return {name: trend_of([s[name] for s in summaries]) for name in METRICS}


def summarize_trend(rows: list[dict]) -> dict:
    """같은 조건 회차끼리, 시나리오별 첫 회차끼리, 회복은 전체 회차로 추세를 낸다"""
    ordered = sorted((r for r in rows if r.get("session_type") not in EXCLUDED_TYPES), key=lambda r: r["at"])
    summaries = [session_summary(r) for r in ordered]
    groups: dict[tuple, list[dict]] = {}
    first_by_scenario: dict[int, dict] = {}
    for row, summary in zip(ordered, summaries, strict=True):
        key = (row.get("scenario_id"), row.get("difficulty"), row.get("ai_personality"))
        groups.setdefault(key, []).append(summary)
        if row.get("scenario_id") is not None:
            first_by_scenario.setdefault(row["scenario_id"], summary)
    first = list(first_by_scenario.values())
    return {
        "version": VERSION,
        "sessions": len(summaries),
        "groups": [
            {
                "scenario_id": scenario_id,
                "difficulty": difficulty,
                "ai_personality": personality,
                "sessions": members,
                "trend": _trends(members),
            }
            for (scenario_id, difficulty, personality), members in groups.items()
        ],
        "first_encounters": {"sessions": first, "trend": _trends(first)},
        "anticipation": trend_of([s["hr_anticipation_delta"] for s in summaries]),
        "recovery": trend_of([s["hr_recovery_pct"] for s in summaries]),
    }
