from scipy.stats import spearmanr

from app.services.biosignal.clock import number
from app.services.biosignal.stats import rounded
from app.services.feedback_points import VOICE_WORK_MIN, voice_band

VERSION = "BIOSIGNAL_FUSION_V1"
BODY_HIGH_VS_BASELINE_BPM = 10.0
BODY_HIGH_VS_CALL_BPM = 3.0
MIN_CORRELATION_TURNS = 5
QUADRANTS = ("body_high_voice_steady", "body_high_voice_shaky", "body_low_voice_steady", "body_low_voice_shaky")
VOICE_VALUES = ("avti", "answer_delay_sec", "syllables_per_sec", "loudness_db")
VOICE_FLAGS = {
    "long_pause": "pause_sec",
    "quiet_drop": "quiet_drop_db",
    "late_answer": "late_answer_sec",
    "fast_rate": "fast_rate_ratio",
}


def _by_turn(turn_metrics: dict | None, name: str) -> dict[str, object]:
    values = (turn_metrics or {}).get(name)
    return {str(k): v for k, v in values.items()} if isinstance(values, dict) else {}


def _turn_ids(*sources: dict) -> list[str]:
    ids = {str(k) for source in sources for k in source}
    return sorted((k for k in ids if k.isdigit()), key=int)


def _body(hr: dict, baseline_measured: bool) -> tuple[str | None, str | None]:
    if hr.get("status") != "MEASURED":
        return None, None
    if baseline_measured and number(hr.get("delta_vs_baseline")) is not None:
        return ("high" if hr["delta_vs_baseline"] >= BODY_HIGH_VS_BASELINE_BPM else "low"), "baseline"
    if number(hr.get("delta_vs_call")) is not None:
        return ("high" if hr["delta_vs_call"] >= BODY_HIGH_VS_CALL_BPM else "low"), "call"
    return None, None


def _voice(avti: object) -> str | None:
    value = number(avti)
    if value is None:
        return None
    return "shaky" if value >= VOICE_WORK_MIN else "steady"


def _spearman(rows: list[dict], x: str, y: str) -> dict:
    points = [(r[x], r[y]) for r in rows if number(r.get(x)) is not None and number(r.get(y)) is not None]
    if not points:
        return {"status": "UNAVAILABLE"}
    if len(points) < MIN_CORRELATION_TURNS:
        return {"status": "INSUFFICIENT_DATA", "turns": len(points)}
    xs, ys = zip(*points, strict=True)
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        return {"status": "INSUFFICIENT_DATA", "turns": len(points), "reason": "constant"}
    return {"status": "MEASURED", "rho": rounded(spearmanr(xs, ys).statistic, 2), "turns": len(points)}


def fuse_turns(*, cardiac: dict, eda: dict | None, turn_metrics: dict | None) -> dict:
    """턴별 몸 반응과 목소리 지표를 한 표로 묶고 2×2 칸과 개인 안 순위상관을 낸다"""
    hr_turns = cardiac.get("turns") or {}
    eda_turns = (eda or {}).get("turns") or {}
    baseline_measured = (cardiac.get("baseline") or {}).get("status") == "MEASURED"
    values = {name: _by_turn(turn_metrics, name) for name in VOICE_VALUES}
    flags = {flag: set(_by_turn(turn_metrics, source)) for flag, source in VOICE_FLAGS.items()}

    rows = []
    for turn in _turn_ids(hr_turns, _by_turn(turn_metrics, "turn_sec"), values["avti"]):
        hr = hr_turns.get(turn) or {}
        skin = eda_turns.get(turn) or {}
        body, basis = _body(hr, baseline_measured)
        voice = _voice(values["avti"].get(turn))
        band = voice_band(number(values["avti"].get(turn)))
        rows.append({
            "turn": int(turn),
            "hr_bpm": hr.get("mean_bpm"),
            "hr_delta_vs_baseline": hr.get("delta_vs_baseline"),
            "hr_delta_vs_call": hr.get("delta_vs_call"),
            "eda_change_vs_baseline_pct": skin.get("change_vs_baseline_pct"),
            "eda_change_vs_call_pct": skin.get("change_vs_call_pct"),
            **{name: values[name].get(turn) for name in VOICE_VALUES},
            "voice_band": band.value if band else None,
            **{flag: turn in hits for flag, hits in flags.items()},
            "body": body,
            "body_basis": basis,
            "voice": voice,
            "quadrant": f"body_{body}_voice_{voice}" if body and voice else None,
        })

    reaction = "hr_delta_vs_baseline" if baseline_measured else "hr_delta_vs_call"
    rated = [r for r in rows if r["body"] is not None and number(r[reaction]) is not None]
    steady_high = [r for r in rated if r["quadrant"] == "body_high_voice_steady"]
    correlations = {f"hr_vs_{name}": _spearman(rows, "hr_bpm", name) for name in VOICE_VALUES}
    correlations["eda_vs_avti"] = _spearman(rows, "eda_change_vs_call_pct", "avti")
    return {
        "version": VERSION,
        "body_basis": "baseline" if baseline_measured else "call",
        "thresholds": {
            "body_high_vs_baseline_bpm": BODY_HIGH_VS_BASELINE_BPM,
            "body_high_vs_call_bpm": BODY_HIGH_VS_CALL_BPM,
            "voice_shaky_avti": VOICE_WORK_MIN,
        },
        "turns": rows,
        "quadrants": {q: sum(r["quadrant"] == q for r in rows) for q in QUADRANTS},
        "praise_turn": max(steady_high, key=lambda r: r[reaction])["turn"] if steady_high else None,
        "strongest_turn": max(rated, key=lambda r: r[reaction])["turn"] if rated else None,
        "correlations": correlations,
    }
