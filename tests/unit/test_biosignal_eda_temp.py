import numpy as np
import pytest

from app.services.biosignal.eda import analyze_eda, rise_times
from app.services.biosignal.skin_temp import analyze_skin_temp
from tests.unit.test_biosignal_cardiac import C0, C1, T0, W2P, _markers, _speaking, _timeline, _times, watch

BASE, ANTICIPATION, LISTEN, SPEAK, RECOVERY = 2.0, 2.4, 2.6, 2.8, 2.2
CLOCK = {"watch_minus_phone_ms": W2P}


def _eda_value(t: float, *, lag: float = 0.0) -> float:
    if C0 + lag <= t < C1 + lag:
        return SPEAK if _speaking(t - lag) else LISTEN
    if t < T0 + 60_000:
        return 3.0
    if t < T0 + 120_000:
        return BASE
    if t < C0:
        return ANTICIPATION
    if t < C1:
        return SPEAK if _speaking(t) else LISTEN
    return RECOVERY


def _eda(*, lag: float = 0.0, override=None) -> list[list]:
    rows = []
    for t in _times():
        value = override(t) if override and override(t) is not None else _eda_value(t, lag=lag)
        rows.append([watch(t), value, 0])
    return rows


def _analyze_eda(**overrides) -> dict:
    args = {"timeline": _timeline(), "eda": _eda(), "acc": [], "clock": CLOCK, "markers": _markers()}
    args.update(overrides)
    return analyze_eda(**args)


def _call_mean() -> float:
    return float(np.mean([_eda_value(t) for t in _times() if C0 <= t < C1]))


# --- EDA -------------------------------------------------------------------


def test_eda_levels_change_from_baseline_by_phase() -> None:
    result = _analyze_eda()
    changes = result["change_vs_baseline_pct"]

    assert result["status"] == "MEASURED"
    assert result["means"]["baseline"]["value"] == BASE
    assert changes["anticipation"] == {"status": "MEASURED", "value": 20.0}
    assert changes["listen"]["value"] == 30.0
    assert changes["speak"]["value"] == 40.0
    assert changes["recovery"]["value"] == 10.0
    assert changes["call"]["value"] == pytest.approx((_call_mean() - BASE) / BASE * 100, abs=0.1)
    assert result["listen_minus_speak_pct"]["value"] == pytest.approx((LISTEN - SPEAK) / _call_mean() * 100, abs=0.02)


def test_eda_rises_are_counted_once_per_step_in_their_window() -> None:
    rises = _analyze_eda()["rises"]

    assert {name: rises[name]["count"] for name in rises} == {
        "baseline": 0, "anticipation": 1, "call": 6, "listen": 1, "speak": 5, "recovery": 0,
    }
    assert rises["call"]["per_min"] == 2.0


def test_rise_needs_three_percent_within_five_seconds_and_one_per_step() -> None:
    t = np.arange(0, 60_000, 1000, dtype=float)

    plateau = np.where(t < 10_000, 2.0, 2.2)
    assert list(rise_times(t, plateau)) == [10_000]

    slow = 2.0 * (1 + 0.005 * t / 1000)
    assert rise_times(t, slow).size == 0

    double = np.where(t < 10_000, 2.0, np.where(t < 12_000, 2.1, 2.2))
    assert list(rise_times(t, double)) == [10_000]

    again = np.where(t < 10_000, 2.0, np.where(t < 30_000, 2.1, np.where(t < 31_000, 2.0, 2.1)))
    assert list(rise_times(t, again)) == [10_000, 31_000]


def test_eda_motion_seconds_are_left_out() -> None:
    spike = range(C0 + 60_000, C0 + 65_000, 1000)
    acc = [[watch(t), 1.0, 0.4 if t in spike else 0.02] for t in range(C0, C1, 1000)]

    result = _analyze_eda(eda=_eda(override=lambda t: 5.0 if t in spike else None), acc=acc)

    assert result["means"]["listen"]["value"] == LISTEN
    assert result["quality"]["eda_motion_excluded"] == 7


def test_eda_turn_reactions_use_a_three_second_lag() -> None:
    turns = _analyze_eda(eda=_eda(lag=3000))["turns"]

    assert set(turns) == {"1", "2", "3", "4", "5", "6"}
    for entry in turns.values():
        assert (entry["status"], entry["mean"], entry["change_vs_baseline_pct"]) == ("MEASURED", SPEAK, 40.0)


def test_eda_trend_is_reported_raw_and_against_baseline() -> None:
    def rising(t: float) -> float | None:
        return 2.5 + 0.06 * (t - C0) / 60_000 if C0 <= t < C1 else None

    trend = _analyze_eda(eda=_eda(override=rising))["trend"]

    assert trend["per_min"]["value"] == pytest.approx(0.06, abs=0.001)
    assert trend["pct_of_baseline_per_min"]["value"] == pytest.approx(3.0, abs=0.05)


def test_eda_without_a_rest_baseline_keeps_relative_measures() -> None:
    result = _analyze_eda(markers=_markers(rest=False))

    assert result["means"]["baseline"] == {"status": "UNAVAILABLE"}
    assert result["change_vs_baseline_pct"]["call"] == {"status": "UNAVAILABLE"}
    assert result["listen_minus_speak_pct"]["status"] == "MEASURED"
    assert result["trend"]["pct_of_baseline_per_min"] == {"status": "UNAVAILABLE"}


def test_no_eda_is_unavailable_everywhere() -> None:
    result = _analyze_eda(eda=[])

    assert result["status"] == "UNAVAILABLE"
    assert all(entry == {"status": "UNAVAILABLE"} for entry in result["means"].values())
    assert all(entry == {"status": "UNAVAILABLE"} for entry in result["rises"].values())
    assert all(entry["status"] == "UNAVAILABLE" for entry in result["turns"].values())


def test_invalid_eda_rows_are_dropped() -> None:
    rows = [[watch(C0 + i * 1000), value, status] for i, (value, status) in enumerate(
        [(2.0, -1), (0.0, 0), (150.0, 0), (True, 0), (2.0, None), (2.1, 0)]
    )]

    quality = _analyze_eda(eda=rows)["quality"]

    assert (quality["eda_rows"], quality["eda_valid"]) == (6, 2)


# --- 피부온도 ----------------------------------------------------------------


def _temp_value(t: float) -> float:
    if t < T0 + 120_000:
        return 33.0
    if t < C0:
        return 32.9
    if t < C1:
        return 33.0 - 0.2 * (t - C0) / 60_000
    return 32.5


def _temps(step_ms: int = 10_000) -> list[list]:
    return [[watch(t), _temp_value(t), 0] for t in range(T0, C1 + 60_000, step_ms)]


def test_skin_temp_tracks_the_slow_drift_during_the_call() -> None:
    result = analyze_skin_temp(timeline=_timeline(), skin_temp=_temps(), clock=CLOCK, markers=_markers())
    call_mean = float(np.mean([_temp_value(t) for t in range(C0, C1, 10_000)]))

    assert result["status"] == "MEASURED"
    assert result["means"]["baseline"]["value"] == 33.0
    assert result["delta_vs_baseline_c"]["anticipation"] == {"status": "MEASURED", "value": -0.1}
    assert result["delta_vs_baseline_c"]["call"]["value"] == pytest.approx(call_mean - 33.0, abs=0.01)
    assert result["delta_vs_baseline_c"]["recovery"]["value"] == -0.5
    assert result["call_trend_c_per_min"]["value"] == pytest.approx(-0.2, abs=0.005)


def test_skin_temp_drops_out_of_range_and_needs_three_samples() -> None:
    rows = [[watch(T0 + 70_000), 15.0, 0], [watch(T0 + 80_000), 50.0, 0], [watch(T0 + 90_000), 33.0, 0]]
    rows += [[watch(T0 + 100_000), 33.1, 0]]

    result = analyze_skin_temp(timeline=_timeline(), skin_temp=rows, clock=CLOCK, markers=_markers())

    assert result["quality"] == {"temp_rows": 4, "temp_valid": 2}
    assert result["means"]["baseline"]["status"] == "INSUFFICIENT_DATA"
    assert result["delta_vs_baseline_c"]["call"] == {"status": "UNAVAILABLE"}

    empty = analyze_skin_temp(timeline=_timeline(), skin_temp=[], clock=CLOCK, markers=_markers())
    assert empty["status"] == "UNAVAILABLE"
    assert empty["call_trend_c_per_min"] == {"status": "UNAVAILABLE"}
