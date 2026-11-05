from app.services.biosignal.cardiac import analyze_cardiac
from app.services.biosignal.eda import analyze_eda
from app.services.biosignal.fusion import fuse_turns
from tests.unit.test_biosignal_cardiac import W2P, _hr, _markers, _timeline
from tests.unit.test_biosignal_eda_temp import _eda


def _hr_turn(delta_baseline: float | None, delta_call: float | None = 0.0, bpm: float = 80.0) -> dict:
    return {"status": "MEASURED", "mean_bpm": bpm, "delta_vs_baseline": delta_baseline, "delta_vs_call": delta_call}


def _cardiac(turns: dict, *, baseline: bool = True) -> dict:
    return {"baseline": {"status": "MEASURED" if baseline else "UNAVAILABLE"}, "turns": turns}


def _row(result: dict, turn: int) -> dict:
    return next(r for r in result["turns"] if r["turn"] == turn)


def test_quadrants_follow_the_baseline_delta_and_avti_thresholds() -> None:
    cardiac = _cardiac({
        "1": _hr_turn(10.0), "2": _hr_turn(9.9), "3": _hr_turn(15.0), "4": _hr_turn(2.0),
    })
    metrics = {"avti": {"1": 4.9, "2": 5.0, "3": 5.5, "4": 1.0}}

    result = fuse_turns(cardiac=cardiac, eda=None, turn_metrics=metrics)

    assert [r["quadrant"] for r in result["turns"]] == [
        "body_high_voice_steady", "body_low_voice_shaky", "body_high_voice_shaky", "body_low_voice_steady",
    ]
    assert result["quadrants"] == {q: 1 for q in result["quadrants"]}
    assert [r["voice_band"] for r in result["turns"]] == ["MIDDLE", "NEEDS_WORK", "NEEDS_WORK", "GOOD"]
    assert result["body_basis"] == "baseline"
    assert (result["praise_turn"], result["strongest_turn"]) == (1, 3)


def test_call_mean_is_the_basis_when_there_is_no_baseline() -> None:
    cardiac = _cardiac({"1": _hr_turn(None, 3.0), "2": _hr_turn(None, 2.9)}, baseline=False)

    result = fuse_turns(cardiac=cardiac, eda=None, turn_metrics={"avti": {"1": 2.0, "2": 2.0}})

    assert result["body_basis"] == "call"
    assert [(r["body"], r["body_basis"]) for r in result["turns"]] == [("high", "call"), ("low", "call")]


def test_unknown_voice_or_body_leaves_the_quadrant_empty() -> None:
    cardiac = _cardiac({"1": _hr_turn(12.0), "2": {"status": "INSUFFICIENT_DATA", "mean_bpm": None}})
    metrics = {"turn_sec": {"1": [0, 3], "2": [5, 8], "3": [10, 12]}, "avti": {"2": 2.0}}

    result = fuse_turns(cardiac=cardiac, eda=None, turn_metrics=metrics)

    assert [r["turn"] for r in result["turns"]] == [1, 2, 3]
    assert [r["quadrant"] for r in result["turns"]] == [None, None, None]
    assert sum(result["quadrants"].values()) == 0
    assert (result["praise_turn"], result["strongest_turn"]) == (None, 1)


def test_voice_flags_come_from_the_judged_metrics_with_any_key_type() -> None:
    metrics = {
        "avti": {1: 2.0, 2: 2.0},
        "pause_sec": {2: [1.7]},
        "quiet_drop_db": {"1": 7.0},
        "late_answer_sec": {"2": 1.4},
        "fast_rate_ratio": {1: 1.3},
        "answer_delay_sec": {"1": 0.8, "2": 1.4},
    }

    result = fuse_turns(cardiac=_cardiac({"1": _hr_turn(1.0), "2": _hr_turn(1.0)}), eda=None, turn_metrics=metrics)

    flags = ("long_pause", "quiet_drop", "late_answer", "fast_rate")
    first, second = _row(result, 1), _row(result, 2)
    assert [first[f] for f in flags] == [False, True, False, True]
    assert [second[f] for f in flags] == [True, False, True, False]
    assert (first["answer_delay_sec"], second["answer_delay_sec"]) == (0.8, 1.4)


def test_rank_correlations_need_five_varying_turns() -> None:
    turns = {str(k): _hr_turn(float(k), bpm=70.0 + k) for k in range(1, 7)}
    metrics = {
        "avti": {str(k): 1.0 + k for k in range(1, 7)},
        "answer_delay_sec": {str(k): 3.0 - 0.1 * k for k in range(1, 7)},
        "loudness_db": {str(k): -20.0 for k in range(1, 7)},
        "syllables_per_sec": {str(k): 4.0 + k for k in range(1, 5)},
    }

    result = fuse_turns(cardiac=_cardiac(turns), eda=None, turn_metrics=metrics)["correlations"]

    assert result["hr_vs_avti"] == {"status": "MEASURED", "rho": 1.0, "turns": 6}
    assert result["hr_vs_answer_delay_sec"]["rho"] == -1.0
    assert result["hr_vs_loudness_db"] == {"status": "INSUFFICIENT_DATA", "turns": 6, "reason": "constant"}
    assert result["hr_vs_syllables_per_sec"] == {"status": "INSUFFICIENT_DATA", "turns": 4}
    assert result["eda_vs_avti"] == {"status": "UNAVAILABLE"}


def test_praise_turn_is_the_biggest_reaction_with_a_steady_voice() -> None:
    cardiac = _cardiac({"1": _hr_turn(12.0), "2": _hr_turn(20.0), "3": _hr_turn(25.0)})

    result = fuse_turns(cardiac=cardiac, eda=None, turn_metrics={"avti": {"1": 2.0, "2": 2.5, "3": 6.0}})

    assert (result["praise_turn"], result["strongest_turn"]) == (2, 3)


def test_fusion_lines_up_with_real_cardiac_and_eda_turns() -> None:
    args = {"timeline": _timeline(), "clock": {"watch_minus_phone_ms": W2P}, "markers": _markers()}
    cardiac = analyze_cardiac(hr=_hr(lag=3000), ibi=[], acc=[], **args)
    eda = analyze_eda(eda=_eda(lag=3000), acc=[], **args)
    avti = {"1": 2.0, "2": 6.5, "3": 2.2, "4": 3.0, "5": 7.0, "6": 1.5}
    metrics = {"turn_sec": {k: [0, 1] for k in avti}, "avti": avti}

    result = fuse_turns(cardiac=cardiac, eda=eda, turn_metrics=metrics)

    assert [r["turn"] for r in result["turns"]] == [1, 2, 3, 4, 5, 6]
    assert {r["hr_bpm"] for r in result["turns"]} == {88.0}
    assert {r["eda_change_vs_baseline_pct"] for r in result["turns"]} == {40.0}
    assert result["quadrants"]["body_high_voice_steady"] == 4
    assert result["quadrants"]["body_high_voice_shaky"] == 2
    assert result["correlations"]["eda_vs_avti"]["status"] == "INSUFFICIENT_DATA"
