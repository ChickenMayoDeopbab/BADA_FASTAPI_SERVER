import numpy as np
import pytest

from app.services.biosignal.cardiac import analyze_cardiac
from app.services.biosignal.clock import estimate_clock
from app.services.biosignal.ibi import Beats, classify_artifacts, hrv_metrics, prepare_beats, removal_mask
from app.services.biosignal.segments import build_segments

T0 = 1_762_300_000_000
P2S = 1500
W2P = -200
LATENCY = 20
C0 = T0 + 150_000
C1 = C0 + 180_000
SPEAK, LISTEN = 88.0, 84.0
SPEAK_WINDOWS = (
    [(C0 + 100, C0 + 20_000)]
    + [(C0 + 35_000 * (k - 1), C0 + 35_000 * (k - 1) + 20_000) for k in range(2, 6)]
    + [(C0 + 175_000, C1)]
)
NO_CLOCK = {"phone_to_server_ms": 0.0, "watch_minus_phone_ms": 0.0}


def phone(server_ms: float) -> float:
    return server_ms - P2S


def watch(server_ms: float) -> float:
    return server_ms - P2S + W2P


def _events() -> list[dict]:
    events = [{"kind": "call_open", "at_ms": C0}, {"kind": "audio_first", "at_ms": C0 + 100}]
    for k in range(1, 6):
        base = C0 + 35_000 * (k - 1)
        events += [
            {"kind": "user_turn_close", "at_ms": base + 20_000, "turn": k},
            {"kind": "ai_audio_start", "at_ms": base + 21_500, "playback": k},
            {"kind": "ai_speaking_end", "at_ms": base + 30_000, "playback": k},
            {"kind": "user_turn_open", "at_ms": base + 30_000, "turn": k + 1},
            {"kind": "ai_playback_done", "at_ms": base + 35_000, "playback": k},
        ]
    events += [
        {"kind": "call_close", "at_ms": C1, "reason": "USER_END"},
        {"kind": "user_turn_close", "at_ms": C1 + 300, "turn": 6},
    ]
    return events


def _timeline(*, pairs: bool = True, events: list[dict] | None = None) -> dict:
    clock_pairs = [[phone(s), s + LATENCY] for s in (C0, C0 + 90_000, C1)] if pairs else []
    return {"v": 1, "intent": {"v": 1}, "events": events or _events(), "anchors": [], "clock_pairs": clock_pairs}


def _markers(*, rest: bool = True) -> dict:
    markers = {"ringing_ms": phone(T0 + 120_000), "meditation": None, "recovery_end_ms": phone(C1 + 60_000)}
    if rest:
        markers["rest_baseline"] = [phone(T0), phone(T0 + 120_000)]
    return markers


def _speaking(t: float) -> bool:
    return any(s <= t < e for s, e in SPEAK_WINDOWS)


def _value(t: float, *, lag: float = 0.0) -> float:
    if C0 + lag <= t < C1 + lag:
        return SPEAK if _speaking(t - lag) else LISTEN
    if t < T0 + 60_000:
        return 90.0
    if t < T0 + 120_000:
        return 70.0
    if t < C0:
        return 78.0
    if t < C1:
        return SPEAK if _speaking(t) else LISTEN
    return 80.0 if t < C1 + 30_000 else 72.0


def _times(step_ms: int = 1000) -> range:
    return range(T0, C1 + 60_000, step_ms)


def _hr(*, lag: float = 0.0, step_ms: int = 1000, override=None) -> list[list]:
    rows = []
    for t in _times(step_ms):
        value = override(t) if override and override(t) is not None else _value(t, lag=lag)
        rows.append([watch(t), value, 1])
    return rows


def _mean(windows: list[tuple[float, float]], *, lag: float = 0.0) -> float:
    values = [_value(t, lag=lag) for t in _times() if any(s <= t < e for s, e in windows)]
    return float(np.mean(values))


def _analyze(**overrides) -> dict:
    args = {
        "timeline": _timeline(),
        "hr": _hr(),
        "ibi": [],
        "acc": [],
        "clock": {"watch_minus_phone_ms": W2P},
        "markers": _markers(),
    }
    args.update(overrides)
    return analyze_cardiac(**args)


def _rr_series(seed: int, n: int, *, mean: float = 857.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 12)
    return mean + 30 * np.sin(np.arange(n) / 6) + noise


def _beats(seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rr = _rr_series(seed, 600)
    times = T0 + np.cumsum(rr)
    keep = times < C1 + 60_000
    return times[keep], rr[keep]


def _ibi_rows(times: np.ndarray, rr: np.ndarray, status: list[int] | None = None) -> list[list]:
    status = status or [0] * len(rr)
    return [[watch(t), float(v), s] for t, v, s in zip(times, rr, status, strict=True)]


def _rmssd_in(times: np.ndarray, rr: np.ndarray, start: float, end: float) -> float:
    inside = (times + LATENCY >= start) & (times + LATENCY < end)
    return float(np.sqrt(np.mean(np.diff(rr[inside]) ** 2)))


# --- 시계 ------------------------------------------------------------------


def test_clock_uses_the_median_of_pairs_and_flags_large_offsets() -> None:
    clock = estimate_clock({"clock_pairs": [[1000, 2510], [2000, 3490], [3000, 4520]]}, {"watch_minus_phone_ms": -40})
    assert clock == {
        "phone_to_server_ms": 1510.0, "watch_minus_phone_ms": -40.0, "source": "pairs", "pairs": 3, "suspicious": False,
    }
    assert estimate_clock({"clock_pairs": [[0, 7000]]}, None)["suspicious"] is True


def test_clock_falls_back_to_ws_events_then_to_zero() -> None:
    events = [{"kind": "call_open", "at_ms": 10_000}, {"kind": "call_close", "at_ms": 90_000}]
    fallback = estimate_clock({"events": events}, {"phone_ws_open_ms": 8_450, "phone_call_end_ms": 88_510})
    assert (fallback["source"], fallback["phone_to_server_ms"], fallback["pairs"]) == ("ws_events", 1520.0, 2)

    nothing = estimate_clock({"events": events}, {"watch_minus_phone_ms": True})
    assert (nothing["source"], nothing["phone_to_server_ms"], nothing["watch_minus_phone_ms"]) == ("none", 0.0, 0.0)


# --- 구간 ------------------------------------------------------------------


def test_segments_split_speak_and_listen_from_playback_and_turn_events() -> None:
    clock = estimate_clock(_timeline(), {"watch_minus_phone_ms": W2P})
    seg = build_segments(_timeline(), _markers(), clock)

    assert seg.call == (C0, C1)
    assert [(turn, s, e) for turn, s, e in seg.speak] == [(k + 1, s, e) for k, (s, e) in enumerate(SPEAK_WINDOWS)]
    assert seg.listen[0] == (C0, C0 + 100)
    assert seg.listen[1] == (C0 + 20_000, C0 + 35_000)
    assert seg.listen[-1] == (C0 + 160_000, C0 + 175_000)
    assert seg.baseline == (T0 + 60_000 + LATENCY, T0 + 120_000 + LATENCY)
    assert seg.anticipation == (T0 + 120_000 + LATENCY, C0)
    assert seg.recovery == (C1, C1 + 60_000)
    assert seg.phase_at(C0 + 40_000) == ("speak", 2)
    assert seg.phase_at(C0 + 25_000) == ("listen", 2)
    assert seg.phase_at(T0 + 90_000) == ("baseline", None)
    assert seg.phase_at(C1 + 1) == ("recovery", None)
    assert seg.phase_at(T0) == (None, None)


def test_turn_start_falls_back_to_barge_in_open_then_previous_close() -> None:
    events = [
        {"kind": "call_open", "at_ms": 0},
        {"kind": "ai_playback_done", "at_ms": 1_000, "playback": 1},
        {"kind": "ai_playback_done", "at_ms": 2_000, "playback": 2},
        {"kind": "user_turn_close", "at_ms": 6_000, "turn": 1},
        {"kind": "user_turn_open", "at_ms": 9_000, "turn": 2},
        {"kind": "user_turn_close", "at_ms": 12_000, "turn": 2},
        {"kind": "user_turn_close", "at_ms": 20_000, "turn": 3},
        {"kind": "call_close", "at_ms": 30_000, "reason": "USER_END"},
    ]
    seg = build_segments({"events": events}, {}, NO_CLOCK)

    assert seg.speak == [(1, 2_000, 6_000), (2, 9_000, 12_000), (3, 12_000, 20_000)]
    assert seg.listen == [(0, 2_000), (6_000, 9_000), (20_000, 30_000)]


def test_short_rest_has_no_baseline_and_meditation_cuts_anticipation() -> None:
    events = [{"kind": "call_open", "at_ms": 200_000}, {"kind": "call_close", "at_ms": 260_000}]
    markers = {"rest_baseline": [0, 100_000], "meditation": [150_000, 190_000], "recovery_end_ms": 270_000}
    seg = build_segments({"events": events}, markers, NO_CLOCK)

    assert seg.baseline is None
    assert seg.anticipation == (190_000, 200_000)
    assert seg.recovery == (260_000, 270_000)

    default = build_segments({"events": events}, {}, NO_CLOCK)
    assert default.anticipation == (170_000, 200_000)
    assert build_segments({"events": events}, {"meditation": [150_000, 197_000]}, NO_CLOCK).anticipation is None


# --- 심박 지표 -------------------------------------------------------------


def test_heart_rate_reactions_follow_the_call_phases() -> None:
    result = _analyze()
    hr = result["hr"]
    call_mean = _mean([(C0, C1)])

    assert result["clock"]["source"] == "pairs"
    assert result["baseline"] == {
        "status": "MEASURED", "value": 70.0, "samples": 60, "coverage_pct": 100.0, "source": "rest", "window_sec": 60.0,
    }
    assert hr["means"]["anticipation"]["value"] == 78.0
    assert hr["anticipation_delta"] == {"status": "MEASURED", "value": 8.0}
    assert hr["call_delta"]["value"] == pytest.approx(call_mean - 70, abs=0.15)
    assert hr["listen_minus_speak"] == {"status": "MEASURED", "value": LISTEN - SPEAK}
    assert hr["means"]["recovery"]["value"] == 76.0
    assert hr["recovery_pct"]["value"] == pytest.approx((call_mean - 72) / (call_mean - 70) * 100, abs=0.5)
    first = _mean([(C0 + 30_000, C0 + 90_000)])
    last = _mean([(C1 - 60_000, C1)])
    assert hr["first_vs_last_min"]["value"] == pytest.approx(last - first, abs=0.15)


def test_peak_is_the_highest_ten_second_mean_inside_a_speak_turn() -> None:
    peak = _analyze()["hr"]["peak"]

    assert (peak["status"], peak["bpm"], peak["delta"], peak["phase"]) == ("MEASURED", SPEAK, SPEAK - 70, "speak")
    assert any(s <= C0 + peak["at_sec"] * 1000 < e for s, e in SPEAK_WINDOWS)


def test_peak_ignores_a_single_second_spike() -> None:
    peak = _analyze(hr=_hr(override=lambda t: 120.0 if t == C0 + 45_000 else None))["hr"]["peak"]

    assert SPEAK < peak["bpm"] < SPEAK + 4


def test_turn_reactions_use_a_three_second_lag() -> None:
    turns = _analyze(hr=_hr(lag=3000))["turns"]

    assert set(turns) == {"1", "2", "3", "4", "5", "6"}
    for entry in turns.values():
        assert entry["status"] == "MEASURED"
        assert entry["mean_bpm"] == SPEAK
        assert entry["delta_vs_baseline"] == SPEAK - 70


def test_trend_is_the_slope_after_the_first_thirty_seconds() -> None:
    def rising(t: float) -> float | None:
        if not C0 <= t < C1:
            return None
        return 110.0 if t < C0 + 30_000 else 80 + 3 * (t - C0) / 60_000

    trend = _analyze(hr=_hr(override=rising))["hr"]["trend_bpm_per_min"]

    assert trend["status"] == "MEASURED"
    assert trend["value"] == pytest.approx(3.0, abs=0.05)


def test_large_movement_seconds_are_left_out_of_heart_rate() -> None:
    spike = range(C0 + 60_000, C0 + 65_000, 1000)
    hr = _hr(override=lambda t: 130.0 if t in spike else None)
    acc = [[watch(t), 1.0, 0.4 if t in spike else 0.02] for t in range(C0, C1, 1000)]

    gated = _analyze(hr=hr, acc=acc)
    ungated = _analyze(hr=hr)

    assert gated["hr"]["means"]["listen"]["value"] == LISTEN
    assert gated["quality"]["hr_motion_excluded"] == 7
    assert gated["motion"]["moving_pct"]["listen"] == pytest.approx(5 / 76 * 100, abs=0.1)
    assert ungated["hr"]["means"]["listen"]["value"] > LISTEN + 2
    assert ungated["motion"] == {"status": "UNAVAILABLE", "threshold_g": 0.15, "moving_pct": {}}


def test_sparse_heart_rate_is_insufficient_not_measured() -> None:
    result = _analyze(hr=_hr(step_ms=3000))

    assert result["hr"]["means"]["call"]["status"] == "INSUFFICIENT_DATA"
    assert result["baseline"]["status"] == "INSUFFICIENT_DATA"
    assert result["hr"]["call_delta"] == {"status": "INSUFFICIENT_DATA"}


def test_previous_sessions_fill_in_a_missing_rest_baseline() -> None:
    result = _analyze(markers=_markers(rest=False), previous_baselines=[60, 70, 71, 69, 73, 90])

    assert result["baseline"] == {"status": "MEASURED", "value": 71.0, "source": "previous_sessions", "sessions": 5}
    assert result["hr"]["call_delta"]["value"] == pytest.approx(_mean([(C0, C1)]) - 71, abs=0.15)


def test_without_any_baseline_only_relative_measures_remain() -> None:
    result = _analyze(markers=_markers(rest=False))

    assert result["baseline"] == {"status": "UNAVAILABLE", "source": "none"}
    assert result["hr"]["call_delta"] == {"status": "UNAVAILABLE"}
    assert result["hr"]["listen_minus_speak"]["value"] == LISTEN - SPEAK
    assert (result["hr"]["peak"]["bpm"], result["hr"]["peak"]["delta"]) == (SPEAK, None)
    assert result["hr"]["recovery_pct"] == {"status": "UNAVAILABLE"}


@pytest.mark.parametrize(
    ("stop_after_ms", "status"),
    [(20_000, "INSUFFICIENT_DATA"), (54_900, "INSUFFICIENT_DATA"), (55_100, "MEASURED"), (59_700, "MEASURED")],
)
def test_recovery_needs_at_least_55_seconds_after_hanging_up(stop_after_ms: int, status: str) -> None:
    result = _analyze(markers={**_markers(), "recovery_end_ms": phone(C1 + stop_after_ms)})

    recovery = result["hr"]["recovery_pct"]
    assert recovery["status"] == status
    if status == "MEASURED":
        assert recovery["value"] > 0
    else:
        assert recovery["reason"] == "recovery_too_short"


def test_no_heart_rate_at_all_is_unavailable_everywhere() -> None:
    result = _analyze(hr=[])

    assert result["hr"]["means"]["call"] == {"status": "UNAVAILABLE"}
    assert result["hr"]["peak"] == {"status": "UNAVAILABLE"}
    assert result["hr"]["trend_bpm_per_min"] == {"status": "UNAVAILABLE"}
    assert all(entry["status"] == "UNAVAILABLE" for entry in result["turns"].values())


def test_invalid_heart_rate_rows_are_dropped() -> None:
    bad = [
        [watch(C0 + 1000), 72, 0], [watch(C0 + 2000), 250, 1], [watch(C0 + 3000), True, 1], [watch(C0 + 4000), 72, 1],
    ]
    quality = _analyze(hr=bad)["quality"]

    assert (quality["hr_rows"], quality["hr_valid"]) == (4, 1)


def test_report_timeline_bins_medians_and_blanks_thin_bins() -> None:
    spike = range(C0 + 60_000, C0 + 65_000, 1000)
    gap = {watch(t) for t in range(C0 + 120_000, C0 + 123_000, 1000)}
    hr = [row for row in _hr(override=lambda t: 130.0 if t in spike else None) if row[0] not in gap]
    acc = [[watch(t), 1.0, 0.4 if t in spike else 0.02] for t in range(C0, C1, 1000)]
    report = _analyze(hr=hr, acc=acc)["report_timeline"]

    assert (report["bin_sec"], report["start_sec"]) == (5, -90.0)
    assert len(report["hr"]) == len(report["phase"]) == 66
    assert (report["phase"][0], report["hr"][0]) == ("baseline", 70.0)
    at_call = 18
    assert (report["phase"][at_call], report["hr"][at_call]) == ("speak", SPEAK)
    assert report["phase"][at_call + 12] == "listen"
    assert report["hr"][at_call + 12] is None
    assert report["hr"][at_call + 11] == LISTEN
    assert report["hr"][at_call + 24] is None
    assert report["phase"][-1] == "recovery"


# --- IBI 정제와 HRV ---------------------------------------------------------


def test_classifier_catches_missed_extra_ectopic_and_long_beats() -> None:
    clean = _rr_series(3, 400, mean=800)

    missed = np.delete(clean.copy(), 101)
    missed[100] = clean[100] + clean[101]
    extra = np.insert(clean.copy(), 201, clean[200] * 0.6)
    extra[200] = clean[200] * 0.4
    ectopic = clean.copy()
    ectopic[300], ectopic[301] = clean[300] * 0.6, clean[301] + clean[300] * 0.4
    long = clean.copy()
    long[150] *= 1.35

    assert 100 in classify_artifacts(missed)["missed"]
    assert 200 in classify_artifacts(extra)["extra"]
    assert 301 in classify_artifacts(ectopic)["ectopic"]
    assert 150 in classify_artifacts(long)["longshort"]


def test_classifier_rarely_flags_clean_beats() -> None:
    flagged = [sum(len(v) for v in classify_artifacts(_rr_series(seed, 400, mean=800)).values()) for seed in range(10)]

    assert np.mean(flagged) / 400 < 0.01


def test_classifier_survives_flat_series_and_tiny_input() -> None:
    flat = np.full(100, 800.0)
    flat[50] = 1600.0

    found = classify_artifacts(flat)

    assert 50 in found["missed"] + found["longshort"]
    assert classify_artifacts(np.array([800.0, 810.0])) == {"ectopic": [], "missed": [], "extra": [], "longshort": []}


def test_removal_drops_neighbours_only_for_ectopic_and_extra() -> None:
    found = {"ectopic": [10], "missed": [20], "extra": [30], "longshort": [40]}

    removed = set(np.flatnonzero(removal_mask(50, found)))

    assert removed == {9, 10, 11, 20, 29, 30, 31, 40}


def test_beats_split_runs_on_bad_status_and_gaps_and_drop_short_runs() -> None:
    times = np.array([0, 800, 1600, 2400, 3200, 4000, 4800, 5600, 6400, 7200, 20_000, 20_800], dtype=float)
    rows = [[t, 800.0, 0] for t in times]
    rows[5][2] = -1
    rows[3][1] = 3000.0

    beats = prepare_beats(rows, NO_CLOCK)

    assert beats.checked == 10
    assert list(np.flatnonzero(beats.keep)) == []
    assert beats.removed == 10

    long_run = [[i * 800.0, 800.0 + (i % 3), 0] for i in range(12)]
    beats = prepare_beats(long_run, NO_CLOCK)
    assert beats.keep.all()
    assert list(beats.linked) == [True] * 11 + [False]


def test_coarse_ppg_steps_are_not_mistaken_for_artifacts() -> None:
    rng = np.random.default_rng(5)
    steps = 800.0 + 40.0 * rng.choice([-1, 0, 0, 0, 1], size=300)

    flagged = sum(len(v) for v in classify_artifacts(steps).values())

    assert flagged / 300 < 0.02


def test_hrv_matches_clean_beats_and_tolerates_one_missed_beat() -> None:
    times, rr = _beats()
    clean = _analyze(ibi=_ibi_rows(times, rr))["hrv"]["baseline"]
    expected = _rmssd_in(times, rr, T0 + 60_000 + LATENCY, T0 + 120_000 + LATENCY)

    assert clean["status"] == "MEASURED"
    assert clean["rmssd_ms"] == pytest.approx(expected, rel=0.05)
    assert clean["ln_rmssd"] == pytest.approx(np.log(clean["rmssd_ms"]), abs=0.01)

    index = int(np.flatnonzero(times > T0 + 90_000)[0])
    broken_times = np.delete(times, index + 1)
    broken_rr = np.delete(rr.copy(), index + 1)
    broken_rr[index] = rr[index] + rr[index + 1]
    broken = _analyze(ibi=_ibi_rows(broken_times, broken_rr))["hrv"]["baseline"]

    assert broken["status"] == "MEASURED"
    assert broken["rmssd_ms"] == pytest.approx(expected, rel=0.1)
    assert broken["removed_pct"] > 0


def test_too_many_removed_beats_make_hrv_insufficient() -> None:
    times, rr = _beats()
    rr = rr.copy()
    start = int(np.flatnonzero(times > T0 + 70_000)[0])
    for offset in (0, 15, 30, 45):
        rr[start + offset] *= 0.55
        rr[start + offset + 1] += rr[start + offset] * 0.8

    baseline = _analyze(ibi=_ibi_rows(times, rr))["hrv"]["baseline"]

    assert baseline["status"] == "INSUFFICIENT_DATA"
    assert baseline["removed_pct"] > 5


def test_hrv_windows_need_sixty_seconds_and_drop_flagged_rows() -> None:
    times, rr = _beats()
    status = [0] * len(rr)
    status[5] = -1
    result = _analyze(ibi=_ibi_rows(times, rr, status))

    assert result["hrv"]["anticipation"]["status"] == "INSUFFICIENT_DATA"
    assert result["hrv"]["speak"]["status"] == "MEASURED"
    assert result["hrv"]["listen"]["status"] == "MEASURED"
    assert result["hrv"]["recovery"]["status"] == "MEASURED"
    assert result["quality"]["ibi_checked"] == len(rr) - 1


def test_heart_rate_only_sources_have_no_hrv() -> None:
    hrv = _analyze(ibi=[])["hrv"]

    assert all(entry == {"status": "UNAVAILABLE"} for entry in hrv.values())


def test_hrv_pairs_do_not_cross_from_one_window_into_the_next() -> None:
    rr = np.array([800.0, 810.0, 800.0, 810.0, 900.0, 910.0, 900.0, 910.0] * 10)
    t = np.cumsum(rr)
    n = rr.size
    beats = Beats(
        t=t, rr=rr, ok=np.ones(n, bool), keep=np.ones(n, bool),
        linked=np.array([True] * (n - 1) + [False]), artifacts={},
    )
    split = float(t[3]) + 1

    joined = hrv_metrics(beats, [(0.0, float(t[-1]) + 1)])
    halves = hrv_metrics(beats, [(0.0, split), (split, float(t[-1]) + 1)])

    assert halves["rmssd_ms"] < joined["rmssd_ms"]


def test_hrv_metrics_without_beats_or_span_is_unavailable() -> None:
    empty = prepare_beats([], NO_CLOCK)

    assert hrv_metrics(empty, [(0, 60_000)]) == {"status": "UNAVAILABLE"}
    assert hrv_metrics(empty, []) == {"status": "UNAVAILABLE"}
