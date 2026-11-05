import numpy as np
import pytest
from pydantic import ValidationError

from app.schemas.biosignal import MAX_PPG_SAMPLES, BiosignalUpload
from app.services.biosignal.ppg import analyze_ppg, chunk_beats
from tests.unit.test_biosignal_api import _env, _payload, _seed
from tests.unit.test_biosignal_cardiac import C0, C1, NO_CLOCK, T0, W2P, _markers, _timeline, watch

HZ = 25
RR_MS = 60_000 / 70
CLOCK = {"watch_minus_phone_ms": W2P}


def _amp(t: float) -> float:
    if t < T0 + 60_000:
        return 120.0
    if t < T0 + 120_000:
        return 100.0
    if t < C0:
        return 95.0
    if t < C1:
        return 80.0
    return 90.0


def _chunks(
    *, start: float = T0, end: float = C1 + 60_000, chunk_sec: float = 60, amp=_amp, gap_ms: float = 0,
) -> list[dict]:
    rng = np.random.default_rng(1)
    chunks, t = [], float(start)
    while t < end:
        n = int(min(chunk_sec, (end - t) / 1000) * HZ)
        ts = t + np.arange(n) * 1000 / HZ
        phase = 2 * np.pi * ts / RR_MS
        wave = np.sin(phase) + 0.3 * np.sin(2 * phase)
        breathing = 10 * np.sin(2 * np.pi * 0.25 * ts / 1000)
        signal = 50_000 + np.array([amp(x) for x in ts]) * wave + breathing + rng.normal(0, 2, n)
        chunks.append({"t0": watch(t), "hz": HZ, "green": signal.round(1).tolist()})
        t += n * 1000 / HZ + gap_ms
    return chunks


def _analyze(**overrides) -> dict:
    args = {"timeline": _timeline(), "ppg": _chunks(), "acc": [], "clock": CLOCK, "markers": _markers()}
    args.update(overrides)
    return analyze_ppg(**args)


def test_pulse_amplitude_drop_is_measured_against_the_rest_baseline() -> None:
    result = _analyze()
    changes = result["change_vs_baseline_pct"]

    assert result["status"] == "MEASURED"
    assert changes["anticipation"]["value"] == pytest.approx(-5.0, abs=1.5)
    for name in ("call", "listen", "speak"):
        assert changes[name]["value"] == pytest.approx(-20.0, abs=1.5)
    assert changes["recovery"]["value"] == pytest.approx(-10.0, abs=1.5)
    quality = result["quality"]
    assert quality["beats_outlier"] == 0
    assert quality["beats"] == pytest.approx((C1 + 60_000 - T0) / RR_MS - quality["chunks_used"], abs=3)


def test_beats_in_large_movement_seconds_are_left_out() -> None:
    window = (C0 + 60_000, C0 + 70_000)
    acc = [[watch(t), 1.0, 0.4 if window[0] <= t < window[1] else 0.02] for t in range(C0, C1, 1000)]

    gated = _analyze(acc=acc)
    ungated = _analyze()

    excluded = gated["quality"]["beats_motion_excluded"]
    assert 12 <= excluded <= 16
    dropped_from_listen = ungated["pulse_amplitude"]["listen"]["beats"] - gated["pulse_amplitude"]["listen"]["beats"]
    assert 12 <= dropped_from_listen <= excluded
    assert gated["change_vs_baseline_pct"]["listen"]["value"] == pytest.approx(-20.0, abs=1.5)


def test_short_or_broken_chunks_are_skipped_and_gaps_are_fine() -> None:
    chunks = _chunks(gap_ms=5000)
    short = {"t0": watch(C0), "hz": HZ, "green": [50_000.0] * int(3 * HZ)}
    broken = {"t0": None, "hz": HZ, "green": [50_000.0] * 200}

    result = _analyze(ppg=[*chunks, short, broken])

    assert result["quality"]["chunks"] == len(chunks) + 2
    assert result["quality"]["chunks_used"] == len(chunks)
    assert result["change_vs_baseline_pct"]["call"]["value"] == pytest.approx(-20.0, abs=1.5)


def test_no_ppg_is_unavailable_everywhere() -> None:
    result = _analyze(ppg=[])

    assert result["status"] == "UNAVAILABLE"
    assert all(entry == {"status": "UNAVAILABLE"} for entry in result["pulse_amplitude"].values())
    assert all(entry == {"status": "UNAVAILABLE"} for entry in result["change_vs_baseline_pct"].values())


def test_too_few_beats_in_a_window_is_insufficient() -> None:
    result = _analyze(ppg=_chunks(start=T0 + 100_000, end=T0 + 110_000))

    assert result["pulse_amplitude"]["baseline"] == {"status": "INSUFFICIENT_DATA", "beats": 10}


def test_one_beat_per_cycle_and_a_giant_beat_is_dropped() -> None:
    steady = _chunks(start=0, end=30_000, chunk_sec=30, amp=lambda t: 100.0)[0]
    giant_cycle = (15_000, 15_000 + RR_MS)
    giant = _chunks(
        start=0, end=30_000, chunk_sec=30, amp=lambda t: 500.0 if giant_cycle[0] <= t < giant_cycle[1] else 100.0,
    )[0]
    for chunk in (steady, giant):
        chunk["t0"] = 0.0

    times, amplitudes, dropped = chunk_beats(steady, NO_CLOCK)
    _, kept, dropped_giant = chunk_beats(giant, NO_CLOCK)

    assert dropped == 0
    assert len(times) == pytest.approx(30_000 / RR_MS - 1, abs=1)
    assert np.all(np.abs(amplitudes / np.median(amplitudes) - 1) < 0.1)
    assert dropped_giant >= 1
    assert np.all(np.abs(kept / np.median(kept) - 1) < 0.2)


def test_ppg_chunks_are_validated() -> None:
    base = {"v": 1, "source": "samsung_sensor_sdk"}

    ok = BiosignalUpload.model_validate({**base, "ppg": [{"t0": 0, "hz": 25, "green": [1.0, 2.0]}]})
    assert ok.ppg[0].hz == 25

    with pytest.raises(ValidationError):
        BiosignalUpload.model_validate({**base, "ppg": [{"t0": 0, "hz": 5, "green": [1.0]}]})
    with pytest.raises(ValidationError):
        half = MAX_PPG_SAMPLES // 2 + 1
        BiosignalUpload.model_validate({**base, "ppg": [{"t0": 0, "hz": 25, "green": [1.0] * half}] * 2})


async def test_upload_with_ppg_stores_the_pulse_amplitude_analysis() -> None:
    async with _env() as env:
        await _seed(env)

        response = await env.client.post("/api/v1/biosignals/sess-1", json=_payload(ppg=_chunks()))

        assert response.status_code == 200
        ppg = response.json()["metrics"]["ppg"]
        assert ppg["status"] == "MEASURED"
        assert ppg["change_vs_baseline_pct"]["call"]["value"] == pytest.approx(-20.0, abs=1.5)


def _bumps(t: np.ndarray, rr: float, *, sigma: float, second: tuple[float, float] | None = None) -> np.ndarray:
    k = np.floor(t / rr)
    out = np.zeros_like(t)
    for d in (-1, 0, 1):
        start = (k + d) * rr + 150
        out += np.exp(-((t - start) ** 2) / (2 * sigma**2))
        if second:
            lag, height = second
            out += height * np.exp(-((t - start - lag) ** 2) / (2 * sigma**2))
    return out


def _raw_chunk(signal: np.ndarray) -> dict:
    return {"t0": 0.0, "hz": HZ, "green": signal.tolist()}


def test_a_double_humped_beat_counts_once() -> None:
    t = np.arange(30 * HZ) * 1000 / HZ
    signal = 50_000 + 100 * _bumps(t, RR_MS, sigma=45, second=(220, 0.9))
    signal += np.random.default_rng(2).normal(0, 1, t.size)

    times, _, _ = chunk_beats(_raw_chunk(signal), NO_CLOCK)

    assert len(times) == pytest.approx(30_000 / RR_MS - 1, abs=1)


def test_noise_bumps_in_a_slow_hearts_long_pause_are_not_beats() -> None:
    rr = 60_000 / 45
    t = np.arange(40 * HZ) * 1000 / HZ
    signal = 50_000 + 100 * _bumps(t, rr, sigma=40) + np.random.default_rng(3).normal(0, 6, t.size)

    times, _, _ = chunk_beats(_raw_chunk(signal), NO_CLOCK)

    assert len(times) == pytest.approx(40_000 / rr - 1, abs=1.5)


def test_beat_times_are_peak_times_moved_to_the_server_clock() -> None:
    steady = _chunks(start=0, end=30_000, chunk_sec=30, amp=lambda t: 100.0)[0]
    steady["t0"] = 0.0
    peak_offset_ms = np.arccos(0.404) / (2 * np.pi) * RR_MS

    times, _, _ = chunk_beats(steady, NO_CLOCK)
    shifted, _, _ = chunk_beats(steady, {"phone_to_server_ms": 1000.0, "watch_minus_phone_ms": -200.0})

    assert times[0] == pytest.approx(RR_MS + peak_offset_ms, abs=60)
    assert np.allclose(shifted - times, 1200.0)
