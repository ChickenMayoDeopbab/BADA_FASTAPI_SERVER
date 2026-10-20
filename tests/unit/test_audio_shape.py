import importlib.util
from pathlib import Path

import numpy as np
import pytest

from app.core import audio_shape
from app.core.audio_shape import RequestMeter

_ROOT = Path(__file__).resolve().parents[2]


def _post():
    spec = importlib.util.spec_from_file_location("tts_bench_post_shape", _ROOT / "scripts" / "tts_bench" / "post.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _random_pcm(rng: np.random.Generator) -> np.ndarray:
    parts = [np.zeros(int(rng.integers(0, 8000)), dtype=np.float32)]
    for _ in range(int(rng.integers(1, 5))):
        n = int(rng.integers(800, 12000))
        x = rng.normal(0, rng.uniform(0.02, 0.4), n).astype(np.float32)
        fade = min(n, int(rng.choice([0, 16, 64, 160, 400, 1200])))
        if fade:
            x[-fade:] *= np.geomspace(1.0, 1e-4, fade).astype(np.float32)
        parts.append(x)
        parts.append(np.zeros(int(rng.integers(0, 6000)), dtype=np.float32))
    return np.clip(np.concatenate(parts), -1, 1)


def _meter(pcm: bytes, rng: np.random.Generator) -> dict:
    meter = RequestMeter()
    pos = 0
    while pos < len(pcm):
        step = int(rng.integers(1, 6000))
        meter.push(pcm[pos : pos + step])
        pos += step
    return meter.result()


def test_matches_bench_oracle_on_random_requests_and_chunking() -> None:
    post = _post()
    rng = np.random.default_rng(11)
    for _ in range(500):
        w = _random_pcm(rng)
        pcm = (w * 32767).astype("<i2").tobytes()
        ref = post.speech_ends(np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768)
        got = _meter(pcm, rng)
        if not ref:
            assert got["decay_ms"] is None
            continue
        assert got["decay_ms"] == pytest.approx(ref[-1][1])
        assert got["cut"] == (ref[-1][1] <= 10.0)
        assert got["inner_ends"] == len(ref) - 1
        assert got["inner_cuts"] == sum(d <= 10.0 for _, d, _ in ref[:-1])


def _trail_reference(pcm: bytes) -> float:
    x = np.frombuffer(pcm[: len(pcm) // 64 * 64], dtype="<i2").astype(np.int32)
    env = np.abs(x).reshape(-1, 32).max(1).astype(np.float64)
    db = 20 * np.log10(env / env.max() + 1e-12)
    sound = np.nonzero(db >= -45)[0]
    return (len(env) - (int(sound[-1]) + 1)) * 2.0


def test_trail_is_time_after_last_block_above_minus_45db() -> None:
    rng = np.random.default_rng(3)
    for _ in range(100):
        pcm = (_random_pcm(rng) * 32767).astype("<i2").tobytes()
        got = _meter(pcm, rng)
        if got["decay_ms"] is None:
            continue
        assert got["trail_ms"] == pytest.approx(_trail_reference(pcm))


def _blocks(levels_db: list[float]) -> bytes:
    out = []
    for i, level in enumerate(levels_db):
        amp = 0 if level is None else int(30000 * 10 ** (level / 20))
        block = np.full(32, amp if i % 2 else -amp, dtype="<i2")
        out.append(block)
    return np.concatenate(out).tobytes()


def _shape(mid_blocks: int) -> dict:
    meter = RequestMeter()
    meter.push(_blocks([0.0] * 50 + [-30.0] * mid_blocks + [-60.0] * 100))
    return meter.result()


def test_decay_boundaries() -> None:
    assert _shape(5)["decay_ms"] == 10.0 and _shape(5)["cut"] is True
    assert _shape(6)["decay_ms"] == 12.0 and _shape(6)["cut"] is False
    assert _shape(2)["hard"] is True and _shape(3)["hard"] is False


def test_silence_has_no_end_and_speech_to_the_end_decays_zero() -> None:
    silent = RequestMeter()
    silent.push(b"\x00" * 6400)
    assert silent.result()["decay_ms"] is None
    talking = RequestMeter()
    talking.push(_blocks([0.0] * 100))
    assert talking.result()["decay_ms"] == 0.0 and talking.result()["trail_ms"] == 0.0


def test_lead_and_inner_pause() -> None:
    meter = RequestMeter()
    meter.push(_blocks([-80.0] * 40 + [0.0] * 50 + [-80.0] * 200 + [0.0] * 50 + [-80.0] * 10))
    got = meter.result()
    assert got["lead_ms"] == 80.0
    assert got["pause_max_ms"] == 400.0
    assert got["inner_ends"] == 1


def test_odd_chunks_are_carried() -> None:
    pcm = _blocks([0.0] * 50 + [-30.0] * 7 + [-60.0] * 100)
    whole, split = RequestMeter(), RequestMeter()
    whole.push(pcm)
    for i in range(0, len(pcm), 7):
        split.push(pcm[i : i + 7])
    assert split.result() == whole.result() and split.samples == whole.samples


def test_failures_are_swallowed(monkeypatch) -> None:
    meter = RequestMeter()
    meter.push(12345)
    assert meter.broken and meter.result() == {"broken": True}

    other = RequestMeter()
    other.push(_blocks([0.0] * 10))

    def _boom(self):
        raise RuntimeError("boom")

    monkeypatch.setattr(audio_shape.RequestMeter, "_result", _boom)
    assert other.result() == {"broken": True}
