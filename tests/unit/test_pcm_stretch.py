import random

import numpy as np
import pytest

from app.services import pcm_stretch
from app.services.pcm_stretch import PcmStretcher, clamp_speed

SR = 16000
TOL = SR * pcm_stretch._TOLERANCE_MS // 1000
WORKER_CHUNKS = [4480] + [5120] * 400


def _sine(freq: float, seconds: float, amp: int = 8000) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype("<i2")


def _harmonic(f0: float, seconds: float) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    x = (
        np.sin(2 * np.pi * f0 * t)
        + 0.6 * np.sin(4 * np.pi * f0 * t + 1.0)
        + 0.4 * np.sin(6 * np.pi * f0 * t + 2.0)
    )
    return (6000 * x / np.max(np.abs(x))).astype("<i2")


def _speechlike(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    f0 = 180 + 40 * np.sin(2 * np.pi * 1.5 * np.arange(n) / SR)
    phase = 2 * np.cumsum(np.pi * f0 / SR)
    env = (np.sin(2 * np.pi * 3 * np.arange(n) / SR) > -0.3).astype(float)
    x = env * (6000 * np.sin(phase) + 2000 * np.sin(2 * phase)) + rng.normal(0, 50, n)
    return np.clip(x, -32768, 32767).astype("<i2")


def _noise(samples: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 3000, samples).astype("<i2")


def _run(stretcher: PcmStretcher, pcm: bytes, sizes: list[int]) -> bytes:
    out, i, k = [], 0, 0
    while i < len(pcm):
        step = sizes[k % len(sizes)]
        out.append(stretcher.push(pcm[i : i + step]))
        i += step
        k += 1
    out.append(stretcher.flush())
    return b"".join(out)


def _samples(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, "<i2").astype(float)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x.astype(float) ** 2)))


def _expected_len(n: int, speed: float, warmup_ms: int = 0) -> int:
    warm = min(n, round(SR * warmup_ms / 1000 / 240) * 240)
    return round(warm + (n - warm) / speed) + TOL


def test_speed_one_passes_bytes_through() -> None:
    st = PcmStretcher(1.0)
    assert st.push(b"\x01") == b"\x01"
    assert st.push(b"\x02\x03\x04") == b"\x02\x03\x04"
    assert st.flush() == b""


@pytest.mark.parametrize("speed", [0.9, 1.1, 1.15, 1.25])
def test_output_length_matches_speed(speed: float) -> None:
    pcm = _speechlike(3.0).tobytes()
    out = _run(PcmStretcher(speed), pcm, [5120])
    assert len(out) == 2 * _expected_len(len(pcm) // 2, speed)


@pytest.mark.parametrize("warmup_ms", [0, 180])
def test_length_matches_speed_for_many_lengths(warmup_ms: int) -> None:
    rng = np.random.default_rng(0)
    r = random.Random(0)
    for speed in (0.8, 1.05, 1.1, 1.15, 1.25):
        for n in [int(v) for v in rng.integers(1, 40000, 25)] + [1, 239, 240, 481, 2880, 2881]:
            pcm = _noise(n, n).tobytes()
            sizes = [r.choice([1, 641, 4480, 5120]) for _ in range(50)]
            st = PcmStretcher(speed, warmup_ms=warmup_ms)
            assert len(_run(st, pcm, sizes)) == 2 * _expected_len(n, st.speed, warmup_ms), (speed, n)


@pytest.mark.parametrize("speed", [1.1, 1.15])
@pytest.mark.parametrize("warmup_ms", [0, 180])
def test_chunked_equals_one_shot(speed: float, warmup_ms: int) -> None:
    pcm = _speechlike(2.5, seed=1).tobytes()
    one = PcmStretcher(speed, warmup_ms=warmup_ms)
    whole = one.push(pcm) + one.flush()
    rnd = random.Random(7)
    sizes = [rnd.choice([1, 3, 333, 641, 1279, 5120]) for _ in range(400)]
    assert _run(PcmStretcher(speed, warmup_ms=warmup_ms), pcm, sizes) == whole


def test_every_output_is_whole_samples() -> None:
    st = PcmStretcher(1.15)
    pcm = _speechlike(1.0).tobytes()
    for i in range(0, len(pcm), 641):
        assert len(st.push(pcm[i : i + 641])) % 2 == 0
    assert len(st.flush()) % 2 == 0


@pytest.mark.parametrize("speed", [1.1, 1.15])
def test_pitch_and_level_are_kept(speed: float) -> None:
    src = _sine(220.0, 2.0)
    y = _samples(_run(PcmStretcher(speed), src.tobytes(), [5120]))
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    assert abs(np.argmax(spec) * SR / len(y) - 220.0) / 220.0 < 0.02, "리샘플이면 220×speed 로 오른다"
    mid = y[len(y) // 4 : 3 * len(y) // 4]
    assert abs(_rms(mid) / _rms(src) - 1) < 0.05


@pytest.mark.parametrize("speed", [0.9, 1.1, 1.15])
@pytest.mark.parametrize("f0", [80.0, 100.0, 220.0])
def test_tone_stays_periodic_without_level_dips(speed: float, f0: float) -> None:
    src = _harmonic(f0, 2.0) if f0 < 200 else _sine(f0, 2.0)
    y = _samples(_run(PcmStretcher(speed), src.tobytes(), [5120]))
    mid = y[len(y) // 8 : -len(y) // 8]
    period = round(SR / f0)
    a, b = mid[:-period], mid[period:]
    assert np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)) > 0.99, "이음매가 주기를 깨면 안 된다"
    w = 3 * period
    local = [_rms(mid[i : i + w]) for i in range(0, len(mid) - w, period)]
    assert min(local) > 0.9 * _rms(src), "역위상 이음매는 국소 레벨을 깎는다"


@pytest.mark.parametrize("value", [1001, -1001])
def test_constant_input_is_reproduced(value: int) -> None:
    y = _samples(_run(PcmStretcher(1.15), np.full(SR, value, dtype="<i2").tobytes(), [5120]))
    assert np.all(y[480:-1500] == value)


@pytest.mark.parametrize("speed", [1.1, 1.15])
def test_output_starts_with_input(speed: float) -> None:
    src = _noise(SR, 1)
    y = _samples(_run(PcmStretcher(speed), src.tobytes(), [5120]))
    assert np.array_equal(y[:240], src[:240].astype(float))


@pytest.mark.parametrize("speed", [1.1, 1.15])
def test_sound_starts_and_ends_on_time(speed: float) -> None:
    gap = np.zeros(SR // 2, dtype="<i2")
    src = np.concatenate([gap, _sine(220.0, 1.0), gap])
    out = np.frombuffer(_run(PcmStretcher(speed), src.tobytes(), [5120]), "<i2")
    loud = np.flatnonzero(np.abs(out) > 1000)
    tol = SR // 100
    assert abs(loud[0] - round(SR * 0.5 / speed)) <= tol
    assert abs(loud[-1] - round(SR * 1.5 / speed)) <= tol


@pytest.mark.parametrize("speed", [1.1, 1.15, 1.25])
def test_sentence_end_decay_is_not_cut(speed: float) -> None:
    t = np.arange(SR) / SR
    x = 6000 * np.sin(2 * np.pi * 200 * t) + 2000 * np.sin(4 * np.pi * 200 * t)
    x[-96:] *= np.linspace(1, 0, 96)
    y = _samples(_run(PcmStretcher(speed), x.astype("<i2").tobytes(), WORKER_CHUNKS))
    assert _rms(y[-40:]) < 50 and abs(y[-1]) < 200, "끝 감쇠가 잘리면 소리 도중에 끝난다"


@pytest.mark.parametrize("speed", [0.9, 1.15])
@pytest.mark.parametrize("warmup_ms", [0, 180])
def test_holdback_is_bounded(speed: float, warmup_ms: int) -> None:
    pcm = _noise(3 * SR, 2).tobytes()
    st = PcmStretcher(speed, warmup_ms=warmup_ms)
    n_in = n_out = pos = 0
    for step in WORKER_CHUNKS[:20]:
        chunk = pcm[pos : pos + step]
        pos += step
        n_out += len(st.push(chunk)) // 2
        n_in += len(chunk) // 2
        assert _expected_len(n_in, speed, warmup_ms) - TOL - n_out < 50 * SR // 1000, "붙잡는 오디오는 50ms 미만"


def test_warmup_keeps_start_at_original_speed() -> None:
    src = _noise(SR, 3)
    y = _samples(_run(PcmStretcher(1.15, warmup_ms=180), src.tobytes(), WORKER_CHUNKS))
    assert np.array_equal(y[:2880], src[:2880].astype(float))


@pytest.mark.parametrize("speed", [1.1, 1.15, 1.25])
def test_warmup_fills_app_prebuffer_in_two_worker_chunks(speed: float) -> None:
    pcm = _speechlike(2.0, seed=4).tobytes()
    prebuffer = SR // 4
    for warmup_ms, enough in ((180, True), (0, False)):
        st = PcmStretcher(speed, warmup_ms=warmup_ms)
        out = len(st.push(pcm[:4480])) + len(st.push(pcm[4480:9600]))
        assert (out // 2 >= prebuffer) is enough, warmup_ms


def test_empty_input() -> None:
    st = PcmStretcher(1.1)
    assert st.push(b"") == b""
    assert st.flush() == b""


def test_lone_odd_byte_waits_for_its_pair() -> None:
    st = PcmStretcher(1.1)
    pcm = _speechlike(0.5).tobytes()
    out = st.push(pcm[:1]) + st.push(pcm[1:]) + st.flush()
    ref = PcmStretcher(1.1)
    assert out == ref.push(pcm) + ref.flush()


def test_speed_is_clamped() -> None:
    assert clamp_speed(0.5) == 0.8
    assert clamp_speed(3.0) == 1.25
    assert clamp_speed(1.1) == 1.1
    assert PcmStretcher(2.0).speed == 1.25


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -1.0])
def test_nonsense_speed_falls_back_to_one(bad: float) -> None:
    assert clamp_speed(bad) == 1.0
    assert PcmStretcher(bad).bypass
