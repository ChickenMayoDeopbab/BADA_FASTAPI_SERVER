import warnings
import wave
from pathlib import Path
from types import SimpleNamespace

import librosa
import numpy as np
import pytest

from app.services import librosa_port, tremor
from app.services.librosa_port import rms, yin
from app.services.tremor import _YIN_FRAME_LENGTH, TremorAnalyzer, TremorConfig

ROOT = Path(__file__).resolve().parents[2]
_SR = 16000
_HOP = 160
_YIN = {"fmin": 70.0, "fmax": 400.0, "sr": _SR, "frame_length": _YIN_FRAME_LENGTH, "hop_length": _HOP}
_RMS = {"frame_length": 4 * _HOP, "hop_length": _HOP}


def _pcm(codes) -> np.ndarray:
    return np.asarray(codes).astype(np.int16).astype(np.float32) / np.iinfo(np.int16).max


def _tone(freq: float) -> np.ndarray:
    t = np.arange(_SR) / _SR
    return _pcm(np.round(0.5 * 32767 * np.sin(2 * np.pi * freq * t)))


def _speechy_codes(seconds: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * _SR)) / _SR
    f0 = 150 + 20 * np.sin(2 * np.pi * 0.5 * t) + 3 * np.sin(2 * np.pi * 6 * t)
    phase = 2 * np.pi * np.cumsum(f0) / _SR
    gate = (np.sin(2 * np.pi * 0.3 * t) > -0.3).astype(float)
    x = gate * (0.5 * np.sin(phase) + 0.2 * np.sin(2 * phase)) + 0.01 * rng.standard_normal(len(t))
    return np.round(np.clip(x, -1, 1) * 32000).astype(np.int16)


def _power_staircase() -> np.ndarray | None:
    x = _pcm(np.arange(-32768, 32768))
    differs = np.flatnonzero(np.power(x, 2, dtype=np.float32) != x * x)
    if differs.size == 0:
        return None
    picked = np.random.default_rng(0).choice(differs, size=min(40, differs.size), replace=False)
    return np.repeat(x[picked], 800)


def _vendor_wav() -> np.ndarray:
    # 48kHz 원본 샘플을 그대로 16kHz 로 넣는다(901프레임, 길이가 hop 의 배수가 아님)
    with wave.open(str(ROOT / "vendor/tremor3.05/sounds/test_F3_I10_envel_A5_I20.wav")) as w:
        return _pcm(np.frombuffer(w.readframes(w.getnframes()), dtype="<i2"))


def _recipes() -> dict[str, np.ndarray]:
    rng = np.random.default_rng
    recipes = {
        "tone420": _tone(420.0),
        "tone66": _tone(66.0),
        "tone60": _tone(60.0),
        "silence": np.zeros(_SR, np.float32),
        "dither05": _pcm(np.round(rng(1).normal(0, 0.5, _SR))),
        "dither15": _pcm(np.round(rng(1).normal(0, 1.5, _SR))),
        "dc8000": _pcm(np.round(8000 + rng(3).normal(0, 1.0, _SR))),
        "full_scale": _pcm(np.full(_SR, 32767)),
        "speechy": _pcm(_speechy_codes(1.0)),
        "vendor_wav": _vendor_wav(),
        "five_seconds": _pcm(_speechy_codes(5.0, seed=2)),
    }
    staircase = _power_staircase()
    if staircase is not None:
        recipes["power_staircase"] = staircase
    for n in (0, 1, 159, 160, 161, 2047, 2048, 2049):
        recipes[f"len{n}"] = _pcm(_speechy_codes(1.0, seed=3)[:n])
    return recipes


RECIPES = _recipes()


def _same_bits(got: np.ndarray, want: np.ndarray) -> None:
    assert (got.dtype, got.shape) == (want.dtype, want.shape)
    assert got.tobytes() == want.tobytes()


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_yin_matches_librosa_bit_for_bit(name: str) -> None:
    y = RECIPES[name]
    _same_bits(yin(y, **_YIN), librosa.yin(y, **_YIN))


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_rms_matches_librosa_bit_for_bit(name: str) -> None:
    y = RECIPES[name]
    _same_bits(rms(y, **_RMS), librosa.feature.rms(y=y, **_RMS)[0])


@pytest.mark.parametrize("chunk_sec", [5.0, 0.5])
@pytest.mark.parametrize("name", sorted(RECIPES))
def test_tremor_chunked_yin_matches_one_librosa_call(name: str, chunk_sec: float) -> None:
    y = RECIPES[name]
    analyzer = TremorAnalyzer(TremorConfig(f0_chunk_sec=chunk_sec))
    _same_bits(analyzer._yin(y, _HOP), librosa.yin(y, **_YIN))


@pytest.mark.parametrize("n", [2048, 2049, 2208])
def test_uncentered_yin_matches_librosa_bit_for_bit(n: int) -> None:
    y = RECIPES["vendor_wav"][:n]
    _same_bits(yin(y, center=False, **_YIN), librosa.yin(y, center=False, **_YIN))


def test_uncentered_yin_rejects_input_shorter_than_a_frame() -> None:
    with pytest.raises(ValueError):
        yin(RECIPES["len2047"], center=False, **_YIN)


@pytest.mark.parametrize("bad", [np.zeros(_SR, np.int16), np.full(_SR, np.nan, np.float32)])
def test_yin_rejects_what_librosa_rejects(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        yin(bad, **_YIN)
    with pytest.raises(librosa.util.exceptions.ParameterError):
        librosa.yin(bad, **_YIN)


def test_power_and_square_stay_where_librosa_has_them(monkeypatch) -> None:
    # np.power 결과만 살짝 흔든다. square 와 power 를 맞바꾸면 CPU 와 상관없이 librosa 와 어긋난다
    real_power = np.power

    def _nudged_power(*args, **kwargs):
        out = real_power(*args, **kwargs)
        return out * out.dtype.type(1 + 2**-8)

    monkeypatch.setattr(np, "power", _nudged_power)
    y = RECIPES["speechy"]

    _same_bits(yin(y, **_YIN), librosa.yin(y, **_YIN))
    _same_bits(rms(y, **_RMS), librosa.feature.rms(y=y, **_RMS)[0])


def _injected_cmnd(dtype) -> np.ndarray:
    lags = 190
    columns = []
    c = np.ones(lags)
    c[50], c[120], c[121:] = 0.1, 0.05, 0.2
    columns.append(c)
    c = np.ones(lags)
    c[30], c[31], c[32] = np.inf, 0.05, 1.0
    columns.append(c)
    c = np.ones(lags)
    c[0] = c[1] = 0.0
    c[-2], c[-1] = 0.08, 0.07
    columns.append(c)
    c = np.ones(lags)
    c[60] = c[61] = 0.05
    columns.append(c)
    rng = np.random.default_rng(42)
    columns += [rng.random(lags) for _ in range(20)]
    columns += list((np.round(rng.random((lags, 20)) * 4) / 8).T)
    return np.stack(columns, axis=1).astype(dtype)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_period_choice_matches_librosa_on_injected_cmnd(monkeypatch, dtype) -> None:
    # 문턱과 같은 골, 고른 지연의 |b|>=|a|, 끝 원소 규칙, 동률 골은 실제 음성으로는 닿지 않는다
    cmnd = _injected_cmnd(dtype)
    fake = lambda *a, **k: cmnd.copy()  # noqa: E731
    monkeypatch.setattr("librosa.core.pitch._cumulative_mean_normalized_difference", fake)
    monkeypatch.setattr(librosa_port, "_cumulative_mean_normalized_difference", fake)
    y = np.zeros(_YIN_FRAME_LENGTH + _HOP * (cmnd.shape[1] - 1), np.float32)

    want = librosa.yin(y, center=False, **_YIN)
    _same_bits(yin(y, center=False, **_YIN), want)
    assert (want[1], want[2]) == (_SR / 71, _SR / 229)


@pytest.mark.parametrize("name", ["silence", "dither05", "len0"])
def test_yin_raises_no_runtime_warning(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        yin(RECIPES[name], **_YIN)


def test_analyze_calls_the_port_with_the_arguments_librosa_got(monkeypatch) -> None:
    assert tremor.librosa_port is librosa_port
    codes = np.concatenate([_speechy_codes(6.0, seed=1), np.zeros(_SR // 2, np.int16)])
    pcm = codes.tobytes()
    analyzer = TremorAnalyzer()
    ported_f0, ported_active = analyzer._extract_f0_cents(analyzer._pcm_to_float(pcm))
    ported = analyzer.analyze(pcm)

    calls: list[tuple[str, dict]] = []

    def _librosa_yin(y, **kwargs):
        calls.append(("yin", kwargs))
        return librosa.yin(y, **kwargs)

    def _librosa_rms(y, **kwargs):
        calls.append(("rms", kwargs))
        return librosa.feature.rms(y=y, **kwargs)[0]

    monkeypatch.setattr(tremor, "librosa_port", SimpleNamespace(yin=_librosa_yin, rms=_librosa_rms))
    f0, active = analyzer._extract_f0_cents(analyzer._pcm_to_float(pcm))

    assert ("yin", {"center": False, **_YIN}) in calls
    assert ("rms", _RMS) in calls
    _same_bits(ported_f0, f0)
    _same_bits(ported_active, active)
    assert analyzer.analyze(pcm) == ported
    assert ported.shake_count > 0
