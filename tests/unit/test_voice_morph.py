import librosa
import numpy as np
import pytest

from app.services.voice_morph import DEFAULT_SEMITONES, morph, morph_pcm, pitch_shift

SR = 16_000


def _voiced(seconds: float = 1.5, f0: float = 180.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    harmonics = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 26))
    envelope = 1 + 0.35 * np.sin(2 * np.pi * 3 * t)
    return (harmonics * envelope / np.abs(harmonics).max() * 0.5).astype(np.float32)


def _median_f0(y: np.ndarray, sr: int = SR) -> float:
    f0, _, _ = librosa.pyin(y, fmin=60, fmax=700, sr=sr, frame_length=2048)
    voiced = f0[~np.isnan(f0)]
    return float(np.median(voiced)) if len(voiced) else float("nan")


def _formant_ratio(ref: np.ndarray, test: np.ndarray, sr: int = SR) -> float:
    grid = np.linspace(np.log(200.0), np.log(min(6000.0, sr / 2 - 1)), 600)
    dlog = grid[1] - grid[0]

    def envelope(y: np.ndarray) -> np.ndarray:
        spec = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        keep = spec.sum(axis=0) > np.percentile(spec.sum(axis=0), 60)
        cep = np.fft.irfft(np.log(spec[:, keep] + 1e-9), axis=0)
        cep[34:-34] = 0
        env = np.fft.rfft(cep, axis=0).real.mean(axis=1)
        on_grid = np.interp(np.exp(grid), librosa.fft_frequencies(sr=sr, n_fft=2048), env)
        return on_grid - np.polyval(np.polyfit(grid, on_grid, 1), grid)

    a, b = envelope(ref), envelope(test)
    lag = np.correlate(b, a, mode="full").argmax() - (len(a) - 1)
    return float(np.exp(lag * dlog))


def test_pitch_lands_on_the_requested_interval() -> None:
    y = _voiced()
    base = _median_f0(y)

    out = morph(y, SR, DEFAULT_SEMITONES)

    want = base * 2 ** (DEFAULT_SEMITONES / 12)
    error = 12 * np.log2(_median_f0(out) / want)
    assert abs(error) < 0.5, f"목표 {want:.0f}Hz 인데 {_median_f0(out):.0f}Hz ({error:+.2f}st)"


def test_psola_leaves_formants_alone() -> None:
    y = _voiced()

    shifted = pitch_shift(y, SR, 2 ** (DEFAULT_SEMITONES / 12))

    ratio = _formant_ratio(y, shifted)
    assert 0.93 < ratio < 1.07, f"피치만 옮겼는데 포먼트가 {ratio:.2f}배 따라갔다"


def test_formants_follow_by_the_cube_root_not_the_whole_shift() -> None:
    y = _voiced()
    alpha = 2 ** (DEFAULT_SEMITONES / 12)

    out = morph(y, SR, DEFAULT_SEMITONES)

    ratio = _formant_ratio(y, out)
    assert ratio < alpha * 0.85, f"포먼트가 α({alpha:.2f})만큼 끌려갔다 — 헬륨 {ratio:.2f}"
    assert ratio > 1.0, f"포먼트가 전혀 안 따라갔다 {ratio:.2f}"


def test_length_is_preserved() -> None:
    y = _voiced(seconds=2.0)
    assert len(morph(y, SR, DEFAULT_SEMITONES)) == len(y)


def test_output_is_not_the_input() -> None:
    y = _voiced()

    out = morph(y, SR, DEFAULT_SEMITONES)

    assert not np.allclose(out, y, atol=1e-3), "변조 결과가 원본과 같다"


def test_pcm_round_trip_keeps_format_and_length() -> None:
    y = _voiced(seconds=1.0)
    pcm = (y * 32767).astype(np.int16).tobytes()

    out = morph_pcm(pcm, SR)

    assert isinstance(out, bytes)
    assert len(out) == len(pcm)
    assert np.abs(np.frombuffer(out, dtype=np.int16)).max() > 0, "무음이 나왔다"


@pytest.mark.parametrize("pcm", [b"", b"\x00"])
def test_degenerate_pcm_does_not_explode(pcm: bytes) -> None:
    assert morph_pcm(pcm, SR) == b""


def test_very_short_audio_passes_through() -> None:
    y = _voiced(seconds=0.05)
    assert len(morph(y, SR, DEFAULT_SEMITONES)) == len(y)


class _FakeS3:
    """get_object 는 성공하지만 내용이 WAV 가 아닌 경우."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def get_object(self, **_kw: object) -> dict:
        return {"Body": _Body(self._body)}


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


def test_corrupt_object_returns_none_like_the_rest_of_the_class() -> None:
    """S3 는 응답했는데 WAV 가 아니면? 이 클래스의 다른 메서드는 전부 None 을 준다.

    여기만 원시 예외를 던지면 ensure_morphed 의 경고 로그를 건너뛰고
    워커의 광범위한 except 로 떨어져 원인 파악이 어려워진다.
    """
    from types import SimpleNamespace

    from app.services.recording_storage import RecordingStorageService

    settings = SimpleNamespace(
        s3_bucket="b", aws_access_key=None, aws_secret_key=None, aws_region="ap-northeast-2"
    )
    storage = RecordingStorageService(settings, client=_FakeS3(b"not a wav at all"))

    assert storage.download_pcm("recordings/x.wav") is None
