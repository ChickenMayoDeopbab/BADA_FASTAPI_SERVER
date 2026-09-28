import librosa
import numpy as np

from app.services.tremor import _YIN_FRAME_LENGTH, TremorAnalyzer, TremorConfig

_SR = 16000
_HOP = 160


def _speechy(seconds: float, seed: int = 0) -> np.ndarray:
    """떨림(6Hz)과 끊김이 섞인 모음 비슷한 신호."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * _SR)) / _SR
    f0 = 150 + 20 * np.sin(2 * np.pi * 0.5 * t) + 3 * np.sin(2 * np.pi * 6 * t)
    phase = 2 * np.pi * np.cumsum(f0) / _SR
    gate = (np.sin(2 * np.pi * 0.3 * t) > -0.3).astype(float)
    x = gate * (0.5 * np.sin(phase) + 0.2 * np.sin(2 * phase)) + 0.01 * rng.standard_normal(len(t))
    return np.clip(x, -1, 1).astype(np.float32)


def _whole_yin(y: np.ndarray) -> np.ndarray:
    return librosa.yin(y, fmin=70.0, fmax=400.0, sr=_SR, frame_length=_YIN_FRAME_LENGTH, hop_length=_HOP)


def test_chunked_yin_matches_single_call() -> None:
    # 묶음 경계가 프레임 수와 딱 맞지 않게 길이를 잡는다.
    y = _speechy(3.37)
    analyzer = TremorAnalyzer(TremorConfig(f0_chunk_sec=0.5))
    chunked = analyzer._yin(y, _HOP)
    np.testing.assert_array_equal(chunked, _whole_yin(y))


def test_short_signal_uses_single_call() -> None:
    y = _speechy(0.8)
    analyzer = TremorAnalyzer(TremorConfig(f0_chunk_sec=5.0))
    np.testing.assert_array_equal(analyzer._yin(y, _HOP), _whole_yin(y))


def test_analyze_result_unchanged_by_chunking() -> None:
    pcm = (_speechy(12.3, seed=1) * 32000).astype(np.int16).tobytes()
    small = TremorAnalyzer(TremorConfig(f0_chunk_sec=1.0)).analyze(pcm)
    whole = TremorAnalyzer(TremorConfig(f0_chunk_sec=1000.0)).analyze(pcm)
    assert small.shake_count == whole.shake_count
    assert small.episodes == whole.episodes
    assert small.good_candidates == whole.good_candidates
    assert small.sustained_spans == whole.sustained_spans
    assert small.voiced_spans == whole.voiced_spans
