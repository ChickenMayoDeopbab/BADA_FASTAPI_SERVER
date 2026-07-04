import numpy as np

from app.services.tremor import TremorAnalyzer, TremorResult


def _pcm_bytes(n_samples: int, extra_byte: bool = False) -> bytes:
    samples = (np.arange(n_samples, dtype=np.int16) % 1000).tobytes()
    return samples + b"\x7f" if extra_byte else samples


def test_pcm_to_float_odd_length_drops_last_byte_only() -> None:
    analyzer = TremorAnalyzer()
    even = analyzer._pcm_to_float(_pcm_bytes(8000))
    odd = analyzer._pcm_to_float(_pcm_bytes(8000, extra_byte=True))
    assert len(odd) == 8000
    np.testing.assert_array_equal(odd, even)


def test_pcm_to_float_even_length_passthrough() -> None:
    analyzer = TremorAnalyzer()
    out = analyzer._pcm_to_float(_pcm_bytes(4000))
    assert len(out) == 4000


def test_analyze_odd_length_returns_result_without_error() -> None:
    analyzer = TremorAnalyzer()
    pcm = np.zeros(8000, dtype=np.int16).tobytes() + b"\x00"
    result = analyzer.analyze(pcm)
    assert isinstance(result, TremorResult)
    assert result.shake_count == 0
