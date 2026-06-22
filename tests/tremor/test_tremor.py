"""TremorAnalyzer 단위 테스트 — 잘한 구간(good_candidates) 산출 로직."""
from __future__ import annotations

import numpy as np

from app.services.tremor import TremorAnalyzer, TremorConfig
from tests.tremor.manual_tremor_check import synth_pcm

# ----------------------------- _subtract (순수 구간 빼기) -----------------------------

def test_subtract_hole_in_middle():
    assert TremorAnalyzer._subtract([(0, 11)], [(3, 5), (8, 9)]) == [(0, 3), (5, 8), (9, 11)]


def test_subtract_no_overlap():
    assert TremorAnalyzer._subtract([(0, 5)], [(10, 12)]) == [(0, 5)]


def test_subtract_hole_covers_whole():
    assert TremorAnalyzer._subtract([(2, 5)], [(0, 10)]) == []


def test_subtract_edges():
    assert TremorAnalyzer._subtract([(0, 10)], [(0, 3)]) == [(3, 10)]
    assert TremorAnalyzer._subtract([(0, 10)], [(7, 10)]) == [(0, 7)]


def test_subtract_no_holes():
    assert TremorAnalyzer._subtract([(0, 5)], []) == [(0, 5)]


# ----------------------------- _speaking_spans (발화 구간 추출) -----------------------------

def _analyzer(min_silence_sec: float = 1.0) -> TremorAnalyzer:
    # contour_fs=100 → 1프레임 = 0.01s, bridge = min_silence_sec * 100 프레임
    return TremorAnalyzer(TremorConfig(min_silence_sec=min_silence_sec))


def _voiced(*runs) -> np.ndarray:
    """(True/False, 프레임수) 들을 이어붙여 bool 배열 생성."""
    return np.concatenate([np.full(n, val, dtype=bool) for val, n in runs])


def test_speaking_spans_continuous():
    spans = _analyzer()._speaking_spans(_voiced((True, 300)))
    assert spans == [(0.0, 3.0)]


def test_speaking_spans_short_gap_bridged():
    # 0.5s 무음(50프레임) < min_silence(1.0s) → 이어붙임
    spans = _analyzer()._speaking_spans(_voiced((True, 100), (False, 50), (True, 100)))
    assert spans == [(0.0, 2.5)]


def test_speaking_spans_long_gap_splits():
    # 1.5s 무음(150프레임) >= min_silence(1.0s) → 분할
    spans = _analyzer()._speaking_spans(_voiced((True, 100), (False, 150), (True, 100)))
    assert spans == [(0.0, 1.0), (2.5, 3.5)]


def test_speaking_spans_leading_silence_excluded():
    spans = _analyzer()._speaking_spans(_voiced((False, 50), (True, 100)))
    assert spans == [(0.5, 1.5)]


# ----------------------------- analyze() 합성 신호 통합 -----------------------------

def _overlap(a, b) -> bool:
    return min(a[1], b[1]) > max(a[0], b[0])


def test_analyze_flat_tone_no_tremor_one_good_span():
    res = TremorAnalyzer().analyze(synth_pcm([], seconds=4.0))
    assert res.shake_count == 0
    assert len(res.good_candidates) == 1
    s, e = res.good_candidates[0]
    assert e - s > 3.5  # 4초 거의 전부가 '잘한 구간'


def test_analyze_tremor_bursts_detected_and_excluded():
    bursts = [(1.0, 3.0), (4.5, 6.5), (8.0, 10.0)]
    res = TremorAnalyzer().analyze(synth_pcm(bursts, seconds=11.0))
    assert res.shake_count >= 1
    # 핵심 불변식: 잘한 구간은 떨림 에피소드와 절대 겹치지 않는다
    tremor = [(ep[0], ep[1]) for ep in res.episodes]
    for g in res.good_candidates:
        assert not any(_overlap(g, t) for t in tremor)
