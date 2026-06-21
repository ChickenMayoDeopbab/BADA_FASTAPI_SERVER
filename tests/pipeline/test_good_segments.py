"""VoicePipeline 잘한 구간 선택 로직 테스트 (_intersect, _pick_good_segments)."""
from __future__ import annotations

from app.services.pipeline import VoicePipeline


def _pipeline_with_turns(turns) -> VoicePipeline:
    """무거운 __init__ 없이 선택 로직만 테스트하기 위한 최소 객체."""
    obj = VoicePipeline.__new__(VoicePipeline)
    obj._user_turn_intervals = turns
    return obj


# ----------------------------- _intersect -----------------------------

def test_intersect_clips_to_turns():
    out = VoicePipeline._intersect([(0, 100)], [(0, 20), (40, 60), (80, 100)])
    assert out == [(0, 20), (40, 60), (80, 100)]


def test_intersect_no_overlap():
    assert VoicePipeline._intersect([(0, 5)], [(10, 20)]) == []


# ----------------------------- _pick_good_segments -----------------------------

def test_pick_excludes_ai_turns():
    # AI 턴(20~40, 60~80)을 가로지르는 후보 → 사용자 턴만 남아야
    p = _pipeline_with_turns([(0, 20), (40, 60), (80, 100)])
    assert p._pick_good_segments([(0, 100)]) == [
        {"start": 0, "end": 20},
        {"start": 40, "end": 60},
        {"start": 80, "end": 100},
    ]


def test_pick_top3_longest_chronological():
    p = _pipeline_with_turns([(0, 90)])
    cands = [(5, 12), (20, 23), (35, 55), (60, 64), (70, 85)]
    # 가장 긴 3개 = (35,55)=20s, (70,85)=15s, (5,12)=7s → 시간순 정렬
    assert p._pick_good_segments(cands) == [
        {"start": 5, "end": 12},
        {"start": 35, "end": 55},
        {"start": 70, "end": 85},
    ]


def test_pick_drops_short_fragments():
    # 0.5s 조각은 min_good(1.0s) 미만 → 탈락
    p = _pipeline_with_turns([(0, 90)])
    assert p._pick_good_segments([(0, 0.5), (10, 13)]) == [{"start": 10, "end": 13}]


def test_pick_empty_when_no_overlap():
    p = _pipeline_with_turns([(50, 60)])
    assert p._pick_good_segments([(0, 10)]) == []
