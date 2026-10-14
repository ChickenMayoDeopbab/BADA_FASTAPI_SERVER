"""짚을 구간 고르기 — 이유끼리 순서 없이 종합해서 고른다."""

import pytest

from app.services.llm import LLMClient
from app.services.pipeline import VoicePipeline


def _pipeline(turns) -> VoicePipeline:
    p = VoicePipeline.__new__(VoicePipeline)
    p._session_id = "test-session"
    p._user_turn_intervals = turns
    p._user_turn_texts = [f"발화{i}" for i in range(len(turns))]
    p._session = {"type": "SCENARIO", "scenario": {"title": "식당 예약"}}
    return p


_FIVE_TURNS = [(0.0, 4.0), (5.0, 9.0), (10.0, 14.0), (15.0, 19.0), (20.0, 24.0)]


def test_a_later_reason_is_not_crowded_out_by_earlier_turns() -> None:
    """앞 턴부터 채우면 1~3번 멈춤에 자리가 다 차서 5번의 작은 목소리를 놓친다."""
    p = _pipeline(_FIVE_TURNS)

    segs = p._pick_segments([], {}, {1: [1.6], 2: [1.7], 3: [1.8]}, {5: 6.5})

    assert [(s["turn"], s["reason"]) for s in segs] == [(2, "pause"), (3, "pause"), (5, "quiet")]


def test_each_reason_gets_a_seat_before_any_repeats() -> None:
    p = _pipeline(_FIVE_TURNS)

    segs = p._pick_segments([], {1: 6.0}, {2: [1.6]}, {}, {3: 1.4}, {5: 1.3})

    assert sorted(s["reason"] for s in segs) == ["late", "pause", "voice"]


def test_turn_with_more_reasons_wins_when_all_are_new() -> None:
    p = _pipeline(_FIVE_TURNS)

    (seg,) = [s for s in p._pick_segments([], {}, {1: [3.0], 4: [1.6]}, {4: 6.5}, {4: 1.3}) if s["turn"] == 4]

    assert (seg["type"], seg["reasons"][0]) == ("IMPROVE", "late")
    assert sorted(seg["reasons"]) == ["late", "pause", "quiet"]


def test_stronger_turn_wins_among_the_same_reason() -> None:
    p = _pipeline(_FIVE_TURNS)

    segs = p._pick_segments([], {}, {1: [1.6], 2: [1.7], 3: [1.8], 4: [4.0], 5: [1.9]}, {})

    assert [s["turn"] for s in segs] == [3, 4, 5]


def test_strongest_reason_leads_the_turn() -> None:
    p = _pipeline([(0.0, 8.0)])

    (seg,) = p._pick_segments([], {1: 5.2}, {1: [1.6]}, {1: 12.0})

    # 크기 12/6 = 2.0, 멈춤 1.6/1.5 ≈ 1.07, 흔들림 5.2/5.0 = 1.04
    assert (seg["reason"], seg["reasons"]) == ("quiet", ["quiet", "pause", "voice"])


def test_praise_fills_the_seats_left_over() -> None:
    p = _pipeline([(0.0, 3.0), (5.0, 15.0), (16.0, 20.0)])

    segs = p._pick_segments([(0.2, 2.8), (5.2, 14.8)], {1: 2.0, 2: 2.0}, {}, {3: 7.0})

    assert [(s["turn"], s["type"]) for s in segs] == [(1, "GOOD"), (2, "GOOD"), (3, "IMPROVE")]


# --- AI 에 넘기는 근거 ------------------------------------------------------


def test_points_follow_the_order_of_reasons() -> None:
    item = {
        "type": "IMPROVE", "utterance": "네", "avti": None, "reason": "late",
        "reasons": ["late", "fast", "pause"], "late_sec": 3.4, "fast": True,
        "pause_count": 1, "pause_longest_sec": 1.6,
    }

    lines = LLMClient._segment_lines([item])

    assert lines[1] == (
        "   짚을 점: 상대 말이 끝나고 약 3초 길게 멈췄다가 대답을 시작함, "
        "다른 말보다 말이 빨라짐, 말 중간에 1번 멈춤 (최대 약 2초)"
    )


@pytest.mark.parametrize(
    ("reasons", "sentence"),
    [
        (["late", "pause"], "상대방이 전화가 끊긴 줄 알았을 수 있어요"),
        (["pause", "late"], "상대방이 말이 끝난 줄 알았을 수 있어요"),
        (["voice", "late", "pause"], "상대방이 전화가 끊긴 줄 알았을 수 있어요"),
    ],
)
def test_sentence_follows_the_leading_silence(reasons: list[str], sentence: str) -> None:
    item = {
        "type": "IMPROVE", "utterance": "네", "avti": None, "reason": reasons[0], "reasons": reasons,
        "late_sec": 3.6, "pause_count": 1, "pause_longest_sec": 1.8,
    }

    lines = LLMClient._segment_lines([item])

    assert f"   넣을 문장: {sentence}" in lines


def test_voice_is_left_out_when_not_among_the_reasons() -> None:
    item = {"type": "IMPROVE", "utterance": "네", "avti": None, "reason": "fast", "reasons": ["fast"], "fast": True}

    assert LLMClient._segment_lines([item])[1] == "   짚을 점: 다른 말보다 말이 빨라짐"
