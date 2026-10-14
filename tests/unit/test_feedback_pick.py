"""짚을 구간 고르기 — 턴마다 좋은 점(+)과 아쉬운 점(−)을 더해, 크기가 큰 3개를 고른다."""

import pytest

from app.services.feedback_points import POINT_CAP, turn_points
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


# --- 항목별 점수 -----------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "points"),
    [
        ({"avti": 6.0}, {"voice": -1.2}),
        ({"pauses": [1.6, 1.8]}, {"pause": -1.2}),        # 가장 긴 멈춤으로
        ({"quiet_db": 9.0}, {"quiet": -1.5}),
        ({"late_sec": 1.8}, {"late": -1.5}),
        ({"fast_ratio": 1.5}, {"fast": -1.2}),
        ({"avti": 2.0}, {"voice": 1.5}),
        ({"clean": True}, {"clean": 1.0}),
    ],
)
def test_each_item_is_scored_by_how_far_past_its_line(kwargs: dict, points: dict) -> None:
    assert turn_points(**kwargs) == points


def test_ambiguous_voice_scores_nothing() -> None:
    assert turn_points(avti=4.0) == {}


@pytest.mark.parametrize(
    ("kwargs", "points"),
    [
        ({"pauses": [6.0]}, {"pause": -POINT_CAP}),
        ({"late_sec": 9.0}, {"late": -POINT_CAP}),
        ({"avti": 1.0}, {"voice": POINT_CAP}),
        ({"avti": 0.0}, {"voice": POINT_CAP}),
    ],
)
def test_one_item_counts_at_most_twice_its_line(kwargs: dict, points: dict) -> None:
    assert turn_points(**kwargs) == points


def test_clean_counts_only_when_voice_was_not_measured() -> None:
    assert turn_points(avti=2.0, clean=True) == {"voice": 1.5}


# --- 고르기 ---------------------------------------------------------------


def test_a_later_turn_is_not_crowded_out_by_earlier_turns() -> None:
    """앞 턴부터 채우면 1~3번 멈춤에 자리가 다 차서 5번의 작은 목소리를 놓친다."""
    p = _pipeline(_FIVE_TURNS)

    segs = p._pick_segments([], {}, {1: [1.6], 2: [1.7], 3: [1.8]}, {5: 9.0})

    assert [(s["turn"], s["reason"]) for s in segs] == [(2, "pause"), (3, "pause"), (5, "quiet")]


def test_praise_and_pointers_compete_on_the_same_scale() -> None:
    p = _pipeline(_FIVE_TURNS)

    segs = p._pick_segments([], {1: 1.2, 4: 2.5}, {2: [1.6]}, {}, {3: 2.4}, {5: 1.3})

    # 1번 +2.0, 2번 −1.07, 3번 −2.0, 4번 +1.2, 5번 −1.04
    assert [(s["turn"], s["type"]) for s in segs] == [(1, "GOOD"), (3, "IMPROVE"), (4, "GOOD")]


def test_no_seat_is_kept_for_praise() -> None:
    p = _pipeline(_FIVE_TURNS)

    segs = p._pick_segments([], {1: 2.8}, {2: [1.9]}, {3: 8.0}, {4: 1.5})

    assert [s["type"] for s in segs] == ["IMPROVE"] * 3


def test_the_same_reason_may_fill_every_seat() -> None:
    p = _pipeline(_FIVE_TURNS)

    segs = p._pick_segments([], {}, {1: [3.0], 2: [2.9], 3: [2.8]}, {4: 7.0})

    assert [s["turn"] for s in segs] == [1, 2, 3]


def test_items_in_one_turn_add_up() -> None:
    p = _pipeline(_FIVE_TURNS)

    segs = p._pick_segments([], {}, {1: [2.9], 2: [1.6], 3: [1.6]}, {4: 6.6}, {4: 1.3})

    # 4번: 크기 −1.1 + 뜸 −1.08 = −2.18 이 1번 멈춤 −1.93 보다 크다.
    assert segs[-1]["turn"] == 4
    assert segs[-1]["reasons"] == ["quiet", "late"]


def test_good_and_bad_in_one_turn_cancel_out() -> None:
    p = _pipeline([(0.0, 8.0), (10.0, 18.0)])

    segs = p._pick_segments([], {1: 2.0, 2: 3.0}, {1: [1.8]}, {}, {2: 1.2})

    # 2번: 목소리 +1.0, 뜸 −1.0 → 0 이라 고르지 않는다.
    assert [s["turn"] for s in segs] == [1]
    # 1번: 목소리 +1.5, 멈춤 −1.2 → +0.3 칭찬. 이긴 쪽 항목만 이유로 남긴다.
    assert (segs[0]["type"], segs[0]["reasons"], segs[0]["score"]) == ("GOOD", ["voice"], 0.3)


def test_bigger_score_wins_among_the_same_reason() -> None:
    p = _pipeline(_FIVE_TURNS)

    segs = p._pick_segments([], {}, {1: [1.6], 2: [1.7], 3: [1.8], 4: [4.0], 5: [1.9]}, {})

    assert [s["turn"] for s in segs] == [3, 4, 5]


def test_earlier_turn_wins_a_tie() -> None:
    p = _pipeline(_FIVE_TURNS)

    segs = p._pick_segments([], {}, {1: [1.8], 2: [1.8], 3: [1.8], 4: [1.8]}, {})

    assert [s["turn"] for s in segs] == [1, 2, 3]


def test_biggest_item_leads_the_turn() -> None:
    p = _pipeline([(0.0, 8.0)])

    (seg,) = p._pick_segments([], {1: 5.2}, {1: [1.6]}, {1: 12.0})

    # 크기 12/6 = 2.0, 멈춤 1.6/1.5 ≈ 1.07, 흔들림 5.2/5.0 = 1.04
    assert (seg["reason"], seg["reasons"]) == ("quiet", ["quiet", "pause", "voice"])


def test_ambiguous_voice_alone_is_not_picked() -> None:
    p = _pipeline([(0.0, 8.0), (10.0, 18.0)])

    segs = p._pick_segments([], {1: 4.0}, {2: [1.8]})

    assert [s["turn"] for s in segs] == [2]


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


@pytest.mark.parametrize(
    ("reasons", "line"),
    [
        (["voice"], "   잘한 점: 목소리가 흔들림 없이 안정적임"),
        (["clean"], "   잘한 점: 끊김 없이 이어 말함"),
    ],
)
def test_praise_segment_states_what_went_well(reasons: list[str], line: str) -> None:
    item = {"type": "GOOD", "utterance": "네", "avti": None, "reason": reasons[0], "reasons": reasons}

    assert LLMClient._segment_lines([item]) == ["1. [칭찬할 구간]", line, '   내가 한 말: "네"']


def test_praise_segment_without_reasons_gets_no_line() -> None:
    item = {"type": "GOOD", "utterance": "네", "avti": None, "reason": "fallback"}

    assert LLMClient._segment_lines([item]) == ["1. [칭찬할 구간]", '   내가 한 말: "네"']
