"""말 속도 — 음절 세기, 말한 시간 재기, 턴별 값 뽑기."""

import pytest

from app.services.feedback_points import (
    ARTICULATION_GAP_SEC,
    MIN_RATE_SPEAKING_SEC,
    MIN_RATE_SYLLABLES,
    count_syllables,
    speaking_seconds,
    speech_rate,
)
from app.services.pipeline import VoicePipeline


def _pipeline(turns, texts) -> VoicePipeline:
    p = VoicePipeline.__new__(VoicePipeline)
    p._session_id = "test-session"
    p._user_turn_intervals = turns
    p._user_turn_texts = texts
    return p


# --- 음절 세기 -------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "count"),
    [
        ("예약하고 싶어서 전화 드렸습니다.", 14),
        ("네, 맞습니다!", 5),
        ("10월 17일 6시쯤", 9),          # 숫자는 글자 수로 어림한다. 읽으면 시월 십칠일 여섯시쯤, 여기서는 딱 맞는다
        ("5월 15일 저녁 7시", 9),        # 읽으면 오월 십오일 저녁 일곱시 = 10음절. 한 음절 어긋나는 경우
        ("A동 101호", 6),
        ("", 0),
        ("... ?!", 0),
    ],
)
def test_syllables_are_counted_from_the_transcript(text: str, count: int) -> None:
    assert count_syllables(text) == count


# --- 말한 시간 -------------------------------------------------------------


def test_speaking_time_adds_up_the_voice_in_the_turn() -> None:
    assert speaking_seconds((0.0, 10.0), [(1.0, 3.0), (5.0, 8.0)]) == pytest.approx(5.0)


def test_short_gap_counts_as_speaking() -> None:
    """낱말 사이나 막히는 소리의 짧은 틈은 말하는 중이다. 딱 기준값부터 쉼으로 뺀다."""
    just_under = 1.0 + 2.0 + (ARTICULATION_GAP_SEC - 0.05) + 2.0
    assert speaking_seconds((0.0, 10.0), [(1.0, 3.0), (3.0 + ARTICULATION_GAP_SEC - 0.05, 5.2)]) == pytest.approx(
        just_under - 1.0
    )
    assert speaking_seconds((0.0, 10.0), [(1.0, 3.0), (3.0 + ARTICULATION_GAP_SEC, 5.25)]) == pytest.approx(4.0)


def test_pause_is_left_out_of_speaking_time() -> None:
    assert speaking_seconds((0.0, 12.0), [(1.0, 3.0), (6.0, 8.0)]) == pytest.approx(4.0)


def test_voice_outside_the_turn_is_not_counted() -> None:
    assert speaking_seconds((5.0, 9.0), [(0.0, 4.0), (4.5, 6.0), (8.0, 11.0)]) == pytest.approx(2.0)
    assert speaking_seconds((5.0, 9.0), []) == 0.0


# --- 말 속도 ---------------------------------------------------------------


def test_rate_is_syllables_per_second_of_speaking() -> None:
    assert speech_rate("예약하고 싶어서 전화 드렸습니다.", (0.0, 5.0), [(1.0, 3.8)]) == 5.0


def test_pause_in_the_turn_does_not_make_the_rate_slower() -> None:
    """멈춘 시간은 빼고 나눈다. 같은 말을 같은 빠르기로 했으면 중간에 멈췄어도 값이 같다."""
    text = "예약하고 싶어서 전화 드렸습니다."

    without_pause = speech_rate(text, (0.0, 10.0), [(1.0, 3.8)])
    with_pause = speech_rate(text, (0.0, 10.0), [(1.0, 2.4), (5.0, 6.4)])

    assert without_pause == with_pause == 5.0


def test_too_little_speech_is_not_measured() -> None:
    assert speech_rate("네, 맞습니다.", (0.0, 5.0), [(1.0, 3.0)]) is None
    assert speech_rate("가" * MIN_RATE_SYLLABLES, (0.0, 5.0), [(1.0, 1.0 + MIN_RATE_SPEAKING_SEC - 0.1)]) is None
    assert speech_rate("가" * MIN_RATE_SYLLABLES, (0.0, 5.0), [(1.0, 1.0 + MIN_RATE_SPEAKING_SEC)]) is not None
    assert speech_rate("예약하고 싶어서 전화 드렸습니다.", (0.0, 5.0), []) is None


# --- 떨림 분석 결과에서 턴별 속도 뽑기 -------------------------------------


class _Tremor:
    def __init__(self, voiced_spans) -> None:
        self.voiced_spans = voiced_spans


def test_rates_are_keyed_by_turn_number() -> None:
    p = _pipeline(
        [(0.0, 3.0), (5.0, 10.0), (12.0, 18.0)],
        ["여보세요.", "예약하고 싶어서 전화 드렸습니다.", "이번 주 토요일 저녁 여섯 시에 네 명이요"],
    )

    rates = p._speech_rates_by_turn(_Tremor([(0.5, 1.5), (6.0, 8.0), (13.0, 17.0)]))

    assert rates == {2: 7.0, 3: 4.0}


def test_missing_analysis_means_no_rates() -> None:
    p = _pipeline([(0.0, 5.0)], ["예약하고 싶어서 전화 드렸습니다."])

    assert p._speech_rates_by_turn(None) == {}
    assert p._speech_rates_by_turn(_Tremor([])) == {}
