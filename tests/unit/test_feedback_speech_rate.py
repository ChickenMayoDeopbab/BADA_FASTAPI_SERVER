"""말 속도 — 음절 세기, 말한 시간 재기, 턴별 값 뽑기."""

import pytest

from app.services.feedback_points import (
    ARTICULATION_GAP_SEC,
    FAST_RATE_RATIO,
    MIN_RATE_SPEAKING_SEC,
    MIN_RATE_SYLLABLES,
    count_syllables,
    fast_turns,
    is_safe,
    speaking_seconds,
    speech_rate,
)
from app.services.llm import LLMClient
from app.services.pipeline import VoicePipeline, _segment_fallback


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


# --- 빨라진 턴 판정 ---------------------------------------------------------


def test_turn_much_faster_than_the_others_is_fast() -> None:
    assert fast_turns({1: 5.0, 2: 5.2, 3: 4.8, 4: 7.0}) == {4: 1.37}


def test_speed_is_fast_from_the_threshold() -> None:
    """딱 기준값부터 빨라진 턴으로 본다."""
    assert fast_turns({1: 4.0, 2: 4.0, 3: 4.0 * FAST_RATE_RATIO}) == {3: FAST_RATE_RATIO}
    assert fast_turns({1: 4.0, 2: 4.0, 3: 4.0 * FAST_RATE_RATIO - 0.1}) == {}


def test_slower_turn_is_not_flagged() -> None:
    assert fast_turns({1: 6.0, 2: 6.0, 3: 3.0}) == {}


def test_too_few_measured_turns_have_nothing_to_compare_with() -> None:
    assert fast_turns({1: 5.0, 2: 9.0}) == {}
    assert fast_turns({}) == {}


def test_fast_turns_are_keyed_by_turn_number_and_kept_for_storage() -> None:
    text = "이번 주 토요일 저녁 여섯 시에 네 명이요"           # 16음절
    p = _pipeline([(0.0, 4.0), (5.0, 9.0), (10.0, 14.0), (15.0, 19.0)], [text] * 4)

    fast = p._fast_by_turn(_Tremor([(0.5, 3.7), (5.5, 8.7), (10.5, 13.7), (15.5, 17.5)]))

    assert fast == {4: 1.6}                                  # 5.0 / 5.0 / 5.0 / 8.0 음절/초
    assert p._turn_metrics_for_storage()["fast_rate_ratio"] == {4: 1.6}


# --- 구간 선택 -------------------------------------------------------------


def _picking_pipeline(turns, texts=None) -> VoicePipeline:
    p = _pipeline(turns, texts if texts is not None else [f"발화{i}" for i in range(len(turns))])
    p._session = {"type": "SCENARIO", "scenario": {"title": "식당 예약"}}
    return p


def test_fast_turn_becomes_an_improve_segment() -> None:
    p = _picking_pipeline([(0.0, 4.0), (8.0, 16.0)])

    segs = p._pick_segments([(0.2, 3.8), (8.2, 15.8)], {}, {}, {}, {}, {2: 1.4})

    by_turn = {s["turn"]: s for s in segs}
    assert (by_turn[2]["type"], by_turn[2]["reason"], by_turn[2]["fast_ratio"]) == ("IMPROVE", "fast", 1.4)
    assert (by_turn[1]["type"], by_turn[1]["reason"], by_turn[1]["fast_ratio"]) == ("GOOD", "clean", None)


@pytest.mark.parametrize(
    ("avti", "kind", "reason"),
    [(2.0, "GOOD", "voice"), (2.9, "IMPROVE", "fast")],
)
def test_stable_voice_and_a_fast_turn_are_added_up(avti: float, kind: str, reason: str) -> None:
    """안정된 목소리(3.0÷AVTI)와 빨라진 정도(÷1.25)를 더해 큰 쪽이 그 구간의 종류가 된다."""
    p = _picking_pipeline([(0.0, 8.0)])

    (seg,) = p._pick_segments([], {1: avti}, {}, {}, {}, {1: 1.4})

    assert (seg["type"], seg["reason"]) == (kind, reason)


def test_fast_mark_is_kept_next_to_other_reasons() -> None:
    p = _picking_pipeline([(0.0, 8.0), (10.0, 18.0), (20.0, 28.0)])

    shaky, quiet, late = p._pick_segments(
        [], {1: 8.0}, {}, {2: 7.5}, {3: 2.0}, {1: 1.3, 2: 1.4, 3: 1.5}
    )

    assert (shaky["reasons"], shaky["fast_ratio"]) == (["voice", "fast"], 1.3)
    assert (quiet["reasons"], quiet["fast_ratio"]) == (["quiet", "fast"], 1.4)
    assert (late["reasons"], late["fast_ratio"]) == (["late", "fast"], 1.5)


def test_without_speed_data_selection_is_unchanged() -> None:
    p = _picking_pipeline([(0.0, 4.0), (8.0, 16.0)])
    args = ([(0.2, 3.8), (8.2, 15.8)], {1: 2.0}, {2: [2.1]}, {}, {})

    assert p._pick_segments(*args) == p._pick_segments(*args, {})


# --- AI 에 넘기는 근거 ------------------------------------------------------


def test_items_say_whether_the_turn_was_fast() -> None:
    p = _picking_pipeline([(0.0, 8.0), (10.0, 18.0)])
    base = {"avti": None, "pauses": [], "quiet_db": None, "late_sec": None}
    fast = {"turn": 1, "start": 0.0, "end": 8.0, "type": "IMPROVE", "reason": "fast", "fast_ratio": 1.4, **base}
    normal = {"turn": 2, "start": 10.0, "end": 18.0, "type": "GOOD", "reason": "clean", "fast_ratio": None, **base}

    first, second = p._feedback_items([fast, normal])

    assert (first["fast"], second["fast"]) == (True, False)


@pytest.mark.parametrize(
    ("item", "line"),
    [
        ({"reason": "fast", "fast": True}, "   짚을 점: 다른 말보다 말이 빨라짐"),
        ({"reason": "fast"}, "   짚을 점: 다른 말보다 말이 빨라짐"),
        ({"reason": "voice", "fast": True}, "   짚을 점: 목소리가 흔들림, 다른 말보다 말이 빨라짐"),
        ({"reason": "voice", "fast": False}, "   짚을 점: 목소리가 흔들림"),
    ],
)
def test_improve_segment_states_the_fast_speech(item: dict, line: str) -> None:
    lines = LLMClient._segment_lines([{"type": "IMPROVE", "utterance": "네", "avti": None, **item}])

    assert (lines[0], lines[1], lines[-1]) == ("1. [짚어줄 구간]", line, '   내가 한 말: "네"')
    assert not any("넣을 문장" in each for each in lines)


# --- 기본 문구와 밖으로 나가는 모양 ----------------------------------------


def test_fast_segment_has_its_own_fallback_text() -> None:
    title, content = _segment_fallback("IMPROVE", "fast")

    assert "빨라" in title and "천천히" in content
    assert not any(word in title + content for word in ("흔들", "멈춘", "작게", "늦"))
    assert is_safe(title, content)


class _SilentLLM:
    async def segment_feedback(self, items, **kwargs):
        self.items = items
        return []


async def test_failed_writing_falls_back_by_reason_and_hides_speed_data() -> None:
    p = _picking_pipeline([(0.0, 8.0)], ["토요일 저녁 여섯 시에 네 명이요"])
    p._llm = _SilentLLM()
    segments = p._pick_segments([], {}, {}, {}, {}, {1: 1.4})

    await p._write_segment_feedback(segments)

    (seg,) = segments
    assert seg["title"] == "말이 빨라졌어요"
    assert set(seg) == {"start", "end", "type", "title", "content"}


async def test_teardown_picks_the_fast_turn_and_tells_the_writer(monkeypatch) -> None:
    """받아쓴 글과 소리 구간으로 턴별 속도를 재고, 다른 턴보다 빨라진 턴을 짚어줄 구간으로 올린다."""
    from tests.unit.test_avti_save import _pipeline as teardown_pipeline

    async def _no_avti(**_kwargs) -> dict:
        return {}

    spring: dict = {}

    async def _capture(session_id, **kwargs):
        spring.update(kwargs)
        return True

    monkeypatch.setattr("app.services.pipeline.run_avti", _no_avti)

    text = "이번 주 토요일 저녁 여섯 시에 네 명이요"           # 16음절
    p = teardown_pipeline(
        seconds=20,
        turns=[(0.0, 4.0), (5.0, 9.0), (10.0, 14.0), (15.0, 19.0)],
        spans=[(0.5, 3.7), (5.5, 8.7), (10.5, 13.7), (15.5, 17.5)],   # 4번 턴만 2초에 말함
    )
    p._user_turn_texts = [text] * 4
    p._spring.notify_session_closed = _capture

    await p._teardown()

    (fast,) = p._llm.last_items
    assert (fast["type"], fast["reason"], fast["fast"]) == ("IMPROVE", "fast", True)
    assert [seg["type"] for seg in spring["good_segments"]] == ["IMPROVE"]
    for seg in spring["good_segments"]:
        assert set(seg) == {"start", "end", "type", "title", "content"}

