"""상대 말이 끝나고 한참 뒤에 대답한 턴 — 재는 규칙, 재생 완료 시각 기록, 구간 선택, AI 에 넘기는 근거."""

import json

import pytest

from app.services.feedback_points import (
    LATE_ANSWER_SEC,
    LONG_SILENCE_SEC,
    answer_delay,
    is_long_silence,
    is_safe,
)
from app.services.llm import LLMClient
from app.services.pipeline import VoicePipeline, _segment_fallback
from tests.unit.test_transcript_frames import _make_pipeline

_BYTES_PER_SEC = 16000 * 2


def _pipeline(turns, texts=None, heard_at=None) -> VoicePipeline:
    p = VoicePipeline.__new__(VoicePipeline)
    p._session_id = "test-session"
    p._session = {"type": "SCENARIO", "scenario": {"title": "식당 예약"}}
    p._user_turn_intervals = turns
    p._user_turn_texts = texts if texts is not None else [f"발화{i}" for i in range(len(turns))]
    p._heard_at_by_turn = heard_at or {}
    return p


# --- 재는 규칙 -------------------------------------------------------------


def test_delay_runs_from_the_end_of_the_other_voice_to_the_first_sound() -> None:
    assert answer_delay((5.0, 12.0), 6.0, [(8.0, 11.0)]) == 2.0


def test_turn_opening_before_the_other_voice_ends_does_not_add_to_the_delay() -> None:
    """턴은 서버가 음성을 다 보낸 때 열린다. 그때부터가 아니라 다 들린 때부터 잰다."""
    assert answer_delay((2.0, 12.0), 6.0, [(6.5, 11.0)]) == 0.5


def test_speaking_over_the_other_voice_is_not_a_delay() -> None:
    """상대 말이 끝나기 전부터 말하고 있었으면 뜸을 들인 게 아니다."""
    assert answer_delay((2.0, 12.0), 6.0, [(5.0, 9.0)]) == 0.0


def test_sound_that_ended_before_the_other_voice_did_is_ignored() -> None:
    assert answer_delay((2.0, 12.0), 6.0, [(3.0, 4.0), (9.0, 11.0)]) == 3.0


def test_answer_before_the_turn_was_opened_still_counts() -> None:
    """재생 완료 신호가 턴이 열리기 전에 올 수도 있다. 그 사이에 낸 소리도 대답의 시작이다."""
    assert answer_delay((7.0, 12.0), 6.0, [(6.4, 9.0)]) == 0.4


def test_no_sound_afterwards_cannot_be_measured() -> None:
    assert answer_delay((2.0, 12.0), 6.0, []) is None
    assert answer_delay((2.0, 12.0), 6.0, [(3.0, 5.0)]) is None
    assert answer_delay((2.0, 12.0), 6.0, [(13.0, 15.0)]) is None


def test_silence_is_called_long_only_from_the_long_threshold() -> None:
    """짚는 기준(1.2초)을 넘었다고 길다고 쓰지 않는다. 길다고 쓰는 기준은 따로 있다."""
    assert not is_long_silence(LATE_ANSWER_SEC)
    assert not is_long_silence(LONG_SILENCE_SEC - 0.1)
    assert is_long_silence(LONG_SILENCE_SEC)


# --- 재생 완료 시각 기록 ---------------------------------------------------


async def _spoken(p, seconds_of_mic: float) -> int:
    """AI 가 말을 다 보냈고, 그동안 마이크 소리가 seconds_of_mic 만큼 들어온 상황."""
    p._begin_playback()
    await p._send_speaking_end()
    p._tremor_buf.extend(b"\x00" * int(seconds_of_mic * _BYTES_PER_SEC))
    return p._playback.turn_id


async def _ack(p, turn_id) -> None:
    await p._handle_client_text(json.dumps({"type": "playback_done", "turn_id": turn_id}))


async def test_playback_ack_marks_where_the_recording_was() -> None:
    p = _make_pipeline()
    turn_id = await _spoken(p, 3.0)

    await _ack(p, turn_id)

    assert p._ai_heard_at == 3.0


async def test_wrong_ack_leaves_no_mark() -> None:
    p = _make_pipeline()
    turn_id = await _spoken(p, 3.0)

    await _ack(p, turn_id + 1)

    assert p._ai_heard_at is None


async def test_repeated_ack_does_not_move_the_mark() -> None:
    p = _make_pipeline()
    turn_id = await _spoken(p, 3.0)
    await _ack(p, turn_id)
    p._tremor_buf.extend(b"\x00" * _BYTES_PER_SEC)

    await _ack(p, turn_id)

    assert p._ai_heard_at == 3.0


async def test_next_ai_speech_clears_an_unused_mark() -> None:
    """앞선 AI 말의 완료 시각이 다음 AI 말 뒤의 턴에 붙으면 뜸이 엉뚱하게 길어진다."""
    p = _make_pipeline()
    await _ack(p, await _spoken(p, 3.0))

    p._begin_playback()

    assert p._ai_heard_at is None


async def test_closing_the_turn_attaches_the_mark_to_that_turn() -> None:
    p = _make_pipeline()
    p._user_turn_texts, p._user_turn_prompts, p._heard_at_by_turn = [], [], {}
    await _ack(p, await _spoken(p, 3.0))
    p._open_user_turn()
    p._tremor_buf.extend(b"\x00" * 4 * _BYTES_PER_SEC)

    p._close_user_turn("네 맞아요")

    assert p._heard_at_by_turn == {1: 3.0}
    assert p._ai_heard_at is None


async def test_turn_without_an_ack_has_no_mark() -> None:
    """통화 첫마디나 끼어든 턴, 신호를 안 보내는 구버전 앱의 턴."""
    p = _make_pipeline()
    p._user_turn_texts, p._user_turn_prompts, p._heard_at_by_turn = [], [], {}
    p._open_user_turn()

    p._close_user_turn("여보세요")

    assert p._heard_at_by_turn == {}


# --- 떨림 분석 결과에서 턴별 뜸 뽑기 ---------------------------------------


class _Tremor:
    def __init__(self, voiced_spans) -> None:
        self.voiced_spans = voiced_spans


def test_late_turns_are_keyed_by_turn_number() -> None:
    p = _pipeline([(0.0, 4.0), (5.0, 12.0), (13.0, 20.0)], heard_at={2: 6.0, 3: 14.0})

    late = p._late_by_turn(_Tremor([(0.5, 3.5), (8.0, 11.0), (14.3, 19.0)]))

    assert late == {2: 2.0}


def test_delay_is_late_from_the_threshold() -> None:
    """딱 기준값부터 늦은 턴으로 본다."""
    p = _pipeline([(0.0, 10.0)], heard_at={1: 2.0})

    assert p._late_by_turn(_Tremor([(2.0 + LATE_ANSWER_SEC, 9.0)])) == {1: LATE_ANSWER_SEC}
    assert p._late_by_turn(_Tremor([(2.0 + LATE_ANSWER_SEC - 0.1, 9.0)])) == {}


def test_turn_without_a_mark_is_not_judged() -> None:
    p = _pipeline([(0.0, 10.0), (11.0, 20.0)], heard_at={2: 12.0})

    assert p._late_by_turn(_Tremor([(5.0, 9.0), (12.2, 19.0)])) == {}


def test_missing_analysis_or_marks_mean_no_late_turn() -> None:
    p = _pipeline([(0.0, 10.0)], heard_at={1: 2.0})

    assert p._late_by_turn(None) == {}
    assert p._late_by_turn(_Tremor([])) == {}
    assert _pipeline([(0.0, 10.0)])._late_by_turn(_Tremor([(5.0, 9.0)])) == {}


# --- 구간 선택 -------------------------------------------------------------


def test_late_turn_becomes_an_improve_segment() -> None:
    p = _pipeline([(0.0, 4.0), (8.0, 16.0)])

    segs = p._pick_segments([(0.2, 3.8), (8.2, 15.8)], {}, {}, {}, {2: 2.0})

    by_turn = {s["turn"]: s for s in segs}
    assert (by_turn[2]["type"], by_turn[2]["reason"], by_turn[2]["late_sec"]) == ("IMPROVE", "late", 2.0)
    assert (by_turn[1]["type"], by_turn[1]["reason"], by_turn[1]["late_sec"]) == ("GOOD", "clean", None)


def test_stable_voice_does_not_hide_a_late_answer() -> None:
    p = _pipeline([(0.0, 8.0)])

    (seg,) = p._pick_segments([], {1: 2.0}, {}, {}, {1: 2.0})

    assert (seg["type"], seg["reason"]) == ("IMPROVE", "late")


def test_late_answer_is_kept_next_to_other_reasons() -> None:
    p = _pipeline([(0.0, 8.0), (10.0, 18.0), (20.0, 28.0)])

    shaky, paused, quiet = p._pick_segments(
        [], {1: 5.5}, {2: [2.4]}, {3: 9.0}, {1: 2.0, 2: 1.3, 3: 1.3}
    )

    # 1번: 뜸 2.0/1.2 이 흔들림 5.5/5.0 보다 기준을 더 넘었다. 2·3번은 반대.
    assert (shaky["reasons"], shaky["late_sec"]) == (["late", "voice"], 2.0)
    assert (paused["reasons"], paused["late_sec"]) == (["pause", "late"], 1.3)
    assert (quiet["reasons"], quiet["late_sec"]) == (["quiet", "late"], 1.3)


def test_late_turn_without_transcript_is_ignored() -> None:
    p = _pipeline([(0.0, 8.0), (10.0, 14.0)], ["", "네 맞아요"])

    segs = p._pick_segments([(10.2, 13.8)], {}, {}, {}, {1: 3.0})

    assert [s["turn"] for s in segs] == [2]


def test_without_delay_data_selection_is_unchanged() -> None:
    p = _pipeline([(0.0, 4.0), (8.0, 16.0)])
    candidates, avti, pauses, quiet = [(0.2, 3.8), (8.2, 15.8)], {1: 2.0}, {2: [2.1]}, {}

    with_none = p._pick_segments(candidates, avti, pauses, quiet)
    with_empty = p._pick_segments(candidates, avti, pauses, quiet, {})

    assert [(s["turn"], s["type"], s["reason"]) for s in with_none] == [
        (1, "GOOD", "voice"), (2, "IMPROVE", "pause"),
    ]
    assert with_none == with_empty


# --- AI 에 넘기는 근거 ------------------------------------------------------


def test_items_carry_how_long_the_answer_took() -> None:
    p = _pipeline([(0.0, 8.0), (10.0, 18.0)])
    late = {"turn": 1, "start": 0.0, "end": 8.0, "type": "IMPROVE", "reason": "late",
            "avti": None, "pauses": [], "quiet_db": None, "late_sec": 2.4}
    normal = {"turn": 2, "start": 10.0, "end": 18.0, "type": "GOOD", "reason": "clean",
              "avti": None, "pauses": [], "quiet_db": None, "late_sec": None}

    first, second = p._feedback_items([late, normal])

    assert (first["late_sec"], second["late_sec"]) == (2.4, 0.0)


@pytest.mark.parametrize(
    ("item", "line"),
    [
        ({"reason": "late", "late_sec": 2.4}, "   짚을 점: 상대 말이 끝나고 약 2초 멈췄다가 대답을 시작함"),
        ({"reason": "voice", "late_sec": 1.2},
         "   짚을 점: 목소리가 흔들림, 상대 말이 끝나고 약 1초 멈췄다가 대답을 시작함"),
        ({"reason": "quiet", "quiet": True, "late_sec": 3.6},
         "   짚을 점: 다른 말보다 목소리가 작아짐, 상대 말이 끝나고 약 4초 길게 멈췄다가 대답을 시작함"),
        ({"reason": "late"}, "   짚을 점: 상대 말이 끝나고 멈췄다가 대답을 시작함"),
        ({"reason": "voice", "late_sec": 0.0}, "   짚을 점: 목소리가 흔들림"),
    ],
)
def test_improve_segment_states_the_late_answer(item: dict, line: str) -> None:
    lines = LLMClient._segment_lines([{"type": "IMPROVE", "utterance": "네", "avti": None, **item}])

    assert (lines[0], lines[1], lines[-1]) == ("1. [짚어줄 구간]", line, '   내가 한 말: "네"')


def test_delay_is_given_without_decimals() -> None:
    """소수는 측정값 유출로 보고 안전 검사가 막는다. AI 에는 어림한 정수만 준다."""
    (_, line, *_) = LLMClient._segment_lines(
        [{"type": "IMPROVE", "utterance": "네", "avti": None, "reason": "late", "late_sec": 1.75}]
    )

    assert "." not in line


@pytest.mark.parametrize(
    ("item", "sentence"),
    [
        ({"reason": "pause", "pause_count": 1, "pause_longest_sec": 2.1}, "상대방이 말이 끝난 줄 알았을 수 있어요"),
        ({"reason": "pause", "pause_count": 1, "pause_longest_sec": 3.4}, "상대방이 전화가 끊긴 줄 알았을 수 있어요"),
        ({"reason": "late", "late_sec": 1.3}, "상대방이 내 말을 못 들은 줄 알았을 수 있어요"),
        ({"reason": "late", "late_sec": 4.2}, "상대방이 전화가 끊긴 줄 알았을 수 있어요"),
    ],
)
def test_paused_segment_is_given_the_sentence_for_its_length(item: dict, sentence: str) -> None:
    """상대가 어떻게 받아들였을지는 멈춘 자리와 길이로 정해 준다. 글 쓰는 쪽이 고르면 짧은 멈춤에도 센 말을 쓴다."""
    lines = LLMClient._segment_lines([{"type": "IMPROVE", "utterance": "네", "avti": None, **item}])

    assert lines[2] == f"   넣을 문장: {sentence}"


@pytest.mark.parametrize(
    "item",
    [{"reason": "voice"}, {"reason": "quiet", "quiet": True}],
)
def test_other_segments_get_no_fixed_sentence(item: dict) -> None:
    lines = LLMClient._segment_lines([{"type": "IMPROVE", "utterance": "네", "avti": None, **item}])

    assert not any("넣을 문장" in line for line in lines)


@pytest.mark.parametrize(
    ("item", "sentence"),
    [
        ({"reason": "voice", "late_sec": 2.6}, "상대방이 내 말을 못 들은 줄 알았을 수 있어요"),
        ({"reason": "quiet", "quiet": True, "late_sec": 4.2}, "상대방이 전화가 끊긴 줄 알았을 수 있어요"),
        ({"reason": "voice", "pause_count": 1, "pause_longest_sec": 2.1}, "상대방이 말이 끝난 줄 알았을 수 있어요"),
        ({"reason": "voice", "pause_count": 1, "pause_longest_sec": 2.1, "late_sec": 4.2},
         "상대방이 말이 끝난 줄 알았을 수 있어요"),
    ],
)
def test_segment_picked_for_another_reason_still_gets_the_sentence_for_its_pause(item: dict, sentence: str) -> None:
    """흔들림으로 고른 구간도 '짚을 점' 에 멈춤이 같이 적히면 글이 그 얘기를 한다. 문장을 안 주면 제멋대로 쓴다."""
    lines = LLMClient._segment_lines([{"type": "IMPROVE", "utterance": "네", "avti": None, **item}])

    assert f"   넣을 문장: {sentence}" in lines


@pytest.mark.parametrize(
    "content",
    [
        "대답하기 전에 멈췄어요. 상대방은 궁금해할 수 있으니 다음엔 짧게 먼저 받아봐요.",
        "기다리는 동안 상대방이 궁금했을 것 같아요. 다음엔 바로 답해봐요.",
        "대답이 늦어서 상대방이 궁금할 수 있어요. 다음엔 바로 답해봐요.",
    ],
)
def test_vague_wording_about_the_listener_is_rejected(content: str) -> None:
    assert not is_safe("대답하기 전에 멈췄어요", content)


def test_my_own_curiosity_is_not_rejected() -> None:
    assert is_safe("궁금한 점을 먼저 물어봤어요", "궁금한 점을 바로 물어봐서 통화가 빨리 끝났어요.")


# --- 기본 문구와 밖으로 나가는 모양 ----------------------------------------


def test_late_segment_has_its_own_fallback_text() -> None:
    title, content = _segment_fallback("IMPROVE", "late")

    assert "늦" in title and "먼저" in content
    assert not any(word in title + content for word in ("흔들", "멈춘", "작"))
    assert is_safe(title, content)


class _SilentLLM:
    async def segment_feedback(self, items, **kwargs):
        self.items = items
        return []


async def test_failed_writing_falls_back_by_reason_and_hides_delay_data() -> None:
    p = _pipeline([(0.0, 8.0)], ["토요일 저녁 여섯 시에 네 명이요"])
    p._llm = _SilentLLM()
    segments = p._pick_segments([], {}, {}, {}, {1: 2.0})

    await p._write_segment_feedback(segments)

    (seg,) = segments
    assert seg["title"] == "대답이 조금 늦었어요"
    assert set(seg) == {"start", "end", "type", "title", "content"}


async def test_teardown_picks_the_late_turn_and_tells_the_writer(monkeypatch) -> None:
    """재생 완료 시각과 떨림 분석이 준 소리 구간으로 뜸을 재고, 늦은 턴을 짚어줄 구간으로 올린다."""
    from tests.unit.test_avti_save import _pipeline as teardown_pipeline

    async def _no_avti(**_kwargs) -> dict:
        return {}

    spring: dict = {}

    async def _capture(session_id, **kwargs):
        spring.update(kwargs)
        return True

    monkeypatch.setattr("app.services.pipeline.run_avti", _no_avti)

    p = teardown_pipeline(
        seconds=20,
        turns=[(1.0, 5.0), (6.0, 16.0)],
        spans=[(1.2, 4.8), (10.5, 15.0)],     # 2번 턴: 상대 말이 8.0초에 끝났고 10.5초에 대답 시작
    )
    p._user_turn_texts = ["예약하려고 전화드렸어요", "토요일 저녁 여섯 시에 네 명이요"]
    p._heard_at_by_turn = {2: 8.0}
    p._spring.notify_session_closed = _capture

    await p._teardown()

    # 이 가짜 분석은 '떨림 없는 구간'을 주지 않아 1번 턴은 고를 근거가 없고, 늦게 대답한 2번 턴만 뽑힌다.
    (late,) = p._llm.last_items
    assert (late["type"], late["reason"], late["late_sec"]) == ("IMPROVE", "late", 2.5)
    assert late["utterance"] == "토요일 저녁 여섯 시에 네 명이요"
    assert [seg["type"] for seg in spring["good_segments"]] == ["IMPROVE"]
    for seg in spring["good_segments"]:
        assert set(seg) == {"start", "end", "type", "title", "content"}
