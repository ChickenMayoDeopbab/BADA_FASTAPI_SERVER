"""다른 턴보다 작게 말한 턴 — 재는 규칙, 구간 선택, AI 에 넘기는 근거."""

import numpy as np
import pytest

from app.services.feedback_points import QUIET_DROP_DB, is_safe, quiet_drops, turn_levels
from app.services.llm import LLMClient
from app.services.pipeline import VoicePipeline, _segment_fallback

_RATE = 16000


def _pcm(seconds: float, voiced: list[tuple[float, float, float]]) -> bytes:
    """(시작초, 끝초, 소리 폭 0~1) 구간에만 소리가 있는 녹음."""
    y = np.zeros(int(seconds * _RATE), dtype=np.float32)
    for start, end, amp in voiced:
        a, b = int(start * _RATE), int(end * _RATE)
        y[a:b] = amp * np.sin(2 * np.pi * 200 * np.arange(b - a) / _RATE)
    return (y * 32767).astype(np.int16).tobytes()


def _pipeline(turns, texts=None) -> VoicePipeline:
    p = VoicePipeline.__new__(VoicePipeline)
    p._session_id = "test-session"
    p._session = {"type": "SCENARIO", "scenario": {"title": "식당 예약"}}
    p._user_turn_intervals = turns
    p._user_turn_texts = texts if texts is not None else [f"발화{i}" for i in range(len(turns))]
    return p


# --- 재는 규칙 -------------------------------------------------------------


def test_half_the_amplitude_measures_six_db_lower() -> None:
    pcm = _pcm(10, [(0.0, 4.0, 0.5), (5.0, 9.0, 0.25)])

    levels = turn_levels(pcm, [(0.0, 4.0), (5.0, 9.0)], [(0.0, 4.0), (5.0, 9.0)])

    assert levels[1] - levels[2] == pytest.approx(6.02, abs=0.05)


def test_pause_inside_a_turn_does_not_lower_the_level() -> None:
    """말소리가 난 부분만 재므로, 같은 크기로 말했으면 중간에 멈췄어도 값이 같다."""
    pcm = _pcm(20, [(0.0, 8.0, 0.5), (10.0, 13.0, 0.5), (15.0, 18.0, 0.5)])

    levels = turn_levels(
        pcm, [(0.0, 8.0), (10.0, 18.0)], [(0.0, 8.0), (10.0, 13.0), (15.0, 18.0)]
    )

    assert levels[1] == pytest.approx(levels[2], abs=0.05)


def test_turn_with_too_little_voice_is_not_measured() -> None:
    """짧은 대답은 크기가 들쭉날쭉해서 재지 않는다."""
    pcm = _pcm(10, [(0.0, 4.0, 0.5), (5.0, 5.5, 0.1)])

    levels = turn_levels(pcm, [(0.0, 4.0), (5.0, 9.0)], [(0.0, 4.0), (5.0, 5.5)])

    assert set(levels) == {1}


def test_voice_outside_the_turn_is_ignored() -> None:
    pcm = _pcm(12, [(0.0, 3.0, 0.9), (4.0, 8.0, 0.25), (9.0, 12.0, 0.9)])
    spans = [(0.0, 3.0), (4.0, 8.0), (9.0, 12.0)]

    only_turn = turn_levels(pcm, [(4.0, 8.0)], spans)
    alone = turn_levels(pcm, [(4.0, 8.0)], [(4.0, 8.0)])

    assert only_turn == alone


def test_silent_or_missing_recording_measures_nothing() -> None:
    assert turn_levels(b"", [(0.0, 4.0)], [(0.0, 4.0)]) == {}
    assert turn_levels(_pcm(5, []), [(0.0, 4.0)], [(0.0, 4.0)]) == {}
    assert turn_levels(_pcm(5, [(0.0, 4.0, 0.5)]), [(0.0, 4.0)], []) == {}


def test_recording_with_a_dangling_byte_is_still_measured() -> None:
    pcm = _pcm(5, [(0.0, 4.0, 0.5)])

    assert turn_levels(pcm + b"\x01", [(0.0, 4.0)], [(0.0, 4.0)]) == turn_levels(
        pcm, [(0.0, 4.0)], [(0.0, 4.0)]
    )


def test_turn_far_below_the_others_is_quiet() -> None:
    assert quiet_drops({1: -20.0, 2: -20.5, 3: -27.0, 4: -19.5}) == {3: 6.8}


def test_drop_is_counted_from_the_threshold() -> None:
    """딱 기준값부터 작아진 턴으로 본다."""
    assert quiet_drops({1: -20.0, 2: -20.0, 3: -20.0 - QUIET_DROP_DB}) == {3: QUIET_DROP_DB}
    assert quiet_drops({1: -20.0, 2: -20.0, 3: -20.0 - QUIET_DROP_DB + 0.1}) == {}


def test_louder_turn_is_not_flagged() -> None:
    assert quiet_drops({1: -20.0, 2: -20.0, 3: -8.0}) == {}


def test_too_few_measured_turns_have_nothing_to_compare_with() -> None:
    """견줄 턴이 모자라면 어느 쪽이 평소 크기인지 알 수 없다."""
    assert quiet_drops({1: -20.0, 2: -30.0}) == {}
    assert quiet_drops({}) == {}


# --- 구간 선택 -------------------------------------------------------------


def test_quiet_turn_becomes_an_improve_segment() -> None:
    p = _pipeline([(0.0, 4.0), (8.0, 16.0)])

    segs = p._pick_segments([(0.2, 3.8), (8.2, 15.8)], {}, {}, {2: 7.5})

    by_turn = {s["turn"]: s for s in segs}
    assert (by_turn[2]["type"], by_turn[2]["reason"], by_turn[2]["quiet_db"]) == ("IMPROVE", "quiet", 7.5)
    assert (by_turn[1]["type"], by_turn[1]["reason"], by_turn[1]["quiet_db"]) == ("GOOD", "clean", None)


@pytest.mark.parametrize(
    ("drop_db", "kind", "reason"),
    [(7.5, "GOOD", "voice"), (12.0, "IMPROVE", "quiet")],
)
def test_stable_voice_and_a_quiet_turn_are_added_up(drop_db: float, kind: str, reason: str) -> None:
    """안정된 목소리 +1.5 와 작아진 정도(÷6)를 더해 큰 쪽이 그 구간의 종류가 된다."""
    p = _pipeline([(0.0, 8.0)])

    (seg,) = p._pick_segments([], {1: 2.0}, {}, {1: drop_db})

    assert (seg["type"], seg["reason"]) == (kind, reason)


def test_ambiguous_voice_with_a_quiet_turn_is_still_picked() -> None:
    p = _pipeline([(0.0, 8.0)])

    (seg,) = p._pick_segments([], {1: 4.0}, {}, {1: 7.5})

    assert (seg["type"], seg["reason"]) == ("IMPROVE", "quiet")


def test_shaky_voice_and_pause_stay_the_reason_and_keep_the_quiet_mark() -> None:
    p = _pipeline([(0.0, 8.0), (10.0, 18.0)])

    shaky, paused = p._pick_segments([], {1: 8.0}, {2: [1.8]}, {1: 7.5, 2: 6.2})

    assert (shaky["reason"], shaky["quiet_db"]) == ("voice", 7.5)
    assert (paused["reason"], paused["quiet_db"]) == ("pause", 6.2)


def test_quiet_turn_without_transcript_is_ignored() -> None:
    p = _pipeline([(0.0, 8.0), (10.0, 14.0)], ["", "네 맞아요"])

    segs = p._pick_segments([(10.2, 13.8)], {}, {}, {1: 9.0})

    assert [s["turn"] for s in segs] == [2]


def test_without_loudness_data_selection_is_unchanged() -> None:
    p = _pipeline([(0.0, 4.0), (8.0, 16.0)])
    candidates, avti, pauses = [(0.2, 3.8), (8.2, 15.8)], {1: 2.0}, {2: [2.1]}

    with_none = p._pick_segments(candidates, avti, pauses)
    with_empty = p._pick_segments(candidates, avti, pauses, {})

    assert [(s["turn"], s["type"], s["reason"]) for s in with_none] == [
        (1, "GOOD", "voice"), (2, "IMPROVE", "pause"),
    ]
    assert with_none == with_empty


# --- 녹음과 떨림 분석 결과에서 턴별 크기 뽑기 ------------------------------


class _Tremor:
    def __init__(self, voiced_spans) -> None:
        self.voiced_spans = voiced_spans


def test_quiet_turns_are_keyed_by_turn_number() -> None:
    spans = [(0.5, 3.5), (5.5, 8.5), (10.5, 13.5)]
    p = _pipeline([(0.0, 4.0), (5.0, 9.0), (10.0, 14.0)])
    pcm = _pcm(15, [(0.5, 3.5, 0.5), (5.5, 8.5, 0.2), (10.5, 13.5, 0.5)])

    assert p._quiet_by_turn(pcm, _Tremor(spans)) == {2: 8.0}


def test_evenly_spoken_call_has_no_quiet_turn() -> None:
    spans = [(0.5, 3.5), (5.5, 8.5), (10.5, 13.5)]
    p = _pipeline([(0.0, 4.0), (5.0, 9.0), (10.0, 14.0)])
    pcm = _pcm(15, [(0.5, 3.5, 0.5), (5.5, 8.5, 0.4), (10.5, 13.5, 0.5)])

    assert p._quiet_by_turn(pcm, _Tremor(spans)) == {}


def test_missing_recording_or_analysis_means_no_quiet_turn() -> None:
    p = _pipeline([(0.0, 4.0)])

    assert p._quiet_by_turn(b"", _Tremor([(0.5, 3.5)])) == {}
    assert p._quiet_by_turn(_pcm(5, [(0.5, 3.5, 0.5)]), None) == {}
    assert p._quiet_by_turn(_pcm(5, [(0.5, 3.5, 0.5)]), _Tremor([])) == {}


def test_broken_measurement_does_not_stop_the_feedback(monkeypatch) -> None:
    """크기를 못 재도 통화 종료는 그대로 진행돼야 한다."""
    def _boom(*_args):
        raise ValueError("bad buffer")

    monkeypatch.setattr("app.services.pipeline.turn_levels", _boom)
    p = _pipeline([(0.0, 4.0)])

    assert p._quiet_by_turn(_pcm(5, [(0.5, 3.5, 0.5)]), _Tremor([(0.5, 3.5)])) == {}


# --- AI 에 넘기는 근거 ------------------------------------------------------


def test_items_say_whether_the_turn_was_quiet() -> None:
    p = _pipeline([(0.0, 8.0), (10.0, 18.0)])
    quiet = {"turn": 1, "start": 0.0, "end": 8.0, "type": "IMPROVE",
             "reason": "quiet", "avti": None, "pauses": [], "quiet_db": 7.5}
    normal = {"turn": 2, "start": 10.0, "end": 18.0, "type": "GOOD",
              "reason": "clean", "avti": None, "pauses": [], "quiet_db": None}

    first, second = p._feedback_items([quiet, normal])

    assert (first["quiet"], second["quiet"]) == (True, False)


@pytest.mark.parametrize(
    ("item", "line"),
    [
        ({"reason": "quiet", "quiet": True}, "   짚을 점: 다른 말보다 목소리가 작아짐"),
        ({"reason": "voice", "quiet": True}, "   짚을 점: 목소리가 흔들림, 다른 말보다 목소리가 작아짐"),
        ({"reason": "pause", "pause_count": 1, "pause_longest_sec": 2.1, "quiet": True},
         "   짚을 점: 말 중간에 1번 멈춤 (최대 약 2초), 다른 말보다 목소리가 작아짐"),
        ({"reason": "quiet"}, "   짚을 점: 다른 말보다 목소리가 작아짐"),
        ({"reason": "voice", "quiet": False}, "   짚을 점: 목소리가 흔들림"),
    ],
)
def test_improve_segment_states_the_quiet_voice(item: dict, line: str) -> None:
    lines = LLMClient._segment_lines([{"type": "IMPROVE", "utterance": "네", "avti": None, **item}])

    assert (lines[0], lines[1], lines[-1]) == ("1. [짚어줄 구간]", line, '   내가 한 말: "네"')


def test_how_much_quieter_is_not_given_to_the_writer() -> None:
    """측정값을 주면 글에 숫자로 샐 수 있다. 작아졌다는 사실만 넘긴다."""
    (_, line, _) = LLMClient._segment_lines(
        [{"type": "IMPROVE", "utterance": "네", "avti": None, "reason": "quiet", "quiet": True}]
    )

    assert not any(ch.isdigit() for ch in line)


# --- 기본 문구와 밖으로 나가는 모양 ----------------------------------------


def test_quiet_segment_has_its_own_fallback_text() -> None:
    """작아서 고른 구간에 '말이 흔들린 부분' 이나 '멈춘 부분' 이라고 쓰면 고른 이유와 어긋난다."""
    title, content = _segment_fallback("IMPROVE", "quiet")

    assert "작" in title and "크게" in content
    assert "흔들" not in title + content and "멈춘" not in title + content
    assert is_safe(title, content)


class _SilentLLM:
    async def segment_feedback(self, items, **kwargs):
        self.items = items
        return []


async def test_failed_writing_falls_back_by_reason_and_hides_loudness_data() -> None:
    p = _pipeline([(0.0, 8.0)], ["토요일 저녁 여섯 시에 네 명이요"])
    p._llm = _SilentLLM()
    segments = p._pick_segments([], {}, {}, {1: 7.5})

    await p._write_segment_feedback(segments)

    (seg,) = segments
    assert seg["title"] == "목소리가 작아졌어요"
    assert set(seg) == {"start", "end", "type", "title", "content", "good_point"}


async def test_teardown_picks_the_quiet_turn_and_tells_the_writer(monkeypatch) -> None:
    """녹음에서 턴별 크기를 재고, 작아진 턴을 짚어줄 구간으로 올려 AI 에 근거를 넘긴다."""
    from tests.unit.test_avti_save import _pipeline as teardown_pipeline

    async def _no_avti(**_kwargs) -> dict:
        return {}

    spring: dict = {}

    async def _capture(session_id, **kwargs):
        spring.update(kwargs)
        return True

    monkeypatch.setattr("app.services.pipeline.run_avti", _no_avti)

    spans = [(1.0, 4.0), (6.0, 9.0), (11.0, 14.0)]
    p = teardown_pipeline(seconds=16, turns=[(0.5, 4.5), (5.5, 9.5), (10.5, 14.5)], spans=spans)
    p._tremor_buf = bytearray(_pcm(16, [(1.0, 4.0, 0.5), (6.0, 9.0, 0.5), (11.0, 14.0, 0.2)]))
    p._user_turn_texts = ["예약하려고 전화드렸어요", "토요일 저녁 여섯 시요", "네 명이요, 창가 자리로 부탁드려요"]
    p._spring.notify_session_closed = _capture

    await p._teardown()

    # 이 가짜 분석은 '떨림 없는 구간'을 주지 않아 다른 턴은 고를 근거가 없고, 작게 말한 3번 턴만 뽑힌다.
    (quiet,) = p._llm.last_items
    assert (quiet["type"], quiet["reason"], quiet["quiet"]) == ("IMPROVE", "quiet", True)
    assert quiet["utterance"] == "네 명이요, 창가 자리로 부탁드려요"
    assert [seg["type"] for seg in spring["good_segments"]] == ["IMPROVE"]
    for seg in spring["good_segments"]:
        assert set(seg) == {"start", "end", "type", "title", "content", "good_point"}
