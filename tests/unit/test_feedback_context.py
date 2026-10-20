"""구간 피드백에 붙는 통화 맥락 — 통화 상대와, 그 구간 직전에 상대가 한 말."""

from types import SimpleNamespace

from app.services.llm import LLMClient
from app.services.pipeline import VoicePipeline

SCENARIO = {"title": "식당 예약", "aiRole": "레스토랑 직원"}
ASK_DATE = "네, 예약 도와드리겠습니다. 원하시는 날짜와 시간이 어떻게 되세요?"
ASK_PEOPLE = "네, 토요일 저녁 6시 가능합니다. 몇 분이 방문하실 예정인가요?"


def _pipeline(turns, texts, *, prompts=None, scenario=SCENARIO) -> VoicePipeline:
    p = VoicePipeline.__new__(VoicePipeline)
    p._session_id = "test-session"
    p._session = {"type": "SCENARIO", "scenario": scenario}
    p._user_turn_intervals = turns
    p._user_turn_texts = texts
    if prompts is not None:
        p._user_turn_prompts = prompts
    return p


def _seg(turn: int | None, start: float, end: float, kind: str = "GOOD") -> dict:
    seg = {"start": start, "end": end, "type": kind, "reason": "clean", "avti": None}
    if turn is not None:
        seg["turn"] = turn
    return seg


def _three_turns(prompts=("", ASK_DATE, ASK_PEOPLE), **overrides) -> VoicePipeline:
    return _pipeline(
        [(0.0, 4.0), (8.0, 12.0), (16.0, 19.0)],
        ["예약하려고 전화드렸어요", "이번 주 토요일 저녁 6시요", "어른 4명이요"],
        prompts=None if prompts is None else list(prompts),
        **overrides,
    )


# --- 턴이 확정되는 순간의 기록 ---------------------------------------------


def _listening(history: list[dict]) -> VoicePipeline:
    p = VoicePipeline.__new__(VoicePipeline)
    p._history = history
    p._tremor_buf = bytearray(b"\x00\x00" * 16000 * 3)
    p._turn_open_at = 1.0
    p._user_turn_intervals, p._user_turn_texts, p._user_turn_prompts = [], [], []
    return p


def test_turn_close_records_what_the_other_side_just_said() -> None:
    history = [{"role": "user", "text": "예약하려고요"}, {"role": "assistant", "text": ASK_DATE}]
    p = _listening(history)

    p._close_user_turn("토요일 저녁 6시요")

    assert p._user_turn_prompts == [ASK_DATE]
    assert p._user_turn_texts == ["토요일 저녁 6시요"]


def test_first_turn_has_no_previous_line() -> None:
    p = _listening([])

    p._close_user_turn("여보세요")

    assert p._user_turn_prompts == [""]


def test_only_the_immediately_preceding_line_counts() -> None:
    """바로 앞에 상대가 말하지 않았으면, 더 예전에 한 말을 끌어오지 않는다."""
    history = [{"role": "assistant", "text": ASK_DATE}, {"role": "user", "text": "네"}]
    p = _listening(history)

    p._close_user_turn("토요일이요")

    assert p._user_turn_prompts == [""]


def test_context_stays_aligned_with_turns_even_without_text() -> None:
    """글이 없는 턴도 한 칸을 차지해서 뒤 턴이 밀리지 않는다."""
    p = _listening([{"role": "assistant", "text": ASK_DATE}])

    p._close_user_turn("")
    p._turn_open_at = 5.0
    p._history.append({"role": "assistant", "text": ASK_PEOPLE})
    p._close_user_turn("어른 4명이요")

    assert len(p._user_turn_intervals) == len(p._user_turn_prompts) == 2
    assert p._user_turn_prompts == [ASK_DATE, ASK_PEOPLE]


# --- AI 에 넘길 재료 -------------------------------------------------------


def test_items_carry_the_previous_line() -> None:
    (item,) = _three_turns()._feedback_items([_seg(2, 8.0, 12.0)])

    assert item["utterance"] == "이번 주 토요일 저녁 6시요"
    assert item["prev_ai"] == ASK_DATE


def test_silent_turn_does_not_shift_the_context() -> None:
    p = _pipeline(
        [(0.0, 4.0), (8.0, 12.0), (16.0, 19.0)],
        ["예약하려고 전화드렸어요", "", "어른 4명이요"],
        prompts=["", ASK_DATE, ASK_PEOPLE],
    )

    (item,) = p._feedback_items([_seg(3, 16.0, 19.0)])

    assert item["utterance"] == "어른 4명이요"
    assert item["prev_ai"] == ASK_PEOPLE


def test_segment_without_a_turn_number_finds_its_turn_by_overlap() -> None:
    """최후 보루로 뽑힌 구간에는 턴 번호가 없다."""
    (item,) = _three_turns()._feedback_items([_seg(None, 8.5, 11.0)])

    assert item["prev_ai"] == ASK_DATE


def test_long_previous_line_is_clipped() -> None:
    p = _three_turns(prompts=("", "가" * 300, ASK_PEOPLE))

    (item,) = p._feedback_items([_seg(2, 8.0, 12.0)])

    assert len(item["prev_ai"]) == 121
    assert item["prev_ai"].endswith("…")


def test_pipeline_without_context_list_gives_empty_context() -> None:
    """맥락을 기록하지 않은 채 만든 객체에서도 재료는 나온다."""
    (item,) = _three_turns(prompts=None)._feedback_items([_seg(2, 8.0, 12.0)])

    assert item["utterance"] == "이번 주 토요일 저녁 6시요"
    assert item["prev_ai"] == ""


def test_segment_outside_every_turn_gets_no_context() -> None:
    (item,) = _three_turns()._feedback_items([_seg(None, 100.0, 101.0)])

    assert (item["utterance"], item["prev_ai"]) == ("", "")


# --- AI 가 받는 글 ---------------------------------------------------------


def test_previous_line_sits_under_each_number() -> None:
    items = _three_turns()._feedback_items(
        [_seg(2, 8.0, 12.0), {**_seg(3, 16.0, 19.0, "IMPROVE"), "avti": 5.4, "reason": "voice"}]
    )

    assert LLMClient._segment_lines(items) == [
        "1. [칭찬할 구간]",
        f'   상대가 한 말: "{ASK_DATE}"',
        '   내가 한 말: "이번 주 토요일 저녁 6시요"',
        "2. [짚어줄 구간, 떨림 5.4/10]",
        f'   상대가 한 말: "{ASK_PEOPLE}"',
        "   짚을 점: 목소리가 흔들림",
        '   내가 한 말: "어른 4명이요"',
    ]


def test_missing_previous_line_is_left_out() -> None:
    lines = LLMClient._segment_lines([{"type": "GOOD", "utterance": "여보세요", "avti": None}])

    assert lines == ["1. [칭찬할 구간]", '   내가 한 말: "여보세요"']


def test_unrecognized_utterance_is_marked() -> None:
    lines = LLMClient._segment_lines([{"type": "IMPROVE", "utterance": " ", "avti": None}])

    assert lines[-1] == '   내가 한 말: "(발화 인식 안 됨)"'


async def test_counterpart_and_previous_line_reach_the_model() -> None:
    sent: dict = {}

    class _Models:
        async def generate_content(self, **kwargs):
            sent.update(kwargs)
            return SimpleNamespace(text="1. 날짜를 바로 답했어요 | 묻는 말에 맞게 답했어요.",
                                   usage_metadata=None, candidates=[])

    llm = LLMClient.__new__(LLMClient)
    llm._model = "gemini-3.5-flash-lite"
    llm._client = SimpleNamespace(aio=SimpleNamespace(models=_Models()))

    pairs = await llm.segment_feedback(
        _three_turns()._feedback_items([_seg(2, 8.0, 12.0)]),
        scenario_title="식당 예약", call_target="레스토랑 직원",
    )

    assert pairs == [("날짜를 바로 답했어요", "묻는 말에 맞게 답했어요.")]
    assert "이번 통화 상황: 식당 예약 / 상대는 레스토랑 직원" in sent["config"].system_instruction
    assert f'상대가 한 말: "{ASK_DATE}"' in sent["contents"]


# --- 통화 종료 흐름 --------------------------------------------------------


class _FakeLLM:
    async def segment_feedback(self, items, **kwargs):
        self.items, self.kwargs = items, kwargs
        return [("날짜를 바로 답했어요", "묻는 말에 맞게 답했어요.")] * len(items)


async def test_ai_role_is_sent_as_the_counterpart() -> None:
    """세션에는 통화 상대가 aiRole 로 온다. callTarget 은 세션에 없는 칸이다."""
    p = _three_turns()
    p._llm = _FakeLLM()

    await p._write_segment_feedback([_seg(2, 8.0, 12.0)])

    assert p._llm.kwargs["call_target"] == "레스토랑 직원"
    assert p._llm.kwargs["scenario_title"] == "식당 예약"
    assert p._llm.items[0]["prev_ai"] == ASK_DATE


async def test_context_never_reaches_spring_or_the_app() -> None:
    """맥락은 AI 에 주는 재료일 뿐, 밖으로 나가는 구간에는 실리지 않는다."""
    p = _three_turns()
    p._llm = _FakeLLM()
    segments = [_seg(2, 8.0, 12.0), _seg(3, 16.0, 19.0, "IMPROVE")]

    await p._write_segment_feedback(segments)

    for seg in segments:
        assert set(seg) == {"start", "end", "type", "title", "content"}


async def test_teardown_gives_context_to_the_writer_and_keeps_spring_payload(monkeypatch) -> None:
    """통화 종료 흐름 전체에서도 맥락은 AI 까지만 가고, Spring 에 가는 구간 모양은 그대로다."""
    from tests.unit.test_avti_save import _pipeline as teardown_pipeline

    async def _no_avti(**_kwargs) -> dict:
        return {}

    spring: dict = {}

    async def _capture(session_id, **kwargs):
        spring.update(kwargs)
        return True

    monkeypatch.setattr("app.services.pipeline.run_avti", _no_avti)

    p = teardown_pipeline(seconds=20, turns=[(1.0, 5.0), (8.0, 12.0)], spans=[(1.2, 4.8), (8.2, 11.8)])
    p._session = {"type": "SCENARIO", "scenario": SCENARIO}
    p._user_turn_texts = ["예약하려고 전화드렸어요", "이번 주 토요일 저녁 6시요"]
    p._user_turn_prompts = ["", ASK_DATE]
    p._spring.notify_session_closed = _capture

    await p._teardown()

    assert [item["prev_ai"] for item in p._llm.last_items] == ["", ASK_DATE]
    assert p._llm.last_kwargs["call_target"] == "레스토랑 직원"
    assert spring["good_segments"]
    for seg in spring["good_segments"]:
        assert set(seg) == {"start", "end", "type", "title", "content"}
