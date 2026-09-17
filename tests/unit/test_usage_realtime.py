import asyncio
import inspect
import json
import logging
from types import SimpleNamespace

import pytest

from app.core.usage import PCM_BYTES_PER_SECOND, SessionUsage
from app.schemas.llm import AiEmotion, LLMEvent, LLMEventType
from app.services import stt as stt_mod
from app.services.pipeline import _State, _TurnTimings
from app.services.qwen_tts import QwenRealtimeTTSClient
from app.services.tts import TTSSession
from tests.unit.test_avti_save import _pipeline as _closing_pipeline
from tests.unit.test_llm_prompt_cache import _Chunk, _ctx, _make_llm, _Usage
from tests.unit.test_llm_thinking_by_model import _llm
from tests.unit.test_pipeline_audio_stats import _make_pipeline, _metric, _TTSClient
from tests.unit.test_qwen_realtime_tts import _settings as _qwen_settings
from tests.unit.test_qwen_realtime_tts import _stream_transport


def _metrics(caplog, name: str):
    return [r for r in caplog.records
            if r.name == "app.metrics" and getattr(r, "metric", None) == name]



def test_session_usage_sums_and_treats_none_as_zero() -> None:
    acc = SessionUsage()
    acc.add_llm_turn({"prompt": 1000, "cached": None, "output": 40, "thought": None})
    acc.add_llm_turn({"prompt": 1200, "cached": 800, "output": 35, "thought": 12})
    acc.add_tts("eleven", chars=30, pcm_bytes=64000)
    acc.add_tts("qwen", chars=12, pcm_bytes=32000)
    acc.add_tts("eleven", chars=5, pcm_bytes=3200)
    acc.add_feedback(SimpleNamespace(
        prompt_token_count=500, candidates_token_count=120,
        cached_content_token_count=None, thoughts_token_count=None,
    ))
    acc.add_feedback(None)
    acc.stt_bytes = 3 * PCM_BYTES_PER_SECOND

    m = acc.as_metrics()
    assert m["turns"] == 2
    assert m["llm_prompt_tokens"] == 2200 and m["llm_cached_tokens"] == 800
    assert m["llm_output_tokens"] == 75 and m["llm_thought_tokens"] == 12
    assert m["feedback_prompt_tokens"] == 500 and m["feedback_output_tokens"] == 120
    assert m["feedback_cached_tokens"] == 0 and m["feedback_thought_tokens"] == 0
    assert m["tts_chars_eleven"] == 35 and m["tts_chars_qwen"] == 12
    assert m["tts_audio_sec_eleven"] == round(67200 / PCM_BYTES_PER_SECOND, 3)
    assert m["tts_audio_sec_qwen"] == 1.0
    assert m["stt_sec"] == 3.0


def test_session_usage_always_emits_both_engine_columns() -> None:
    m = SessionUsage().as_metrics()
    for key in ("tts_chars_eleven", "tts_chars_qwen", "tts_audio_sec_eleven", "tts_audio_sec_qwen"):
        assert key in m and m[key] == 0
    acc = SessionUsage()
    acc.add_tts("other", chars=3)
    assert acc.as_metrics()["tts_chars_other"] == 3


class _FullUsage(_Usage):
    def __init__(self, prompt, cached, output, thought) -> None:
        super().__init__(prompt, cached)
        self.candidates_token_count = output
        self.thoughts_token_count = thought


@pytest.mark.asyncio
async def test_turn_end_carries_output_and_thought_tokens() -> None:
    chunks = [
        _Chunk("[EMOTION:NEUTRAL]\n안녕"),
        _Chunk("하세요", usage=_FullUsage(1500, 1200, 42, 7)),
    ]
    events = [ev async for ev in _make_llm(chunks).stream(_ctx())]
    turn_end = next(ev for ev in events if ev.type == LLMEventType.TURN_END)
    assert turn_end.output_tokens == 42
    assert turn_end.thought_tokens == 7


@pytest.mark.asyncio
async def test_turn_end_output_tokens_none_when_sdk_omits_them() -> None:
    chunks = [_Chunk("[EMOTION:NEUTRAL]\n네", usage=_Usage(prompt=900, cached=None))]
    events = [ev async for ev in _make_llm(chunks).stream(_ctx())]
    turn_end = next(ev for ev in events if ev.type == LLMEventType.TURN_END)
    assert turn_end.output_tokens is None and turn_end.thought_tokens is None


def _feedback_llm(usage_metadata):
    class _Models:
        async def generate_content(self, **kwargs):
            return type("R", (), {
                "text": "1. 잘했어요 | 또박또박 말했어요.",
                "usage_metadata": usage_metadata,
            })()

    llm = _llm("gemini-3.5-flash-lite")
    llm._client = type("C", (), {"aio": type("A", (), {"models": _Models()})()})()
    return llm


@pytest.mark.asyncio
async def test_segment_feedback_reports_usage_to_callback() -> None:
    usage = SimpleNamespace(prompt_token_count=321, candidates_token_count=45,
                            cached_content_token_count=None, thoughts_token_count=None)
    seen: list = []
    pairs = await _feedback_llm(usage).segment_feedback(
        [{"type": "GOOD", "utterance": "안녕하세요", "avti": None}], on_usage=seen.append
    )
    assert pairs[0][0]
    assert seen == [usage]


@pytest.mark.asyncio
async def test_segment_feedback_survives_a_broken_usage_callback() -> None:
    def _boom(_usage):
        raise RuntimeError("집계 버그")

    pairs = await _feedback_llm(None).segment_feedback(
        [{"type": "GOOD", "utterance": "안녕하세요", "avti": None}], on_usage=_boom
    )
    assert pairs[0][0], "콜백 예외가 피드백을 없애면 안 된다"


@pytest.mark.asyncio
async def test_elevenlabs_session_counts_chars_sent() -> None:
    class _WS:
        def __init__(self) -> None:
            self.sent: list[dict] = []

        async def send(self, raw: str) -> None:
            self.sent.append(json.loads(raw))

    ws = _WS()
    session = TTSSession(ws, {"stability": 0.5}, difficulty=None)
    await session.begin(AiEmotion.NEUTRAL)

    async def source():
        yield "안녕하세요. 반갑"
        yield "습니다"

    await session._send_text(source())

    texts = [m["text"] for m in ws.sent if m.get("text") not in (" ", "")]
    assert texts, "문장 payload 가 전송돼야 한다"
    assert session.chars_sent == sum(len(t) for t in texts)


@pytest.mark.asyncio
async def test_qwen_session_counts_chars_sent() -> None:
    calls: list[dict] = []
    client = QwenRealtimeTTSClient(_qwen_settings(), transport=_stream_transport(calls))
    session = await client.open("voice")
    try:
        await session.begin()

        async def source():
            yield "안녕하세요. 반갑"
            yield "습니다"

        [chunk async for chunk in session.stream(source())]
    finally:
        await session.aclose()

    assert calls
    assert session.chars_sent == sum(len(c["text"]) for c in calls)


def test_stt_clients_start_their_byte_counter_at_zero() -> None:
    for cls in (stt_mod.GoogleSTTClient, stt_mod.GeminiLiveSTTClient):
        assert "self.audio_bytes_sent = 0" in inspect.getsource(cls.__init__), cls.__name__


@pytest.mark.asyncio
async def test_google_stt_counts_audio_bytes_sent() -> None:
    client = stt_mod.GoogleSTTClient.__new__(stt_mod.GoogleSTTClient)
    client.audio_bytes_sent = 0
    client._build_config_request = lambda: object()
    queue: asyncio.Queue = asyncio.Queue()
    for chunk in (b"\x00" * 3200, b"\x00" * 1600, stt_mod.AUDIO_EOS):
        queue.put_nowait(chunk)

    requests = [r async for r in client._request_generator(queue, first_chunk=b"\x00" * 320)]

    assert len(requests) == 4
    assert client.audio_bytes_sent == 320 + 3200 + 1600


@pytest.mark.asyncio
async def test_gemini_stt_counts_audio_bytes_sent() -> None:
    class _Sess:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def send_realtime_input(self, **kwargs) -> None:
            self.calls.append(kwargs)

    client = stt_mod.GeminiLiveSTTClient.__new__(stt_mod.GeminiLiveSTTClient)
    client._mime_type = "audio/pcm;rate=16000"
    client.audio_bytes_sent = 0
    sess = _Sess()
    queue: asyncio.Queue = asyncio.Queue()
    for chunk in (b"\x00" * 3200, b"\x00" * 1600, stt_mod.AUDIO_EOS):
        queue.put_nowait(chunk)

    await client._send_audio(sess, queue, first_chunk=b"\x00" * 320)

    assert client.audio_bytes_sent == 320 + 3200 + 1600
    assert sess.calls[-1] == {"audio_stream_end": True}


class _CountingTTSSession:
    def __init__(self) -> None:
        self.chars_sent = 0

    async def begin(self, emotion) -> None:
        pass

    async def stream(self, text_source):
        async for chunk in text_source:
            self.chars_sent += len(chunk)
        yield b"\x00" * 3200

    async def aclose(self) -> None:
        pass


_AI_TEXT = "네, 예약 도와드릴게요."


class _TokenLLM:
    async def stream(self, ctx):
        yield LLMEvent(type=LLMEventType.EMOTION_RESOLVED, emotion=AiEmotion.NEUTRAL)
        yield LLMEvent(type=LLMEventType.TEXT_DELTA, text=_AI_TEXT)
        yield LLMEvent(
            type=LLMEventType.TURN_END,
            prompt_tokens=1500, cached_tokens=900, output_tokens=30, thought_tokens=None,
        )


@pytest.mark.asyncio
async def test_voice_turn_reports_tokens_and_chars_and_accumulates(caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    p = _make_pipeline(_TokenLLM(), _TTSClient(_CountingTTSSession()))

    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)

    rec = _metric(caplog, "voice_turn")
    assert rec.llm_prompt_tokens == 1500 and rec.llm_cached_tokens == 900
    assert rec.llm_output_tokens == 30
    assert rec.llm_thought_tokens is None
    assert rec.tts_chars == len(_AI_TEXT)

    m = p._usage.as_metrics()
    assert m["turns"] == 1
    assert m["llm_prompt_tokens"] == 1500 and m["llm_cached_tokens"] == 900
    assert m["llm_output_tokens"] == 30 and m["llm_thought_tokens"] == 0
    assert m["tts_chars_eleven"] == len(_AI_TEXT) and m["tts_chars_qwen"] == 0
    assert m["tts_audio_sec_eleven"] == 0.1


@pytest.mark.asyncio
async def test_turn_accounting_failure_does_not_break_the_turn(caplog, monkeypatch) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")

    def _boom(self, usage):
        raise RuntimeError("집계 버그")

    monkeypatch.setattr(SessionUsage, "add_llm_turn", _boom)
    p = _make_pipeline(_TokenLLM(), _TTSClient(_CountingTTSSession()))

    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)

    assert _metric(caplog, "voice_turn").tts_chars == len(_AI_TEXT)
    assert p._state == _State.LISTENING


async def _no_avti(**kwargs):
    return {}


def _closing(monkeypatch, *, seconds: float) -> object:
    monkeypatch.setattr("app.services.pipeline.run_avti", _no_avti)
    p = _closing_pipeline(seconds=seconds, turns=[(1.0, 3.0)], spans=[(0.0, 3.0)])
    p._session = {"type": "SCENARIO", "userId": 77, "scenarioId": 3}
    p._settings.stt_engine = "chirp"
    p._settings.llm_realtime_model = "gemini-3.5-flash-lite"
    p._settings.elevenlabs_model = "eleven_flash_v2_5"
    return p


@pytest.mark.asyncio
async def test_session_usage_metric_on_close_uses_stt_counter(caplog, monkeypatch) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    p = _closing(monkeypatch, seconds=4)
    p._stt = SimpleNamespace(audio_bytes_sent=2 * PCM_BYTES_PER_SECOND)
    acc = SessionUsage()
    acc.add_llm_turn({"prompt": 100, "cached": 0, "output": 10, "thought": None})
    acc.add_tts("qwen", chars=20, pcm_bytes=PCM_BYTES_PER_SECOND)
    p._usage = acc

    await p._teardown()

    [rec] = _metrics(caplog, "session_usage")
    assert rec.user_id == 77 and rec.scenario_id == 3 and rec.reason == "USER_END"
    assert rec.stt_sec == 2.0, "녹음 4초가 아니라 STT 로 보낸 2초"
    assert rec.turns == 1 and rec.llm_prompt_tokens == 100 and rec.llm_output_tokens == 10
    assert rec.tts_chars_qwen == 20 and rec.tts_audio_sec_qwen == 1.0
    assert rec.tts_chars_eleven == 0
    assert rec.stt_engine == "chirp" and rec.llm_model == "gemini-3.5-flash-lite"
    assert rec.tts_model == "eleven_flash_v2_5"


@pytest.mark.asyncio
async def test_session_usage_falls_back_to_recording_length(caplog, monkeypatch) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    p = _closing(monkeypatch, seconds=4)

    await p._teardown()

    [rec] = _metrics(caplog, "session_usage")
    assert rec.stt_sec == 4.0
    assert rec.turns == 0 and rec.llm_prompt_tokens == 0


@pytest.mark.asyncio
async def test_write_segment_feedback_wires_usage_callback(monkeypatch) -> None:
    p = _closing(monkeypatch, seconds=1)
    p._session = {"scenario": {"title": "t", "callTarget": "c", "callPurpose": "p"}}
    p._utterance_for = lambda start, end: "발화"
    captured: dict = {}

    class _LLM:
        async def segment_feedback(self, items, **kwargs):
            captured.update(kwargs)
            kwargs["on_usage"](SimpleNamespace(
                prompt_token_count=11, candidates_token_count=5,
                cached_content_token_count=None, thoughts_token_count=None,
            ))
            return [("제목", "내용입니다.")] * len(items)

    p._llm = _LLM()
    await p._write_segment_feedback([{"type": "GOOD", "start": 0.0, "end": 1.0}])

    assert "on_usage" in captured
    assert p._usage.feedback_prompt_tokens == 11
    assert p._usage.feedback_output_tokens == 5


@pytest.mark.asyncio
async def test_fallback_speech_counts_chars_and_audio(caplog) -> None:
    from app.services.pipeline import _TURN_FALLBACK_TEXT

    caplog.set_level(logging.INFO, logger="app.metrics")
    p = _make_pipeline(_TokenLLM(), _TTSClient(_CountingTTSSession()))

    await asyncio.wait_for(p._speak_fallback(), timeout=2.0)

    [rec] = _metrics(caplog, "fallback_audio")
    assert rec.chars == len(_TURN_FALLBACK_TEXT)
    m = p._usage.as_metrics()
    assert m["tts_chars_eleven"] == len(_TURN_FALLBACK_TEXT)
    assert m["tts_audio_sec_eleven"] == 0.1
    assert m["turns"] == 0, "폴백은 LLM 턴이 아니다"


@pytest.mark.asyncio
async def test_fallback_without_char_counter_assumes_full_text(caplog) -> None:
    from app.services.pipeline import _TURN_FALLBACK_TEXT
    from tests.unit.test_pipeline_audio_stats import _ThreeChunkTTSSession

    caplog.set_level(logging.INFO, logger="app.metrics")
    p = _make_pipeline(_TokenLLM(), _TTSClient(_ThreeChunkTTSSession()))

    await asyncio.wait_for(p._speak_fallback(), timeout=2.0)

    [rec] = _metrics(caplog, "fallback_audio")
    assert rec.chars == len(_TURN_FALLBACK_TEXT)
