import asyncio

import httpx
import pytest

from app.core.audio_stats import BYTES_PER_MS
from app.core.metrics import now_ms
from app.schemas.llm import AiEmotion, LLMEvent, LLMEventType
from app.services import qwen_tts as qwen_mod
from app.services.pipeline import _State, _TurnTimings
from app.services.qwen_tts import (
    QwenRealtimeTTSClient,
    QwenTTSRunawayError,
    QwenTTSUnavailableError,
    runaway_cap_ms,
)
from tests.unit.test_pipeline_qwen_engine import (
    _capture_metrics,
    _FakeELClient,
    _FakeQwenClient,
    _HappyLLM,
    _make_pipeline,
)
from tests.unit.test_qwen_realtime_tts import _settings
from tests.unit.test_turn_fallback import _FALLBACK_MARKER, _LogWS

_BABBLE_CHUNK = b"\x01\x00" * 8000  # 500ms


class _ClosableLogWS(_LogWS):
    async def close(self, code: int = 1000) -> None:
        self.events.append(("close", code))


def _babble_transport(total_ms: int) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        async def _chunks():
            for _ in range(total_ms // 100):
                yield b"\x00\x01" * (BYTES_PER_MS // 2 * 100)

        return httpx.Response(200, content=_chunks())

    return httpx.MockTransport(handler)


async def _stream_one(
    text: str, transport: httpx.MockTransport, got: list[bytes] | None = None
) -> list[bytes]:
    session = await QwenRealtimeTTSClient(_settings(), transport=transport).open()
    await session.begin()
    got = [] if got is None else got
    try:
        async def source():
            yield text

        async for chunk in session.stream(source()):
            got.append(chunk)
    finally:
        await session.aclose()
    return got


def test_cap_counts_stripped_text() -> None:
    assert runaway_cap_ms("안녕하세요. ") == runaway_cap_ms("안녕하세요.")


async def test_sentence_over_cap_raises_runaway_without_yielding_past_cap(monkeypatch) -> None:
    monkeypatch.setattr(qwen_mod, "_RUNAWAY_FLOOR_MS", 1000)
    monkeypatch.setattr(qwen_mod, "_RUNAWAY_MS_PER_CHAR", 100)
    got: list[bytes] = []

    with pytest.raises(QwenTTSRunawayError) as info:
        await asyncio.wait_for(
            _stream_one("안녕하세요.", _babble_transport(60_000), got), timeout=2.0
        )

    exc = info.value
    assert (exc.chars, exc.cap_ms, exc.worker) == (6, 1600, "http://tts.test")
    assert exc.pcm_bytes > 1600 * BYTES_PER_MS
    assert sum(map(len, got)) == 1600 * BYTES_PER_MS, "상한까지만 내보내고 넘긴 청크는 내보내면 안 된다"


async def test_sentence_under_cap_passes_through() -> None:
    out = await asyncio.wait_for(_stream_one("안녕하세요.", _babble_transport(1_000)), timeout=2.0)
    assert sum(map(len, out)) == 1000 * BYTES_PER_MS


async def test_ordinary_failure_is_not_runaway() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    with pytest.raises(QwenTTSUnavailableError) as info:
        await asyncio.wait_for(_stream_one("안녕하세요.", httpx.MockTransport(handler)), timeout=2.0)
    assert not isinstance(info.value, QwenTTSRunawayError)


class _QwenSession:
    def __init__(self, error: Exception, chunks: int) -> None:
        self._error, self._chunks = error, chunks
        self.chars_sent = 0

    async def begin(self, emotion=AiEmotion.NEUTRAL) -> None:
        pass

    async def stream(self, text_source):
        async for t in text_source:
            self.chars_sent += len(t)
        for _ in range(self._chunks):
            yield _BABBLE_CHUNK
        raise self._error

    async def aclose(self) -> None:
        pass


class _QwenClient(_FakeQwenClient):
    def __init__(self, error: Exception, chunks: int = 5) -> None:
        super().__init__()
        self._error, self._chunks = error, chunks

    async def open(self, voice_id=None):
        self.open_calls += 1
        return _QwenSession(self._error, self._chunks)


def _runaway() -> QwenTTSRunawayError:
    return QwenTTSRunawayError(
        worker="http://w:1", chars=13, pcm_bytes=80000, cap_ms=2000, wall_ms=900.0
    )


def _pipeline(llm, qwen):
    p = _make_pipeline(llm, _FakeELClient(), qwen=qwen)
    p._ws = _ClosableLogWS()
    return p


def _kinds(p) -> list[str]:
    return [k if k != "json" else payload.get("type") for k, payload in p._ws.events]


async def test_runaway_interrupts_after_babble_and_before_fallback(monkeypatch) -> None:
    records = _capture_metrics(monkeypatch)
    qwen = _QwenClient(_runaway())
    p = _pipeline(_HappyLLM(), qwen)

    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)

    kinds = _kinds(p)
    assert kinds.count("interrupt") == 1
    i = kinds.index("interrupt")
    assert "pcm" in kinds[:i], "옹알이가 먼저 나가 있어야 버릴 게 있다"
    assert kinds[i + 1] == "emotion", "interrupt 와 폴백 emotion 사이에 앱이 마이크를 열면 안 된다"
    after = kinds[i + 1:]
    assert "pcm" in after
    assert after[-1] == "speaking_end"
    fallback_pcm = [d for k, d in p._ws.events[i + 1:] if k == "pcm"]
    assert fallback_pcm == [b"\xee\xee"], "폴백 멘트는 ElevenLabs 에서 나와야 한다"
    assert any(_FALLBACK_MARKER in f.get("text", "") for f in p._ws.frames_of("transcript"))
    assert p._state == _State.LISTENING

    assert qwen.released is True and p._qwen_tts is None
    metrics = dict(records)
    assert metrics["realtime_tts_switch"]["reason"] == "synth_runaway"
    runaway = metrics["realtime_tts_runaway"]
    assert (runaway["worker"], runaway["chars"], runaway["cap_ms"]) == ("http://w:1", 13, 2000)
    assert runaway["sent_audio_ms"] == 2500.0
    assert metrics["voice_turn"]["tts_failed"] is True


async def test_runaway_interrupts_even_when_fallback_tts_fails(monkeypatch) -> None:
    class _BrokenEL:
        async def open(self, voice_id=None):
            raise RuntimeError("EL down")

    _capture_metrics(monkeypatch)
    p = _make_pipeline(_HappyLLM(), _BrokenEL(), qwen=_QwenClient(_runaway()))
    p._ws = _ClosableLogWS()

    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)

    kinds = _kinds(p)
    assert kinds.count("interrupt") == 1, "폴백이 실패해도 쌓인 옹알이는 버려야 한다"
    assert kinds.index("interrupt") < len(kinds) - 1 and kinds[-1] == "speaking_end"


async def test_ordinary_failure_after_partial_audio_does_not_interrupt(monkeypatch) -> None:
    records = _capture_metrics(monkeypatch)
    p = _pipeline(_HappyLLM(), _QwenClient(QwenTTSUnavailableError("ReadTimeout")))

    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)

    assert "interrupt" not in _kinds(p), "정상 부분 음성일 수 있는 실패는 자르지 않는다"
    assert dict(records)["realtime_tts_switch"]["reason"] == "synth_failed"
    assert "realtime_tts_runaway" not in dict(records)


async def test_runaway_before_any_audio_does_not_interrupt(monkeypatch) -> None:
    _capture_metrics(monkeypatch)
    p = _pipeline(_HappyLLM(), _QwenClient(_runaway(), chunks=0))

    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)

    assert "interrupt" not in _kinds(p), "앱에 보낸 게 없으면 재생기를 리셋할 이유가 없다"


async def test_fallback_turn_does_not_inherit_dropped_babble(monkeypatch) -> None:
    _capture_metrics(monkeypatch)
    p = _pipeline(_HappyLLM(), _QwenClient(_runaway(), chunks=8))

    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)

    remaining = p._playback.expected_end_ms - now_ms()
    assert remaining < 500, f"앱이 버린 옹알이 길이가 폴백 턴에 이월됐다: {remaining:.0f}ms"


async def test_runaway_on_closing_turn_does_not_wait_for_dropped_audio(monkeypatch) -> None:
    class _EndLLM:
        async def stream(self, ctx):
            yield LLMEvent(type=LLMEventType.EMOTION_RESOLVED, emotion=AiEmotion.NEUTRAL)
            yield LLMEvent(type=LLMEventType.TEXT_DELTA, text="안녕히 계세요.")
            yield LLMEvent(type=LLMEventType.END_CALL)

    _capture_metrics(monkeypatch)
    p = _pipeline(_EndLLM(), _QwenClient(_runaway(), chunks=8))
    p._playback_ack_supported = True

    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)

    assert p._closing.is_set()
    assert "interrupt" in _kinds(p)
