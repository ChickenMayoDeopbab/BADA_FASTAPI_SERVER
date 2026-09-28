import asyncio
from types import SimpleNamespace

from app.schemas.llm import AiEmotion
from app.services import pipeline as pipeline_mod
from app.services.pipeline import _State, _TurnTimings
from app.services.qwen_tts import QwenTTSUnavailableError
from tests.unit.test_pipeline_qwen_engine import (
    _capture_metrics,
    _FakeELClient,
    _FakeQwenClient,
    _HappyLLM,
    _make_pipeline,
)


class _FakeCsmSession:
    accepts_user_turn = True

    def __init__(self, client: "_FakeCsmClient") -> None:
        self._client = client
        self.closed = False

    async def begin(self, emotion=AiEmotion.NEUTRAL, user_turn=None) -> None:
        self._client.begins.append((emotion, user_turn))

    async def stream(self, text_source):
        if self._client.fail:
            raise QwenTTSUnavailableError("worker down")
        async for _ in text_source:
            pass
        yield b"\x0c\x0d"

    async def aclose(self) -> None:
        self.closed = True


class _FakeCsmClient:
    engine_name = "csm"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.released = False
        self.begins: list[tuple] = []

    async def open(self, voice_id=None) -> _FakeCsmSession:
        return _FakeCsmSession(self)

    def release_slot(self) -> None:
        self.released = True


async def test_csm_turn_passes_user_pcm_slice_and_tags_engine(monkeypatch) -> None:
    records = _capture_metrics(monkeypatch)
    csm = _FakeCsmClient()
    p = _make_pipeline(_HappyLLM(), _FakeELClient(), qwen=csm)
    p._tremor_buf = bytearray(b"\x00\x00" * 4000 + b"\x11\x22" * 8000 + b"\x00\x00" * 4000)
    p._user_turn_intervals = [(0.0, 0.1), (0.25, 0.75)]

    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)

    assert p._ws.pcm == [b"\x0c\x0d"]
    assert dict(records)["voice_turn"]["tts_engine"] == "csm"
    assert len(csm.begins) == 1
    emotion, user_turn = csm.begins[0]
    assert emotion == AiEmotion.NEUTRAL
    assert user_turn == (b"\x11\x22" * 8000, "여보세요"), "구간 [0.25, 0.75) s 를 16 kHz int16 로 잘라 넘긴다"
    assert csm.released is False and p._qwen_tts is csm


async def test_csm_begin_gets_no_user_turn_without_intervals(monkeypatch) -> None:
    _capture_metrics(monkeypatch)
    csm = _FakeCsmClient()
    p = _make_pipeline(_HappyLLM(), _FakeELClient(), qwen=csm)

    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)

    assert csm.begins == [(AiEmotion.NEUTRAL, None)]


async def test_qwen_session_still_gets_plain_begin(monkeypatch) -> None:
    _capture_metrics(monkeypatch)
    qwen = _FakeQwenClient()
    p = _make_pipeline(_HappyLLM(), _FakeELClient(), qwen=qwen)
    p._user_turn_intervals = [(0.0, 0.1)]
    p._tremor_buf = bytearray(b"\x00\x00" * 3200)

    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)

    assert p._ws.pcm == [b"\x0a\x0b"], "accepts_user_turn 이 없는 세션은 begin(emotion) 만 받는다"


async def test_csm_failure_switches_to_eleven(monkeypatch) -> None:
    records = _capture_metrics(monkeypatch)
    csm = _FakeCsmClient(fail=True)
    p = _make_pipeline(_HappyLLM(), _FakeELClient(), qwen=csm)

    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)

    assert not p._closing.is_set() and p._state == _State.LISTENING
    assert csm.released is True and p._qwen_tts is None
    assert dict(records)["realtime_tts_switch"]["reason"] == "synth_failed"


async def test_init_prefers_csm_then_qwen(monkeypatch) -> None:
    records = _capture_metrics(monkeypatch)
    csm, qwen = _FakeCsmClient(), _FakeQwenClient()
    p = _make_pipeline(_HappyLLM(), _FakeELClient())
    p._settings = SimpleNamespace(csm_tts_realtime_enabled=True)

    async def acquire_csm(settings):
        return csm, None

    async def acquire_qwen(settings):
        return qwen, None

    monkeypatch.setattr(pipeline_mod, "try_acquire_realtime_csm", acquire_csm)
    monkeypatch.setattr(pipeline_mod, "try_acquire_realtime_tts", acquire_qwen)
    await p._init_qwen_tts()
    assert p._qwen_tts is csm and p._current_tts_engine() == "csm"
    assert records[-1] == ("realtime_tts_engine", {"session_id": "sess-qwen", "engine": "csm", "skip_reason": None})

    async def csm_busy(settings):
        return None, "busy"

    monkeypatch.setattr(pipeline_mod, "try_acquire_realtime_csm", csm_busy)
    await p._init_qwen_tts()
    assert p._qwen_tts is qwen and p._current_tts_engine() == "qwen"

    p._settings = SimpleNamespace(csm_tts_realtime_enabled=False)
    monkeypatch.setattr(pipeline_mod, "try_acquire_realtime_csm", acquire_csm)
    await p._init_qwen_tts()
    assert p._qwen_tts is qwen, "꺼져 있으면 CSM 은 묻지도 않는다"
