import asyncio
from types import SimpleNamespace

from app.schemas.llm import AiEmotion, LLMEvent, LLMEventType
from app.services import pipeline as pipeline_mod
from app.services.pipeline import VoicePipeline, _State, _TurnTimings
from app.services.qwen_tts import QwenTTSUnavailableError


class _FakeWS:
    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.pcm: list[bytes] = []
        self.closed = False

    async def send_json(self, payload: dict) -> None:
        self.frames.append(payload)

    async def send_bytes(self, data: bytes) -> None:
        self.pcm.append(data)

    async def close(self, code: int = 1000) -> None:
        self.closed = True


class _FakeELSession:
    def __init__(self) -> None:
        self.closed = False

    async def begin(self, emotion=AiEmotion.NEUTRAL) -> None:
        pass

    async def stream(self, text_source):
        async for _ in text_source:
            pass
        yield b"\xee\xee"

    async def aclose(self) -> None:
        self.closed = True


class _FakeELClient:
    def __init__(self) -> None:
        self.open_calls = 0

    async def open(self, voice_id=None) -> _FakeELSession:
        self.open_calls += 1
        return _FakeELSession()


class _FakeQwenSession:
    def __init__(self, fail: bool) -> None:
        self._fail = fail
        self.closed = False

    async def begin(self, emotion=AiEmotion.NEUTRAL) -> None:
        pass

    async def stream(self, text_source):
        if self._fail:
            raise QwenTTSUnavailableError("boom")
        async for _ in text_source:
            pass
        yield b"\x0a\x0b"

    async def aclose(self) -> None:
        self.closed = True


class _FakeQwenClient:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.released = False
        self.open_calls = 0

    async def open(self, voice_id=None) -> _FakeQwenSession:
        self.open_calls += 1
        return _FakeQwenSession(fail=self.fail)

    def release_slot(self) -> None:
        self.released = True


class _HappyLLM:
    async def stream(self, ctx):
        yield LLMEvent(type=LLMEventType.EMOTION_RESOLVED, emotion=AiEmotion.NEUTRAL)
        yield LLMEvent(type=LLMEventType.TEXT_DELTA, text="네, 알겠습니다.")
        yield LLMEvent(type=LLMEventType.TURN_END)


def _make_pipeline(llm, tts, qwen=None) -> VoicePipeline:
    p = VoicePipeline.__new__(VoicePipeline)
    p._ws = _FakeWS()
    p._session_id = "sess-qwen"
    p._session = {}
    p._llm = llm
    p._tts = tts
    p._qwen_tts = qwen
    p._state = _State.THINKING
    p._history = []
    p._current_step = 1
    p._ws_alive = True
    p._time_up = False
    p._closing = asyncio.Event()
    p._turn_task = None
    p._listening_since = None
    p._tremor_buf = bytearray()
    p._user_turn_intervals = []
    p._turn_open_at = None
    p._script_len = 0
    p._ai_pcm_bytes = 0
    p._server_wait_duration_ms = 0
    p._completed_script_steps = 0
    return p


def _capture_metrics(monkeypatch) -> list[tuple[str, dict]]:
    records: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        pipeline_mod, "log_metric", lambda name, **kw: records.append((name, kw))
    )
    return records


async def test_happy_qwen_turn_keeps_slot_and_tags_engine(monkeypatch) -> None:
    records = _capture_metrics(monkeypatch)
    qwen = _FakeQwenClient()
    p = _make_pipeline(_HappyLLM(), _FakeELClient(), qwen=qwen)

    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)

    assert p._ws.pcm == [b"\x0a\x0b"], "오디오는 Qwen 세션에서 나와야 한다"
    assert qwen.released is False and p._qwen_tts is qwen
    assert not p._closing.is_set()
    turn = dict(records)["voice_turn"]
    assert turn["tts_engine"] == "qwen"
    assert turn["tts_failed"] is False


async def test_qwen_failure_switches_to_eleven_and_keeps_call(monkeypatch) -> None:
    records = _capture_metrics(monkeypatch)
    qwen = _FakeQwenClient(fail=True)
    el = _FakeELClient()
    p = _make_pipeline(_HappyLLM(), el, qwen=qwen)

    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)

    assert not p._closing.is_set(), "Qwen 장애로 통화가 끊기면 안 된다"
    assert p._state == _State.LISTENING
    assert qwen.released is True and p._qwen_tts is None, "슬롯 반납 + 편도 전환"
    assert el.open_calls == 1, "폴백 멘트는 ElevenLabs 로 나가야 한다"
    assert p._ws.pcm == [b"\xee\xee"], "Qwen PCM 없이 EL 멘트만"
    names = [name for name, _ in records]
    assert "realtime_tts_switch" in names
    turn = dict(records)["voice_turn"]
    assert turn["tts_engine"] == "qwen", "실패한 턴은 시도한 엔진으로 기록"
    assert turn["tts_failed"] is True
    assert turn["error"] is False


async def test_open_tts_prefers_qwen_then_eleven_after_switch() -> None:
    qwen = _FakeQwenClient()
    el = _FakeELClient()
    p = _make_pipeline(_HappyLLM(), el, qwen=qwen)

    session = await p._open_tts()
    assert isinstance(session, _FakeQwenSession)

    p._switch_to_eleven("synth_failed")
    assert qwen.released is True and p._qwen_tts is None

    session = await p._open_tts()
    assert isinstance(session, _FakeELSession)


async def test_open_tts_falls_back_when_qwen_open_fails() -> None:
    class _BrokenQwen(_FakeQwenClient):
        async def open(self, voice_id=None):
            raise QwenTTSUnavailableError("open boom")

    qwen = _BrokenQwen()
    p = _make_pipeline(_HappyLLM(), _FakeELClient(), qwen=qwen)

    session = await p._open_tts()

    assert isinstance(session, _FakeELSession)
    assert qwen.released is True and p._qwen_tts is None


async def test_init_qwen_tts_records_engine_metric(monkeypatch) -> None:
    records = _capture_metrics(monkeypatch)
    qwen = _FakeQwenClient()
    qwen.speed = 1.15

    async def _acquire(_settings):
        return qwen, None

    monkeypatch.setattr(pipeline_mod, "try_acquire_realtime_tts", _acquire)
    p = _make_pipeline(_HappyLLM(), _FakeELClient())
    p._settings = SimpleNamespace()

    await p._init_qwen_tts()

    assert p._qwen_tts is qwen
    metric = dict(records)["realtime_tts_engine"]
    assert (metric["engine"], metric["speed"]) == ("qwen", 1.15)


async def test_run_releases_slot_on_exit(monkeypatch) -> None:
    qwen = _FakeQwenClient()

    async def _acquire(_settings):
        return qwen, None

    monkeypatch.setattr(pipeline_mod, "try_acquire_realtime_tts", _acquire)
    p = _make_pipeline(_HappyLLM(), _FakeELClient())
    p._settings = SimpleNamespace()
    p._max_duration = None

    async def _noop():
        return None

    p._llm = SimpleNamespace(warmup=_noop)
    p._recv_loop = _noop
    p._stt_consumer = _noop

    async def _teardown(*tasks):
        for task in tasks:
            if task is not None:
                task.cancel()

    p._teardown = _teardown
    p._closing.set()

    await asyncio.wait_for(p.run(), timeout=2.0)

    assert qwen.released is True, "통화 종료 시 슬롯을 반납해야 한다"
    assert p._qwen_tts is None


async def test_run_reports_teardown_failure_and_releases_slot(monkeypatch) -> None:
    qwen = _FakeQwenClient()

    async def _acquire(_settings):
        return qwen, None

    async def _noop():
        return None

    async def _failing_teardown(*tasks):
        for task in tasks:
            if task is not None:
                task.cancel()
        raise RuntimeError("teardown failed")

    monkeypatch.setattr(pipeline_mod, "try_acquire_realtime_tts", _acquire)
    p = _make_pipeline(_HappyLLM(), _FakeELClient())
    p._settings = SimpleNamespace()
    p._max_duration = None
    p._llm = SimpleNamespace(warmup=_noop)
    p._recv_loop = _noop
    p._stt_consumer = _noop
    p._teardown = _failing_teardown
    p._closing.set()

    await asyncio.wait_for(p.run(), timeout=2.0)

    assert p._ws.frames == [
        {"type": "error", "code": "SESSION_CLOSE_FAILED"}
    ]
    assert p._ws.closed is True
    assert qwen.released is True
    assert p._qwen_tts is None


class _RecordingQwen(_FakeQwenClient):
    def __init__(self, voices=None) -> None:
        super().__init__()
        self.voices = voices
        self.opened_with: list = []

    async def open(self, voice_id=None):
        self.opened_with.append(voice_id)
        return await super().open(voice_id)


async def test_open_tts_sends_the_scenario_qwen_voice() -> None:
    qwen = _RecordingQwen()
    p = _make_pipeline(_HappyLLM(), _FakeELClient(), qwen=qwen)
    p._qwen_voice = "male_old_rough"
    await p._open_tts()
    p._qwen_voice = None
    await p._open_tts()
    assert qwen.opened_with == ["male_old_rough", None]


async def test_init_drops_voice_missing_on_worker_and_records_it(monkeypatch) -> None:
    records = _capture_metrics(monkeypatch)
    qwen = _RecordingQwen(voices=frozenset({"ai", "user"}))

    async def _acquire(_settings):
        return qwen, None

    monkeypatch.setattr(pipeline_mod, "try_acquire_realtime_tts", _acquire)
    p = _make_pipeline(_HappyLLM(), _FakeELClient())
    p._settings = SimpleNamespace()
    p._qwen_voice = "male_old_rough"

    await p._init_qwen_tts()

    assert p._qwen_voice is None, "워커에 없는 보이스면 그 통화는 ai"
    voice_metric = dict(records)["realtime_tts_voice"]
    assert voice_metric == {"session_id": "sess-qwen", "voice": "male_old_rough", "missing": True}
    engine_keys = set(dict(records)["realtime_tts_engine"])
    assert engine_keys == {"session_id", "engine", "skip_reason", "csm_skip_reason", "speed"}


async def test_init_keeps_voice_present_on_worker(monkeypatch) -> None:
    records = _capture_metrics(monkeypatch)
    qwen = _RecordingQwen(voices=frozenset({"ai", "male_old_rough"}))

    async def _acquire(_settings):
        return qwen, None

    monkeypatch.setattr(pipeline_mod, "try_acquire_realtime_tts", _acquire)
    p = _make_pipeline(_HappyLLM(), _FakeELClient())
    p._settings = SimpleNamespace()
    p._qwen_voice = "male_old_rough"
    await p._init_qwen_tts()
    assert p._qwen_voice == "male_old_rough" and dict(records)["realtime_tts_voice"]["missing"] is False


async def test_init_without_scenario_voice_records_no_voice_metric(monkeypatch) -> None:
    records = _capture_metrics(monkeypatch)

    async def _acquire(_settings):
        return _RecordingQwen(voices=frozenset({"ai"})), None

    monkeypatch.setattr(pipeline_mod, "try_acquire_realtime_tts", _acquire)
    p = _make_pipeline(_HappyLLM(), _FakeELClient())
    p._settings = SimpleNamespace()
    await p._init_qwen_tts()
    assert "realtime_tts_voice" not in dict(records)


async def test_init_treats_worker_without_voice_list_as_missing(monkeypatch) -> None:
    records = _capture_metrics(monkeypatch)

    async def _acquire(_settings):
        return _RecordingQwen(voices=None), None

    monkeypatch.setattr(pipeline_mod, "try_acquire_realtime_tts", _acquire)
    p = _make_pipeline(_HappyLLM(), _FakeELClient())
    p._settings = SimpleNamespace()
    p._qwen_voice = "male_old_rough"
    await p._init_qwen_tts()
    assert p._qwen_voice is None and dict(records)["realtime_tts_voice"]["missing"] is True


async def test_init_without_worker_records_no_voice_metric(monkeypatch) -> None:
    records = _capture_metrics(monkeypatch)

    async def _acquire(_settings):
        return None, "busy"

    monkeypatch.setattr(pipeline_mod, "try_acquire_realtime_tts", _acquire)
    p = _make_pipeline(_HappyLLM(), _FakeELClient())
    p._settings = SimpleNamespace()
    p._qwen_voice = "male_old_rough"
    await p._init_qwen_tts()
    assert "realtime_tts_voice" not in dict(records), "ElevenLabs 통화를 칸 보이스로 집계하지 않는다"


class _MeteredQwenSession(_FakeQwenSession):
    def __init__(self, metrics_fail: bool = False) -> None:
        super().__init__(fail=False)
        self.request_log = [{"outcome": "eos", "raw_ms": 100.0}, {"outcome": "eos", "raw_ms": 200.0}]
        self._metrics_fail = metrics_fail

    def turn_metrics(self) -> dict:
        if self._metrics_fail:
            raise RuntimeError("boom")
        return {"tts_req": 2, "tts_req_cut": 1}


class _MeteredQwenClient(_FakeQwenClient):
    def __init__(self, metrics_fail: bool = False) -> None:
        super().__init__()
        self._metrics_fail = metrics_fail

    async def open(self, voice_id=None) -> _MeteredQwenSession:
        self.open_calls += 1
        return _MeteredQwenSession(self._metrics_fail)


async def test_request_lines_come_before_voice_turn(monkeypatch) -> None:
    records = _capture_metrics(monkeypatch)
    p = _make_pipeline(_HappyLLM(), _FakeELClient(), qwen=_MeteredQwenClient())

    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)

    names = [name for name, _ in records]
    reqs = [kw for name, kw in records if name == "realtime_tts_req"]
    assert len(reqs) == 2 and names.index("voice_turn") > max(i for i, n in enumerate(names) if n == "realtime_tts_req")
    assert [(r["i"], r["last"], r["raw_ms"]) for r in reqs] == [(0, False, 100.0), (1, True, 200.0)]
    assert all(r["session_id"] == "sess-qwen" and r["step"] == 1 for r in reqs)
    turn = dict(records)["voice_turn"]
    assert (turn["tts_req"], turn["tts_req_cut"]) == (2, 1)


async def test_turn_metrics_failure_keeps_voice_turn_and_audio(monkeypatch) -> None:
    records = _capture_metrics(monkeypatch)
    p = _make_pipeline(_HappyLLM(), _FakeELClient(), qwen=_MeteredQwenClient(metrics_fail=True))

    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)

    assert p._ws.pcm == [b"\x0a\x0b"]
    turn = dict(records)["voice_turn"]
    assert "tts_req" not in turn


async def test_sessions_without_request_log_add_no_fields(monkeypatch) -> None:
    records = _capture_metrics(monkeypatch)
    p = _make_pipeline(_HappyLLM(), _FakeELClient())

    await asyncio.wait_for(p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0)

    assert all(name != "realtime_tts_req" for name, _ in records)
    assert "tts_req" not in dict(records)["voice_turn"]


async def test_engine_metric_records_worker_when_known(monkeypatch) -> None:
    records = _capture_metrics(monkeypatch)
    qwen = _FakeQwenClient()
    qwen._worker_url = "http://w2.test"

    async def _acquire(_settings):
        return qwen, None

    monkeypatch.setattr(pipeline_mod, "try_acquire_realtime_tts", _acquire)
    p = _make_pipeline(_HappyLLM(), _FakeELClient())
    p._settings = SimpleNamespace()

    await p._init_qwen_tts()

    assert dict(records)["realtime_tts_engine"]["worker"] == "http://w2.test"
