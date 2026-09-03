"""AVTI 저장과 백그라운드 워커 테스트."""
import asyncio
import sys
from types import SimpleNamespace

from app.schemas.frames import EndReason
from app.schemas.training_analysis import AnalysisQualityStatus
from app.services.avti import AvtiPart, AvtiStatus
from app.services.avti_service import save_avti_metrics
from app.services.pipeline import VoicePipeline
from app.workers import avti_worker
from tests.unit.test_avti import stub_script


class _FakeResult:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


class _FakeSession:
    def __init__(self, store: dict, rowcount: int = 1, raise_on_execute: bool = False) -> None:
        self._store = store
        self._rowcount = rowcount
        self._raise = raise_on_execute

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info) -> bool:
        return False

    async def execute(self, stmt):
        if self._raise:
            raise RuntimeError("DB 연결 끊김")
        self._store["params"] = stmt.compile().params
        return _FakeResult(self._rowcount)

    async def commit(self) -> None:
        self._store["committed"] = True


def _factory(store: dict, **kwargs):
    return lambda: _FakeSession(store, **kwargs)


def _settings(**overrides) -> SimpleNamespace:
    base = {
        "avti_enabled": True,
        "praat_bin": "praat",
        "avti_script_path": None,
        "avti_window_sec": 3.0,
        "avti_min_sustained_sec": 3.0,
        "avti_timeout_sec": 10.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeWS:
    def __init__(self, log: list | None = None) -> None:
        self._log = log

    async def send_json(self, payload: dict) -> None:
        if self._log is not None and payload.get("type") == "end":
            self._log.append("end_frame")

    async def close(self, code: int = 1000) -> None:
        pass


class _FakeSpring:
    def __init__(self, log: list | None = None) -> None:
        self._log = log

    async def notify_session_closed(self, *args, **kwargs) -> bool:
        if self._log is not None:
            self._log.append("spring")
        return True


class _FakeTremor:
    def __init__(self, spans: list) -> None:
        self._spans = spans

    def analyze(self, pcm: bytes):
        return SimpleNamespace(
            shake_count=0,
            good_candidates=[],
            sustained_spans=self._spans,
            voiced_spans=self._spans,
            episodes=[],
        )


class _FakeStorage:
    def upload_pcm(self, session_id: str, pcm: bytes) -> str:
        return "rec.wav"


class _NullStorage:
    def upload_pcm(
        self,
        session_id: str,
        pcm: bytes,
    ) -> None:
        return None


class _FakeLLM:
    async def segment_feedback(self, items, **kwargs):
        self.last_items = items
        self.last_kwargs = kwargs
        return [("용건을 말했어요", "무슨 일로 걸었는지 밝혔어요.")] * len(items)


def _pipeline(*, seconds: float, turns: list, spans: list, log: list | None = None) -> VoicePipeline:
    """무거운 __init__ 없이 _teardown 에 필요한 속성만 채운 최소 객체."""
    p = VoicePipeline.__new__(VoicePipeline)
    p._ws, p._ws_alive = _FakeWS(log), True
    p._session_id, p._session = "sess-1", {"type": "SCENARIO"}
    p._history, p._silence_total = [], 1.0
    p._turn_task, p._end_reason = None, EndReason.USER_END
    p._recording_storage, p._tremor = _FakeStorage(), _FakeTremor(spans)
    p._tremor_buf = bytearray(b"\x00\x00" * int(16000 * seconds))
    p._llm, p._spring = _FakeLLM(), _FakeSpring(log)
    p._user_turn_intervals = turns
    p._user_turn_texts = [f"발화{i}" for i in range(len(turns))]
    p._turn_open_at = None
    p._script_len = 0
    p._ai_pcm_bytes = 0
    p._server_wait_duration_ms = 0
    p._completed_script_steps = 0
    p._settings = _settings()
    return p


# --- 저장 ---------------------------------------------------------------


async def test_unmeasured_values_are_null_not_zero() -> None:
    """논문이 명시적으로 경고한 부분 — 0 으로 채우면 분포가 왜곡된다."""
    store: dict = {}
    parts = [AvtiPart(0, 0.0, 30.0, AvtiStatus.NO_SUSTAINED)]

    await save_avti_metrics(session_id="sess-1", parts=parts, session_factory=_factory(store))

    params = store["params"]
    for key in ("avti", "ftrcip", "atri", "fcohnr", "fcom"):
        values = [v for k, v in params.items() if k.startswith(key)]
        assert values and all(v is None for v in values), key


async def test_measured_values_and_status_are_stored() -> None:
    store: dict = {}
    parts = [
        AvtiPart(0, 1.0, 4.0, AvtiStatus.OK, avti=3.21,
                 ftrcip=1.0, atri=2.0, fcohnr=3.0, fcom=0.5)
    ]

    await save_avti_metrics(session_id="sess-1", parts=parts, session_factory=_factory(store))

    values = set(store["params"].values())
    assert 3.21 in values
    assert "OK" in values
    assert "3.05" in values  # script_version
    assert store["committed"] is True


async def test_empty_parts_skip_the_database() -> None:
    store: dict = {}
    saved = await save_avti_metrics(session_id="sess-1", parts=[], session_factory=_factory(store))
    assert saved == 0
    assert store == {}


async def test_database_failure_is_swallowed() -> None:
    parts = [AvtiPart(0, 0.0, 3.0, AvtiStatus.OK, avti=1.0)]
    saved = await save_avti_metrics(
        session_id="sess-1",
        parts=parts,
        session_factory=_factory({}, raise_on_execute=True),
    )
    assert saved == 0


# --- 워커 ---------------------------------------------------------------


def test_disabled_flag_skips_the_task() -> None:
    task = avti_worker.schedule_avti(
        settings=_settings(avti_enabled=False),
        session_id="sess-1",
        pcm=b"\x00\x00",
        parts=[(0.0, 1.0)],
        sustained_spans=[],
    )
    assert task is None


def test_empty_recording_skips_the_task() -> None:
    task = avti_worker.schedule_avti(
        settings=_settings(),
        session_id="sess-1",
        pcm=b"",
        parts=[(0.0, 1.0)],
        sustained_spans=[],
    )
    assert task is None


async def test_missing_script_does_not_raise(monkeypatch) -> None:
    """세션 종료 흐름은 AVTI 가 어떻게 되든 영향받지 않는다."""
    calls: list = []
    monkeypatch.setattr(
        avti_worker, "save_avti_metrics",
        lambda **kw: calls.append(kw) or asyncio.sleep(0),
    )

    task = avti_worker.schedule_avti(
        settings=_settings(avti_script_path=None),
        session_id="sess-1",
        pcm=b"\x00\x00" * 16000,
        parts=[(0.0, 1.0)],
        sustained_spans=[],
    )
    await task

    assert task.done() and task.exception() is None


async def test_running_task_keeps_a_reference(tmp_path, monkeypatch) -> None:
    """참조를 놓으면 GC 가 실행 중인 태스크를 조용히 취소한다."""
    script = stub_script(tmp_path)

    started, release = asyncio.Event(), asyncio.Event()

    async def slow_save(**_kwargs):
        started.set()
        await release.wait()

    monkeypatch.setattr(avti_worker, "save_avti_metrics", slow_save)
    monkeypatch.setattr(avti_worker.AvtiAnalyzer, "_run_praat", lambda self, d: {})

    task = avti_worker.schedule_avti(
        settings=_settings(avti_script_path=script, praat_bin=sys.executable),
        session_id="sess-1",
        pcm=b"\x00\x00" * 16000 * 5,
        parts=[(0.0, 5.0)],
        sustained_spans=[],
    )
    await started.wait()
    assert task in avti_worker._running

    release.set()
    await task
    assert task not in avti_worker._running


# --- 파이프라인 연결 ----------------------------------------------------


async def test_avti_and_spring_callback_run_before_the_end_frame(monkeypatch) -> None:
    """앱이 후속 API를 호출하기 전에 훈련 기록 생성이 완료되어야 한다."""
    order: list[str] = []

    async def _fake_run(**kwargs):
        order.append("avti")
        return {0: 4.2}

    monkeypatch.setattr("app.services.pipeline.run_avti", _fake_run)

    p = _pipeline(seconds=5, turns=[(1.0, 4.0)], spans=[(0.0, 4.0)], log=order)
    await p._teardown()

    assert order == ["avti", "spring", "end_frame"]


async def test_pipeline_passes_whole_session_as_part_zero(monkeypatch) -> None:
    captured: dict = {}

    async def _fake_run(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr("app.services.pipeline.run_avti", _fake_run)

    p = _pipeline(seconds=10, turns=[(1.0, 4.0), (5.0, 9.0)], spans=[(0.5, 4.0)])
    await p._teardown()

    assert captured["parts"][0] == (0.0, 10.0)  # 0번 = 세션 전체
    assert captured["parts"][1:] == [(1.0, 4.0), (5.0, 9.0)]
    assert captured["session_id"] == "sess-1"


async def test_no_recording_means_no_avti(monkeypatch) -> None:
    calls: list = []

    async def _fake_run(**kwargs):
        calls.append(kwargs)
        return {}

    monkeypatch.setattr("app.services.pipeline.run_avti", _fake_run)

    p = _pipeline(seconds=0, turns=[], spans=[])
    await p._teardown()

    assert not calls


async def test_null_recording_key_does_not_block_analysis(
    monkeypatch,
) -> None:
    captured: dict = {}

    async def _fake_run(**kwargs):
        return {}

    async def _capture(*args, **kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(
        "app.services.pipeline.run_avti",
        _fake_run,
    )

    pipeline = _pipeline(
        seconds=8,
        turns=[
            (0.0, 4.0),
            (4.0, 8.0),
        ],
        spans=[
            (0.0, 8.0),
        ],
    )
    pipeline._recording_storage = _NullStorage()
    pipeline._end_reason = EndReason.SCENARIO_DONE
    pipeline._spring.notify_session_closed = _capture

    await pipeline._teardown()

    assert captured["recording_key"] is None

    analysis = captured["analysis"]
    assert analysis is not None
    assert (
        analysis.analysis_quality_status
        is AnalysisQualityStatus.PASS
    )
    assert analysis.stability_score == 100.0
    assert analysis.conversation_score == 100.0
    assert analysis.fluency_score == 100.0


async def test_analysis_failure_does_not_block_session_close(
    monkeypatch,
) -> None:
    order: list[str] = []
    captured: dict = {}

    def _raise_analysis_error(*args, **kwargs):
        raise RuntimeError("analysis failed")

    async def _capture(*args, **kwargs):
        order.append("spring")
        captured.update(kwargs)
        return True

    monkeypatch.setattr(
        "app.services.pipeline.TrainingPerformanceAnalyzer.analyze",
        _raise_analysis_error,
    )

    p = _pipeline(seconds=0, turns=[], spans=[], log=order)
    p._spring.notify_session_closed = _capture

    await p._teardown()

    assert order == ["spring", "end_frame"]
    assert captured["analysis"] is None


async def test_slow_avti_does_not_block_the_session(monkeypatch) -> None:
    """AVTI 가 매달려도 end 프레임과 훈련 기록은 나간다."""
    order: list[str] = []

    async def _hang(**kwargs):
        order.append("avti")
        await asyncio.sleep(60)

    monkeypatch.setattr("app.services.pipeline.run_avti", _hang)
    monkeypatch.setattr("app.services.pipeline._AVTI_WAIT_SECONDS", 0.05)

    p = _pipeline(seconds=5, turns=[(1.0, 4.0)], spans=[(0.0, 4.0)], log=order)
    await p._teardown()

    assert order == ["avti", "spring", "end_frame"]


async def test_segment_text_reaches_the_callback(monkeypatch) -> None:
    """AI 가 쓴 구간 문구가 그대로 기록으로 넘어간다."""
    captured: dict = {}

    async def _fake_run(**kwargs):
        return {1: 8.5}

    async def _capture(*args, **kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr("app.services.pipeline.run_avti", _fake_run)

    p = _pipeline(seconds=60, turns=[(1.0, 40.0)], spans=[(1.0, 40.0)])
    p._spring.notify_session_closed = _capture
    p._history = [{"role": "user", "text": "예약하려고 전화했어요"}]
    await p._teardown()

    seg = captured["good_segments"][0]
    assert seg["type"] == "IMPROVE"        # AVTI 8.5 → 짚어줄 구간
    assert seg["title"] == "용건을 말했어요"
    assert seg["content"] == "무슨 일로 걸었는지 밝혔어요."
    assert seg["good_point"] == seg["content"]   # 기존 Spring 매핑 호환
    assert "reason" not in seg


async def test_turn_avti_reaches_the_model(monkeypatch) -> None:
    """구간 문구를 쓸 때 그 턴의 측정값이 근거로 넘어간다."""
    async def _fake_run(**kwargs):
        return {1: 8.5}

    monkeypatch.setattr("app.services.pipeline.run_avti", _fake_run)

    p = _pipeline(seconds=60, turns=[(1.0, 40.0)], spans=[(1.0, 40.0)])
    p._history = [{"role": "user", "text": "예약하려고요"}]
    await p._teardown()

    item = p._llm.last_items[0]
    assert item["type"] == "IMPROVE"
    assert item["avti"] == 8.5


async def test_internal_keys_never_leave_the_server(monkeypatch) -> None:
    """turn·avti·reason 은 구간을 고르는 재료다. 사용자에게도 Spring 에도 안 나간다."""
    import copy

    sent: list = []
    captured: dict = {}

    async def _fake_run(**kwargs):
        return {1: 8.5}

    async def _capture(*args, **kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr("app.services.pipeline.run_avti", _fake_run)

    p = _pipeline(seconds=20, turns=[(1.0, 9.0)], spans=[(1.0, 9.0)])
    p._ws.send_json = lambda payload: sent.append(copy.deepcopy(payload)) or _noop()
    p._spring.notify_session_closed = _capture
    p._history = [{"role": "user", "text": "여보세요"}]
    await p._teardown()

    internal = {"turn", "avti", "reason"}
    end_segments = sent[-1]["feedback"]["good_segments"]
    assert end_segments and all(not (internal & set(s)) for s in end_segments)
    # end 프레임에는 문구가 아직 없다 — 공개 필드만
    assert all(set(s) == {"start", "end", "type"} for s in end_segments)

    assert all(not (internal & set(s)) for s in captured["good_segments"])


async def test_callback_failure_sends_error_instead_of_end() -> None:
    sent: list[dict] = []

    async def _capture_frame(payload: dict) -> None:
        sent.append(payload)

    async def _fail_callback(*args, **kwargs) -> bool:
        return False

    p = _pipeline(seconds=0, turns=[], spans=[])
    p._ws.send_json = _capture_frame
    p._spring.notify_session_closed = _fail_callback

    await p._teardown()

    assert sent == [{"type": "error", "code": "SESSION_CLOSE_FAILED"}]


async def test_pipeline_error_is_preserved_when_callback_also_fails() -> None:
    sent: list[dict] = []

    async def _capture_frame(payload: dict) -> None:
        sent.append(payload)

    async def _fail_callback(*args, **kwargs) -> bool:
        return False

    p = _pipeline(seconds=0, turns=[], spans=[])
    p._end_reason = EndReason.ERROR
    p._ws.send_json = _capture_frame
    p._spring.notify_session_closed = _fail_callback

    await p._teardown()

    assert sent == [{"type": "error", "code": "PIPELINE_ERROR"}]


async def _noop() -> None:
    return None
