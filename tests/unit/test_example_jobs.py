import asyncio
import gc
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import numpy as np
import pytest

from app.services import example_service as es
from app.services.qwen_tts import NsResult, QwenUnreachableError, RefsState


def _speech(sec: float = 0.6, fade: bool = True) -> bytes:
    sr = 16_000
    t = np.arange(int(sr * sec)) / sr
    x = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    if fade:
        n = int(sr * 0.06)
        x[-n:] *= np.geomspace(1.0, 1e-3, n).astype(np.float32)
    pad = np.zeros(int(sr * 0.3), dtype=np.float32)
    return (np.concatenate([pad, x, pad]) * 32767).astype("<i2").tobytes()


_GOOD = _speech()
_D1 = [{"speaker": "ai", "text": "네, 바다식당입니다."}, {"speaker": "user", "text": "예약할게요."}]
_D2 = [{"speaker": "ai", "text": "네, 바다병원입니다."}, {"speaker": "user", "text": "진료 예약이요."}]
_DC = [{"speaker": "ai", "text": "네, 집주인입니다."}, {"speaker": "user", "text": "월세 날짜요."}]


def _row(sid: int, *, custom: bool = False, dialogue=None, url=None, warmup: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        scenario_id=sid, is_custom=custom, user_id=7, tts_voice_id=None, example_dialogue=dialogue,
        example_audio_url=url, title="t", call_target="ct", call_purpose="cp",
        script=[{"step": 1, "ai_goal": "인사", "hint": "인사"}], is_warmup=warmup, deleted_at=None,
    )


class _DB:
    def __init__(self, rows: dict) -> None:
        self.rows = rows
        self.commits = 0

    async def get(self, _model, pk):
        return self.rows.get(pk)

    async def commit(self) -> None:
        self.commits += 1

    async def refresh(self, obj, attrs=None) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a) -> bool:
        return False


class _Storage:
    def __init__(self, objects: dict | None = None) -> None:
        self.objects = dict(objects or {})
        self.uploads: list[str] = []
        self.copies: list[tuple[str, str]] = []
        self.deleted: list[str] = []
        self.markers: list[str] = []
        self.fail_copy: Exception | None = None
        self.fail_head: Exception | None = None

    def upload_wav(self, key: str, pcm: bytes) -> str:
        self.uploads.append(key)
        self.objects[key] = pcm
        return key

    def exists(self, key: str) -> bool:
        return key in self.objects

    def strict_exists(self, key: str) -> bool:
        if self.fail_head is not None:
            raise self.fail_head
        return key in self.objects

    def put_marker(self, key: str) -> bool:
        self.markers.append(key)
        self.objects[key] = b""
        return True

    def copy(self, src: str, dst: str) -> bool:
        if self.fail_copy is not None:
            raise self.fail_copy
        if src not in self.objects:
            return False
        self.objects[dst] = self.objects[src]
        self.copies.append((src, dst))
        return True

    def delete(self, key: str) -> bool:
        self.deleted.append(key)
        self.objects.pop(key, None)
        return True

    def presigned_url(self, key: str, expires_in: int = 600) -> str:
        return f"https://signed.test/{key}"


class _Qwen:
    def __init__(self, *, refs=None, states=("ok",), unreachable=(), missing=(), delay=0.0,
                 script=None, ns=True, synth_refs=None) -> None:
        self.refs = refs or {"ai": "aaaaaaaa", "user": "bbbbbbbb"}
        self.synth_refs = synth_refs or self.refs
        self.states = list(states)
        self.unreachable = tuple(unreachable)
        self.missing = tuple(missing)
        self.delay = delay
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.ns = ns
        self.calls: list[tuple[str, str]] = []
        self.connects: list[set] = []
        self.health_calls = 0
        self.enabled = True
        self.worker_states: dict[str, str] = {}
        self.health_delay = 0.0
        self.probed: list[str] = []

    @asynccontextmanager
    async def connect(self, exclude=()):
        self.connects.append(set(exclude))
        free = [u for u in ("http://w1.test", "http://w2.test") if u not in set(exclude)]
        if not free:
            raise es.QwenTTSUnavailableError("쓸 수 있는 워커 없음(모두 제외)")
        self._session_url = free[0]
        yield self

    async def synth_ns(self, voice: str, text: str) -> NsResult:
        self.calls.append((voice, text))
        if self.delay:
            await asyncio.sleep(self.delay)
        queue = self.script.get(text)
        item = queue.pop(0) if queue else _GOOD
        if isinstance(item, BaseException):
            raise item
        return NsResult(item, self.ns, self.synth_refs.get(voice), "http://w1.test")

    async def health_refs(self, needed) -> RefsState:
        self.health_calls += 1
        if self.health_delay:
            await asyncio.sleep(self.health_delay)
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        refs = {v: self.refs[v] for v in needed if v in self.refs} if state not in ("down", "no_refs") else {}
        return RefsState(state, refs, self.unreachable, self.missing)

    async def healthy(self) -> bool:
        return False


class _Probe:
    def __init__(self, qwen: _Qwen, url: str) -> None:
        self.qwen, self.url = qwen, url

    async def health_state(self) -> str:
        self.qwen.probed.append(self.url)
        state = self.qwen.worker_states.get(self.url, "ready")
        if isinstance(state, list):
            return state.pop(0) if len(state) > 1 else state[0]
        return state


class _EL:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.opens = 0

    async def open(self, voice_id=None):
        self.opens += 1
        if self.fail:
            raise RuntimeError("EL 폭발")
        session = SimpleNamespace()

        async def begin(emotion=None):
            return None

        async def stream(source):
            async for _ in source:
                pass
            yield b"\x01\x02"

        async def aclose():
            return None

        session.begin, session.stream, session.aclose = begin, stream, aclose
        return session


def _settings(**kw) -> SimpleNamespace:
    base = dict(
        s3_bucket="b", example_qwen_ns=True, example_request_wait_sec=1.0, example_turn_prior_sec=0.01,
        example_ns_skip_presets="", qwen_tts_url=None, qwen_tts_urls="http://w1.test,http://w2.test",
        qwen_tts_timeout=30.0, qwen_tts_health_timeout=1.0, elevenlabs_voice_id="voice-default",
        anthropic_api_key="k", llm_analysis_model="m", qwen_scenario_voices=False,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _wire(monkeypatch, rows: dict, *, storage=None, qwen=None, el=None, settings=None, presets=None,
          llm_payload=None, llm_delay: float = 0.0):
    storage = storage or _Storage()
    qwen = qwen or _Qwen()
    el = el or _EL()
    settings = settings or _settings()
    metrics: list[tuple[str, dict]] = []
    usage: list[tuple[tuple, dict]] = []

    class _Messages:
        async def create(self, **_kw):
            if llm_delay:
                await asyncio.sleep(llm_delay)
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(llm_payload or _DC, ensure_ascii=False))])

    class _Anthropic:
        def __init__(self, api_key: str) -> None:
            self.messages = _Messages()

    monkeypatch.setattr(es, "get_settings", lambda: settings)
    monkeypatch.setattr(es, "AsyncSessionLocal", lambda: _DB(rows))
    monkeypatch.setattr(es, "RecordingStorageService", lambda _s: storage)
    monkeypatch.setattr(es, "QwenTTSClient", lambda *_a, **k: _Probe(qwen, k["base_url"]) if "base_url" in k else qwen)
    monkeypatch.setattr(es, "ElevenLabsTTSClient", lambda _s: el)
    monkeypatch.setattr(es, "has_free_worker", lambda _s: True)
    monkeypatch.setattr(es, "PRESET_MAP", presets or {1: {"example_dialogue": _D1}})
    monkeypatch.setattr(es, "AsyncAnthropic", _Anthropic)
    monkeypatch.setattr(es, "log_metric", lambda name, **kw: metrics.append((name, kw)))
    monkeypatch.setattr(es, "log_tts_usage", lambda *a, **kw: usage.append((a, kw)))
    return SimpleNamespace(storage=storage, qwen=qwen, el=el, settings=settings, metrics=metrics, usage=usage)


@pytest.fixture(autouse=True)
async def _clean():
    yield
    await es.cancel_example_jobs()
    loop = asyncio.get_running_loop()
    es._cooldown_by_loop.pop(loop, None)
    es._drivers.pop(loop, None)
    gc.collect()


def _named(metrics, name):
    return [kw for n, kw in metrics if n == name]


async def _job(sid: int):
    job = es._loop_jobs().get(sid)
    key = None if job is None else await asyncio.wait_for(asyncio.shield(job.task), 2)
    await asyncio.sleep(0)
    return key


async def _bake(sid: int, limit: float = 2):
    task = await es.bake_example_audio(sid)
    return None if task is None else await asyncio.wait_for(asyncio.shield(task), limit)


async def test_switch_off_request_uses_plain_bake(monkeypatch) -> None:
    rows = {1: _row(1)}
    env = _wire(monkeypatch, rows, settings=_settings(example_qwen_ns=False))
    resp = await asyncio.wait_for(es.get_example_conversation(_DB(rows), 1, user_id=7), 2)
    assert resp.audio_url and "-qn" not in resp.audio_url and env.el.opens == 2
    assert es._loop_jobs() == {} and env.qwen.calls == []


async def test_switch_off_bake_example_audio_uses_plain_bake(monkeypatch) -> None:
    rows = {42: _row(42, custom=True, dialogue=_DC)}
    env = _wire(monkeypatch, rows, settings=_settings(example_qwen_ns=False))
    await _bake(42)
    assert rows[42].example_audio_url and "-qn" not in rows[42].example_audio_url
    assert env.qwen.calls == []


async def test_request_waits_for_a_fast_job_and_serves_ns_key(monkeypatch) -> None:
    rows = {1: _row(1)}
    env = _wire(monkeypatch, rows)
    resp = await asyncio.wait_for(es.get_example_conversation(_DB(rows), 1, user_id=7), 2)
    assert resp.audio_url.endswith("-qn.wav")
    assert rows[1].example_audio_url == env.storage.uploads[0] and env.storage.uploads[0].endswith("-qn.wav")
    assert _named(env.metrics, "example_wait")[0]["served"] is True


async def test_request_times_out_returns_dialogue_and_job_keeps_going(monkeypatch) -> None:
    rows = {1: _row(1)}
    env = _wire(monkeypatch, rows, qwen=_Qwen(delay=0.15), settings=_settings(example_request_wait_sec=0.1))
    resp = await asyncio.wait_for(es.get_example_conversation(_DB(rows), 1, user_id=7), 2)
    assert resp.audio_url is None and [t.text for t in resp.dialogue] == [t["text"] for t in _D1]
    assert _named(env.metrics, "example_wait")[0]["timed_out"] is True
    key = await _job(1)
    assert key and rows[1].example_audio_url == key


async def test_request_returns_at_once_when_estimate_exceeds_wait(monkeypatch) -> None:
    rows = {1: _row(1)}
    env = _wire(monkeypatch, rows, qwen=_Qwen(delay=0.05), settings=_settings(example_turn_prior_sec=30.0))
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    resp = await asyncio.wait_for(es.get_example_conversation(_DB(rows), 1, user_id=7), 2)
    assert resp.audio_url is None and loop.time() - t0 < 0.1
    wait = _named(env.metrics, "example_wait")[0]
    assert wait["early_return"] is True and wait["waited_ms"] < 100


async def test_request_during_dialogue_generation_returns_at_once(monkeypatch) -> None:
    rows = {42: _row(42, custom=True)}
    env = _wire(monkeypatch, rows, llm_delay=0.2)
    resp = await asyncio.wait_for(es.get_example_conversation(_DB(rows), 42, user_id=7), 2)
    assert resp.audio_url is None and resp.dialogue == []
    assert _named(env.metrics, "example_wait")[0]["waited_ms"] < 100
    assert await _job(42)
    assert rows[42].example_dialogue == _DC


async def test_cancelled_job_still_answers_200(monkeypatch) -> None:
    rows = {1: _row(1)}
    _wire(monkeypatch, rows, qwen=_Qwen(delay=0.2))
    request = asyncio.ensure_future(es.get_example_conversation(_DB(rows), 1, user_id=7))
    await asyncio.sleep(0.05)
    es._loop_jobs()[1].task.cancel()
    resp = await asyncio.wait_for(request, 2)
    assert resp.audio_url is None


async def test_own_cancellation_is_reraised_and_job_survives(monkeypatch) -> None:
    rows = {1: _row(1)}
    _wire(monkeypatch, rows, qwen=_Qwen(delay=0.1))
    request = asyncio.ensure_future(es.get_example_conversation(_DB(rows), 1, user_id=7))
    await asyncio.sleep(0.05)
    job = es._loop_jobs()[1]
    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request
    assert await asyncio.wait_for(asyncio.shield(job.task), 2)


async def test_two_requests_and_creation_bake_share_one_job(monkeypatch) -> None:
    rows = {42: _row(42, custom=True, dialogue=_DC)}
    env = _wire(monkeypatch, rows, qwen=_Qwen(delay=0.02))
    await asyncio.wait_for(
        asyncio.gather(
            _bake(42),
            es.get_example_conversation(_DB(rows), 42, user_id=7),
            es.get_example_conversation(_DB(rows), 42, user_id=7),
        ),
        3,
    )
    assert len(env.qwen.connects) == 1 and len(env.storage.uploads) == 1


async def test_queue_wait_does_not_eat_the_budget(monkeypatch) -> None:
    rows = {42: _row(42, custom=True, dialogue=_DC)}
    env = _wire(monkeypatch, rows)
    monkeypatch.setattr(es, "_CUSTOM_NS_BUDGET_SEC", 0.2)

    async def _hold():
        async with es._prebake_semaphore():
            await asyncio.sleep(0.3)

    holder = asyncio.ensure_future(_hold())
    await asyncio.sleep(0)
    await _bake(42)
    await holder
    assert rows[42].example_audio_url.endswith("-qn.wav") and env.el.opens == 0


async def test_preset_failure_is_deferred_not_fallen_back(monkeypatch) -> None:
    rows = {1: _row(1, url="examples/1-old.wav")}
    env = _wire(monkeypatch, rows, qwen=_Qwen(ns=False))
    job = es._ensure_job(1, "preset_prebake")
    assert await asyncio.wait_for(job.task, 2) is None
    assert rows[1].example_audio_url == "examples/1-old.wav" and env.storage.uploads == [] and env.el.opens == 0
    assert _named(env.metrics, "example_tts_deferred")[0]["reason"] == "no_ns"


async def test_custom_failure_falls_back_to_plain_with_reason(monkeypatch) -> None:
    rows = {42: _row(42, custom=True, dialogue=_DC)}
    env = _wire(monkeypatch, rows, qwen=_Qwen(ns=False))
    await _bake(42)
    assert rows[42].example_audio_url and "-qn" not in rows[42].example_audio_url and env.el.opens == 2
    plain = [m for m in _named(env.metrics, "example_tts") if m.get("ns_fallback_reason")]
    assert plain and plain[0]["ns_fallback_reason"] == "no_ns"


async def test_failed_fallback_starts_a_cooldown(monkeypatch) -> None:
    rows = {42: _row(42, custom=True, dialogue=_DC)}
    env = _wire(monkeypatch, rows, qwen=_Qwen(ns=False), el=_EL(fail=True))
    await _bake(42)
    calls = len(env.qwen.calls)
    resp = await asyncio.wait_for(es.get_example_conversation(_DB(rows), 42, user_id=7), 2)
    assert resp.audio_url is None and len(env.qwen.calls) == calls and es._loop_jobs() == {}


async def test_custom_budget_timeout_falls_back(monkeypatch) -> None:
    rows = {42: _row(42, custom=True, dialogue=_DC)}
    env = _wire(monkeypatch, rows, qwen=_Qwen(delay=0.2))
    monkeypatch.setattr(es, "_CUSTOM_NS_BUDGET_SEC", 0.05)
    await _bake(42)
    assert "-qn" not in rows[42].example_audio_url
    assert [m for m in _named(env.metrics, "example_tts") if m.get("ns_fallback_reason") == "timeout"]


async def test_unreachable_worker_is_excluded_on_the_retry(monkeypatch) -> None:
    rows = {1: _row(1)}
    first = _D1[0]["text"]
    env = _wire(monkeypatch, rows, qwen=_Qwen(script={first: [QwenUnreachableError("x", worker="http://w1.test")]}))
    key = await asyncio.wait_for(es._ensure_job(1, "request").task, 2)
    assert key and env.qwen.connects == [set(), {"http://w1.test"}]


async def test_skip_list_keeps_plain_path(monkeypatch) -> None:
    rows = {1: _row(1)}
    env = _wire(monkeypatch, rows, settings=_settings(example_ns_skip_presets="1, 3"))
    resp = await asyncio.wait_for(es.get_example_conversation(_DB(rows), 1, user_id=7), 2)
    assert resp.audio_url and "-qn" not in resp.audio_url and env.qwen.calls == []


async def test_ns_bake_logs_usage_and_metric(monkeypatch) -> None:
    rows = {1: _row(1, url="examples/1-old.wav")}
    env = _wire(monkeypatch, rows)
    key = await asyncio.wait_for(es._ensure_job(1, "preset_prebake").task, 2)
    (args, kw), = env.usage
    assert args == ("qwen", "qwen-ns1", "example_audio")
    assert (kw["attempts"], kw["turns"], kw["trigger"]) == (2, 2, "preset_prebake")
    metric = [m for m in _named(env.metrics, "example_tts") if m.get("method") == "ns1"][0]
    assert metric["prev_key"] == "examples/1-old.wav" and metric["v8"] == key.rsplit("-", 2)[-2]


async def test_warmup_row_is_not_baked(monkeypatch) -> None:
    rows = {42: _row(42, custom=True, dialogue=_DC, warmup=True)}
    env = _wire(monkeypatch, rows)
    await _bake(42)
    assert env.qwen.calls == [] and env.storage.uploads == []


def _expected(sid: int, dialogue, refs=None) -> str:
    refs = refs or {"ai": "aaaaaaaa", "user": "bbbbbbbb"}
    levels = {"ai": -16.94749568714168, "user": -22.544892951720513}
    return es._ns_key(sid, dialogue, es._ns_v8({"ai": "ai", "user": "user"}, refs, levels))


_TWO = {1: {"example_dialogue": _D1}, 2: {"example_dialogue": _D2}}


async def test_prebake_gate_off_does_nothing(monkeypatch) -> None:
    env = _wire(monkeypatch, {1: _row(1)}, settings=_settings(example_qwen_ns=False))
    await asyncio.wait_for(es.prebake_presets(), 2)
    assert env.qwen.health_calls == 0


async def test_prebake_hits_everything_without_synthesis(monkeypatch) -> None:
    rows = {1: _row(1, url=_expected(1, _D1)), 2: _row(2, url=_expected(2, _D2))}
    env = _wire(monkeypatch, rows, presets=_TWO)
    await asyncio.wait_for(es.prebake_presets(), 2)
    assert env.qwen.calls == []
    (summary,) = _named(env.metrics, "example_preset_prebake")
    assert (summary["hit"], summary["baked"], summary["reused"], summary["deferred"]) == (2, 0, 0, 0)


async def test_prebake_rebakes_when_ref_fingerprint_changed(monkeypatch) -> None:
    old = _expected(1, _D1, {"ai": "99999999", "user": "bbbbbbbb"})
    rows = {1: _row(1, url=old)}
    env = _wire(monkeypatch, rows)
    await asyncio.wait_for(es.prebake_presets(), 2)
    assert rows[1].example_audio_url == _expected(1, _D1)
    assert _named(env.metrics, "example_preset_prebake")[0]["baked"] == 1


async def test_prebake_reuses_existing_object_without_synthesis(monkeypatch) -> None:
    want = _expected(1, _D1)
    rows = {1: _row(1, url="examples/1-old.wav")}
    env = _wire(monkeypatch, rows, storage=_Storage({want: b"x"}))
    await asyncio.wait_for(es.prebake_presets(), 2)
    assert rows[1].example_audio_url == want and env.qwen.calls == []
    assert _named(env.metrics, "example_preset_prebake")[0]["reused"] == 1


async def test_prebake_waits_while_down_then_bakes(monkeypatch) -> None:
    rows = {1: _row(1)}
    env = _wire(monkeypatch, rows, qwen=_Qwen(states=("down", "rolling", "ok")))
    monkeypatch.setattr(es, "_NS_PRESET_RETRY_SEC", 0.01)
    monkeypatch.setattr(es, "_NS_PRESET_ROUNDS", 1)
    await asyncio.wait_for(es.prebake_presets(), 2)
    waits = [m.get("waiting") for m in _named(env.metrics, "example_preset_prebake")]
    assert waits[:2] == ["down", "rolling"] and rows[1].example_audio_url == _expected(1, _D1)


async def test_prebake_ends_on_old_workers_without_refs(monkeypatch) -> None:
    rows = {1: _row(1)}
    env = _wire(monkeypatch, rows, qwen=_Qwen(states=("no_refs",)))
    await asyncio.wait_for(es.prebake_presets(), 2)
    assert env.qwen.calls == [] and _named(env.metrics, "example_preset_prebake")[0]["reason"] == "no_refs"


async def test_prebake_ref_mismatch_stops_the_round(monkeypatch) -> None:
    rows = {1: _row(1), 2: _row(2)}
    qwen = _Qwen(synth_refs={"ai": "cccccccc", "user": "bbbbbbbb"})
    env = _wire(monkeypatch, rows, qwen=qwen, presets=_TWO)
    monkeypatch.setattr(es, "_NS_PRESET_ROUNDS", 1)
    await asyncio.wait_for(es.prebake_presets(), 2)
    assert len(qwen.calls) == 1
    summary = _named(env.metrics, "example_preset_prebake")[0]
    assert summary["deferred"] == 2 and rows[1].example_audio_url is None


async def test_prebake_skips_rejected_render(monkeypatch) -> None:
    want = _expected(1, _D1)
    rows = {1: _row(1, url="examples/1-old.wav")}
    env = _wire(monkeypatch, rows, storage=_Storage({es._rejected_key(want): b"x"}))
    monkeypatch.setattr(es, "_NS_PRESET_ROUNDS", 1)
    await asyncio.wait_for(es.prebake_presets(), 2)
    assert env.qwen.calls == [] and rows[1].example_audio_url == "examples/1-old.wav"


async def test_custom_created_mid_round_waits_behind_one_preset(monkeypatch) -> None:
    rows = {1: _row(1), 2: _row(2), 42: _row(42, custom=True, dialogue=_DC)}
    qwen = _Qwen(delay=0.02)
    _wire(monkeypatch, rows, qwen=qwen, presets=_TWO)
    driver = asyncio.ensure_future(es.prebake_presets())
    while not qwen.calls:
        await asyncio.sleep(0.005)
    await asyncio.wait_for(asyncio.gather(_bake(42, 3), driver), 3)
    texts = [text for _, text in qwen.calls]
    owners = {t["text"]: sid for sid, d in ((1, _D1), (2, _D2), (42, _DC)) for t in d}
    order = [owners[text] for text in texts]
    assert order == [1, 1, 42, 42, 2, 2]


async def test_trigger_prebake_joins_a_running_driver(monkeypatch) -> None:
    rows = {1: _row(1)}
    _wire(monkeypatch, rows, qwen=_Qwen(delay=0.05))
    driver = asyncio.ensure_future(es.prebake_presets())
    await asyncio.sleep(0.01)
    assert es.trigger_prebake() is True
    await asyncio.wait_for(driver, 2)


async def test_unreachable_worker_is_waited_for_then_excluded(monkeypatch) -> None:
    rows = {1: _row(1)}
    qwen = _Qwen(unreachable=("http://w2.test",))
    env = _wire(monkeypatch, rows, qwen=qwen)
    monkeypatch.setattr(es, "_UNREACHABLE_GRACE_SEC", 0.0)
    await asyncio.wait_for(es.prebake_presets(), 2)
    assert qwen.connects == [{"http://w2.test"}]
    assert _named(env.metrics, "example_preset_prebake")[-1]["unreachable"] == "http://w2.test"


async def test_rebake_backs_up_then_bakes_fresh(monkeypatch) -> None:
    want = _expected(1, _D1)
    rows = {1: _row(1, url=want)}
    env = _wire(monkeypatch, rows, storage=_Storage({want: b"old"}))
    result = await asyncio.wait_for(es.rebake_preset(1, stamp="T1"), 2)
    assert result == {"key": want, "backup": f"examples/prev/{want.rsplit('/', 1)[-1]}.T1"}
    await _job(1)
    assert env.storage.copies == [(want, result["backup"])] and env.storage.uploads == [want]


async def test_rebake_copy_failure_does_not_bake(monkeypatch) -> None:
    storage = _Storage()
    storage.fail_copy = RuntimeError("권한 없음")
    env = _wire(monkeypatch, {1: _row(1)}, storage=storage)
    with pytest.raises(es.ExampleOpError) as err:
        await es.rebake_preset(1, stamp="T1")
    assert err.value.status == 503 and es._loop_jobs() == {} and env.qwen.calls == []


async def test_ops_refuse_when_switch_off(monkeypatch) -> None:
    _wire(monkeypatch, {1: _row(1)}, settings=_settings(example_qwen_ns=False))
    for call in (es.rebake_preset(1, stamp="T"), es.restore_preset(1, "examples/prev/x", stamp="T"),
                 es.reject_preset(1, "examples/1-old.wav")):
        with pytest.raises(es.ExampleOpError) as err:
            await call
        assert err.value.status == 409
    with pytest.raises(es.ExampleOpError):
        es.trigger_prebake()


async def test_restore_only_accepts_backups_of_the_expected_key(monkeypatch) -> None:
    want = _expected(1, _D1)
    prev = f"examples/prev/{want.rsplit('/', 1)[-1]}.T0"
    rows = {1: _row(1, url=want)}
    env = _wire(monkeypatch, rows, storage=_Storage({want: b"new", prev: b"old"}))
    with pytest.raises(es.ExampleOpError) as err:
        await es.restore_preset(1, "examples/prev/other.wav.T0", stamp="T1")
    assert err.value.status == 409
    await es.restore_preset(1, prev, stamp="T1")
    backup = f"examples/prev/{want.rsplit('/', 1)[-1]}.T1"
    assert env.storage.objects[want] == b"old" and env.storage.objects[backup] == b"new"
    assert rows[1].example_audio_url == want


async def test_reject_moves_render_aside_and_restores_r0(monkeypatch) -> None:
    current = _expected(1, _D1)
    rows = {1: _row(1, url=current)}
    env = _wire(monkeypatch, rows, storage=_Storage({current: b"bad", "examples/1-old.wav": b"el"}))
    await es.reject_preset(1, "examples/1-old.wav")
    assert rows[1].example_audio_url == "examples/1-old.wav"
    assert env.storage.objects[es._rejected_key(current)] == b"bad" and env.storage.deleted == [current]


async def test_reject_refuses_missing_restore_key_or_non_ns_row(monkeypatch) -> None:
    current = _expected(1, _D1)
    rows = {1: _row(1, url=current)}
    _wire(monkeypatch, rows, storage=_Storage({current: b"bad"}))
    with pytest.raises(es.ExampleOpError):
        await es.reject_preset(1, "examples/1-missing.wav")
    rows[1].example_audio_url = "examples/1-old.wav"
    with pytest.raises(es.ExampleOpError):
        await es.reject_preset(1, "examples/1-old.wav")


async def test_unreachable_worker_is_waited_for_within_grace(monkeypatch) -> None:
    class _Recovering(_Qwen):
        async def health_refs(self, needed):
            state = await super().health_refs(needed)
            self.unreachable = ()
            return state

    rows = {1: _row(1)}
    qwen = _Recovering(unreachable=("http://w2.test",))
    env = _wire(monkeypatch, rows, qwen=qwen)
    monkeypatch.setattr(es, "_NS_PRESET_RETRY_SEC", 0.01)
    await asyncio.wait_for(es.prebake_presets(), 2)
    summaries = _named(env.metrics, "example_preset_prebake")
    assert summaries[0]["waiting"] == "unreachable" and qwen.connects == [set()]
    assert rows[1].example_audio_url == _expected(1, _D1)


async def _status(call) -> int | None:
    try:
        await call
    except es.ExampleOpError as exc:
        return exc.status
    return None


async def _running_job(sid: int, qwen: _Qwen):
    job = es._ensure_job(sid, "rebake", force=True)
    while not qwen.calls:
        await asyncio.sleep(0.005)
    return job


async def test_ops_refuse_while_a_job_runs_and_change_nothing(monkeypatch) -> None:
    current = _expected(1, _D1)
    prev = f"examples/prev/{current.rsplit('/', 1)[-1]}.T0"
    rows = {1: _row(1, url=current)}
    qwen = _Qwen(delay=0.05)
    storage = _Storage({current: b"now", prev: b"old", "examples/1-old.wav": b"el"})
    env = _wire(monkeypatch, rows, storage=storage, qwen=qwen)
    job = await _running_job(1, qwen)
    assert await _status(es.reject_preset(1, "examples/1-old.wav")) == 409
    assert await _status(es.restore_preset(1, prev, stamp="T1")) == 409
    assert await _status(es.rebake_preset(1, stamp="T1")) == 409
    assert env.storage.copies == [] and env.storage.deleted == [] and env.storage.markers == []
    assert rows[1].example_audio_url == current
    await asyncio.wait_for(asyncio.shield(job.task), 2)
    await es.reject_preset(1, "examples/1-old.wav")
    assert rows[1].example_audio_url == "examples/1-old.wav"


async def test_rebake_and_restore_refuse_skip_listed_preset_but_reject_works(monkeypatch) -> None:
    current = _expected(1, _D1)
    prev = f"examples/prev/{current.rsplit('/', 1)[-1]}.T0"
    rows = {1: _row(1, url=current)}
    storage = _Storage({current: b"now", prev: b"old", "examples/1-old.wav": b"el"})
    env = _wire(monkeypatch, rows, storage=storage, settings=_settings(example_ns_skip_presets="1"))
    assert await _status(es.rebake_preset(1, stamp="T1")) == 409
    assert await _status(es.restore_preset(1, prev, stamp="T1")) == 409
    assert es._loop_jobs() == {} and env.qwen.calls == [] and env.storage.copies == []
    await es.reject_preset(1, "examples/1-old.wav")
    assert rows[1].example_audio_url == "examples/1-old.wav"


async def test_trigger_during_a_round_runs_one_more_round(monkeypatch) -> None:
    rows = {1: _row(1, url=_expected(1, _D1))}
    qwen = _Qwen()
    qwen.health_delay = 0.05
    env = _wire(monkeypatch, rows, qwen=qwen)
    driver = asyncio.ensure_future(es.prebake_presets())
    while qwen.health_calls == 0:
        await asyncio.sleep(0.005)
    assert es.trigger_prebake() is True
    await asyncio.wait_for(driver, 2)
    assert qwen.health_calls == 2
    assert [m["rounds"] for m in _named(env.metrics, "example_preset_prebake")] == [1, 2]


async def test_trigger_starts_a_single_counted_round(monkeypatch) -> None:
    rows = {1: _row(1)}
    env = _wire(monkeypatch, rows, qwen=_Qwen(ns=False))
    monkeypatch.setattr(es, "_NS_PRESET_RETRY_SEC", 0.0)
    assert es.trigger_prebake() is False
    driver = es._loop_driver().task
    await asyncio.wait_for(asyncio.shield(driver), 2)
    assert len(_named(env.metrics, "example_preset_prebake")) == 1


async def test_shutdown_cancels_the_driver_before_jobs(monkeypatch) -> None:
    rows = {1: _row(1)}
    qwen = _Qwen()
    qwen.health_delay = 0.1
    _wire(monkeypatch, rows, qwen=qwen)
    es.trigger_prebake()
    driver = es._loop_driver().task
    while qwen.health_calls == 0:
        await asyncio.sleep(0.005)
    await es.cancel_example_jobs()
    assert driver.done()
    await asyncio.sleep(0.15)
    assert es._loop_jobs() == {} and qwen.calls == []


async def test_creation_bake_returns_without_waiting_for_the_job(monkeypatch) -> None:
    rows = {42: _row(42, custom=True, dialogue=_DC)}
    _wire(monkeypatch, rows, qwen=_Qwen(delay=0.05))
    task = await asyncio.wait_for(es.bake_example_audio(42), 0.5)
    assert task is not None and not task.done()
    assert (await asyncio.wait_for(task, 2)).endswith("-qn.wav")


async def test_custom_skips_a_taken_worker_that_is_not_ready(monkeypatch) -> None:
    rows = {42: _row(42, custom=True, dialogue=_DC)}
    qwen = _Qwen()
    qwen.worker_states = {"http://w1.test": "warming"}
    _wire(monkeypatch, rows, qwen=qwen)
    assert (await _bake(42)).endswith("-qn.wav")
    assert qwen.probed == ["http://w1.test", "http://w2.test"]
    assert qwen.connects == [set(), {"http://w1.test"}]


async def test_custom_waits_out_a_short_busy_on_the_taken_worker(monkeypatch) -> None:
    rows = {42: _row(42, custom=True, dialogue=_DC)}
    qwen = _Qwen()
    qwen.worker_states = {"http://w1.test": ["busy", "busy", "ready"]}
    _wire(monkeypatch, rows, qwen=qwen)
    monkeypatch.setattr(es, "_BUSY_PROBE_SEC", 0.01)
    assert (await _bake(42)).endswith("-qn.wav")
    assert qwen.probed == ["http://w1.test"] * 3 and qwen.connects == [set()]


async def test_custom_skips_a_worker_stuck_busy(monkeypatch) -> None:
    rows = {42: _row(42, custom=True, dialogue=_DC)}
    qwen = _Qwen()
    qwen.worker_states = {"http://w1.test": "busy"}
    _wire(monkeypatch, rows, qwen=qwen)
    monkeypatch.setattr(es, "_BUSY_PROBE_SEC", 0.0)
    assert (await _bake(42)).endswith("-qn.wav")
    assert qwen.probed.count("http://w1.test") == es._BUSY_PROBES
    assert qwen.connects == [set(), {"http://w1.test"}]


async def test_custom_with_no_ready_worker_falls_back(monkeypatch) -> None:
    rows = {42: _row(42, custom=True, dialogue=_DC)}
    qwen = _Qwen()
    qwen.worker_states = {"http://w1.test": "warming", "http://w2.test": "down"}
    env = _wire(monkeypatch, rows, qwen=qwen)
    key = await _bake(42)
    assert "-qn" not in key and qwen.calls == [] and env.el.opens == 2


async def test_preset_request_without_any_synthesis_does_not_cool_down(monkeypatch) -> None:
    rows = {1: _row(1)}
    qwen = _Qwen()
    qwen.worker_states = {"http://w1.test": "warming", "http://w2.test": "down"}
    env = _wire(monkeypatch, rows, qwen=qwen)
    await asyncio.wait_for(es.get_example_conversation(_DB(rows), 1, user_id=7), 2)
    await _job(1)
    assert _named(env.metrics, "example_tts_deferred")[0]["ns_attempts_wasted"] == 0
    assert not es._cooling(1)


async def test_preset_rounds_do_not_probe_worker_health(monkeypatch) -> None:
    rows = {1: _row(1)}
    qwen = _Qwen()
    _wire(monkeypatch, rows, qwen=qwen)
    await asyncio.wait_for(es.prebake_presets(), 2)
    assert qwen.probed == [] and rows[1].example_audio_url == _expected(1, _D1)


async def test_reject_without_source_object_still_marks_the_render(monkeypatch) -> None:
    current = _expected(1, _D1)
    rows = {1: _row(1, url=current)}
    env = _wire(monkeypatch, rows, storage=_Storage({"examples/1-old.wav": b"el"}))
    monkeypatch.setattr(es, "_NS_PRESET_ROUNDS", 1)
    await es.reject_preset(1, "examples/1-old.wav")
    assert env.storage.markers == [es._rejected_key(current)]
    await asyncio.wait_for(es.prebake_presets(), 2)
    assert env.qwen.calls == [] and rows[1].example_audio_url == "examples/1-old.wav"


async def test_rejected_preset_does_not_keep_the_driver_going(monkeypatch) -> None:
    want = _expected(1, _D1)
    rows = {1: _row(1, url="examples/1-old.wav")}
    env = _wire(monkeypatch, rows, storage=_Storage({es._rejected_key(want): b""}))
    monkeypatch.setattr(es, "_NS_PRESET_RETRY_SEC", 0.0)
    await asyncio.wait_for(es.prebake_presets(), 2)
    (summary,) = _named(env.metrics, "example_preset_prebake")
    assert (summary["rejected"], summary["deferred"]) == (1, 0)


async def test_deferred_preset_cools_down_until_triggered(monkeypatch) -> None:
    rows = {1: _row(1)}
    env = _wire(monkeypatch, rows, qwen=_Qwen(ns=False))
    await asyncio.wait_for(es.get_example_conversation(_DB(rows), 1, user_id=7), 2)
    await _job(1)
    calls = len(env.qwen.calls)
    resp = await asyncio.wait_for(es.get_example_conversation(_DB(rows), 1, user_id=7), 2)
    assert resp.audio_url is None and len(env.qwen.calls) == calls and es._loop_jobs() == {}
    assert _named(env.metrics, "example_tts_deferred")[0]["ns_attempts_wasted"] == 1
    es.trigger_prebake()
    assert not es._cooling(1)
    await asyncio.wait_for(asyncio.shield(es._loop_driver().task), 2)


async def test_rebake_clears_the_cooldown(monkeypatch) -> None:
    rows = {1: _row(1)}
    _wire(monkeypatch, rows)
    es._loop_cooldown()[1] = 10**12
    await es.rebake_preset(1, stamp="T1")
    assert not es._cooling(1)
    await _job(1)


async def test_round_summary_names_v8_and_mismatch(monkeypatch) -> None:
    rows = {1: _row(1), 2: _row(2)}
    qwen = _Qwen(synth_refs={"ai": "cccccccc", "user": "bbbbbbbb"})
    env = _wire(monkeypatch, rows, qwen=qwen, presets=_TWO)
    monkeypatch.setattr(es, "_NS_PRESET_ROUNDS", 1)
    await asyncio.wait_for(es.prebake_presets(), 2)
    summary = _named(env.metrics, "example_preset_prebake")[0]
    v8 = {sid: _expected(sid, d).rsplit("-", 2)[-2] for sid, d in ((1, _D1), (2, _D2))}
    assert summary["v8"] == f"1:{v8[1]},2:{v8[2]}" and summary["mismatch"] == "1:voice_ref"
    assert summary["rounds"] == 1


async def test_fallback_metric_counts_wasted_renders(monkeypatch) -> None:
    rows = {42: _row(42, custom=True, dialogue=_DC)}
    env = _wire(monkeypatch, rows, qwen=_Qwen(ns=False))
    await _bake(42)
    plain = [m for m in _named(env.metrics, "example_tts") if m.get("ns_fallback_reason")]
    assert plain[0]["ns_attempts_wasted"] == 1


async def test_s3_head_error_defers_instead_of_overwriting(monkeypatch) -> None:
    want = _expected(1, _D1)
    rows = {1: _row(1, url="examples/1-old.wav")}
    storage = _Storage({want: b"approved"})
    storage.fail_head = RuntimeError("S3 503")
    env = _wire(monkeypatch, rows, storage=storage)
    monkeypatch.setattr(es, "_NS_PRESET_ROUNDS", 1)
    await asyncio.wait_for(es.prebake_presets(), 2)
    assert env.qwen.calls == [] and env.storage.objects[want] == b"approved"
    summary = _named(env.metrics, "example_preset_prebake")[0]
    assert summary["deferred"] == 1 and summary["s3_error"] == "1"


async def test_restore_copy_error_is_a_503(monkeypatch) -> None:
    want = _expected(1, _D1)
    prev = f"examples/prev/{want.rsplit('/', 1)[-1]}.T0"
    storage = _Storage({want: b"new", prev: b"old"})
    storage.fail_copy = RuntimeError("권한 없음")
    _wire(monkeypatch, {1: _row(1, url=want)}, storage=storage)
    assert await _status(es.restore_preset(1, prev, stamp="T1")) == 503


@pytest.mark.parametrize("op", ["restore", "rebake"])
async def test_ops_recheck_for_a_job_started_while_reading_refs(monkeypatch, op) -> None:
    current = _expected(1, _D1)
    prev = f"examples/prev/{current.rsplit('/', 1)[-1]}.T0"
    rows = {1: _row(1, url=current)}
    qwen = _Qwen(delay=0.05)
    qwen.health_delay = 0.05
    env = _wire(monkeypatch, rows, qwen=qwen, storage=_Storage({current: b"now", prev: b"old"}))
    call = es.restore_preset(1, prev, stamp="T1") if op == "restore" else es.rebake_preset(1, stamp="T1")
    pending = asyncio.ensure_future(call)
    while qwen.health_calls == 0:
        await asyncio.sleep(0.005)
    job = es._ensure_job(1, "request", force=True)
    assert await _status(asyncio.wait_for(pending, 2)) == 409 and env.storage.copies == []
    await asyncio.wait_for(asyncio.shield(job.task), 2)


async def test_restore_refuses_a_rejection_marker(monkeypatch) -> None:
    want = _expected(1, _D1)
    marker = es._rejected_key(want)
    rows = {1: _row(1, url="examples/1-old.wav")}
    env = _wire(monkeypatch, rows, storage=_Storage({marker: b""}))
    assert await _status(es.restore_preset(1, marker, stamp="T1")) == 409
    assert want not in env.storage.objects and rows[1].example_audio_url == "examples/1-old.wav"


async def test_reuse_rechecks_the_marker_under_the_scene_lock(monkeypatch) -> None:
    want = _expected(1, _D1)

    class _Racing(_Storage):
        def strict_exists(self, key: str) -> bool:
            found = super().strict_exists(key)
            if key == want and found:
                self.objects[es._rejected_key(want)] = b""
            return found

    rows = {1: _row(1, url="examples/1-old.wav")}
    env = _wire(monkeypatch, rows, storage=_Racing({want: b"approved"}))
    monkeypatch.setattr(es, "_NS_PRESET_ROUNDS", 1)
    await asyncio.wait_for(es.prebake_presets(), 2)
    assert rows[1].example_audio_url == "examples/1-old.wav" and env.qwen.calls == []
    (summary,) = _named(env.metrics, "example_preset_prebake")
    assert (summary["reused"], summary["rejected"]) == (0, 1)
