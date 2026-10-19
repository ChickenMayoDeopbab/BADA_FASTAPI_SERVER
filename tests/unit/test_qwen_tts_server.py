import asyncio
import gc
import importlib.util
import json
import socket
import threading
import time
import types
from pathlib import Path

import httpx
import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "qwen_tts_server" / "server.py"
_STEP_S = 0.002
_SLOW_MACHINE_S = 60.0
_TEXT60 = "가" * 60


class _T:
    def __init__(self, a) -> None:
        self.a = np.asarray(a, dtype=np.float32)

    def numpy(self) -> np.ndarray:
        return self.a


def _resample(t: _T, a: int, b: int) -> _T:
    x = t.a
    n = int(np.ceil(len(x) * b / a))
    if not len(x):
        return _T(x)
    return _T(np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x))


@pytest.fixture(scope="module")
def srv():
    torch = types.ModuleType("torch")
    torch.from_numpy = _T
    torch.bfloat16 = "bf16"
    taf = types.ModuleType("torchaudio.functional")
    taf.resample = _resample
    ta = types.ModuleType("torchaudio")
    ta.functional = taf
    qt = types.ModuleType("qwen_tts")
    qt.Qwen3TTSModel = object
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("QWEN_STOP_MODE", raising=False)
        for name, mod in {
            "torch": torch,
            "torchaudio": ta,
            "torchaudio.functional": taf,
            "qwen_tts": qt,
        }.items():
            mp.setitem(__import__("sys").modules, name, mod)
        spec = importlib.util.spec_from_file_location("qwen_tts_server_under_test", _SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return mod


class _Fake:
    """포크 stream_generate_pcm 과 같은 꼴"""

    def __init__(
        self, eos_after: int | None = None, step_s: float = _STEP_S, flush_scale: float = 1.0
    ) -> None:
        self.eos_after, self.step_s, self.flush_scale = eos_after, step_s, flush_scale
        self.calls: list[dict] = []
        self.frames = 0
        self.closed = 0
        self.threads: set[str] = set()

    def stream_generate_voice_clone(self, *, text, emit_every_frames=8, max_frames=10000, **kw):
        self.calls.append({"text": text, "max_frames": max_frames, **kw})
        buf = 0
        try:
            for i in range(max_frames):
                self.threads.add(threading.current_thread().name)
                if self.eos_after is not None and i >= self.eos_after:
                    break
                if self.step_s:
                    time.sleep(self.step_s)
                self.frames += 1
                buf += 1
                if buf == emit_every_frames:
                    buf = 0
                    yield np.full(1920 * emit_every_frames, 0.1, np.float32), 24000
            if buf:  # 포크 flush 는 decode 가 첫 코드북 0 인 프레임만큼 잘라 더 짧을 수 있다
                yield np.full(int(1920 * buf * self.flush_scale), 0.1, np.float32), 24000
        except GeneratorExit:
            self.closed += 1
            raise


@pytest.fixture(autouse=True)
def _state(srv):
    saved = (dict(srv.S), srv.STOP_MODE, srv.STREAM_LOCK_TIMEOUT)
    gc.disable()
    try:
        yield
        for th in threading.enumerate():
            if th.name == "tts-stream":
                th.join(timeout=_SLOW_MACHINE_S)
        assert not [t for t in threading.enumerate() if t.name == "tts-stream"], "생산 스레드가 남았다"
    finally:
        gc.enable()
        srv.S.clear()
        srv.S.update(saved[0])
        srv.STOP_MODE, srv.STREAM_LOCK_TIMEOUT = saved[1], saved[2]


def _reset(srv, model: _Fake, mode: str = "drain") -> None:
    srv.S.update(model=model, voices={"ai": ("r.wav", "t")}, ready=True, lock=threading.Lock())
    srv.STOP_MODE = mode


def _wait_unlocked(srv, limit: float) -> float | None:
    t = time.perf_counter()
    while srv.S["lock"].locked():
        if time.perf_counter() - t > limit:
            return None
        time.sleep(0.005)
    return time.perf_counter() - t


def _cap_wall(srv, text: str) -> float:
    return max(_SLOW_MACHINE_S, srv.max_frames_for(text) * _STEP_S * 1.5 + 1.0)


async def _drive(srv, text: str, spec_version: str, send_mode: str, keep: list, peer_closed=None):
    resp = srv.tts_stream(srv.Req(voice="ai", text=text))
    keep.append(resp)
    got = asyncio.Event()
    n = {"recv": 0, "body": 0}

    async def receive():
        n["recv"] += 1
        if n["recv"] == 1:
            return {"type": "http.request", "body": b"", "more_body": False}
        if send_mode == "complete":
            await asyncio.sleep(3600)
        if send_mode == "block":
            await peer_closed.wait()
            return {"type": "http.disconnect"}
        await got.wait()
        return {"type": "http.disconnect"}

    async def send(m):
        if m["type"] == "http.response.body" and m.get("body"):
            n["body"] += len(m["body"])
            if got.is_set() and send_mode == "oserror":
                raise OSError("peer gone")
            if got.is_set() and send_mode == "block":
                await peer_closed.wait()
                return
            got.set()

    err = None
    try:
        await asyncio.wait_for(
            resp({"type": "http", "asgi": {"spec_version": spec_version}}, receive, send),
            timeout=30.0,
        )
    except Exception as e:
        err = type(e).__name__
    return err, n["body"]


def test_cap_formula_covers_observed_speech(srv) -> None:
    observed = {13: 1.44, 20: 2.88, 24: 2.48, 25: 6.64, 27: 5.28, 30: 4.80, 32: 6.88,
                43: 7.36, 70: 9.12, 75: 9.28}
    worst = max(sec / (srv.max_frames_for("x" * k) * srv.FRAME_S) for k, sec in observed.items())
    assert worst <= 0.6, f"정상 발화가 상한에 너무 가깝다: {worst:.2f}"
    assert srv.max_frames_for("x" * srv.MAX_CHARS) < 10000, "포크 기본값보다 작아야 의미가 있다"


def test_cap_ends_on_emit_boundary(srv) -> None:
    assert all(srv.max_frames_for("x" * n) % srv.EMIT == 0 for n in range(srv.MAX_CHARS + 1))


def test_cap_detected_even_when_fork_flush_is_short(srv) -> None:
    m = _Fake(step_s=0, flush_scale=0.1)
    _reset(srv, m)

    r = TestClient(srv.app).post("/v1/tts", json={"voice": "ai", "text": _TEXT60})

    assert r.status_code == 502, "옹알이가 정상(200)으로 구워지면 안 된다"


def test_default_stop_mode_is_drain(srv) -> None:
    assert srv.STOP_MODE == "drain", "close 는 GPU 점검 뒤 env 로만 켠다"


def test_generate_passes_max_frames_for_stream_and_batch(srv) -> None:
    text = "네, 확인해 드리겠습니다."
    m = _Fake(eos_after=10, step_s=0)
    _reset(srv, m)

    TestClient(srv.app).post("/v1/tts", json={"voice": "ai", "text": text})
    asyncio.run(_drive(srv, text, "2.3", "complete", []))

    assert [c["max_frames"] for c in m.calls] == [srv.max_frames_for(text)] * 2


def test_disconnect_spec23_close_mode_releases_lock_without_gc(srv) -> None:
    m = _Fake()
    _reset(srv, m, mode="close")

    asyncio.run(_drive(srv, _TEXT60, "2.3", "ignore", keep := []))

    assert _wait_unlocked(srv, 5.0) is not None, "끊긴 스트림이 락을 쥐고 있다"
    assert m.closed == 1 and m.frames < 20
    assert m.threads == {"tts-stream"}, "생성은 생산 스레드 하나에서만 돌아야 한다"
    assert keep


def test_disconnect_spec23_drain_mode_runs_to_cap_without_close(srv) -> None:
    m = _Fake()
    _reset(srv, m, mode="drain")

    asyncio.run(_drive(srv, _TEXT60, "2.3", "ignore", []))

    assert _wait_unlocked(srv, _cap_wall(srv, _TEXT60)) is not None
    assert m.frames == srv.max_frames_for(_TEXT60) and m.closed == 0


def test_busy_stays_true_until_producer_finishes(srv) -> None:
    m = _Fake(step_s=0.005)
    _reset(srv, m, mode="drain")
    client = TestClient(srv.app)

    asyncio.run(_drive(srv, _TEXT60, "2.3", "ignore", []))

    assert client.get("/health").json()["busy"] is True, "소비자가 떠나도 생성 중이면 busy"
    assert _wait_unlocked(srv, _SLOW_MACHINE_S) is not None
    assert client.get("/health").json()["busy"] is False


def test_abandoned_iterator_spec24_oserror_released_within_cap(srv) -> None:
    m = _Fake()
    _reset(srv, m, mode="close")

    keep: list = []

    async def run():
        r = await _drive(srv, _TEXT60, "2.4", "oserror", keep)
        return r, await asyncio.to_thread(_wait_unlocked, srv, _cap_wall(srv, _TEXT60))

    (_err, _body), waited = asyncio.run(run())

    assert waited is not None, "Starlette 가 이터레이터를 방치해도 상한 안에 풀려야 한다"
    assert m.frames == srv.max_frames_for(_TEXT60) and m.closed == 0, "방치 경로를 탔어야 한다"


def test_send_ignored_spec24_stops_at_cap_and_aborts(srv) -> None:
    m = _Fake()
    _reset(srv, m)

    err, _body = asyncio.run(_drive(srv, _TEXT60, "2.4", "ignore", []))

    assert m.frames == srv.max_frames_for(_TEXT60), "포크 기본 10000 프레임까지 돌면 안 된다"
    assert err == "CapHitError", "상한은 정상 종료가 아니라 스트림 중단이어야 호출측이 폴백한다"
    assert _wait_unlocked(srv, 1.0) is not None


def test_normal_eos_streams_full_body_and_unlocks(srv) -> None:
    text = "네, 확인해 드리겠습니다."
    m = _Fake(eos_after=41)
    _reset(srv, m)

    err, body = asyncio.run(_drive(srv, text, "2.3", "complete", []))

    assert err is None
    assert body == 41 * 1280 * 2, "24k 1920 샘플 → 16k 1280 샘플 × 2바이트"
    assert not srv.S["lock"].locked(), "응답이 끝났으면 이미 비어 있어야 한다"


def test_busy_aborts_stream_and_keeps_foreign_lock(srv) -> None:
    m = _Fake()
    _reset(srv, m)
    srv.STREAM_LOCK_TIMEOUT = 0.2
    srv.S["lock"].acquire()
    try:
        err, _ = asyncio.run(_drive(srv, _TEXT60, "2.3", "complete", []))
        assert err == "RuntimeError" and not m.calls
        assert srv.S["lock"].locked(), "남이 쥔 락을 풀면 안 된다"
    finally:
        srv.S["lock"].release()


def test_cancel_while_waiting_for_lock_never_generates(srv) -> None:
    m = _Fake()
    _reset(srv, m)
    srv.S["lock"].acquire()

    async def run():
        agen = srv._stream_pcm("ai", _TEXT60)
        task = asyncio.ensure_future(agen.__anext__())
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    srv.S["lock"].release()

    for th in threading.enumerate():
        if th.name == "tts-stream":
            th.join(timeout=10.0)
    assert not m.calls, "소비자가 떠났으면 락을 얻은 뒤에도 생성을 시작하면 안 된다"
    assert not srv.S["lock"].locked()


def test_batch_babble_returns_502_and_unlocks(srv) -> None:
    m = _Fake(step_s=0)
    _reset(srv, m)

    r = TestClient(srv.app).post("/v1/tts", json={"voice": "ai", "text": _TEXT60})

    assert r.status_code == 502
    assert m.frames == srv.max_frames_for(_TEXT60)
    assert not srv.S["lock"].locked()


def test_batch_eos_returns_full_pcm(srv) -> None:
    m = _Fake(eos_after=30, step_s=0)
    _reset(srv, m)

    r = TestClient(srv.app).post("/v1/tts", json={"voice": "ai", "text": "네, 안녕하세요."})

    assert r.status_code == 200 and len(r.content) == 30 * 1280 * 2


def test_eos_at_cap_minus_one_is_not_flagged(srv) -> None:
    cap = srv.max_frames_for("네")
    m = _Fake(eos_after=cap - 1, step_s=0)
    _reset(srv, m)

    r = TestClient(srv.app).post("/v1/tts", json={"voice": "ai", "text": "네"})

    assert r.status_code == 200


def _wav(path: Path, seed: int = 0, seconds: float = 0.5) -> Path:
    import soundfile as sf

    x = np.random.default_rng(seed).normal(0, 0.1, int(24000 * seconds)).astype(np.float32)
    sf.write(str(path), x, 24000, subtype="PCM_16")
    return path


def test_warmup_cap_hit_does_not_abort_startup(srv, tmp_path, monkeypatch) -> None:
    voices = tmp_path / "voices.json"
    ref = _wav(tmp_path / "r.wav")
    voices.write_text(json.dumps({"ai": {"ref_audio": str(ref), "ref_text": "t"}}))
    fake = _Fake(step_s=0)
    fake.enable_streaming_optimizations = lambda: None
    monkeypatch.setattr(srv, "VOICES_F", str(voices))
    monkeypatch.setattr(
        srv, "Qwen3TTSModel", types.SimpleNamespace(from_pretrained=lambda *a, **k: fake)
    )
    srv.S.update(ready=False, lock=threading.Lock())

    with TestClient(srv.app) as client:
        assert client.get("/health").json() == {
            "ready": True, "voices": ["ai"], "busy": False, "refs": {"ai": srv.ref_fingerprint(ref.read_bytes(), "t")},
        }


def test_fastapi_sentence_guard_always_trips_before_worker_cap(srv) -> None:
    from app.services.qwen_tts import runaway_cap_ms

    margin_ms = 2000
    for n in range(1, srv.MAX_CHARS + 1):
        for text in ("가" * n, "가" * n + " "):  # FastAPI 는 끝 공백까지 붙여 보낸다
            worker_ms = srv.max_frames_for(text) * srv.FRAME_S * 1000
            assert runaway_cap_ms(text) + margin_ms <= worker_ms, f"{n}자에서 워커가 먼저 끊는다"


@pytest.mark.parametrize("mode", ["drain", "close"])
def test_stalled_reader_spec23_releases_lock_at_cap_while_send_blocks(srv, mode) -> None:
    m = _Fake()
    _reset(srv, m, mode=mode)

    async def run():
        peer_closed = asyncio.Event()
        task = asyncio.ensure_future(_drive(srv, _TEXT60, "2.3", "block", [], peer_closed))
        while not m.frames:
            await asyncio.sleep(0.001)
        waited = await asyncio.to_thread(_wait_unlocked, srv, _cap_wall(srv, _TEXT60))
        still_streaming = not task.done()
        peer_closed.set()
        err, _ = await task
        return waited, still_streaming, err

    waited, still_streaming, err = asyncio.run(run())

    assert waited is not None, "send 가 막혀도 락은 상한 안에 풀려야 한다"
    assert still_streaming, "락이 풀릴 때 HTTP 쪽은 아직 send 에서 막혀 있어야 이 경로를 탄 것이다"
    assert m.frames == srv.max_frames_for(_TEXT60) and m.closed == 0
    assert err == "CapHitError", "상대가 닫히면 남은 큐를 비우고 상한 예외로 끝난다"


@pytest.mark.parametrize("http", ["h11", "httptools"])
@pytest.mark.parametrize("mode", ["drain", "close"])
def test_real_uvicorn_client_disconnect_releases_lock(srv, mode, http) -> None:
    if http == "httptools":
        pytest.importorskip("httptools")
    m = _Fake()
    _reset(srv, m, mode=mode)
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(srv.app, lifespan="off", http=http, log_config=None, access_log=False))
    th = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    th.start()
    try:
        deadline = time.perf_counter() + 5
        while not server.started and time.perf_counter() < deadline:
            time.sleep(0.01)
        url = f"http://127.0.0.1:{port}"
        with httpx.Client() as c, c.stream("POST", url + "/v1/tts/stream", json={"voice": "ai", "text": _TEXT60}) as r:
            assert r.status_code == 200
            next(r.iter_raw())
        limit = 5.0 if mode == "close" else _cap_wall(srv, _TEXT60)
        assert _wait_unlocked(srv, limit) is not None
        if mode == "close":
            assert m.closed == 1 and m.frames < 20, "끊김(spec 2.3 취소)이 바로 stop 을 세워야 한다"
        else:
            assert m.frames == srv.max_frames_for(_TEXT60) and m.closed == 0
        assert httpx.get(url + "/health").json()["busy"] is False
    finally:
        server.should_exit = True
        th.join(timeout=5)


# ── 계획 0071 E 기능 1: 요청 단위 non_streaming_mode ──


def _post(srv, path: str, body: dict):
    return TestClient(srv.app).post(path, json={"voice": "ai", **body})


@pytest.mark.parametrize("path", ["/v1/tts", "/v1/tts/stream"])
def test_request_without_field_calls_fork_exactly_as_before(srv, path) -> None:
    m = _Fake(eos_after=10, step_s=0)
    _reset(srv, m)

    r = _post(srv, path, {"text": "네, 확인해 드리겠습니다."})

    assert r.status_code == 200
    assert "non_streaming_mode" not in m.calls[0], "옛 FastAPI 요청은 포크 인자가 예전과 같아야 한다"
    assert r.headers["x-non-streaming"] == "0"


@pytest.mark.parametrize("path", ["/v1/tts", "/v1/tts/stream"])
def test_non_streaming_flag_reaches_fork_with_same_cap(srv, path) -> None:
    text = "네, 확인해 드리겠습니다."
    m = _Fake(eos_after=10, step_s=0)
    _reset(srv, m)

    r = _post(srv, path, {"text": text, "non_streaming_mode": True})

    assert r.status_code == 200
    assert m.calls[0]["non_streaming_mode"] is True
    assert m.calls[0]["max_frames"] == srv.max_frames_for(text), "상한은 모드와 무관하게 같은 식"
    assert r.headers["x-non-streaming"] == "1", "호출측은 이 헤더로만 N 이 적용됐다고 본다"


def test_unknown_request_fields_are_ignored(srv) -> None:
    m = _Fake(eos_after=10, step_s=0)
    _reset(srv, m)

    r = _post(srv, "/v1/tts", {"text": "네.", "future_field": 1})

    assert r.status_code == 200, "새 FastAPI 가 옛 워커에 새 필드를 보내도 422 가 나면 안 된다(롤링 배포)"


def test_cap_formula_covers_observed_n_speech(srv) -> None:
    # 학교 3090 ai N 460요청(가드 걸린 1건 제외) 길이별 최장 — 뒤 무음 ~0.9s 포함
    observed = {7: 3.68, 9: 2.88, 11: 3.2, 12: 3.36, 14: 3.52, 15: 6.16, 21: 4.4, 24: 4.8, 27: 6.32, 46: 7.44}
    worst = max(sec / (srv.max_frames_for("x" * k) * srv.FRAME_S) for k, sec in observed.items())
    assert worst <= 0.6, f"N 정상 발화가 상한에 너무 가깝다: {worst:.2f}"


class _Rec(_Fake):
    def __init__(self, srv, **k) -> None:
        super().__init__(**k)
        self.srv = srv
        self.ready_at_call: list[bool] = []

    def stream_generate_voice_clone(self, **kw):
        self.ready_at_call.append(self.srv.S["ready"])
        return super().stream_generate_voice_clone(**kw)


def _boot(srv, tmp_path, monkeypatch, voices: dict, fake) -> dict:
    vf = tmp_path / "voices.json"
    paths = {k: _wav(tmp_path / a, seed=i) for i, (k, a) in enumerate(voices.items())}
    vf.write_text(json.dumps({k: {"ref_audio": str(p), "ref_text": "t"} for k, p in paths.items()}))
    fake.enable_streaming_optimizations = lambda: None
    monkeypatch.setattr(srv, "VOICES_F", str(vf))
    monkeypatch.setattr(srv, "Qwen3TTSModel", types.SimpleNamespace(from_pretrained=lambda *a, **k: fake))
    srv.S.update(ready=False, lock=threading.Lock())
    with TestClient(srv.app) as client:
        return client.get("/health").json()


@pytest.mark.parametrize("voices", [{"user": "u.wav", "ai": "a.wav"}, {"ai": "a.wav"}])
def test_warmup_runs_non_streaming_twice_with_distinct_lengths_before_ready(srv, tmp_path, monkeypatch, voices) -> None:
    fake = _Rec(srv, eos_after=6, step_s=0)

    health = _boot(srv, tmp_path, monkeypatch, voices, fake)

    s0 = [c for c in fake.calls if "non_streaming_mode" not in c]
    ns = [c for c in fake.calls if c.get("non_streaming_mode") is True]
    snap = {v: srv.S["voices"][v][0] for v in voices}

    def _owner(call):
        return next(v for v, ref in snap.items() if call["ref_audio"] is ref)

    assert [_owner(c) for c in s0] == list(voices), "보이스마다 S0 워밍업은 그대로"
    assert len(ns) == 2 and len({len(c["text"]) for c in ns}) == 2, "N 워밍업은 길이가 다른 글 2개"
    assert {_owner(c) for c in ns} == {next(iter(voices))}
    order = [id(c) for c in fake.calls]
    assert order.index(id(ns[0])) > order.index(id(s0[-1])), "S0 워밍업 뒤에"
    assert fake.ready_at_call == [False] * len(fake.calls), "워밍업은 전부 ready 전에 끝난다"
    assert health == {"ready": True, "voices": list(voices), "busy": False, "refs": dict(srv.S["refs"])}
    assert set(health["refs"]) == set(voices)


def _load_check_ref():
    spec = importlib.util.spec_from_file_location("check_ref_under_test", _SCRIPT.parent / "check_ref.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fingerprint_depends_on_bytes_and_text_not_path(srv, tmp_path) -> None:
    a = _wav(tmp_path / "x.wav", seed=1).read_bytes()
    b = _wav(tmp_path / "x.wav", seed=2).read_bytes()
    assert srv.ref_fingerprint(a, "t") != srv.ref_fingerprint(b, "t")
    assert srv.ref_fingerprint(a, "t") != srv.ref_fingerprint(a, "u")
    assert len(srv.ref_fingerprint(a, "t")) == 8


def test_check_ref_fingerprint_matches_worker(srv) -> None:
    check_ref = _load_check_ref()
    rng = np.random.default_rng(5)
    for _ in range(20):
        data = rng.bytes(int(rng.integers(1, 5000)))
        text = "안녕하세요" * int(rng.integers(1, 4))
        assert check_ref.ref_fingerprint(data, text) == srv.ref_fingerprint(data, text)


def test_snapshot_keeps_boot_audio_when_file_is_overwritten(srv, tmp_path, monkeypatch) -> None:
    fake = _Fake(eos_after=6, step_s=0)
    health = _boot(srv, tmp_path, monkeypatch, {"ai": "a.wav"}, fake)
    booted = srv.S["voices"]["ai"][0]
    before = booted[0].copy()
    _wav(tmp_path / "a.wav", seed=99)
    fake.calls.clear()
    srv.S["ready"] = True

    r = TestClient(srv.app).post("/v1/tts", json={"voice": "ai", "text": "네"})

    assert r.status_code == 200
    assert fake.calls[-1]["ref_audio"] is booted and np.array_equal(booted[0], before)
    assert r.headers["x-voice-ref"] == health["refs"]["ai"]


def test_snapshot_decodes_like_a_path_load(srv, tmp_path) -> None:
    import librosa

    path = _wav(tmp_path / "s.wav", seed=3)
    voices, refs = srv._snapshot({"ai": {"ref_audio": str(path), "ref_text": "t"}})
    wav, sr = voices["ai"][0]
    ref_wav, ref_sr = librosa.load(str(path), sr=None, mono=True)
    assert sr == ref_sr and np.array_equal(wav, ref_wav)
    assert refs == {"ai": srv.ref_fingerprint(path.read_bytes(), "t")}


@pytest.mark.parametrize("path", ["/v1/tts", "/v1/tts/stream"])
def test_voice_ref_header_on_both_paths(srv, path) -> None:
    _reset(srv, _Fake(eos_after=6, step_s=0))
    srv.S["refs"] = {"ai": "0a1b2c3d"}
    r = TestClient(srv.app).post(path, json={"voice": "ai", "text": "네"})
    assert r.status_code == 200 and r.headers["x-voice-ref"] == "0a1b2c3d"


def test_no_voice_ref_header_when_unknown(srv) -> None:
    _reset(srv, _Fake(eos_after=6, step_s=0))
    srv.S["refs"] = {}
    r = TestClient(srv.app).post("/v1/tts", json={"voice": "ai", "text": "네"})
    assert r.status_code == 200 and "x-voice-ref" not in r.headers


def test_unknown_voice_still_404(srv) -> None:
    _reset(srv, _Fake(eos_after=6, step_s=0))
    srv.S["refs"] = {"ai": "0a1b2c3d"}
    assert TestClient(srv.app).post("/v1/tts", json={"voice": "nope", "text": "네"}).status_code == 404


def test_missing_ref_file_fails_before_model_load(srv, tmp_path, monkeypatch) -> None:
    vf = tmp_path / "voices.json"
    vf.write_text(json.dumps({"ai": {"ref_audio": str(tmp_path / "없음.wav"), "ref_text": "t"}}))
    loaded: list[int] = []
    monkeypatch.setattr(srv, "VOICES_F", str(vf))
    monkeypatch.setattr(srv, "Qwen3TTSModel", types.SimpleNamespace(from_pretrained=lambda *a, **k: loaded.append(1)))
    srv.S.update(ready=False, lock=threading.Lock())
    with pytest.raises(FileNotFoundError), TestClient(srv.app):
        pass
    assert loaded == []


def test_check_ref_reports_and_refuses_existing_path(tmp_path, capsys) -> None:
    import soundfile as sf

    check_ref = _load_check_ref()
    good = tmp_path / "good.wav"
    sf.write(str(good), np.random.default_rng(1).normal(0, 0.1, 24000 * 12).astype(np.float32), 24000, subtype="PCM_16")
    refs = tmp_path / "refs"
    refs.mkdir()
    args = [str(good), "--voice", "user", "--text", "안녕", "--refs-dir", str(refs), "--date", "20261020"]
    assert check_ref.main(args) == 0
    fp = check_ref.ref_fingerprint(good.read_bytes(), "안녕")
    out = capsys.readouterr().out
    assert f"user-20261020-{fp}.wav" in out and '"ref_text": "안녕"' in out
    (refs / f"user-20261020-{fp}.wav").write_bytes(good.read_bytes())
    assert check_ref.main(args) == 2


def test_check_ref_fails_stereo_and_too_short_and_warns_silence(tmp_path, capsys) -> None:
    import soundfile as sf

    check_ref = _load_check_ref()
    stereo = tmp_path / "stereo.wav"
    sf.write(str(stereo), np.zeros((24000 * 5, 2), dtype=np.float32) + 0.1, 24000)
    short = tmp_path / "short.wav"
    sf.write(str(short), np.full(24000 * 2, 0.1, dtype=np.float32), 24000)
    quiet = tmp_path / "quiet.wav"
    x = np.concatenate([np.zeros(24000 * 2), np.random.default_rng(0).normal(0, 0.1, 24000 * 11)]).astype(np.float32)
    sf.write(str(quiet), x, 44100)
    assert check_ref.main([str(stereo), "--voice", "v", "--text", "t", "--refs-dir", str(tmp_path)]) == 1
    assert check_ref.main([str(short), "--voice", "v", "--text", "t", "--refs-dir", str(tmp_path)]) == 1
    capsys.readouterr()
    assert check_ref.main([str(quiet), "--voice", "v", "--text", "t", "--refs-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "샘플레이트 44100Hz" in out and "앞뒤 무음" in out


def test_voice_ref_header_is_per_voice(srv) -> None:
    _reset(srv, _Fake(eos_after=6, step_s=0))
    srv.S["voices"] = {"ai": ("r.wav", "t"), "user": ("u.wav", "t")}
    srv.S["refs"] = {"ai": "11111111", "user": "22222222"}
    r = TestClient(srv.app).post("/v1/tts", json={"voice": "user", "text": "네"})
    assert r.headers["x-voice-ref"] == "22222222"
