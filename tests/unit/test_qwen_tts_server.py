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
                th.join(timeout=10.0)
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
    return srv.max_frames_for(text) * _STEP_S * 1.5 + 1.0


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

    assert _wait_unlocked(srv, 1.0) is not None, "끊긴 스트림이 락을 쥐고 있다"
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
    assert _wait_unlocked(srv, srv.max_frames_for(_TEXT60) * 0.005 * 1.5 + 1.0) is not None
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
            th.join(timeout=2.0)
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


def test_warmup_cap_hit_does_not_abort_startup(srv, tmp_path, monkeypatch) -> None:
    voices = tmp_path / "voices.json"
    voices.write_text(json.dumps({"ai": {"ref_audio": "r.wav", "ref_text": "t"}}))
    fake = _Fake(step_s=0)
    fake.enable_streaming_optimizations = lambda: None
    monkeypatch.setattr(srv, "VOICES_F", str(voices))
    monkeypatch.setattr(
        srv, "Qwen3TTSModel", types.SimpleNamespace(from_pretrained=lambda *a, **k: fake)
    )
    srv.S.update(ready=False, lock=threading.Lock())

    with TestClient(srv.app) as client:
        assert client.get("/health").json() == {"ready": True, "voices": ["ai"], "busy": False}


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
        limit = 0.5 if mode == "close" else _cap_wall(srv, _TEXT60)
        assert _wait_unlocked(srv, limit) is not None
        if mode == "close":
            assert m.closed == 1 and m.frames < 20, "끊김(spec 2.3 취소)이 바로 stop 을 세워야 한다"
        else:
            assert m.frames == srv.max_frames_for(_TEXT60) and m.closed == 0
        assert httpx.get(url + "/health").json()["busy"] is False
    finally:
        server.should_exit = True
        th.join(timeout=5)
