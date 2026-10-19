import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from app.services import pause_compress as pc_mod
from app.services import qwen_tts as qwen_mod
from app.services.pcm_stretch import PcmStretcher
from app.services.qwen_tts import NS_HEADER, QwenRealtimeTTSClient, QwenTTSRunawayError, QwenTTSUnavailableError
from tests.unit.test_qwen_realtime_tts import _real_settings

SR = 16000


def ms(n: float) -> int:
    return int(SR * n / 1000)


def speech(dur: float, db: float = -20.0) -> np.ndarray:
    t = np.arange(ms(dur)) / SR
    return 10 ** (db / 20) * np.sqrt(2) * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t) ** 2)


def pcm(x: np.ndarray) -> bytes:
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


RAW = pcm(np.concatenate([speech(500), np.zeros(ms(900))]))


def chunked(b: bytes, first: int = 4480, rest: int = 5120) -> list[bytes]:
    return [b[:first]] + [b[i : i + rest] for i in range(first, len(b), rest)]


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class _Worker:
    def __init__(self, body: bytes = RAW, *, header: str | None = "1", clock: _Clock | None = None,
                 ttfb_ms: float = 0.0, rtf: float = 0.0, chunks=None, error: Exception | None = None,
                 eos_delay_ms: float = 0.0) -> None:
        self.body = body
        self.eos_delay_ms = eos_delay_ms
        self.header = header
        self.clock = clock
        self.ttfb_ms = ttfb_ms
        self.rtf = rtf
        self.chunks = chunks
        self.error = error
        self.bodies: list[dict] = []

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            self.bodies.append(json.loads(request.read()))

            async def gen():
                if self.clock is not None:
                    self.clock.t += self.ttfb_ms / 1000
                for c in self.chunks or chunked(self.body):
                    yield c
                    if self.clock is not None:
                        self.clock.t += len(c) / 32 / 1000 * self.rtf
                if self.clock is not None:
                    self.clock.t += self.eos_delay_ms / 1000
                if self.error is not None:
                    raise self.error

            headers = {} if self.header is None else {NS_HEADER: self.header}
            return httpx.Response(200, content=gen(), headers=headers)

        return httpx.MockTransport(handler)


def _settings(on: bool = True, speed: float = 1.0, voices: str = "ai", **kw) -> SimpleNamespace:
    return SimpleNamespace(
        qwen_tts_url="http://tts.test", qwen_tts_urls=None, qwen_tts_timeout=5.0, qwen_tts_health_timeout=1.0,
        qwen_tts_realtime_enabled=True, qwen_tts_speed=speed, qwen_tts_non_streaming=on,
        qwen_tts_non_streaming_voices=voices, **kw,
    )


async def _turn(session, sentences=("네.",)) -> bytes:
    await session.begin()

    async def source():
        for s in sentences:
            yield s

    try:
        return b"".join([c async for c in session.stream(source())])
    finally:
        await session.aclose()


def _session(worker: _Worker, *, speed: float = 1.0, clock=None, call=None, **kw):
    client = httpx.AsyncClient(transport=worker.transport())
    return qwen_mod.QwenRealtimeTTSSession(
        client, "http://tts.test", speed, "ai", ns="want", call=call or qwen_mod._CallState(),
        pause=qwen_mod._PauseConfig(**kw), clock=clock or worker.clock or _Clock(),
    )


def _stretched(raw: bytes, speed: float, chunks=None) -> bytes:
    st = PcmStretcher(speed, warmup_ms=180)
    out = b"".join(st.push(c) for c in (chunks or chunked(raw)))
    return out + st.flush()


@pytest.mark.parametrize(("on", "voice"), [(False, None), (False, "ai"), (True, "male_old_rough")])
@pytest.mark.parametrize("speed", [1.0, 1.1])
async def test_off_or_slot_voice_sends_plain_body_and_same_bytes(on, voice, speed) -> None:
    worker = _Worker()
    client = QwenRealtimeTTSClient(_settings(on, speed), transport=worker.transport())
    session = await client.open(voice)
    out = await _turn(session)
    assert worker.bodies == [{"voice": voice or "ai", "text": "네. "}]
    assert out == _stretched(RAW, speed)
    assert session.turn_metrics()["tts_ns"] == ("voice" if on else "off")
    assert "e" not in session.request_log[0] and "drop_ms" not in session.request_log[0]


async def test_on_ai_with_header_trims_end_pause_to_keep() -> None:
    worker = _Worker(clock=_Clock())
    session = _session(worker)
    out = await _turn(session)
    assert worker.bodies == [{"voice": "ai", "text": "네. ", "non_streaming_mode": True}]
    assert out == RAW[: ms(500 + 160 + 20) * 2]
    rec = session.request_log[0]
    assert rec["e"] == "on" and rec["drop_ms"] == 720 and rec["extra_ms"] == 0
    assert rec["out_trail_ms"] == pytest.approx(180, abs=4)
    metrics = session.turn_metrics()
    assert metrics["tts_ns"] == "on" and metrics["tts_drop_ms"] == 720 and metrics["tts_extra_ms"] == 0


async def test_on_through_client_compresses_ai_turns() -> None:
    worker = _Worker()
    client = QwenRealtimeTTSClient(_settings(), transport=worker.transport())
    out = await _turn(await client.open())
    assert worker.bodies[0]["non_streaming_mode"] is True
    assert len(out) < ms(1000) * 2


@pytest.mark.parametrize("header", [None, "0"])
async def test_missing_header_passes_through_and_turns_off_for_the_call(header) -> None:
    worker = _Worker(header=header)
    client = QwenRealtimeTTSClient(_settings(), transport=worker.transport())
    first = await client.open()
    assert await _turn(first) == RAW
    second = await client.open()
    assert await _turn(second) == RAW
    assert worker.bodies[0] == {"voice": "ai", "text": "네. ", "non_streaming_mode": True}
    assert worker.bodies[1] == {"voice": "ai", "text": "네. "}
    assert first.request_log[0]["e"] == "no_header" and "e" not in second.request_log[0]
    assert first.turn_metrics()["tts_ns"] == second.turn_metrics()["tts_ns"] == "no_header"

    fresh = _Worker()
    other = QwenRealtimeTTSClient(_settings(), transport=fresh.transport())
    await _turn(await other.open())
    assert fresh.bodies[0]["non_streaming_mode"] is True


async def test_off_state_holds_within_a_turn_too() -> None:
    worker = _Worker(header=None)
    client = QwenRealtimeTTSClient(_settings(), transport=worker.transport())
    await _turn(await client.open(), ("안녕하세요. ", "반갑습니다."))
    assert "non_streaming_mode" in worker.bodies[0] and "non_streaming_mode" not in worker.bodies[1]


async def test_speed_scales_keep_so_heard_keep_stays_160() -> None:
    speed = 1.15
    worker = _Worker(clock=_Clock())
    out = await _turn(_session(worker, speed=speed))
    expected = _stretched(RAW[: ms(500 + 20 + 180) * 2], speed)
    assert abs(len(out) - len(expected)) <= ms(5) * 2
    plain_keep = _stretched(RAW[: ms(500 + 20 + 160) * 2], speed)
    assert abs(len(out) - len(plain_keep)) > ms(10) * 2


def _spy_finish(monkeypatch) -> list[bool]:
    calls: list[bool] = []
    real = pc_mod.PauseCompressor.finish

    def finish(self, eos=True, spare_ms=0.0):
        calls.append(eos)
        return real(self, eos, spare_ms)

    monkeypatch.setattr(pc_mod.PauseCompressor, "finish", finish)
    return calls


async def test_guard_counts_raw_bytes_and_aborts_without_flush(monkeypatch) -> None:
    calls = _spy_finish(monkeypatch)
    body = pcm(np.concatenate([speech(400), np.zeros(ms(900))] * 4))
    errors = []
    for on in (True, False):
        worker = _Worker(body)
        session = await QwenRealtimeTTSClient(_settings(on), transport=worker.transport()).open()
        with pytest.raises(QwenTTSRunawayError) as exc:
            await _turn(session)
        errors.append(exc.value.pcm_bytes)
        assert session.request_log[0]["outcome"] == "runaway"
    assert errors[0] == errors[1]
    assert calls == [False]


async def test_transport_error_aborts_without_flush(monkeypatch) -> None:
    calls = _spy_finish(monkeypatch)
    worker = _Worker(chunks=chunked(RAW)[:3], error=httpx.ReadError("boom"))
    session = await QwenRealtimeTTSClient(_settings(speed=1.15), transport=worker.transport()).open()
    with pytest.raises(QwenTTSUnavailableError):
        await _turn(session)
    assert calls == [False]
    assert session.request_log[0]["outcome"] == "error"


async def test_consumer_stop_aborts_without_flush(monkeypatch) -> None:
    calls = _spy_finish(monkeypatch)
    worker = _Worker()
    session = await QwenRealtimeTTSClient(_settings(), transport=worker.transport()).open()
    await session.begin()

    async def source():
        yield "네."

    stream = session.stream(source())
    await stream.__anext__()
    await stream.aclose()
    await session.aclose()
    assert calls == [False]


async def test_fast_worker_keeps_only_keep() -> None:
    worker = _Worker(clock=_Clock(), rtf=0.3)
    session = _session(worker)
    out = await _turn(session)
    assert out == RAW[: ms(680) * 2] and session.request_log[0]["extra_ms"] == 0


async def test_slow_worker_keeps_extra_pause_for_a_thin_buffer() -> None:
    worker = _Worker(clock=_Clock(), rtf=1.3)
    session = _session(worker)
    out = await _turn(session)
    rec = session.request_log[0]
    assert rec["extra_ms"] > 0 and len(out) == ms(680 + rec["extra_ms"]) * 2
    assert out == RAW[: len(out)]
    assert session.turn_metrics()["tts_end_ahead_ms"] < 0


async def test_buffer_floor_applies_even_with_zero_ttfb() -> None:
    body = pcm(np.concatenate([speech(60), np.zeros(ms(900))]))
    worker = _Worker(body, clock=_Clock(), chunks=chunked(body, 640, 640))
    session = _session(worker)
    await _turn(session)
    assert session.request_log[0]["extra_ms"] == pytest.approx(300 - (60 + 20 + 160), abs=20)


@pytest.mark.parametrize(("ttfb", "extra"), [(0.0, 0), (400.0, 120), (1000.0, 1320), (1500.0, 1320)])
async def test_target_buffer_grows_with_ttfb_up_to_a_cap(ttfb, extra) -> None:
    body = pcm(np.concatenate([speech(500), np.zeros(ms(3000))]))
    worker = _Worker(body, clock=_Clock(), ttfb_ms=ttfb)
    session = _session(worker)
    await _turn(session)
    assert session.request_log[0]["extra_ms"] == pytest.approx(extra, abs=20)


def _fail_push(monkeypatch, at: int) -> None:
    real = pc_mod.PauseCompressor.push
    count = {"n": 0}

    def push(self, data, spare_ms=0.0):
        count["n"] += 1
        if count["n"] == at:
            raise RuntimeError("boom")
        return real(self, data, spare_ms)

    monkeypatch.setattr(pc_mod.PauseCompressor, "push", push)


@pytest.mark.parametrize("at", [2, 3])
async def test_compressor_error_passes_rest_raw_and_turns_off(monkeypatch, at) -> None:
    _fail_push(monkeypatch, at)

    chunks = chunked(RAW, first=4481, rest=5121)
    worker = _Worker(chunks=chunks)
    client = QwenRealtimeTTSClient(_settings(), transport=worker.transport())
    first = await client.open()
    assert await _turn(first) == RAW
    second = await client.open()
    await _turn(second)
    assert first.request_log[0]["e"] == "comp_error"
    assert "non_streaming_mode" not in worker.bodies[1]
    assert second.turn_metrics()["tts_ns"] == "comp_error"


async def test_compressor_finish_error_keeps_partial_frame(monkeypatch) -> None:
    def finish(self, eos=True, spare_ms=0.0):
        if eos:
            raise RuntimeError("boom")
        return b""

    monkeypatch.setattr(pc_mod.PauseCompressor, "finish", finish)
    body = RAW + b"\x01\x02\x03\x04"
    worker = _Worker(body, clock=_Clock())
    session = _session(worker)
    out = await _turn(session)
    assert out.endswith(b"\x01\x02\x03\x04") and len(out) == ms(680) * 2 + 4
    assert session.request_log[0]["e"] == "comp_error"


async def test_compression_works_without_request_metric() -> None:
    worker = _Worker()
    client = QwenRealtimeTTSClient(_settings(qwen_tts_req_metric=False), transport=worker.transport())
    session = await client.open()
    out = await _turn(session)
    assert len(out) < ms(1000) * 2 and session.request_log == [] and session.turn_metrics() == {}


def test_env_reaches_realtime_client(monkeypatch) -> None:
    for name in ("QWEN_TTS_NON_STREAMING", "QWEN_TTS_NON_STREAMING_VOICES", "QWEN_TTS_PAUSE_KEEP_MS"):
        monkeypatch.delenv(name, raising=False)
    default = QwenRealtimeTTSClient(_real_settings())
    assert default._ns_on is False and default._ns_voices == frozenset({"ai"})
    monkeypatch.setenv("QWEN_TTS_NON_STREAMING", "true")
    monkeypatch.setenv("QWEN_TTS_NON_STREAMING_VOICES", "ai, user")
    monkeypatch.setenv("QWEN_TTS_PAUSE_KEEP_MS", "200")
    client = QwenRealtimeTTSClient(_real_settings())
    assert client._ns_on is True and client._ns_voices == frozenset({"ai", "user"})
    assert client._pause == qwen_mod._PauseConfig(keep_ms=200.0)


async def test_compressor_error_then_transport_error_still_turns_off(monkeypatch) -> None:
    _fail_push(monkeypatch, 2)
    worker = _Worker(chunks=chunked(RAW)[:4], error=httpx.ReadError("boom"))
    client = QwenRealtimeTTSClient(_settings(), transport=worker.transport())
    with pytest.raises(QwenTTSUnavailableError):
        await _turn(await client.open())
    worker.error = None
    await _turn(await client.open())
    assert "non_streaming_mode" not in worker.bodies[1]


async def test_late_eos_releases_held_end_for_a_dry_buffer() -> None:
    worker = _Worker(clock=_Clock(), eos_delay_ms=3000)
    session = _session(worker)
    out = await _turn(session)
    assert session.request_log[0]["extra_ms"] == 90 and out == RAW[: ms(680 + 90) * 2]


@pytest.mark.parametrize("speed", [1.0, 1.25])
async def test_adaptive_keep_fills_the_heard_buffer_to_target(speed) -> None:
    body = pcm(np.concatenate([speech(500), np.zeros(ms(3000))]))
    worker = _Worker(body, clock=_Clock(), ttfb_ms=1000)
    session = _session(worker, speed=speed)
    await _turn(session)
    assert session.turn_metrics()["tts_end_ahead_ms"] == pytest.approx(2000, abs=40)


async def test_turn_drop_counts_inner_caps_and_end_trims() -> None:
    body = pcm(np.concatenate([speech(500), np.zeros(ms(900)), speech(500), np.zeros(ms(900))]))
    worker = _Worker(body, clock=_Clock())
    session = _session(worker)
    await _turn(session)
    rec = session.request_log[0]
    assert (rec["drop_ms"], rec["cap_ms"]) == (720, 630)
    assert session.turn_metrics()["tts_drop_ms"] == 1350


@pytest.mark.parametrize(("speed", "extra"), [(1.0, 300), (1.25, 370)])
async def test_spare_is_converted_to_raw_time_by_speed(speed, extra) -> None:
    body = pcm(np.concatenate([speech(500), np.zeros(ms(2000))]))
    worker = _Worker(body, clock=_Clock(), chunks=[body])
    session = _session(worker, speed=speed)
    await _turn(session)
    assert session.request_log[0]["extra_ms"] == extra


@pytest.mark.parametrize(("speed", "cap"), [(1.0, 330), (1.25, 270)])
async def test_inner_cap_scales_with_speed(speed, cap) -> None:
    body = pcm(np.concatenate([speech(500), np.zeros(ms(600)), speech(500), np.zeros(ms(900))]))
    session = _session(_Worker(body, clock=_Clock()), speed=speed)
    await _turn(session)
    assert session.request_log[0]["cap_ms"] == pytest.approx(cap, abs=5)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -50.0, "abc"])
async def test_unusable_pause_settings_fall_back_to_defaults(bad) -> None:
    worker = _Worker()
    settings = _settings(qwen_tts_pause_keep_ms=bad, qwen_tts_pause_cap_to_ms=bad)
    client = QwenRealtimeTTSClient(settings, transport=worker.transport())
    assert client._pause == qwen_mod._PauseConfig()
    session = await client.open()
    out = await _turn(session)
    assert len(out) < ms(1000) * 2 and session.request_log[0]["e"] == "on"


async def test_compressor_creation_error_passes_raw_without_switching(monkeypatch) -> None:
    def _boom(self, **_kw):
        raise ValueError("boom")

    monkeypatch.setattr(pc_mod.PauseCompressor, "__init__", _boom)
    worker = _Worker()
    client = QwenRealtimeTTSClient(_settings(), transport=worker.transport())
    first = await client.open()
    assert await _turn(first) == RAW
    second = await client.open()
    await _turn(second)
    assert first.request_log[0]["e"] == "comp_error" and "non_streaming_mode" not in worker.bodies[1]
    assert second.turn_metrics()["tts_ns"] == "comp_error"


async def test_error_before_the_header_is_not_labelled_no_header() -> None:
    worker = _Worker()
    client = QwenRealtimeTTSClient(
        _settings(), transport=httpx.MockTransport(lambda r: httpx.Response(500))
    )
    session = await client.open()
    with pytest.raises(QwenTTSUnavailableError):
        await _turn(session)
    assert "e" not in session.request_log[0] and session.ns_mode == "on"
    assert worker.bodies == []
