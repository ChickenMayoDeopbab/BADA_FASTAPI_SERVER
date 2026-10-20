import asyncio
import json
import time

import pytest

from tests.perf.ws_bench import _recv_loop


class _FakeWS:

    def __init__(self, script: list[tuple[float, object]]) -> None:
        self._script = list(script)

    async def recv(self):
        if not self._script:
            await asyncio.sleep(3600)
        delay, msg = self._script.pop(0)
        await asyncio.sleep(delay)
        return json.dumps(msg) if isinstance(msg, dict) else msg


def _user(text: str) -> dict:
    return {"type": "transcript", "role": "user", "text": text}


def _ai(text: str) -> dict:
    return {"type": "transcript", "role": "ai", "text": text}


async def test_stt_final_measured_from_speech_end_and_transcript_kept() -> None:
    ws = _FakeWS([
        (0.05, _user("안녕하세요")), (0.01, _ai("네")), (0.01, b"\x00\x00"), (0.01, {"type": "speaking_end"}),
    ])
    speech_end = time.perf_counter()
    res = await _recv_loop(ws, speech_end=speech_end, timeout=2.0)
    assert res["terminal"] is None
    assert res["transcript"] == "안녕하세요"
    assert 40 <= res["stt_final_ms"] <= 400
    assert res["client_response_ms"] is not None and res["client_turn_ms"] is not None


async def test_no_user_transcript_leaves_stt_final_none() -> None:
    ws = _FakeWS([(0.01, b"\x00\x00"), (0.01, {"type": "speaking_end"})])
    res = await _recv_loop(ws, speech_end=time.perf_counter(), timeout=2.0)
    assert res["stt_final_ms"] is None
    assert res["transcript"] is None


async def test_timeout_without_final_is_reported() -> None:
    ws = _FakeWS([])
    res = await _recv_loop(ws, speech_end=time.perf_counter(), timeout=0.05)
    assert res["terminal"] == "TIMEOUT"
    assert res["stt_final_ms"] is None


async def test_end_frame_terminates_with_reason() -> None:
    ws = _FakeWS([(0.01, {"type": "end", "reason": "ERROR"})])
    res = await _recv_loop(ws, speech_end=time.perf_counter(), timeout=1.0)
    assert res["terminal"] == "end"


@pytest.mark.parametrize("bad", [b"", "not json"])
async def test_unparseable_text_is_ignored(bad) -> None:
    ws = _FakeWS([(0.0, "{}"), (0.01, {"type": "speaking_end"})])
    res = await _recv_loop(ws, speech_end=time.perf_counter(), timeout=1.0)
    assert res["terminal"] is None


async def test_ai_transcript_before_user_is_not_counted_as_final() -> None:
    ws = _FakeWS([(0.0, _ai("여보세요")), (0.06, _user("안녕하세요")), (0.01, {"type": "speaking_end"})])
    res = await _recv_loop(ws, speech_end=time.perf_counter(), timeout=1.0)
    assert res["transcript"] == "안녕하세요"
    assert res["stt_final_ms"] >= 50


async def test_run_turn_reports_closed_when_send_fails() -> None:
    from websockets.exceptions import ConnectionClosedOK

    from tests.perf.ws_bench import _run_turn

    class _ClosedWS:
        async def send(self, data):
            raise ConnectionClosedOK(None, None)

        async def recv(self):
            raise ConnectionClosedOK(None, None)

    res = await _run_turn(_ClosedWS(), audio=b"\x00" * 3200, silence=b"", chunk_bytes=3200, chunk_s=0.0, timeout=1.0)
    assert res["terminal"] == "CLOSED"
    assert res["stt_final_ms"] is None


async def test_turn_gap_silence_is_sent_after_response() -> None:
    from tests.perf.ws_bench import _run_turn

    class _WS:
        def __init__(self) -> None:
            self.sent: list[bytes] = []
            self._replied = False

        async def send(self, data):
            self.sent.append(data)

        async def recv(self):
            if not self._replied:
                self._replied = True
                return json.dumps({"type": "speaking_end"})
            await asyncio.sleep(3600)

    ws = _WS()
    gap = b"\x00" * 6400
    res = await _run_turn(
        ws, audio=b"\x01" * 3200, silence=b"", chunk_bytes=3200, chunk_s=0.0, timeout=1.0, turn_gap=gap
    )
    assert res["terminal"] is None
    assert sum(len(b) for b in ws.sent if b and b[0] == 0) == 6400


def test_noise_loop_wraps_and_continues_between_takes() -> None:
    from tests.perf.ws_bench import _NoiseLoop

    bed = _NoiseLoop(b"\x01\x02\x03\x04")
    assert bed.take(6) == b"\x01\x02\x03\x04\x01\x02"
    assert bed.take(4) == b"\x03\x04\x01\x02"
    assert _NoiseLoop(b"").take(4) == b"\x00" * 4


def test_noise_turn_cuts_at_utterance_end_and_keeps_clip_tail() -> None:
    from tests.perf.ws_bench import _noise_turn

    clip = b"L1L2" + b"S1S2S3" + b"T1T2"  # 앞 잡음 2샘플 + 발화 3샘플 + 꼬리 2샘플
    assert _noise_turn(clip, 5 / 16000) == (b"L1L2S1S2S3", b"T1T2")


async def test_tail_more_keeps_sending_until_reply() -> None:
    from tests.perf.ws_bench import _run_turn

    ws = _BenchWS(reply_after=3200 + 1600 + 8000)
    res = await asyncio.wait_for(
        _run_turn(ws, audio=b"\x01" * 3200, silence=b"\x02" * 1600, chunk_bytes=1600, chunk_s=0.005, timeout=2.0,
                  tail_more=lambda n: b"\x07" * n),
        5.0,
    )
    sent = b"".join(ws.sent)
    assert res["terminal"] is None and res["transcript"] == "가"
    assert sent[:4800] == b"\x01" * 3200 + b"\x02" * 1600
    assert len(sent) >= 3200 + 1600 + 8000 and set(sent[4800:]) == {7}
    assert len(sent) <= ws.reply_at * 1600 + 2 * 1600  # 응답 뒤엔 멈춘다


def _noise_manifest(tmp_path):
    import numpy as np

    from scripts.stt_noise import mix_noise

    src = tmp_path / "src"
    src.mkdir()
    t = np.arange(3200) / 16000
    for name, hz in (("a.wav", 500.0), ("b.wav", 900.0)):
        mix_noise.write_pcm16(src / name, np.concatenate([np.zeros(800), 0.2 * np.sin(2 * np.pi * hz * t)]))
    (src / "manifest.json").write_text(json.dumps({"a.wav": "가", "b.wav": "나"}), encoding="utf-8")
    mix_noise.build(src, tmp_path / "audio", noises=["pink"], snrs=[10.0, 0.0], lead_ms=100, tail_ms=200,
                    noise_only_ms=300, target_db=-26.0, talkers=2, seed=0, variants=2)
    return src, tmp_path / "audio" / "manifest.json"


def _pcm(path) -> bytes:
    from tests.perf.ws_bench import _load_audio

    return _load_audio(str(path))


def test_noise_condition_picks_requested_snr_and_cycles_variants(tmp_path) -> None:
    from tests.perf.ws_bench import _NoiseCondition

    _, manifest = _noise_manifest(tmp_path)
    base = manifest.parent
    cond = _NoiseCondition(str(manifest), "pink", 10.0)
    assert (cond.noise, cond.snr_db, cond.lead_ms) == ("pink", 10.0, 100)
    cut = (1600 + 4000) * 2
    for cycle, variant in ((0, 1), (1, 2), (2, 1)):
        v, audio, tail = cond.turn("a.wav", cycle)
        clip = _pcm(base / f"pink_snr10/v{variant}/a.wav")
        assert v == variant
        assert (audio, tail) == (clip[:cut], clip[cut:])
        assert len(tail) == 200 * 32 and any(tail)
    beds = _pcm(base / "noise_only/v1/pink_snr10.wav") + _pcm(base / "noise_only/v2/pink_snr10.wav")
    assert cond.bed.take(len(beds)) == beds

    clean = _NoiseCondition(str(manifest), "clean", None)
    v, audio, tail = clean.turn("b.wav")
    assert v is None and len(audio) == cut and not any(audio[:3200]) and any(audio[3200:])
    assert tail == b"\x00" * (200 * 32) and clean.bed.take(3200) == b"\x00" * 3200
    with pytest.raises(SystemExit):
        _NoiseCondition(str(manifest), "white", 10.0)


def test_noise_condition_refuses_overlapping_clips(tmp_path) -> None:
    from tests.perf.ws_bench import _NoiseCondition

    _, manifest = _noise_manifest(tmp_path)
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    meta["clips"] += [c for c in meta["clips"] if c["noise"] == "pink" and c["snr_db"] == 10.0]
    manifest.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SystemExit):
        _NoiseCondition(str(manifest), "pink", 10.0)


class _BenchWS:
    """턴마다 보낸 이진 바이트가 reply_after 를 넘으면 사용자 전사·speaking_end 를 돌려준다"""

    def __init__(self, reply_after: int, turns: int = 1) -> None:
        self.sent: list[bytes] = []
        self.reply_at: int | None = None
        self._reply_after = reply_after
        self._since = 0
        self._ready = asyncio.Event()
        self._replies = [r for _ in range(turns)
                         for r in ({"type": "transcript", "role": "user", "text": "가"}, {"type": "speaking_end"})]

    async def send(self, data):
        if isinstance(data, bytes):
            self.sent.append(data)
            self._since += len(data)
            if self._since >= self._reply_after:
                self._ready.set()

    async def recv(self):
        await self._ready.wait()
        if not self._replies:
            await asyncio.sleep(3600)
        reply = self._replies.pop(0)
        if reply["type"] == "speaking_end":
            self.reply_at = len(self.sent)
            self._since = 0
            self._ready.clear()
        return json.dumps(reply)


def _bench_args(tmp_path, audio, **kw):
    import argparse

    args = argparse.Namespace(
        url="ws://local/ws/voice/s1", base_url="ws://localhost:8000", session_id=None, token="t", audio=[str(audio)],
        engine="chirp", json=True, json_out=str(tmp_path / "out.json"), turns=1, chunk_ms=50, tail_silence_ms=100,
        turn_timeout=2.0, turn_gap_ms=0, timeout=1.0, noise_manifest=None, noise=None, snr=None,
    )
    for k, v in kw.items():
        setattr(args, k, v)
    return args


async def _bench(monkeypatch, args, reply_after: int) -> tuple[_BenchWS, dict]:
    import contextlib

    from tests.perf import ws_bench

    ws = _BenchWS(reply_after, turns=args.turns)

    @contextlib.asynccontextmanager
    async def connect(uri, **kw):
        yield ws

    monkeypatch.setattr(ws_bench.websockets, "connect", connect)
    await asyncio.wait_for(ws_bench._run(args), 10.0)
    return ws, json.loads(open(args.json_out, encoding="utf-8").read())


async def test_default_run_sends_digital_silence_tail_and_null_noise_fields(tmp_path, monkeypatch) -> None:
    src, _ = _noise_manifest(tmp_path)
    audio_len = 4000 * 2
    ws, payload = await _bench(monkeypatch, _bench_args(tmp_path, src / "a.wav"), audio_len + 3200)
    sent = b"".join(ws.sent)
    assert len(sent) == audio_len + 3200
    assert sent[audio_len:] == b"\x00" * 3200
    assert payload["turns"][0]["transcript"] == "가"
    assert "variant" not in payload["turns"][0]
    assert (payload["noise"], payload["snr_db"], payload["lead_ms"]) == (None, None, 0)
    assert (payload["engine"], payload["tail_silence_ms"], payload["chunk_ms"], payload["turn_gap_ms"]) == (
        "chirp", 100, 50, 0
    )


async def test_noise_run_sends_clip_then_noise_until_reply_and_records_condition(tmp_path, monkeypatch) -> None:
    src, manifest = _noise_manifest(tmp_path)
    clip = _pcm(manifest.parent / "pink_snr10/v1/a.wav")
    bed = _pcm(manifest.parent / "noise_only/v1/pink_snr10.wav")
    reply_after = len(clip) + 6400  # 클립 꼬리 200ms 를 넘어 잡음만 클립 200ms 까지 보낸 뒤 응답
    args = _bench_args(tmp_path, src / "a.wav", noise_manifest=str(manifest), noise="pink", snr=10.0)
    ws, payload = await _bench(monkeypatch, args, reply_after)
    sent = b"".join(ws.sent)
    assert len(sent) >= reply_after  # --tail-silence-ms(100) 에서 끊기지 않는다
    assert sent[: len(clip)] == clip
    assert sent[len(clip) : reply_after] == bed[:6400]
    assert (payload["noise"], payload["snr_db"], payload["lead_ms"], payload["tail_silence_ms"]) == (
        "pink", 10.0, 100, None
    )
    assert payload["turns"][0]["audio"] == "a.wav" and payload["turns"][0]["variant"] == 1


async def test_noise_run_fills_turn_gap_with_noise(tmp_path, monkeypatch) -> None:
    src, manifest = _noise_manifest(tmp_path)
    clip = _pcm(manifest.parent / "pink_snr10/v1/a.wav")
    args = _bench_args(tmp_path, src / "a.wav", noise_manifest=str(manifest), noise="pink", snr=10.0,
                       turn_gap_ms=200)
    ws, _ = await _bench(monkeypatch, args, len(clip))
    after = b"".join(ws.sent[ws.reply_at :])
    assert len(after) >= 6400 and any(after[-6400:])


async def test_noise_run_moves_to_next_variant_each_cycle(tmp_path, monkeypatch) -> None:
    src, manifest = _noise_manifest(tmp_path)
    v1, v2 = (_pcm(manifest.parent / f"pink_snr10/v{v}/a.wav") for v in (1, 2))
    cut = (1600 + 4000) * 2
    args = _bench_args(tmp_path, src / "a.wav", noise_manifest=str(manifest), noise="pink", snr=10.0, turns=2)
    ws, payload = await _bench(monkeypatch, args, len(v1))
    sent = b"".join(ws.sent)
    assert [t["variant"] for t in payload["turns"]] == [1, 2]
    assert sent.startswith(v1) and v2[:cut] in sent[len(v1) :]
