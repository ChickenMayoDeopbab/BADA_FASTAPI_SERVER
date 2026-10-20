import argparse
import asyncio
import dataclasses
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.services.stt import AUDIO_EOS, GoogleSTTClient, STTError, STTEvent, STTEventType, STTStreamAbortedError
from scripts.stt_compare import wilson
from scripts.stt_noise import mix_noise, run_chirp, score

SR = mix_noise.SR
BPS = mix_noise.BYTES_PER_S


def _tone(seconds: float, hz: float, *, pad_s: float = 0.1) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    x = np.sin(2 * np.pi * hz * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    pad = np.zeros(int(SR * pad_s))
    return np.concatenate([pad, 0.2 * x, pad])


def _src(tmp_path: Path, tones: dict[str, float], seconds: float = 0.6) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    texts = {}
    for i, (name, hz) in enumerate(tones.items()):
        mix_noise.write_pcm16(src / name, _tone(seconds + 0.1 * i, hz))
        texts[name] = f"문장 {i}"
    (src / "manifest.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
    return src


def _build(tmp_path: Path, src: Path, **kw) -> dict:
    opts = dict(noises=list(mix_noise.NOISES), snrs=[20.0, 10.0, 5.0, 0.0], lead_ms=300, tail_ms=500,
                noise_only_ms=1000, target_db=-26.0, talkers=4, seed=1, variants=2)
    opts.update(kw)
    return mix_noise.build(src, tmp_path / "out", **opts)


def _frame_power(x: np.ndarray) -> np.ndarray:
    k = len(x) // 320
    return np.mean(x[: k * 320].reshape(k, 320) ** 2, axis=1)


def _read(path: Path) -> np.ndarray:
    return mix_noise.read_pcm16(path)


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> tuple[Path, dict]:
    tmp = tmp_path_factory.mktemp("mix")
    src = _src(tmp, {"a.wav": 500.0, "b.wav": 1300.0, "c.wav": 2100.0})
    return src, _build(tmp, src)


@pytest.mark.parametrize("kind", mix_noise.NOISES)
def test_mixed_clip_hits_target_snr_within_half_db(built, kind) -> None:
    src, manifest = built
    base = src.parent / "out"
    clean = {c["utt"]: c for c in manifest["clips"] if c["noise"] == "clean"}
    lead = int(round(clean["a.wav"]["lead_s"] * SR))
    for c in manifest["clips"]:
        if c["noise"] != kind or c["noise_only"]:
            continue
        utt = mix_noise.read_pcm16(src / c["utt"])
        active = 10 * np.log10(_frame_power(utt) + 1e-20) >= -45.0
        k_clean = 10 ** (clean[c["utt"]]["clip_gain_db"] / 20)
        k_mix = 10 ** (c["clip_gain_db"] / 20)
        speech = _read(base / clean[c["utt"]]["file"]) / k_clean * k_mix
        noise = _read(base / c["file"]) - speech
        seg = slice(lead, lead + len(utt))
        measured = 10 * np.log10(_frame_power(speech[seg])[active].mean() / _frame_power(noise[seg])[active].mean())
        assert abs(measured - c["snr_db"]) <= 0.5, (c["file"], measured)


def test_clean_clip_has_digital_silence_lead_and_tail_and_target_level(built) -> None:
    src, manifest = built
    base = src.parent / "out"
    c = next(c for c in manifest["clips"] if c["noise"] == "clean" and c["utt"] == "a.wav")
    x = _read(base / c["file"])
    utt = mix_noise.read_pcm16(src / "a.wav")
    lead, tail = 300 * SR // 1000, 500 * SR // 1000
    assert len(x) == lead + len(utt) + tail
    assert not x[:lead].any() and not x[-tail:].any()
    assert c["utt_end_s"] == pytest.approx((lead + len(utt)) / SR, abs=1e-3)
    assert c["lead_s"] < c["speech_start_s"] < c["speech_end_s"] <= c["utt_end_s"]
    assert c["speech_start_s"] == pytest.approx(0.3 + 0.1, abs=0.03)
    active = 10 * np.log10(_frame_power(utt) + 1e-20) >= -45.0
    level = 10 * np.log10(_frame_power(x[lead : lead + len(utt)])[active].mean())
    assert level == pytest.approx(-26.0, abs=0.3)


def test_noise_clip_lead_and_tail_carry_noise(built) -> None:
    src, manifest = built
    c = next(c for c in manifest["clips"] if c["noise"] == "pink" and c["snr_db"] == 10.0 and not c["noise_only"])
    x = _read(src.parent / "out" / c["file"])
    lead = int(round(c["lead_s"] * SR))
    tail = len(x) - int(round(c["utt_end_s"] * SR))
    assert tail == 500 * SR // 1000
    assert np.sqrt(np.mean(x[:lead] ** 2)) > 0 and np.sqrt(np.mean(x[-tail:] ** 2)) > 0
    assert c["dur_s"] == pytest.approx(len(x) / SR, abs=1e-3)


def test_noise_only_clip_level_is_speech_level_minus_snr(built) -> None:
    src, manifest = built
    only = [c for c in manifest["clips"] if c["noise_only"]]
    assert len(only) == 3 * 4 * 2
    for c in only:
        x = _read(src.parent / "out" / c["file"])
        assert len(x) == SR
        assert 10 * np.log10(np.mean(x**2)) == pytest.approx(-26.0 - c["snr_db"], abs=0.5)


def test_babble_excludes_the_target_utterance(built) -> None:
    src, manifest = built
    c = next(c for c in manifest["clips"] if c["noise"] == "babble" and c["utt"] == "a.wav" and c["snr_db"] == 0.0)
    x = _read(src.parent / "out" / c["file"])
    lead = int(round(c["lead_s"] * SR))
    spec = np.abs(np.fft.rfft(x[:lead])) ** 2
    f = np.fft.rfftfreq(lead, 1 / SR)

    def band(hz: float) -> float:
        return float(spec[(f > hz - 60) & (f < hz + 60)].sum())

    assert band(1300.0) + band(2100.0) > 1000 * band(500.0)


def _noise_of(base: Path, manifest: dict, noise: str, snr: float, variant: int, utt: str = "a.wav") -> np.ndarray:
    clean = next(c for c in manifest["clips"] if c["noise"] == "clean" and c["utt"] == utt)
    row = next(c for c in manifest["clips"] if not c["noise_only"] and c["noise"] == noise and c["snr_db"] == snr
               and c["variant"] == variant and c["utt"] == utt)
    speech = _read(base / clean["file"]) / 10 ** (clean["clip_gain_db"] / 20) * 10 ** (row["clip_gain_db"] / 20)
    return _read(base / row["file"]) - speech


def test_variants_are_different_noise_and_snr_steps_share_one(built) -> None:
    src, manifest = built
    base = src.parent / "out"
    assert manifest["variants"] == 2
    assert {c["variant"] for c in manifest["clips"] if c["noise"] == "clean"} == {None}
    assert {c["variant"] for c in manifest["clips"] if c["noise"] != "clean"} == {1, 2}
    v1_10, v1_0 = _noise_of(base, manifest, "pink", 10.0, 1), _noise_of(base, manifest, "pink", 0.0, 1)
    v2_10 = _noise_of(base, manifest, "pink", 10.0, 2)
    assert np.corrcoef(v1_10, v1_0)[0, 1] > 0.99
    assert abs(np.corrcoef(v1_10, v2_10)[0, 1]) < 0.2
    only = {(c["variant"], c["snr_db"]): _read(base / c["file"]) for c in manifest["clips"]
            if c["noise_only"] and c["noise"] == "white"}
    assert abs(np.corrcoef(only[(1, 10.0)], only[(2, 10.0)])[0, 1]) < 0.2


def test_rep_seconds_counts_clean_and_one_variant(built) -> None:
    _, manifest = built
    clean_s = sum(c["dur_s"] for c in manifest["clips"] if c["noise"] == "clean")
    total_s = sum(c["dur_s"] for c in manifest["clips"])
    assert mix_noise.rep_seconds(manifest) == pytest.approx(clean_s + (total_s - clean_s) / 2)


def test_pink_has_equal_energy_per_octave_and_white_does_not() -> None:
    rng = np.random.default_rng(0)
    n = SR * 8

    def octaves(x: np.ndarray) -> list[float]:
        spec = np.abs(np.fft.rfft(x)) ** 2
        f = np.fft.rfftfreq(len(x), 1 / SR)
        return [10 * np.log10(spec[(f >= lo) & (f < 2 * lo)].sum()) for lo in (250, 500, 1000, 2000)]

    pink = octaves(mix_noise.pink(rng, n))
    white = octaves(mix_noise.white(rng, n))
    assert max(pink) - min(pink) < 1.5
    assert white[-1] - white[0] == pytest.approx(9.0, abs=1.0)


def test_mix_clip_rejects_short_noise_and_missing_snr() -> None:
    utt = _tone(0.3, 500.0)
    with pytest.raises(ValueError):
        mix_noise.mix_clip(utt, np.ones(10), 10.0, lead=100, tail=100, target_db=-26.0)
    with pytest.raises(ValueError):
        mix_noise.mix_clip(utt, np.ones(len(utt) + 200), None, lead=100, tail=100, target_db=-26.0)


def test_peak_guard_scales_whole_clip_and_keeps_snr() -> None:
    utt = _tone(0.5, 500.0)
    noise = mix_noise.white(np.random.default_rng(3), len(utt) + 400)
    mixed, speech, info = mix_noise.mix_clip(utt, noise, 0.0, lead=200, tail=200, target_db=-3.0)
    assert np.max(np.abs(mixed)) <= 10 ** (-1 / 20) + 1e-9
    assert info["clip_gain_db"] < 0
    active = 10 * np.log10(_frame_power(utt) + 1e-20) >= -45.0
    seg = slice(200, 200 + len(utt))
    snr = 10 * np.log10(_frame_power(speech[seg])[active].mean() / _frame_power((mixed - speech)[seg])[active].mean())
    assert snr == pytest.approx(0.0, abs=0.01)


def test_mix_noise_main_dry_run_writes_all_conditions_and_prints_cost(tmp_path, monkeypatch, capsys) -> None:
    src = _src(tmp_path, {"a.wav": 500.0, "b.wav": 900.0})
    out = tmp_path / "audio"
    monkeypatch.setattr(sys, "argv", ["mix_noise.py", "--src", str(src), "--out", str(out), "--snr", "10", "0",
                                      "--lead-ms", "200", "--tail-ms", "300", "--noise-only-ms", "500",
                                      "--variants", "2"])
    mix_noise.main()
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    conds = {(c["noise"], c["snr_db"], c["variant"], c["noise_only"]) for c in manifest["clips"]}
    assert ("clean", None, None, False) in conds
    want = {(k, s, v, f) for k in mix_noise.NOISES for s in (10.0, 0.0) for v in (1, 2) for f in (False, True)}
    assert want <= conds
    assert all((out / c["file"]).exists() for c in manifest["clips"])
    assert "$" in capsys.readouterr().out


class _FakeSTT:
    """스트림마다 정해진 오디오 위치에서 이벤트를 낸다. 큐를 운영 클라이언트처럼 직접 읽는다.
    FINAL 의 end_offset_sec 는 '지금 스트림 위치보다 몇 초 앞에서 끝났나'로 받아 스트림 기준 오프셋으로 바꾼다"""

    multi_utterance = False

    def __init__(self, script, *, on_eos=(), hang_on_eos=False, raise_at=None) -> None:
        self.script = list(script)
        self.on_eos = list(on_eos)
        self.hang_on_eos = hang_on_eos
        self.raise_at = raise_at
        self.opened = 0
        self.total = 0
        self.chunk_times: list[float] = []
        self.audio_bytes_sent = 0

    async def stream(self, queue, first_chunk=None):
        self.opened += 1
        no = self.opened
        pending = [(at, ev) for s, at, ev in self.script if s == no]
        got = 0
        chunk = first_chunk
        while chunk is not AUDIO_EOS:
            got += len(chunk)
            self.total += len(chunk)
            self.audio_bytes_sent += len(chunk)
            self.chunk_times.append(time.perf_counter())
            if self.raise_at and self.raise_at[0] == no and self.total / BPS >= self.raise_at[1]:
                raise STTStreamAbortedError("끊김")
            while pending and self.total / BPS >= pending[0][0] - 1e-9:
                yield _placed(pending.pop(0)[1], got)
            chunk = await queue.get()
        if self.hang_on_eos:
            await asyncio.sleep(3600)
        for ev in self.on_eos:
            yield _placed(ev, got)


def _placed(ev: STTEvent, got: int) -> STTEvent:
    if ev.type != STTEventType.FINAL:
        return ev
    return dataclasses.replace(ev, end_offset_sec=round(got / BPS - (ev.end_offset_sec or 0.0), 6))


class _EchoSTT:
    """스트림마다 받은 오디오가 at_s 를 넘으면 say 를 FINAL 로 낸다(None 이면 아무것도 안 낸다)"""

    multi_utterance = False

    def __init__(self, say: str | None, at_s: float = 0.1) -> None:
        self.say = say
        self.at_s = at_s
        self.audio_bytes_sent = 0

    async def stream(self, queue, first_chunk=None):
        got = 0
        chunk = first_chunk
        while chunk is not AUDIO_EOS:
            got += len(chunk)
            self.audio_bytes_sent += len(chunk)
            if self.say is not None and got / BPS >= self.at_s:
                yield STTEvent(type=STTEventType.FINAL, text=self.say, end_offset_sec=got / BPS)
            chunk = await queue.get()


class _BrokenSTT:
    """열 때마다 첫 청크에서 STTError"""

    multi_utterance = False

    def __init__(self) -> None:
        self.opened = 0
        self.audio_bytes_sent = 0

    async def stream(self, queue, first_chunk=None):
        self.opened += 1
        self.audio_bytes_sent += len(first_chunk or b"")
        if first_chunk is not None:
            raise STTError("PERMISSION_DENIED")
        yield _final("")


def _final(text: str, lag: float = 0.0) -> STTEvent:
    return STTEvent(type=STTEventType.FINAL, text=text, end_offset_sec=lag)


_BEGIN = STTEvent(type=STTEventType.SPEECH_BEGIN)
_END = STTEvent(type=STTEventType.SPEECH_END)
_CLIP = b"\x01\x00" * SR  # 1초, 말소리는 0.1~0.6초로 친다
_STOP = 0.6 - score.END_MARGIN_S


def _row(res: dict, *, speech_end_s: float = 0.6, speech_start_s: float = 0.1, text: str = "안녕하세요",
         noise_only: bool = False, file: str = "white_snr10/v1/u.wav") -> dict:
    return {"file": file, "utt": "u.wav", "text": text, "noise": "white", "snr_db": 10.0, "denoise": False,
            "rep": 1, "noise_only": noise_only, "speech_start_s": speech_start_s, "speech_end_s": speech_end_s,
            **res}


def _ev_final(text: str, *, t: float, end: float) -> dict:
    return {"t": t, "type": "final", "text": text, "stream": 1, "stream_open_s": 0.0, "end_offset_s": end,
            "after_eos": False}


async def _clip(stt, pcm: bytes = _CLIP, stop_after_s: float | None = _STOP, drain_s: float = 0.5) -> dict:
    return await asyncio.wait_for(
        run_chirp.run_clip(stt, pcm, chunk_ms=20, stop_after_s=stop_after_s, drain_s=drain_s), 5.0
    )


async def test_empty_final_reopens_and_turn_is_first_nonempty_final_covering_speech_end() -> None:
    stt = _FakeSTT([
        (1, 0.06, _BEGIN), (1, 0.08, _final("")),
        (2, 0.20, _BEGIN), (2, 0.22, STTEvent(type=STTEventType.INTERIM, text="안")), (2, 0.62, _END),
        (2, 0.70, _final("안녕하세요", lag=0.1)),
    ])
    res = await _clip(stt)
    assert res["stopped"] == "turn"
    assert stt.opened == res["streams"] == 2
    finals = [e for e in res["events"] if e["type"] == "final"]
    assert [(e["text"], e["stream"]) for e in finals] == [("", 1), ("안녕하세요", 2)]
    assert finals[1]["stream_open_s"] == pytest.approx(0.08, abs=0.03)
    assert score.final_end_s(finals[1]) == pytest.approx(0.6, abs=0.03)
    assert res["interims"] == 1
    kinds = [e["type"] for e in res["events"]]
    assert kinds.count("speech_begin") == 2 and kinds.count("speech_end") == 1
    assert 0.68 <= res["audio_sent_s"] <= 0.9  # 턴에서 멈춤, 1초를 다 보내지 않는다
    assert res["elapsed_s"] >= 0.68  # 실시간 속도
    row = _row(res)
    assert score.turn_of(row)["text"] == "안녕하세요"
    s = score.summarize_utterances([row])
    assert s["finals"] == 1 and s["mismatches"] == 0 and s["mean_cer"] == 0.0
    assert 80 <= s["final_p50"] <= 400
    assert s["turn_end_gap_p50"] == pytest.approx(0.0, abs=30)
    assert s["noise_turns"] == 0 and s["early_turns"] == 0 and s["split"] == 0
    assert s["empty_finals_per_clip"] == 1 and s["speech_begin_per_clip"] == 2


async def test_fragment_final_arriving_after_speech_end_is_early_turn_not_complete() -> None:
    stt = _FakeSTT([(1, 0.70, _final("안녕", lag=0.45)), (2, 0.90, _final("음"))])
    res = await _clip(stt)
    assert res["stopped"] == "turn"
    assert res["streams"] == 2  # 조각에서 멈추지 않는다
    row = _row(res)
    assert [score.kind_of(row, e) for e in res["events"] if e["type"] == "final"] == ["fragment", "noise"]
    assert score.turn_of(row)["text"] == "안녕"
    s = score.summarize_utterances([row])
    assert s["early_turns"] == 1 and s["noise_turns"] == 0 and s["split"] == 0
    assert s["final_p50"] is None and s["mismatches"] == 1


async def test_fragment_before_speech_end_then_rest_counts_as_split() -> None:
    stt = _FakeSTT([(1, 0.30, _final("안녕", lag=0.05)), (2, 0.74, _final("하세요", lag=0.1))])
    res = await _clip(stt)
    assert res["stopped"] == "turn"
    assert res["streams"] == 2
    row = _row(res)
    assert [score.kind_of(row, e) for e in res["events"] if e["type"] == "final"] == ["fragment", "complete"]
    assert score.turn_of(row)["text"] == "안녕"
    s = score.summarize_utterances([row])
    assert s["split"] == 1 and s["early_turns"] == 1 and s["noise_turns"] == 0
    assert s["final_p50"] is None
    assert s["mismatches"] == 1 and s["mean_cer"] > 0


async def test_lead_noise_final_is_noise_turn_not_split() -> None:
    stt = _FakeSTT([(1, 0.06, _final("음")), (2, 0.70, _final("안녕하세요", lag=0.1))])
    res = await _clip(stt)
    assert res["stopped"] == "turn" and res["streams"] == 2
    row = _row(res)
    assert score.turn_of(row)["text"] == "음"
    s = score.summarize_utterances([row])
    assert s["noise_turns"] == 1 and s["early_turns"] == 0 and s["split"] == 0
    assert s["final_p50"] is None and s["mismatches"] == 1


async def test_empty_final_after_speech_end_does_not_end_measurement() -> None:
    stt = _FakeSTT([(1, 0.62, _final("")), (2, 0.80, _final("어"))])
    res = await _clip(stt)
    assert res["stopped"] == "turn"
    assert res["streams"] == 2
    s = score.summarize_utterances([_row(res)])
    assert s["finals"] == 1 and s["noise_turns"] == 1  # 말소리가 끝난 뒤 열린 스트림이라 잡음
    assert s["final_p50"] is None


def test_final_end_falls_back_to_arrival_without_offset() -> None:
    assert score.final_end_s({"t": 0.9, "stream_open_s": 0.2, "end_offset_s": 0.5}) == pytest.approx(0.7)
    assert score.final_end_s({"t": 0.9, "stream_open_s": 0.2, "end_offset_s": 0.0}) == 0.9
    assert score.final_end_s({"t": 0.9, "stream_open_s": 0.2, "end_offset_s": None}) == 0.9


async def test_noise_only_clip_runs_to_end_and_after_eos_final_is_not_counted() -> None:
    stt = _FakeSTT(
        [(1, 0.10, _BEGIN), (1, 0.15, _final("어")), (2, 0.30, _BEGIN), (2, 0.32, _final(""))],
        on_eos=[_final("음")],
    )
    res = await _clip(stt, pcm=b"\x01\x00" * (SR * 2 // 5), stop_after_s=None)
    assert res["stopped"] == "audio_end"
    assert res["audio_sent_s"] == pytest.approx(0.4)
    assert [e["after_eos"] for e in res["events"] if e["type"] == "final"] == [False, False, True]
    s = score.summarize_noise_only([_row(res, speech_end_s=0.0, noise_only=True)])
    assert s["false_finals"] == 1 and s["false_texts"] == ["어"]
    assert s["clips_with_false"] == 1 and s["empty_finals"] == 1 and s["speech_begins"] == 2
    assert s["false_per_min"] == pytest.approx(1 / (0.4 / 60))


async def test_no_final_ends_after_audio_and_counts_as_missing_turn() -> None:
    stt = _FakeSTT([], on_eos=[_final("뒤늦은")])
    res = await _clip(stt, pcm=b"\x01\x00" * (SR // 5), stop_after_s=0.1)
    assert res["stopped"] == "audio_end"
    s = score.summarize_utterances([_row(res, speech_end_s=0.15, speech_start_s=0.05)])
    assert s["finals"] == 0 and s["final_rate"] == 0.0
    assert s["mean_cer"] is None and s["mean_cer_all"] == 1.0


async def test_hung_stream_after_eos_is_cut_by_drain_timeout() -> None:
    stt = _FakeSTT([], hang_on_eos=True)
    t0 = time.perf_counter()
    res = await _clip(stt, pcm=b"\x01\x00" * (SR // 5), stop_after_s=None, drain_s=0.1)
    elapsed = time.perf_counter() - t0
    assert res["stopped"] == "drain_timeout"
    assert 0.28 <= elapsed <= 1.5


async def test_chunks_are_paced_at_real_time() -> None:
    stt = _FakeSTT([])
    await run_chirp.run_clip(stt, b"\x01\x00" * (SR // 4), chunk_ms=25, stop_after_s=None, drain_s=0.2)
    assert len(stt.chunk_times) == 10
    span = stt.chunk_times[-1] - stt.chunk_times[0]
    assert 9 * 0.025 * 0.9 <= span <= 9 * 0.025 + 0.3


async def test_stream_error_is_recorded_and_stream_reopens() -> None:
    stt = _FakeSTT([(2, 0.30, _final("네"))], raise_at=(1, 0.1))
    res = await _clip(stt, pcm=b"\x01\x00" * (SR // 2), stop_after_s=0.2, drain_s=0.3)
    assert res["stopped"] == "turn"
    assert [e["type"] for e in res["events"]] == ["error", "final"]
    assert res["streams"] == 2
    assert score.failed(_row(res))  # 오디오를 잃은 클립이라 채점에서 뺀다


async def test_persistent_stream_error_gives_up_after_three_and_is_not_scored() -> None:
    stt = _BrokenSTT()
    res = await _clip(stt, stop_after_s=None)
    assert res["stopped"] == "error"
    assert stt.opened == res["streams"] == run_chirp.MAX_STREAM_ERRORS
    assert [e["type"] for e in res["events"]] == ["error"] * run_chirp.MAX_STREAM_ERRORS
    assert "PERMISSION_DENIED" in res["events"][0]["error"]
    assert res["audio_sent_s"] < 0.5
    bad = _row(res, file="a.wav")
    good = _row({"stopped": "turn", "audio_sent_s": 1.0, "events": [_ev_final("안녕하세요", t=0.8, end=0.6)]},
                file="b.wav")
    s = score.summarize_utterances([bad, good])
    assert s["n"] == 1 and s["errors"] == 1 and s["final_rate"] == 1.0 and s["mean_cer_all"] == 0.0
    only = score.summarize_noise_only([{**bad, "noise_only": True}, {**good, "noise_only": True}])
    assert only["clips"] == 1 and only["errors"] == 1 and only["false_finals"] == 1


def test_ci_sample_size_is_distinct_audio_not_repeats() -> None:
    def rows(files: list[str]) -> list[dict]:
        out = []
        for i, f in enumerate(files):
            hit = i % 4 != 3
            events = [_ev_final("안녕하세요", t=0.8, end=0.6)] if hit else []
            out.append(_row({"stopped": "turn" if hit else "audio_end", "audio_sent_s": 1.0, "events": events},
                            file=f))
        return out

    same = score.summarize_utterances(rows([f"u{i}.wav" for i in range(8)] * 3))
    assert same["n"] == 24 and same["n_audio"] == 8 and same["final_rate"] == pytest.approx(0.75)
    assert (same["final_rate_lo"], same["final_rate_hi"]) == pytest.approx(wilson(6, 8)[1:])
    distinct = score.summarize_utterances(rows([f"u{i}.wav" for i in range(24)]))
    assert distinct["n_audio"] == 24
    assert (distinct["final_rate_lo"], distinct["final_rate_hi"]) == pytest.approx(wilson(18, 24)[1:])


def _settings(engine: str = "chirp") -> SimpleNamespace:
    return SimpleNamespace(stt_engine=engine, google_project_id="p", google_stt_location="asia-northeast1",
                           google_stt_model="chirp_3", google_stt_language="ko-KR")


def test_denoise_client_only_adds_denoiser_to_operational_config(monkeypatch) -> None:
    monkeypatch.setattr("app.services.stt.SpeechAsyncClient", lambda **kw: object())
    clients = run_chirp.make_clients(_settings(), [False, True], 0.0)
    assert type(clients[False]) is GoogleSTTClient
    assert type(clients[True]) is run_chirp.DenoiseSTTClient
    assert clients[True].snr_threshold == 0.0
    op, dn = run_chirp.config_dict(clients[False]), run_chirp.config_dict(clients[True])
    assert "denoiser_config" not in op["streaming_config"]["config"]
    assert dn["streaming_config"]["config"].pop("denoiser_config") == {"denoise_audio": True, "snr_threshold": 0.0}
    assert dn == op
    features = op["streaming_config"]["streaming_features"]
    assert features["interim_results"] is True
    assert features["enable_voice_activity_events"] is True


def test_non_chirp_operational_engine_is_refused(monkeypatch) -> None:
    monkeypatch.setattr(run_chirp, "build_stt_client", lambda settings: object())
    with pytest.raises(SystemExit):
        run_chirp.make_clients(_settings("gemini_live"), [False], 0.0)


def _manifest() -> dict:
    def clip(noise, snr, utt, only=False, dur=2.0):
        return {"file": f"{noise}/{utt}", "utt": None if only else utt, "text": "t", "noise": noise, "snr_db": snr,
                "noise_only": only, "dur_s": dur, "speech_end_s": 1.0}

    return {"clips": [
        clip("clean", None, "a.wav"), clip("clean", None, "b.wav"),
        clip("pink", 10.0, "a.wav"), clip("pink", 0.0, "a.wav"), clip("white", 10.0, "a.wav"),
        clip("pink", 10.0, "n", only=True, dur=10.0),
    ]}


def test_plan_jobs_interleaves_denoise_and_filters_conditions() -> None:
    m = _manifest()
    jobs = run_chirp.plan_jobs(m, reps=2, modes=[False, True])
    assert len(jobs) == 2 * 6 * 2
    assert [(j.rep, j.clip["file"], j.denoise) for j in jobs[:3]] == [
        (1, "clean/a.wav", False), (1, "clean/a.wav", True), (1, "clean/b.wav", False),
    ]
    picked = run_chirp.plan_jobs(m, reps=1, modes=[True], noises=["clean", "pink"], snrs=[10.0])
    assert [j.clip["file"] for j in picked] == ["clean/a.wav", "clean/b.wav", "pink/a.wav", "pink/n"]
    no_only = run_chirp.plan_jobs(m, reps=1, modes=[False], noises=["pink"], noise_only=False, utts=["a.wav"])
    assert [j.clip["file"] for j in no_only] == ["pink/a.wav", "pink/a.wav"]
    assert run_chirp.warmup_clip(m)["file"] == "clean/a.wav"
    assert run_chirp.estimate_s(picked, run_chirp.warmup_clip(m), 1, [True]) == pytest.approx(2 * 3 + 10 + 2)


def test_plan_jobs_uses_next_noise_variant_each_rep() -> None:
    def clip(noise, snr, v, only=False):
        return {"file": f"{noise}/v{v}", "utt": None if only else "a.wav", "text": "t", "noise": noise,
                "snr_db": snr, "variant": v, "noise_only": only, "dur_s": 1.0, "speech_end_s": 0.5}

    m = {"variants": 2, "clips": [clip("clean", None, None), clip("pink", 10.0, 1), clip("pink", 10.0, 2),
                                  clip("pink", 10.0, 1, only=True), clip("pink", 10.0, 2, only=True)]}
    jobs = run_chirp.plan_jobs(m, reps=3, modes=[False])
    assert [(j.rep, j.clip["file"], j.clip["noise_only"]) for j in jobs] == [
        (1, "clean/vNone", False), (1, "pink/v1", False), (1, "pink/v1", True),
        (2, "clean/vNone", False), (2, "pink/v2", False), (2, "pink/v2", True),
        (3, "clean/vNone", False), (3, "pink/v1", False), (3, "pink/v1", True),
    ]


def test_run_chirp_dry_run_prints_cost_without_settings_or_network(tmp_path, monkeypatch, capsys) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")

    def boom():
        raise AssertionError("dry-run 은 설정·클라이언트를 만들지 않는다")

    monkeypatch.setattr(run_chirp, "get_settings", boom)
    monkeypatch.setattr(sys, "argv", ["run_chirp.py", "--manifest", str(path), "--dry-run"])
    run_chirp.main()
    out = capsys.readouterr().out
    assert "작업 36개" in out and "$" in out


def test_run_chirp_refuses_over_budget(tmp_path, monkeypatch) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_manifest()), encoding="utf-8")
    monkeypatch.setattr(run_chirp, "get_settings", lambda: pytest.fail("예산 초과면 시작하지 않는다"))
    monkeypatch.setattr(sys, "argv", ["run_chirp.py", "--manifest", str(path), "--max-usd", "0.001"])
    with pytest.raises(SystemExit):
        run_chirp.main()


async def test_run_jobs_rows_carry_condition_fields(tmp_path) -> None:
    audio = tmp_path / "audio"
    for rel in ("clean/a.wav", "pink_snr10/v1/a.wav"):
        mix_noise.write_pcm16(audio / rel, np.full(SR // 5, 0.1))
    clips = [
        {"file": "clean/a.wav", "utt": "a.wav", "text": "네", "noise": "clean", "snr_db": None, "variant": None,
         "noise_only": False, "lead_s": 0.0, "speech_start_s": 0.0, "speech_end_s": 0.05, "utt_end_s": 0.2,
         "dur_s": 0.2},
        {"file": "pink_snr10/v1/a.wav", "utt": "a.wav", "text": "네", "noise": "pink", "snr_db": 10.0, "variant": 1,
         "noise_only": False, "lead_s": 0.0, "speech_start_s": 0.0, "speech_end_s": 0.05, "utt_end_s": 0.2,
         "dur_s": 0.2},
    ]
    jobs = run_chirp.plan_jobs({"clips": clips}, reps=1, modes=[False, True])
    denoiser = _EchoSTT(None)
    denoiser.snr_threshold = 5.0
    clients = {False: _EchoSTT("네"), True: denoiser}
    seen = []
    rows = await run_chirp.run_jobs(jobs, clients, audio, chunk_ms=20, drain_s=0.2, concurrency=2,
                                    on_result=lambda i, r: seen.append(i))
    assert sorted(seen) == [0, 1, 2, 3]
    assert [(r["noise"], r["snr_db"], r["variant"], r["denoise"], r["denoise_snr_threshold"], r["chunk_ms"],
             r["rep"]) for r in rows] == [
        ("clean", None, None, False, None, 20, 1), ("clean", None, None, True, 5.0, 20, 1),
        ("pink", 10.0, 1, False, None, 20, 1), ("pink", 10.0, 1, True, 5.0, 20, 1),
    ]
    assert [r["stopped"] for r in rows] == ["turn", "audio_end", "turn", "audio_end"]
    assert all(r["audio_sent_s"] > 0 and "events" in r for r in rows)


async def test_run_jobs_lets_noise_only_clip_run_to_end_and_counts_every_false_final(tmp_path) -> None:
    audio = tmp_path / "audio"
    mix_noise.write_pcm16(audio / "noise_only/v1/pink_snr10.wav", np.full(SR // 2, 0.1))
    clip = {"file": "noise_only/v1/pink_snr10.wav", "utt": None, "text": None, "noise": "pink", "snr_db": 10.0,
            "variant": 1, "noise_only": True, "dur_s": 0.5}
    stt = _FakeSTT([(1, 0.10, _final("어")), (2, 0.30, _final("음"))])
    jobs = run_chirp.plan_jobs({"clips": [clip]}, reps=1, modes=[False])
    rows = await asyncio.wait_for(run_chirp.run_jobs(jobs, {False: stt}, audio, chunk_ms=20, drain_s=0.3), 5.0)
    assert [r["stopped"] for r in rows] == ["audio_end"]
    assert rows[0]["audio_sent_s"] == pytest.approx(0.5)
    assert [e["text"] for e in rows[0]["events"] if e["type"] == "final"] == ["어", "음"]
    s = score.summarize_noise_only(rows)
    assert s["false_finals"] == 2 and s["clips_with_false"] == 1


async def test_run_jobs_stops_once_final_covers_speech_end_minus_margin(tmp_path) -> None:
    audio = tmp_path / "audio"
    mix_noise.write_pcm16(audio / "clean/a.wav", np.full(SR * 3 // 5, 0.1))
    clip = {"file": "clean/a.wav", "utt": "a.wav", "text": "네", "noise": "clean", "snr_db": None,
            "noise_only": False, "speech_start_s": 0.0, "speech_end_s": 0.4, "dur_s": 0.6}
    stt = _FakeSTT([(1, 0.20, _final("네"))])  # 끝 0.2초: 말소리 끝(0.4) − 0.3 을 넘는다
    jobs = run_chirp.plan_jobs({"clips": [clip]}, reps=1, modes=[False])
    rows = await asyncio.wait_for(run_chirp.run_jobs(jobs, {False: stt}, audio, chunk_ms=20, drain_s=0.2), 5.0)
    assert rows[0]["stopped"] == "turn" and rows[0]["audio_sent_s"] < 0.35


async def test_full_tail_keeps_sending_after_turn(tmp_path) -> None:
    audio = tmp_path / "audio"
    mix_noise.write_pcm16(audio / "clean/a.wav", np.full(SR // 2, 0.1))
    clip = {"file": "clean/a.wav", "utt": "a.wav", "text": "네", "noise": "clean", "snr_db": None,
            "noise_only": False, "speech_start_s": 0.0, "speech_end_s": 0.05, "dur_s": 0.5}
    jobs = run_chirp.plan_jobs({"clips": [clip]}, reps=1, modes=[False])
    early = await run_chirp.run_jobs(jobs, {False: _EchoSTT("네")}, audio, chunk_ms=20, drain_s=0.2)
    full = await run_chirp.run_jobs(jobs, {False: _EchoSTT("네")}, audio, chunk_ms=20, drain_s=0.2, full_tail=True)
    assert early[0]["stopped"] == "turn" and early[0]["audio_sent_s"] < 0.3
    assert full[0]["stopped"] == "audio_end" and full[0]["audio_sent_s"] == pytest.approx(0.5)
    assert sum(e["type"] == "final" for e in full[0]["events"]) >= 3


def _tiny_manifest(tmp_path: Path) -> tuple[Path, dict]:
    audio = tmp_path / "audio"
    clips = []
    for rel, noise, snr, v in (("clean/a.wav", "clean", None, None), ("pink_snr10/v1/a.wav", "pink", 10.0, 1)):
        mix_noise.write_pcm16(audio / rel, np.full(SR // 2, 0.1))
        clips.append({"file": rel, "utt": "a.wav", "text": "네", "noise": noise, "snr_db": snr, "variant": v,
                      "noise_only": False, "lead_s": 0.0, "speech_start_s": 0.0, "speech_end_s": 0.05,
                      "utt_end_s": 0.5, "dur_s": 0.5})
    manifest = {"variants": 1, "clips": clips}
    path = audio / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def _run_args(tmp_path: Path, manifest_path: Path) -> argparse.Namespace:
    return argparse.Namespace(manifest=str(manifest_path), out=str(tmp_path / "runs" / "chirp.json"),
                              denoise_snr_threshold=5.0, chunk_ms=20, reps=1, drain_s=0.2, full_tail=False,
                              concurrency=4, warmup=1)


async def test_run_writes_warmup_git_stt_configs_and_cost(tmp_path, monkeypatch) -> None:
    path, manifest = _tiny_manifest(tmp_path)
    op, dn = _EchoSTT("네"), _EchoSTT("네")
    dn.snr_threshold = 5.0
    monkeypatch.setattr(run_chirp, "get_settings", _settings)
    monkeypatch.setattr(run_chirp, "make_clients", lambda settings, modes, thr: {False: op, True: dn})
    monkeypatch.setattr(run_chirp, "config_dict", lambda c: {"denoise": c is dn})
    args = _run_args(tmp_path, path)
    jobs = run_chirp.plan_jobs(manifest, reps=1, modes=[False, True])
    await asyncio.wait_for(run_chirp._run(args, manifest, jobs, [False, True]), 10.0)
    payload = json.loads(Path(args.out).read_text(encoding="utf-8"))
    assert [(r["file"], r["denoise"], r["rep"]) for r in payload["warmup"]] == [
        ("clean/a.wav", False, 0), ("clean/a.wav", True, 0),
    ]
    assert set(payload["git"]) == {"commit", "dirty"}
    assert payload["stt"]["configs"] == {"operational": {"denoise": False}, "denoise": {"denoise": True}}
    assert payload["stt"]["model"] == "chirp_3"
    sent = (op.audio_bytes_sent + dn.audio_bytes_sent) / BPS
    assert sent > 0 and payload["stt_audio_s"] == pytest.approx(sent, abs=1e-3)
    assert payload["est_usd"] > 0 and payload["est_usd"] == pytest.approx(mix_noise.stt_usd(sent), abs=1e-4)
    assert payload["finished_at"] is not None
    assert len(payload["clips"]) == 4
    assert {(r["denoise"], r["denoise_snr_threshold"]) for r in payload["clips"]} == {(False, None), (True, 5.0)}
    assert {r["variant"] for r in payload["clips"]} == {None, 1}


async def test_warmup_stream_error_stops_before_measurement(tmp_path, monkeypatch) -> None:
    path, manifest = _tiny_manifest(tmp_path)
    broken = {False: _BrokenSTT(), True: _BrokenSTT()}
    monkeypatch.setattr(run_chirp, "get_settings", _settings)
    monkeypatch.setattr(run_chirp, "make_clients", lambda settings, modes, thr: broken)
    monkeypatch.setattr(run_chirp, "config_dict", lambda c: {})
    args = _run_args(tmp_path, path)
    jobs = run_chirp.plan_jobs(manifest, reps=1, modes=[False, True])
    with pytest.raises(SystemExit, match="PERMISSION_DENIED"):
        await asyncio.wait_for(run_chirp._run(args, manifest, jobs, [False, True]), 10.0)
    assert [c.opened for c in broken.values()] == [run_chirp.MAX_STREAM_ERRORS] * 2
    payload = json.loads(Path(args.out).read_text(encoding="utf-8"))
    assert [r["stopped"] for r in payload["warmup"]] == ["error", "error"] and payload["clips"] == []


def test_score_groups_by_condition_and_reads_ws_bench(tmp_path) -> None:
    def clip(noise, snr, denoise, text):
        return {"file": f"{noise}/u.wav", "utt": "u.wav", "text": "네", "noise": noise, "snr_db": snr,
                "denoise": denoise, "denoise_snr_threshold": 0.0 if denoise else None, "chunk_ms": 100,
                "noise_only": False, "speech_start_s": 0.2, "speech_end_s": 1.0, "audio_sent_s": 2.0, "rep": 1,
                "stopped": "turn", "events": [_ev_final(text, t=2.0, end=1.0)]}

    run = tmp_path / "run.json"
    run.write_text(json.dumps({"clips": [clip("pink", 10.0, False, "네"), clip("pink", 10.0, True, "내"),
                                         clip("clean", None, False, "네")]}), encoding="utf-8")
    ws = tmp_path / "ws.json"
    ws.write_text(json.dumps({"engine": "chirp", "noise": "pink", "snr_db": 10.0, "tail_silence_ms": None,
                              "chunk_ms": 100, "lead_ms": 1000,
                              "turns": [{"audio": "u.wav", "stt_final_ms": 900.0, "transcript": "네"}]}),
                  encoding="utf-8")
    clips, ws_runs = score.load([str(run), str(ws)])
    utterances, noise_only = score.group(clips)
    assert sorted(utterances, key=score._sort_key) == [
        ("clean", None, False, 100, None), ("pink", 10.0, False, 100, None), ("pink", 10.0, True, 100, 0.0),
    ]
    assert not noise_only
    assert score.summarize_utterances(utterances[("pink", 10.0, True, 100, 0.0)])["mismatches"] == 1
    assert score.summarize_utterances(utterances[("pink", 10.0, False, 100, None)])["final_p50"] == pytest.approx(
        1000.0
    )
    assert list(ws_runs) == [("chirp", "pink", 10.0, None, 100, 1000)]


def test_score_keeps_chunk_and_denoise_threshold_apart() -> None:
    base = {"noise": "pink", "snr_db": 10.0}
    keys = {score.cond_key({**base, "denoise": d, "chunk_ms": ch, "denoise_snr_threshold": thr})
            for d, ch, thr in [(False, 100, None), (False, 256, None), (True, 100, 0.0), (True, 100, 5.0),
                               (False, 100, 5.0)]}
    assert keys == {("pink", 10.0, False, 100, None), ("pink", 10.0, False, 256, None),
                    ("pink", 10.0, True, 100, 0.0), ("pink", 10.0, True, 100, 5.0)}
    assert sorted(keys | {("clean", None, False, None, None)}, key=score._sort_key)[0][0] == "clean"


def test_ws_key_separates_tail_chunk_lead_and_labels_raw() -> None:
    legacy = {"engine": "chirp", "tail_silence_ms": 800, "turns": []}
    assert score.ws_key(legacy) == ("chirp", "raw", None, 800, None, None)
    assert score.ws_key({**legacy, "tail_silence_ms": 1500}) != score.ws_key(legacy)
    noisy = {"engine": "chirp", "noise": "clean", "snr_db": None, "tail_silence_ms": None, "chunk_ms": 100,
             "lead_ms": 1000}
    assert score.ws_key(noisy) == ("chirp", "clean", None, None, 100, 1000)
    assert score.ws_key({**noisy, "chunk_ms": 256}) != score.ws_key(noisy)
    keys = [score.ws_key(legacy), score.ws_key(noisy), score.ws_key({**noisy, "noise": "pink", "snr_db": 0.0})]
    assert [k[1] for k in sorted(keys, key=lambda k: (k[0], *score._sort_key(k[1:])))] == ["raw", "clean", "pink"]


def test_git_info_reports_commit() -> None:
    info = run_chirp.git_info()
    assert set(info) == {"commit", "dirty"}
    assert info["commit"] is None or len(info["commit"]) == 40


def test_stt_usd_uses_chirp_price() -> None:
    assert mix_noise.stt_usd(60.0) == pytest.approx(0.016)
    assert math.isfinite(mix_noise.stt_usd(1.0))
