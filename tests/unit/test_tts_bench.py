import argparse
import asyncio
import base64
import collections
import importlib.util
import io
import json
import re
import sys
import types
import wave
import zipfile
from pathlib import Path

import numpy as np
import pytest

from app.core.config import Settings
from app.services.pcm_stretch import PcmStretcher
from app.services.qwen_tts import _SPEED_WARMUP_MS, runaway_cap_ms
from app.services.tts import ElevenLabsTTSClient, _SentenceBuffer

_ROOT = Path(__file__).resolve().parents[2]
_BENCH = _ROOT / "scripts" / "tts_bench"
_SR = 16000


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen_el = _load("tts_bench_gen_el", _BENCH / "gen_el.py")
gen_qwen = _load("tts_bench_gen_qwen", _BENCH / "gen_qwen.py")
post = _load("tts_bench_post", _BENCH / "post.py")


def _wav(path: Path, x: np.ndarray) -> None:
    post.write_wav(path, (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def _frames(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, _SR)
        return w.readframes(w.getnframes())


def _tone(seconds: float, hz: float = 200.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * _SR)) / _SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * _SR), dtype=np.float32)


# ───────── 문장·턴 데이터 ─────────
def test_sentences_are_52_unique_clean_lines():
    s = post.load_tsv(_BENCH / "sentences.tsv")
    assert list(s) == [f"s{i:02d}" for i in range(1, 53)]
    assert len(set(s.values())) == 52
    for text in s.values():
        assert text and text == text.strip() and "\\" not in text and "\t" not in text


def test_turns_use_all_44_dialogue_sentences_46_times():
    s = post.load_tsv(_BENCH / "sentences.tsv")
    turns = {k: v.split() for k, v in post.load_tsv(_BENCH / "turns.tsv").items()}
    assert list(turns) == [f"t{i:02d}" for i in range(1, 17)]
    assert sum(map(len, turns.values())) == 46
    assert {sid for ids in turns.values() for sid in ids} == {f"s{i:02d}" for i in range(1, 45)}
    assert all(sid in s for ids in turns.values() for sid in ids)


def test_production_splitter_cuts_each_turn_into_its_sentences():
    s = post.load_tsv(_BENCH / "sentences.tsv")
    for tid, ids in ((k, v.split()) for k, v in post.load_tsv(_BENCH / "turns.tsv").items()):
        buf, got = _SentenceBuffer(), []
        for word in " ".join(s[i] for i in ids).split(" "):
            cut = buf.feed(word + " ")
            if cut:
                got.append(cut.strip())
        rest = buf.flush()
        if rest:
            got.append(rest.strip())
        assert got == [s[i] for i in ids], tid


# ───────── Qwen: 운영 워커와 같은가 ─────────
class _T:
    def __init__(self, a) -> None:
        self.a = np.asarray(a, dtype=np.float32)

    def numpy(self) -> np.ndarray:
        return self.a


@pytest.fixture(scope="module")
def srv():
    torch = types.ModuleType("torch")
    torch.from_numpy = _T
    torch.bfloat16 = "bf16"
    taf = types.ModuleType("torchaudio.functional")
    taf.resample = lambda t, a, b: t
    ta = types.ModuleType("torchaudio")
    ta.functional = taf
    qt = types.ModuleType("qwen_tts")
    qt.Qwen3TTSModel = object
    with pytest.MonkeyPatch.context() as mp:
        for name, mod in {"torch": torch, "torchaudio": ta, "torchaudio.functional": taf, "qwen_tts": qt}.items():
            mp.setitem(sys.modules, name, mod)
        return _load("qwen_tts_server_for_bench", _ROOT / "scripts" / "qwen_tts_server" / "server.py")


_TEXTS = ["네. ", "아, 3,000분이요? ", "가" * 60 + " ", "저희 ", "하아... "]


def test_qwen_generation_kwargs_match_worker(srv):
    seen = {}

    class _Model:
        def stream_generate_voice_clone(self, **kw):
            seen.update(kw)
            return iter(())

    srv.S["model"], srv.S["voices"] = _Model(), {"ai": ("/r/ai.wav", "참조 전사")}
    for text in _TEXTS:
        seen.clear()
        srv._generate("ai", text, srv.max_frames_for(text))
        assert seen == gen_qwen.gen_kwargs(text, "/r/ai.wav", "참조 전사")


def test_qwen_caps_and_pcm_match_worker_and_guard(srv):
    for n in range(1, 301):
        text = "가" * n + " "
        assert gen_qwen.max_frames_for(text) == srv.max_frames_for(text)
        assert gen_qwen.guard_cap_ms(text) == runaway_cap_ms(text)
    for n_samples, cap in ((1000, 10), (19200, 10), (18241, 10), (18240, 10), (0, 2)):
        assert gen_qwen.hit_cap(n_samples, 24000, cap) == srv._hit_cap(n_samples, 24000, cap)
    w = np.random.default_rng(0).uniform(-1.5, 1.5, 4000).astype(np.float32)
    assert gen_qwen.to_pcm16(w) == srv._pcm16(w)
    assert gen_qwen.OUT_SR == srv.OUT_SR == _SR


class _Gen:
    """포크 stream_generate_voice_clone 꼴: 24kHz, emit 2프레임마다 한 청크"""

    def __init__(self, frames: int | None) -> None:
        self.frames, self.kw = frames, None

    def stream_generate_voice_clone(self, **kw):
        self.kw = kw
        total = (self.frames if self.frames is not None else kw["max_frames"]) * 1920
        for start in range(0, total, 3840):
            yield np.full(min(3840, total - start), 0.25, dtype=np.float32), 24000


def _down(w: np.ndarray, sr: int) -> np.ndarray:
    assert sr == 24000
    n = len(w) * 2 // 3
    return np.interp(np.linspace(0, len(w) - 1, n), np.arange(len(w)), w).astype(np.float32)


def test_qwen_synth_sends_sentence_with_space_and_flags_eos():
    model = _Gen(20)
    pcm, info = gen_qwen.synth(model, _down, "/r.wav", "전사", "네, 알겠습니다.")
    assert model.kw["text"] == "네, 알겠습니다. "
    assert model.kw["max_frames"] == gen_qwen.max_frames_for("네, 알겠습니다. ")
    assert info == {"eos": True, "frames": 20, "guard": False, "audio_s": 1.6}
    assert len(pcm) == 20 * 1920 * 2 // 3 * 2

    model = _Gen(None)
    _, info = gen_qwen.synth(model, _down, "/r.wav", "전사", "네, 알겠습니다.")
    assert info["eos"] is False and info["frames"] == model.kw["max_frames"]


def test_qwen_synth_cuts_at_fastapi_guard():
    pcm, info = gen_qwen.synth(_Gen(60), _down, "/r.wav", "전사", "네.")
    assert info["eos"] is True and info["guard"] is True
    assert len(pcm) == runaway_cap_ms("네. ") * _SR // 1000 * 2


def test_qwen_parse_voices_only_and_ref(tmp_path):
    vj = tmp_path / "voices.json"
    vj.write_text(json.dumps({"ai": {"ref_audio": "/a.wav", "ref_text": "에이"},
                              "user": {"ref_audio": "/u.wav", "ref_text": "유저"}}), encoding="utf-8")
    got = gen_qwen.parse_voices(str(vj), "ai", ["new=/mnt/c/ref.wav:안녕: 하세요 "])
    assert got == {"ai": ("/a.wav", "에이"), "new": ("/mnt/c/ref.wav", "안녕: 하세요")}


# ───────── ElevenLabs: 운영 클라이언트 경로 ─────────
def _el_settings():
    fields = {k: f.default for k, f in Settings.model_fields.items() if k.startswith("elevenlabs_")}
    fields.update(elevenlabs_api_key="key", elevenlabs_voice_id="voice-1")
    return types.SimpleNamespace(**fields)


class _FakeWS:
    def __init__(self, uri: str, chunks: list[bytes]) -> None:
        self.uri, self.chunks, self.sent, self.closed = uri, chunks, [], False
        self._end = asyncio.Event()

    async def send(self, msg: str) -> None:
        self.sent.append(json.loads(msg))
        if self.sent[-1].get("text") == "":
            self._end.set()

    def __aiter__(self):
        return self._recv()

    async def _recv(self):
        await self._end.wait()
        for c in self.chunks:
            yield json.dumps({"audio": base64.b64encode(c).decode()})
        yield json.dumps({"isFinal": True})

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def el_env(tmp_path, monkeypatch):
    sents = tmp_path / "s.tsv"
    sents.write_text("s01\t네, 알겠습니다.\ns02\t저희\n", encoding="utf-8")
    turns = tmp_path / "t.tsv"
    turns.write_text("t01\ts01 s02\n", encoding="utf-8")
    calls, plan = [], {"fail": 0, "chunks": [b"\x01\x00" * 800, b"\x02\x00" * 800]}
    tts_mod = sys.modules[ElevenLabsTTSClient.__module__]

    async def connect(uri, additional_headers=None):
        calls.append((uri, additional_headers))
        if plan["fail"]:
            plan["fail"] -= 1
            raise OSError("down")
        ws = _FakeWS(uri, plan["chunks"])
        plan.setdefault("ws", []).append(ws)
        return ws

    monkeypatch.setattr(tts_mod.websockets, "connect", connect)
    monkeypatch.setattr(gen_el, "RETRY_WAIT_S", 0.0)
    client = ElevenLabsTTSClient(_el_settings())
    args = argparse.Namespace(out=tmp_path / "out", sentences=sents, turns=turns, only=None, no_turns=False,
                              voice_id=None, force=False)
    return client, args, calls, plan


def test_el_uses_production_client_and_writes_wavs(el_env):
    client, args, calls, plan = el_env
    assert asyncio.run(gen_el.run(args, client)) == 0
    assert [u for u, _ in calls] == [client._build_uri()] * 3
    assert "output_format=pcm_16000" in calls[0][0] and calls[0][1] == {"xi-api-key": "key"}
    s01, s02, t01 = plan["ws"]
    neutral = {"text": " ", "voice_settings": client._voice_settings}
    assert s01.sent == [neutral, {"text": "네, 알겠습니다. "}, {"text": ""}]
    assert s02.sent == [neutral, {"text": "저희 "}, {"text": ""}]
    assert t01.sent == [neutral, {"text": "네, 알겠습니다. "}, {"text": "저희 "}, {"text": ""}]
    assert all(ws.closed for ws in plan["ws"])
    audio = b"".join(plan["chunks"])
    for rel in ("el/s01.wav", "el/s02.wav", "el_turns/t01.wav"):
        assert _frames(args.out / rel) == audio
    rows = [json.loads(x) for x in (args.out / "el" / "gen.jsonl").read_text("utf-8").splitlines()]
    assert rows == [{"id": "s01", "audio_s": 0.1, "chars": 10}, {"id": "s02", "audio_s": 0.1, "chars": 3}]

    assert asyncio.run(gen_el.run(args, client)) == 0
    assert len(calls) == 3


def test_el_retries_then_gives_up(el_env):
    client, args, calls, plan = el_env
    plan["fail"] = 1
    args.no_turns, args.only = True, "s01"
    assert asyncio.run(gen_el.run(args, client)) == 0
    assert len(calls) == 2 and (args.out / "el" / "s01.wav").exists()

    plan["fail"], args.only = 99, "s02"
    assert asyncio.run(gen_el.run(args, client)) == 1
    assert len(calls) == 2 + gen_el.RETRIES + 1 and not (args.out / "el" / "s02.wav").exists()


def test_el_empty_audio_is_a_failure(el_env):
    client, args, _, plan = el_env
    plan["chunks"] = []
    assert asyncio.run(gen_el.run(args, client)) == 1
    assert not (args.out / "el" / "s01.wav").exists()


# ───────── 바다 지표 ─────────
def test_abrupt_end_is_a_cut_and_faded_end_is_not(tmp_path):
    fade = (10 ** (-3 * np.arange(int(0.06 * _SR)) / (0.06 * _SR))).astype(np.float32)  # 60ms 동안 -60dB
    tone = _tone(0.5)
    cases = {
        "abrupt": np.concatenate([_silence(0.2), tone, _silence(0.3)]),
        "faded": np.concatenate([_silence(0.2), tone[: -len(fade)], tone[-len(fade):] * fade, _silence(0.3)]),
        "loud_to_end": np.concatenate([_silence(0.2), tone]),
    }
    got = {}
    for name, x in cases.items():
        _wav(tmp_path / f"{name}.wav", x)
        got[name] = post.file_metrics(tmp_path / f"{name}.wav", "가나다")["end_cut"]
    assert got == {"abrupt": True, "faded": False, "loud_to_end": True}


def test_pause_cut_inside_a_turn_is_counted(tmp_path):
    x = np.concatenate([_tone(0.4), _silence(0.3), _tone(0.4), _silence(0.3)])
    _wav(tmp_path / "t.wav", x)
    m = post.file_metrics(tmp_path / "t.wav", "가나다")
    assert (m["mid_pauses"], m["mid_cuts"], m["end_cut"]) == (1, 1, True)


def test_lead_tail_and_rate(tmp_path):
    _wav(tmp_path / "x.wav", np.concatenate([_silence(0.3), _tone(1.0), _silence(0.2)]))
    m = post.file_metrics(tmp_path / "x.wav", "가나다라마, 3시!")
    assert (m["lead_s"], m["tail_s"], m["audio_s"]) == (0.3, 0.2, 1.5)
    assert m["rate"] == pytest.approx(6.0, abs=0.05)
    assert m["level_db"] == pytest.approx(20 * np.log10(0.5 / np.sqrt(2)), abs=0.2)


def test_silent_file_has_no_cut_and_no_rate(tmp_path):
    _wav(tmp_path / "z.wav", _silence(0.5))
    m = post.file_metrics(tmp_path / "z.wav", "가나다")
    assert (m["end_cut"], m["rate"], m["level_db"]) == (False, None, None)


# ───────── 조건 만들기 ─────────
def _bench_out(tmp_path: Path) -> tuple[Path, dict, dict]:
    out = tmp_path / "out"
    sents = {"s01": "가나다.", "s02": "라마바사."}
    turns = {"t01": ["s01", "s02"]}
    for cond, amp in (("el", 0.4), ("qwen_ai", 0.2), ("qwen_new", 0.3)):
        _wav(out / cond / "s01.wav", np.concatenate([_tone(0.6, amp=amp), _silence(0.2)]))
        _wav(out / cond / "s02.wav", np.concatenate([_tone(0.8, amp=amp), _silence(0.2)]))
    _wav(out / "el_turns" / "t01.wav", np.concatenate([_tone(1.4, amp=0.4), _silence(0.2)]))
    return out, sents, turns


def test_qwen_turns_are_sentence_audio_joined_without_gaps(tmp_path):
    out, sents, turns = _bench_out(tmp_path)
    post.build_conditions(out, sents, turns, None)
    for cond in ("qwen_ai", "qwen_new"):
        joined = _frames(out / cond / "s01.wav") + _frames(out / cond / "s02.wav")
        assert _frames(out / f"{cond}_turns" / "t01.wav") == joined
    assert post.conditions(out) == (["el", "qwen_ai", "qwen_new"], ["el_turns", "qwen_ai_turns", "qwen_new_turns"])


def test_qwen_speed_uses_production_stretcher_with_turn_warmup(tmp_path):
    out, sents, turns = _bench_out(tmp_path)
    post.build_conditions(out, sents, turns, 1.1)
    s01, s02 = _frames(out / "qwen_ai" / "s01.wav"), _frames(out / "qwen_ai" / "s02.wav")

    def run(pcm: bytes, warmup_ms: int) -> bytes:
        st = PcmStretcher(1.1, warmup_ms=warmup_ms)
        return st.push(pcm) + st.flush()

    assert _frames(out / "qwen_ai_x1.1" / "s01.wav") == run(s01, _SPEED_WARMUP_MS)
    assert _frames(out / "qwen_ai_x1.1_turns" / "t01.wav") == run(s01, _SPEED_WARMUP_MS) + run(s02, 0)
    assert post.turn_audio([b"", s02], 1.1) == run(s02, _SPEED_WARMUP_MS)
    post.build_conditions(out, sents, turns, 1.1)
    names = post.conditions(out)
    assert names[0] == ["el", "qwen_ai", "qwen_ai_x1.1", "qwen_new", "qwen_new_x1.1"]
    assert "qwen_ai_x1.1_turns" in names[1] and not (out / "qwen_ai_x1.1_x1.1").exists()


def test_set_meta_points_prompt_at_new_ref(tmp_path):
    ref = tmp_path / "my.WAV"
    ref.write_bytes(b"RIFF")
    post.write_set(tmp_path / "out", {"s01": "가나다.", "s02": "라마."}, ref, " 안녕,  하세요 ")
    assert (tmp_path / "out" / "set" / "ref.wav").read_bytes() == b"RIFF"
    lines = (tmp_path / "out" / "set" / "meta.lst").read_text("utf-8").splitlines()
    assert lines == ["s01|안녕,  하세요|ref.wav|가나다.", "s02|안녕,  하세요|ref.wav|라마."]


@pytest.mark.parametrize(("sents", "ref_text", "named"), [
    ({"s01": "가|나다."}, "전사", "s01"),
    ({"s01": "가나\n다."}, "전사", "s01"),
    ({"s01": "가나\r다."}, "전사", "s01"),
    ({"s|1": "가나다."}, "전사", "s|1"),
    ({"s01": "가나다."}, "안녕 | 하세요", "ref_text"),
    ({"s01": "가나다."}, "안녕\n하세요", "ref_text"),
])
def test_set_meta_refuses_fields_that_break_the_pipe_format(tmp_path, sents, ref_text, named):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    with pytest.raises(ValueError) as err:
        post.write_set(tmp_path / "out", {"s00": "정상.", **sents}, ref, ref_text)
    assert named in str(err.value) and "s00" not in str(err.value)
    assert not (tmp_path / "out" / "set").exists()


# ───────── 블라인드 청감 ─────────
def test_blind_page_hides_names_and_maps_back(tmp_path):
    out, sents, turns = _bench_out(tmp_path)
    post.build_conditions(out, sents, turns, None)
    names = post.conditions(out)[0]
    mets = post.metrics(out, names, sents)
    page = post.blind_page(out, [("문장", names, sents, mets)], seed=3)
    visible = page.split("<script>")[0]
    assert "qwen" not in visible and "el/" not in visible
    data = json.loads(re.search(r"DATA=(\[.*?\]);", page).group(1))
    assert [d["row"] for d in data] == ["0-s01", "0-s02"]
    for d in data:
        assert sorted(d["order"]) == names
        key = d["row"].split("-", 1)[1]
        for i, cond in enumerate(d["order"]):
            label = chr(65 + i)
            assert f"src='blind/{d['row']}-{label}.wav'" in visible
            assert _frames(out / "blind" / f"{d['row']}-{label}.wav") == _frames(out / cond / f"{key}.wav")
    gains = [float(g) for g in re.findall(r"data-gain='([0-9.]+)'", visible)]
    assert min(gains) < 1.0 and max(gains) == 1.0
    def tag(p: str) -> str:
        return re.search(r"const TAG='([0-9a-f]+)'", p).group(1)

    assert tag(page) == tag(post.blind_page(out, [("문장", names, sents, mets)], 3))
    assert tag(page) != tag(post.blind_page(out, [("문장", names, sents, mets)], 4))
    assert "location.pathname" in page


def test_blind_page_skips_rows_with_one_condition(tmp_path):
    out, sents, _ = _bench_out(tmp_path)
    (out / "qwen_ai" / "s02.wav").unlink()
    (out / "qwen_new" / "s02.wav").unlink()
    names = post.conditions(out)[0]
    page = post.blind_page(out, [("문장", names, sents, post.metrics(out, names, sents))], seed=0)
    assert "0-s01" in page and "0-s02" not in page


# ───────── 보고서 ─────────
def test_paired_ci_of_constant_difference():
    m, lo, hi = post.paired_ci([1.1, 2.1, 3.1], [1.0, 2.0, 3.0])
    assert (m, lo, hi) == pytest.approx((0.1, 0.1, 0.1))


def test_g1_section_pairs_each_condition_against_el():
    per = {
        "el": {f"s{i}": {"cer": 0.02, "utmos": 3.0} for i in range(6)},
        "qwen_new": {f"s{i}": {"cer": 0.05, "utmos": 2.5} for i in range(6)},
    }
    summary = [dict(name="el", n=6, cer_mean=2.0, cer_ci=[1.0, 3.0], wer_mean=5.0, utmos=3.0, sim=0.7,
                    eos_fail=None, level_p50=-20.0),
               dict(name="qwen_new", n=6, cer_mean=5.0, cer_ci=[4.0, 6.0], wer_mean=9.0, utmos=2.5, sim=0.9,
                    eos_fail=0.0, level_p50=-22.0)]
    text = "\n".join(post.g1_section({"summary": summary, "per": per}))
    assert "| qwen_new | 6 | **5.00** [4.0, 6.0] | 9.00 | 2.50 | 0.900 | 0.0 | -22.0 |" in text
    assert "- qwen_new CER +3.00%p [+3.00, +3.00] (n=6)" in text
    assert "- qwen_new UTMOS -0.50 [-0.50, -0.50] (n=6)" in text


def test_main_writes_report_page_and_metrics(tmp_path, monkeypatch, capsys):
    out, _, _ = _bench_out(tmp_path)
    (tmp_path / "s.tsv").write_text("s01\t가나다.\ns02\t라마바사.\n", encoding="utf-8")
    (tmp_path / "t.tsv").write_text("t01\ts01 s02\n", encoding="utf-8")
    ref = tmp_path / "ref.wav"
    _wav(ref, _tone(1.0))
    (out / "qwen_new" / "gen.jsonl").write_text(  # 같은 id 는 마지막 줄(다시 돌린 결과)
        '{"id": "s02", "eos": false, "guard": false}\n{"id": "s01", "eos": true, "guard": false}\n'
        '{"id": "s02", "eos": true, "guard": true}\n', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["post.py", "--out", str(out), "--sentences", str(tmp_path / "s.tsv"),
                                      "--turns", str(tmp_path / "t.tsv"), "--ref", str(ref), "--ref-text", "전사"])
    assert post.main() == 0
    md = (out / "bench.md").read_text("utf-8")
    assert "g1_score 결과 없음" in md
    assert re.search(r"\| qwen_new \| 2 \| .* \| 0 · 1 / 2 \|", md)
    assert re.search(r"\| el_turns \| 1 \|", md)
    assert (out / "listen_blind.html").exists() and (out / "set" / "meta.lst").exists()
    saved = json.loads((out / "bada_metrics.json").read_text("utf-8"))
    assert set(saved["sentences"]) == {"el", "qwen_ai", "qwen_new"}
    assert set(saved["turns"]) == {"el_turns", "qwen_ai_turns", "qwen_new_turns"}


def test_main_drops_turns_whose_sentences_are_not_listed(tmp_path, monkeypatch):
    out, _, _ = _bench_out(tmp_path)
    (tmp_path / "s.tsv").write_text("s01\t가나다.\n", encoding="utf-8")
    (tmp_path / "t.tsv").write_text("t01\ts01 s02\nt02\ts01\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["post.py", "--out", str(out), "--sentences", str(tmp_path / "s.tsv"),
                                      "--turns", str(tmp_path / "t.tsv")])
    assert post.main() == 0
    assert sorted(p.name for p in (out / "qwen_ai_turns").iterdir()) == ["t02.wav"]
    saved = json.loads((out / "bada_metrics.json").read_text("utf-8"))
    assert set(saved["turns"]["qwen_ai_turns"]) == {"t02"} and set(saved["turns"]["el_turns"]) == set()



design_voices = _load("tts_bench_design_voices", _BENCH / "design_voices.py")
prep_ref = _load("tts_bench_prep_ref", _BENCH / "prep_ref.py")
_HANGUL = re.compile("[가-힣]")


def _wav_sr(path: Path, x: np.ndarray, sr: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def _tone_sr(seconds: float, sr: int, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _gap(seconds: float, sr: int) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def test_design_slots_are_the_gender_age_tone_grid():
    grid = [f"{g}_{a}_{t}" for g in ("female", "male") for a in ("young", "middle", "old") for t in ("soft", "rough")]
    assert grid == design_voices.SLOTS
    assert list(design_voices.INSTRUCTS) == design_voices.SLOTS
    rough_words = ("hoarse", "raspy", "gravelly", "husky", "rough", "gruff")
    age_words = {"young": "20s", "middle": "40s", "old": "70s"}
    for slot, text in design_voices.INSTRUCTS.items():
        gender, age, tone = slot.split("_")
        assert (" woman " in text) == (gender == "female") and (" man " in text) == (gender == "male"), slot
        assert age_words[age] in text and (("elderly" in text) == (age == "old")), slot
        assert any(w in text for w in rough_words) == (tone == "rough"), slot
        assert ("soft" in text) == (tone == "soft"), slot


@pytest.mark.parametrize("slot", design_voices.SLOTS)
def test_design_script_lands_in_reference_length_at_expected_pace(slot):
    text = design_voices.script_for(slot)
    _, age, tone = slot.split("_")
    assert text in (design_voices.SCRIPTS[tone], design_voices.SCRIPTS[f"{tone}_old"])
    syllables = len(_HANGUL.findall(text))
    fast, slow = (5.0, 3.6) if age == "old" else (6.8, 5.0)
    lo, hi = design_voices.REF_RANGE_S
    assert lo <= syllables / fast + 2 * design_voices.PAD_S and syllables / slow + 2 * design_voices.PAD_S <= hi
    assert design_voices.judge({"speech_s": syllables / fast}, text)[0]
    assert design_voices.judge({"speech_s": syllables / slow}, text)[0]
    assert text == text.strip() and "\n" not in text and "  " not in text
    assert "안녕히" not in text and "끊" not in text and not text.endswith("감사합니다."), "끝인사는 출력에 샐 수 있다"


def test_design_parsers_reject_unknown_slots_and_names():
    assert design_voices.parse_slots(None) == design_voices.SLOTS
    assert design_voices.parse_slots(" male_old_rough ,female_old_soft") == ["male_old_rough", "female_old_soft"]
    for bad in ("male_old_neutral", "male_elder_soft", ","):
        with pytest.raises(ValueError):
            design_voices.parse_slots(bad)
    got = design_voices.parse_instructs(["male_old_rough=An old man, very hoarse."])
    assert got["male_old_rough"] == "An old man, very hoarse."
    assert got["female_old_soft"] == design_voices.INSTRUCTS["female_old_soft"]
    for bad in ("male_old=x", "male_old_rough=  "):
        with pytest.raises(ValueError):
            design_voices.parse_instructs([bad])
    assert design_voices.parse_refs(["female_young_soft__ai=/x/ref.wav:네, 안녕: 하세요 "]) == {
        "female_young_soft__ai": ("/x/ref.wav", "네, 안녕: 하세요")}
    for bad in ("female_young_soft=/x.wav:t", "female_young_soft__=/x.wav:t", "ai__x=/x.wav:t",
                "female_young_soft__ai=/x.wav:  "):
        with pytest.raises(ValueError):
            design_voices.parse_refs([bad])


def test_design_measure_speech_span_and_judge_by_syllables():
    sr = 24000
    m = design_voices.measure(np.concatenate([_gap(1.0, sr), _tone_sr(10.0, sr), _gap(0.7, sr)]), sr)
    assert m["dur_s"] == pytest.approx(11.7, abs=0.01)
    assert m["speech_s"] == pytest.approx(10.0, abs=0.03)
    assert m["lead_s"] == pytest.approx(1.0, abs=0.03) and m["tail_s"] == pytest.approx(0.7, abs=0.03)
    assert m["ref_s"] == pytest.approx(11.0, abs=0.03)
    text = "가" * 45  # 5~18초가 정상
    assert design_voices.judge(m, text) == (True, None)
    short = design_voices.judge(design_voices.measure(_tone_sr(4.5, sr), sr), text)
    long = design_voices.judge(design_voices.measure(_tone_sr(19.0, sr), sr), text)
    assert short[0] is False and "끝 잘림" in short[1]
    assert long[0] is False and "EOS" in long[1]
    silent = design_voices.measure(_gap(3.0, sr), sr)
    assert silent["speech_s"] == 0.0 and design_voices.judge(silent, text)[0] is False


class _DesignModel:
    def __init__(self, seconds: dict[str, float] | None = None) -> None:
        self.seconds = seconds or {}
        self.calls: list[dict] = []
        self.seed: int | None = None

    def set_seed(self, seed: int) -> None:
        self.seed = seed

    def generate_voice_design(self, **kw):
        self.calls.append({**kw, "seed_at_call": self.seed})
        key = (kw["instruct"][:12], self.seed)
        secs = self.seconds.get(key, 10.0)
        return [np.concatenate([_gap(0.3, 24000), _tone_sr(secs, 24000), _gap(0.3, 24000)])], 24000


def _run_design(out: Path, model: _DesignModel, slots: list[str], n: int, instructs: dict | None = None) -> list[str]:
    instructs = instructs or design_voices.INSTRUCTS
    names = design_voices.pending(out, slots, n, instructs)
    design_voices.design(model, model.set_seed, out, names, instructs)
    return names


def test_design_seeds_each_candidate_before_generating_and_skips_done(tmp_path):
    rough = design_voices.INSTRUCTS["male_old_rough"][:12]
    model = _DesignModel({(rough, 2672): 2.0})
    names = _run_design(tmp_path, model, ["male_old_rough", "female_young_soft"], 2)
    assert names == ["male_old_rough__c1", "male_old_rough__c2", "female_young_soft__c1", "female_young_soft__c2"]
    assert [c["seed_at_call"] for c in model.calls] == [2671, 2672, 2671, 2672]
    scripts = design_voices.SCRIPTS
    assert [c["text"] for c in model.calls] == [scripts["rough_old"]] * 2 + [scripts["soft"]] * 2
    assert model.calls[0]["instruct"] == design_voices.INSTRUCTS["male_old_rough"]
    assert {(c["language"], c["temperature"], c["top_k"], c["max_new_tokens"]) for c in model.calls} == {
        ("Korean", 0.7, 20, 375)}
    log = design_voices.load_log(tmp_path)
    assert [log[x]["seed"] for x in names] == [2671, 2672, 2671, 2672]
    assert log["male_old_rough__c2"]["ok"] is False and log["male_old_rough__c1"]["ok"] is True
    pend = design_voices.pending
    assert pend(tmp_path, ["male_old_rough", "female_young_soft"], 2, design_voices.INSTRUCTS) == []
    assert pend(tmp_path, ["male_old_rough"], 3, design_voices.INSTRUCTS) == ["male_old_rough__c3"]
    (tmp_path / "male_old_rough__c1.wav").unlink()
    assert pend(tmp_path, ["male_old_rough"], 2, design_voices.INSTRUCTS) == ["male_old_rough__c1"]
    with pytest.raises(ValueError):
        design_voices.design(model, model.set_seed, tmp_path, ["male_old_rough__c1"],
                             design_voices.parse_instructs(["male_old_rough=other"]))


def test_design_changed_instruct_makes_new_tagged_candidates_shown_side_by_side(tmp_path):
    _run_design(tmp_path, _DesignModel(), ["male_young_rough"], 2)
    custom = design_voices.parse_instructs(["male_young_rough=CUSTOM <gruff> & husky"])
    tag = design_voices.instruct_tag("male_young_rough", custom["male_young_rough"])
    assert len(tag) == 4
    assert design_voices.instruct_tag("male_young_rough", design_voices.INSTRUCTS["male_young_rough"]) == ""
    model = _DesignModel()
    names = _run_design(tmp_path, model, ["male_young_rough"], 2, custom)
    assert names == [f"male_young_rough__{tag}c1", f"male_young_rough__{tag}c2"]
    assert [c["seed_at_call"] for c in model.calls] == [2671, 2672] and model.calls[0]["instruct"].startswith("CUSTOM")
    assert design_voices.pending(tmp_path, ["male_young_rough"], 2, custom) == []
    html_text = design_voices.page(design_voices.load_log(tmp_path))
    assert "CUSTOM &lt;gruff&gt; &amp; husky" in html_text and "CUSTOM <gruff>" not in html_text
    assert design_voices.INSTRUCTS["male_young_rough"] in html_text
    assert f"설명 {tag} · seed 2671" in html_text and "설명 기본 · seed 2671" in html_text
    assert len(design_voices.write_candidates(tmp_path)) == 4


def test_design_candidates_json_feeds_gen_qwen(tmp_path):
    soft_old = design_voices.INSTRUCTS["female_old_soft"][:12]
    _run_design(tmp_path, _DesignModel({(soft_old, 2671): 25.0}), ["female_old_soft"], 3)
    cands = design_voices.write_candidates(tmp_path)
    assert list(cands) == ["female_old_soft__c2", "female_old_soft__c3"]
    voices = gen_qwen.parse_voices(str(tmp_path / "candidates.json"), "female_old_soft__c3", [])
    path, text = voices["female_old_soft__c3"]
    assert Path(path).is_absolute() and Path(path).exists() and text == design_voices.SCRIPTS["soft_old"]
    (tmp_path / "female_old_soft__c2.wav").unlink()
    assert list(design_voices.write_candidates(tmp_path)) == ["female_old_soft__c3"]


def test_design_page_groups_slots_and_shows_existing_ref(tmp_path):
    ai = tmp_path / "ref_ai.wav"
    _wav_sr(ai, np.concatenate([_gap(0.6, 24000), _tone_sr(14.4, 24000), _gap(0.5, 24000)]), 24000)
    ai_text = "가" * 85
    design_voices.add_refs(tmp_path, {"female_young_soft__ai": (str(ai), ai_text)})
    design_voices.add_refs(tmp_path, {"female_young_soft__ai": (str(ai), ai_text)})
    assert len((tmp_path / "design.jsonl").read_text("utf-8").splitlines()) == 1
    rough = design_voices.INSTRUCTS["male_young_rough"][:12]
    _run_design(tmp_path, _DesignModel({(rough, 2671): 3.0}), ["male_young_rough", "female_young_soft"], 1)
    html_text = design_voices.page(design_voices.load_log(tmp_path))
    assert html_text.index("id='female_young_soft'") < html_text.index("id='male_young_rough'")
    assert "src='female_young_soft__ai.wav'" in html_text and "기존 참조" in html_text
    assert "15.4s <span class='warn'>" in html_text
    assert "class='bad'" in html_text and "끝 잘림" in html_text
    assert design_voices.write_candidates(tmp_path)["female_young_soft__ai"]["ref_text"] == ai_text


def test_design_main_skips_model_when_nothing_pending(tmp_path, monkeypatch):
    _run_design(tmp_path, _DesignModel(), ["male_middle_soft"], 2)
    monkeypatch.setitem(sys.modules, "qwen_tts", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr(sys, "argv", ["design_voices.py", "--out", str(tmp_path), "--n", "2",
                                      "--slots", "male_middle_soft"])
    design_voices.main()
    assert list(json.loads((tmp_path / "candidates.json").read_text("utf-8"))) == [
        "male_middle_soft__c1", "male_middle_soft__c2"]
    assert (tmp_path / "listen.html").exists()


def test_audition_sentences_are_a_subset_of_bench_sentences():
    full = post.load_tsv(_BENCH / "sentences.tsv")
    sub = post.load_tsv(_BENCH / "audition.tsv")
    assert 10 <= len(sub) <= 16 and all(full[k] == v for k, v in sub.items())
    assert min(len(v) for v in sub.values()) <= 3 and max(len(v) for v in sub.values()) >= 40


def _speech_db(w: np.ndarray, sr: int) -> float:
    db = design_voices.frames_db(w, sr)
    loud = db[db >= -45.0]
    return float(10 * np.log10(np.mean(10 ** (loud / 10))))


def _noise(n: int, db: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 10 ** (db / 20), n).astype(np.float32)


def _talk(sr: int, amp: float, noise_db: float, lead: float = 1.0, seed: int = 0) -> np.ndarray:
    x = np.concatenate([_gap(lead, sr), _tone_sr(5.0, sr, amp), _gap(0.4, sr), _tone_sr(5.0, sr, amp), _gap(1.0, sr)])
    return x + _noise(len(x), noise_db, seed)


def test_prep_trims_to_half_second_pads_resamples_and_levels():
    x = np.concatenate([_gap(2.0, 16000), _tone_sr(11.0, 16000, amp=0.05), _gap(1.5, 16000)])
    x += _noise(len(x), -80, 0)
    out, info = prep_ref.prep(x, 16000, -21.0)
    m = design_voices.measure(out, prep_ref.SR)
    assert m["lead_s"] == pytest.approx(0.5, abs=0.03) and m["tail_s"] == pytest.approx(0.5, abs=0.03)
    assert info["dur_s"] == pytest.approx(12.0, abs=0.05) and len(out) / prep_ref.SR == pytest.approx(12.0, abs=0.05)
    assert _speech_db(out, prep_ref.SR) == pytest.approx(-21.0, abs=0.2)
    assert info["level_db"] == pytest.approx(-21.0, abs=0.2)
    assert info["gain_db"] == pytest.approx(-21.0 - 20 * np.log10(0.05 / np.sqrt(2)), abs=0.3)
    assert info["floor_db"] < -60 and info["warnings"] == []
    assert np.abs(out[:100]).max() == 0 and np.abs(out[-100:]).max() == 0


def test_prep_noise_floor_is_measured_in_pauses_and_follows_the_gain():
    clean, info = prep_ref.prep(_talk(24000, 0.1, -75), 24000, -21.0)
    assert info["floor_db"] == pytest.approx(-75 + info["gain_db"], abs=1.5) and info["warnings"] == []
    quiet = prep_ref.prep(_talk(24000, 0.02, -63, seed=1), 24000, -21.0)[1]
    assert quiet["gain_db"] > 10 and quiet["floor_db"] == pytest.approx(-63 + quiet["gain_db"], abs=1.5)
    assert any("잡음 바닥" in w for w in quiet["warnings"])
    x = _talk(24000, 0.1, -50, lead=1.0, seed=2)
    x[:24000] = _noise(24000, -95, 3)
    assert any("잡음 바닥" in w for w in prep_ref.prep(x, 24000, -21.0)[1]["warnings"])
    noisy = prep_ref.prep(_talk(24000, 0.3, -38, seed=4), 24000, -21.0)[1]
    assert any("잡음 바닥" in w for w in noisy["warnings"])


def test_prep_short_dips_between_syllables_are_not_the_noise_floor():
    sr = 24000
    syllable = np.concatenate([_tone_sr(0.2, sr, 0.1), _tone_sr(0.04, sr, 0.0025)])  # 음절 사이 40ms 골(-55dBFS)
    floor = lambda s: _noise(int(s * sr), -80, 5)  # noqa: E731
    x = np.concatenate([floor(0.1), *[syllable] * 20, floor(0.2), *[syllable] * 20, floor(0.1)])
    out, info = prep_ref.prep(x, sr, -21.0)
    assert info["floor_db"] == pytest.approx(-80 + info["gain_db"], abs=2.0)
    assert not any("잡음 바닥" in w for w in info["warnings"])


def test_prep_warns_on_length_and_peak():
    short = prep_ref.prep(_tone_sr(6.0, 24000), 24000, -21.0)[1]
    assert any("밖" in w for w in short["warnings"])
    loud, info = prep_ref.prep(_tone_sr(11.0, 24000, amp=0.05), 24000, -2.0)
    assert any("피크" in w for w in info["warnings"])
    assert 20 * np.log10(np.abs(loud).max()) == pytest.approx(-1.0, abs=0.05)
    with pytest.raises(ValueError):
        prep_ref.prep(_gap(12.0, 24000), 24000, -21.0)


def _cands_dir(tmp_path: Path) -> Path:
    d = tmp_path / "vd"
    d.mkdir()
    cands = {}
    for name, secs in (("female_young_soft__c2", 10.5), ("male_old_rough__c4", 12.0), ("male_old_rough__c1", 11.0)):
        _wav_sr(d / f"{name}.wav", np.concatenate([_gap(0.4, 24000), _tone_sr(secs, 24000, 0.1), _gap(0.4, 24000)]),
                24000)
        cands[name] = {"ref_audio": f"/mnt/c/qwen-out/voice_design/{name}.wav", "ref_text": f"{name} 전사"}
    (d / "candidates.json").write_text(json.dumps(cands, ensure_ascii=False), encoding="utf-8")
    (d / "design.jsonl").write_text(json.dumps({"name": "male_old_rough__c4", "seed": 2674, "instruct": "old man"})
                                    + "\n", encoding="utf-8")
    return d


def test_prep_run_writes_worker_fragment_with_absolute_paths_and_check_set(tmp_path):
    d = _cands_dir(tmp_path)
    out = tmp_path / "refs_v1"
    report = prep_ref.run(d / "candidates.json", ["female_young_soft__c2,male_old_rough__c4"],
                          "/home/u/bada-qwen3-tts/refs/", "v1", out, -21.0)
    part = json.loads((out / "voices.part.json").read_text("utf-8"))
    assert part == {
        "female_young_soft": {"ref_audio": "/home/u/bada-qwen3-tts/refs/female_young_soft_v1.wav",
                              "ref_text": "female_young_soft__c2 전사"},
        "male_old_rough": {"ref_audio": "/home/u/bada-qwen3-tts/refs/male_old_rough_v1.wav",
                           "ref_text": "male_old_rough__c4 전사"},
    }
    assert report["male_old_rough"]["source"] == "male_old_rough__c4" and report["male_old_rough"]["seed"] == 2674
    assert report["female_young_soft"]["seed"] is None
    with wave.open(str(out / "male_old_rough_v1.wav"), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 24000)
    assert json.loads((out / "prep.json").read_text("utf-8"))["female_young_soft"]["file"] == "female_young_soft_v1.wav"
    assert (out / "set" / "meta.lst").read_text("utf-8").splitlines() == [
        "female_young_soft|female_young_soft__c2 전사|female_young_soft.wav|female_young_soft__c2 전사",
        "male_old_rough|male_old_rough__c4 전사|male_old_rough.wav|male_old_rough__c4 전사"]
    assert (out / "set" / "male_old_rough.wav").read_bytes() == (out / "male_old_rough_v1.wav").read_bytes()
    again = tmp_path / "refs_v1_school"
    prep_ref.run(d / "candidates.json", ["female_young_soft__c2", "male_old_rough__c4"], "/mnt/c/x", "v1", again, -21.0)
    assert (again / "male_old_rough_v1.wav").read_bytes() == (out / "male_old_rough_v1.wav").read_bytes()


def test_prep_rejects_relative_dest_unknown_and_duplicate_picks(tmp_path):
    d = _cands_dir(tmp_path)

    def run(picks, dest="/home/u/refs", tag="v1"):
        return prep_ref.run(d / "candidates.json", picks, dest, tag, tmp_path / "o", -21.0)

    for dest in ("refs", "~/bada-qwen3-tts/refs", "/home/~u/refs"):
        with pytest.raises(ValueError):
            run(["male_old_rough__c4"], dest=dest)
    for picks in (["male_old_rough__c9"], ["male_old_rough__c4", "male_old_rough__c1"], [], [" , "]):
        with pytest.raises(ValueError):
            run(picks)
    with pytest.raises(ValueError):
        run(["male_old_rough__c4"], tag="a/b")
    assert not (tmp_path / "o" / "voices.part.json").exists()



# ───────── 화자 12칸: AIHub 71631 실제 사람 후보 (pick_aihub_voices.py) ─────────
pick_aihub = _load("tts_bench_pick_aihub_voices", _BENCH / "pick_aihub_voices.py")


@pytest.mark.parametrize("raw,cell", [("20대", "young"), ("39", "young"), ("40대", "middle"), ("59세", "middle"),
                                      ("60대 이상", "old"), ("72", "old"), ("미상", None), ("1990", None), ("", None)])
def test_aihub_age_cells(raw, cell):
    assert pick_aihub.parse_age(raw) == cell


def test_aihub_gender_snr_and_speaker_filters():
    raw = ("남", "남성", "M", "male", "여", "여자", "F", "Female", "기타", None)
    assert [pick_aihub.parse_gender(g) for g in raw] == ["male"] * 4 + ["female"] * 4 + [None, None]
    snr = ("+31dB", "0dB", "-3dB", "+101dB", None, "")
    assert [pick_aihub.parse_snr(v) for v in snr] == [31, 0, -3, 101, None, None]
    ok = pick_aihub.usable_speaker
    assert ok({"Mask": "미착용", "Condition": "정상"}) and ok({})
    assert not ok({"Mask": "착용", "Condition": "정상"}) and not ok({"Mask": "미착용", "Condition": "목잠김"})


def _u(spk: int, s: float, e: float, text: str = "네 그럼 그렇게 할게요.", emo: str = "없음") -> dict:
    return {"SpeakerNo": f"Speaker{spk}", "StartTime": f"{s:,.2f}", "EndTime": f"{e:,.2f}", "Text": text,
            "VerifyEmotionTarget": emo}


def _block(spk: int, t0: float, emo: str = "없음") -> list[dict]:
    """t0 부터 11초 — 발화 셋, 사이 0.3초"""
    return [_u(spk, t0, t0 + 3.6, emo=emo), _u(spk, t0 + 3.9, t0 + 7.4, emo=emo), _u(spk, t0 + 7.7, t0 + 11.0, emo=emo)]


def _spans(us, spk):
    return [(w["start"], w["end"]) for w in pick_aihub.windows(pick_aihub.utterances({"Conversation": us}), spk)]


def test_aihub_window_extends_over_glued_own_utterances_instead_of_cutting_them():
    # 같은 화자 발화가 0.2초 간격으로 붙어 있으면 그 사이를 경계로 쓰지 않는다
    us = [_u(1, 0.0, 0.8), _u(1, 1.0, 5.0), _u(1, 5.3, 9.0), _u(1, 9.2, 12.0), _u(1, 12.2, 13.0)]
    assert _spans(us, "Speaker1") == [(0.0, 13.0)]


def test_aihub_window_other_speaker_gap_inner_gap_and_markup():
    base = _block(1, 2.0)  # 2~13
    assert _spans(base + [_u(2, 13.3, 14.0)], "Speaker1") == []  # 상대가 0.3초 뒤
    assert _spans(base + [_u(2, 13.6, 14.0)], "Speaker1") == [(2.0, 13.0)]  # 0.6초 뒤
    assert _spans([_u(1, 2.0, 6.0), _u(1, 7.6, 13.0)], "Speaker1") == []  # 안쪽 쉼 1.6초
    assert _spans([_u(1, 2.0, 6.0), _u(1, 7.4, 13.0)], "Speaker1") == [(2.0, 13.0)]  # 1.4초
    assert _spans([_u(1, 2.0, 6.0), _u(1, 6.3, 13.0, "#@이름# 씨 그럼요.")], "Speaker1") == []
    assert _spans([_u(1, 2.0, 6.0), _u(1, 6.3, 13.0, "(웃음) 그럼요.")], "Speaker1") == []


def test_aihub_window_ends_and_starts_at_sentence_boundaries():
    # 마지막 발화가 문장 끝 부호로 끝나야 — 문장 중간에서 끝난 참조는 클론 끝을 자른다
    assert _spans([_u(1, 2.0, 6.0, "그래서 제가."), _u(1, 6.3, 13.0, "갔는데 그게")], "Speaker1") == []
    assert _spans([_u(1, 2.0, 6.0, "그래서 제가."), _u(1, 6.3, 13.0, "갔어요?")], "Speaker1") == [(2.0, 13.0)]
    # 앞 같은 화자 발화가 문장 중간이고 1초 안이면 그 뒤에서 시작하지 않는다(처음부터 포함한 구간만)
    mid = [_u(1, 0.5, 1.7, "그런데"), _u(1, 2.0, 6.0, "저는 그."), _u(1, 6.3, 13.0, "갔어요.")]
    assert _spans(mid, "Speaker1") == [(0.5, 13.0)]
    done = [_u(1, 0.5, 1.7, "그런데."), _u(1, 2.0, 6.0, "저는 그."), _u(1, 6.3, 13.0, "갔어요.")]
    assert _spans(done, "Speaker1") == [(0.5, 13.0), (2.0, 13.0)]
    rested = [_u(1, 0.0, 0.9, "그런데"), _u(1, 2.0, 6.0, "저는 그."), _u(1, 6.3, 13.0, "갔어요.")]
    assert (2.0, 13.0) in _spans(rested, "Speaker1")  # 1.1초 쉬었으면 새로 시작한 말


@pytest.mark.parametrize("word", ["했어요", "합니다", "그렇죠", "할게", "갈까", "하십니까", "좋아", "그랬어", "거야",
                                  "그렇지", "네", "갔어요.", "뭐?", "그렇대요~", "뭐하냐"])
def test_aihub_sentence_end_by_ending_when_unpunctuated(word):
    assert pick_aihub.sentence_end("저는 " + word)


@pytest.mark.parametrize("word", ["그래서", "하시더니", "당근", "그렇게", "그러니까", "했으니까", "하니까", "어", "음",
                                  "하면서", "했는데", "그랬고", "했는데요,", "계", "좋은데", "20대"])
def test_aihub_sentence_middle_by_ending(word):
    assert not pick_aihub.sentence_end("저는 " + word)


def test_aihub_unpunctuated_transcripts_still_get_windows():
    # 71631 화자 79% 는 문장부호를 안 쓴다 — 종결어미로 끝나면 구간이 된다
    ok = [_u(1, 2.0, 6.0, "그래서 제가 갔어요"), _u(1, 6.3, 13.0, "다 팔렸더라고요")]
    mid = [_u(1, 2.0, 6.0, "그래서 제가 갔는데"), _u(1, 6.3, 13.0, "그랬더니 다 팔려서")]
    assert _spans(ok, "Speaker1") == [(2.0, 13.0)] and _spans(mid, "Speaker1") == []


def test_aihub_window_closest_to_target_and_english_anger_hint():
    us = [_u(1, 1.0, 6.0, emo="Neutral"), _u(1, 6.4, 11.5, emo="Angry"), _u(1, 11.9, 14.8, emo="Angry")]
    assert _spans(us, "Speaker1") == [(1.0, 11.5)]  # 10.5·13.8 중 12초에 가까운 10.5 (6.4 시작은 8.4초라 짧음)
    later = [_u(1, 1.0, 6.0, emo="없음"), _u(1, 6.4, 10.5, emo="화남"), _u(1, 10.9, 13.5, emo="화남")]
    got = pick_aihub.windows(pick_aihub.utterances({"Conversation": later}), "Speaker1")
    assert [(w["start"], w["end"]) for w in got] == [(1.0, 13.5)]  # 9.5 보다 12초에 가까운 12.5
    assert got[0]["emotions"] == {"화남": 2}  # '없음' 은 감정으로 세지 않는다
    win = pick_aihub.windows(pick_aihub.utterances({"Conversation": us}), "Speaker1")[0]
    assert win["voiced_s"] == pytest.approx(10.1) and win["emotions"] == {"Neutral": 1, "Angry": 1}
    rows = pick_aihub.shortlist({"S": {"Speaker1": {"ID": "a", "Gender": "여성", "Age": "25"}, "Speaker2": {},
                                       "Conversation": us}}, 1)
    assert rows["female_young"][0]["windows"][0]["emotions"]["Angry"] == 1
    assert pick_aihub.rough_hint({"Angry": 1}) and pick_aihub.rough_hint({"화남": 2})
    assert not pick_aihub.rough_hint({"기쁨": 3, "Neutral": 1}) and not pick_aihub.rough_hint({})


_SR16 = 16000


def _stereo(d: dict, length: float = 70.0, swap: bool = False, mixed: bool = False, noise_db: float = -80.0,
            extra: tuple = (), seed: int = 0) -> bytes:
    """화자 소리는 자기 채널, 상대 채널엔 -20dB 로 샌다. extra = (채널, 시작, 끝) 라벨 없는 소리"""
    n = int(length * _SR16)
    x = np.zeros((n, 2), dtype=np.float32)

    def put(ch: int, s: float, e: float) -> None:
        a, b = int(s * _SR16), min(n, int(e * _SR16))
        if a >= b:
            return
        tone = 0.3 * np.sin(2 * np.pi * (220 if ch == 0 else 150) * np.arange(b - a) / _SR16)
        x[a:b, ch] += tone
        x[a:b, 1 - ch] += 0.1 * tone if mixed else 0.03 * tone

    for u in d["Conversation"]:
        put(int(u["SpeakerNo"][-1]) - 1, float(u["StartTime"].replace(",", "")), float(u["EndTime"].replace(",", "")))
    for ch, s, e in extra:
        put(ch, s, e)
    if mixed:
        x[:, 1] = x[:, 0] = x.sum(1) / 2
    x += np.random.default_rng(seed).normal(0, 10 ** (noise_db / 20), x.shape).astype(np.float32)
    if swap:
        x = x[:, ::-1]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(_SR16)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def _sp(sid: str, gender: str, age: str, **kw) -> dict:
    return {"ID": sid, "Gender": gender, "Age": age, "Mask": "미착용", "Condition": "정상", "RecDevice": "PC", **kw}


def _sess(s1: dict, s2: dict, snr1: str = "+30dB", snr2: str = "+25dB", emo2: str = "없음", more: tuple = ()) -> dict:
    """S1 2~13초, S2 20~31초 구간 + 짧은 상대 발화"""
    conv = _block(1, 2.0) + [_u(2, 15.0, 15.8, "네")] + _block(2, 20.0, emo2) + [_u(1, 33.0, 33.8, "아")] + list(more)
    return {"Speaker1": s1, "Speaker2": s2, "Noise": {"Speaker1SNR": snr1, "Speaker2SNR": snr2}, "Conversation": conv}


def _aihub(tmp_path: Path) -> tuple[list[str], list[str]]:
    f1 = _sp("F1", "여성", "20대")
    sessions = {
        "A": (_sess(f1, _sp("M9", "남성", "60대 이상"), emo2="화남"), {}),
        "B": (_sess(f1, _sp("M2", "남성", "40대", Mask="착용"), snr1="+10dB"), {}),
        # 뒤바뀐 채널 + 상대 채널 딸깍 25ms(60ms 미만이라 통과)
        "C": (_sess(_sp("F3", "여성", "40대"), _sp("X", "기타", "30대")), dict(swap=True, extra=[(1, 9.452, 9.477)])),
        "H": (_sess(_sp("F9", "여성", "40대"), _sp("Y", "기타", "30대")), dict(noise_db=-70.0, seed=3)),
        "D": (_sess(_sp("F4", "여성", "50대"), _sp("M5", "남성", "30대")), dict(noise_db=-40.0, seed=1)),
        "E": (_sess(_sp("F6", "여성", "20대"), _sp("M6", "남성", "20대", Condition="목잠김")), dict(mixed=True)),
        "F": (_sess(_sp("M7", "남성", "40대"), _sp("F7", "여성", "50대")),
              dict(extra=[(1, 5.6, 5.9)], noise_db=-65.0, seed=2)),  # S1 구간 안 쉼에 라벨 없는 상대 소리 0.3초
        # S2 의 12초짜리 구간(53.5~65.5)은 wav(60초) 밖 → 20~31 구간으로 넘어간다
        "G": (_sess(_sp("F8", "여성", "60대 이상"), _sp("M8", "남성", "60대 이상"),
                    more=(_u(2, 53.5, 57.5), _u(2, 57.8, 61.6), _u(2, 61.9, 65.5))), dict(length=60.0)),
    }
    for k in range(9):  # wav 가 없는 세션 — SNR 라벨이 높아도 후보 자리를 차지하면 안 된다
        sessions[f"L{k}"] = (_sess(_sp(f"Q{k}", "여성", "20대"), _sp(f"R{k}", "여성", "20대"), "+40dB", "+40dB"), None)
    lz, wz = tmp_path / "TL_01.zip", tmp_path / "TS_01.zip"
    with zipfile.ZipFile(lz, "w") as zl, zipfile.ZipFile(wz, "w") as zw:
        for name, (d, kw) in sessions.items():
            zl.writestr(f"라벨/{name}.json", json.dumps(d, ensure_ascii=False).encode("utf-8-sig"))
            if kw is not None:
                zw.writestr(f"원천/{name}.wav", _stereo(d, **kw))
    return [str(lz)], [str(wz)]


def test_aihub_end_to_end_picks_cells_channels_and_filters(tmp_path, monkeypatch, capsys):
    labels, wavs = _aihub(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["pick_aihub_voices.py", "--label", *labels, "--wav", *wavs, "--out", str(out),
                                      "--per-cell", "2"])
    pick_aihub.main()
    log = capsys.readouterr().out
    assert "'잡음': 2" in log and "'채널 섞임': 1" in log and "'상대 소리': 1" in log and "'wav 밖': 1" in log
    rows = [json.loads(x) for x in (out / "picks.jsonl").read_text("utf-8").splitlines()]
    cells = collections.defaultdict(list)
    for r in rows:
        cells[r["cell"]].append(r["spk_id"])
    # wav 없는 고-SNR 화자에 밀리지 않은 F1 · 깨끗한 순 F3(-80) > F9(-70), F7(-65) 은 칸당 2명 상한으로 빠짐
    # 마스크·기타 성별·목잠김·시끄러움·섞임·상대 소리는 제외
    assert dict(cells) == {"female_young": ["F1"], "female_middle": ["F3", "F9"], "female_old": ["F8"],
                           "male_old": cells["male_old"]}
    assert sorted(cells["male_old"]) == ["M8", "M9"]
    by = {r["spk_id"]: r for r in rows}
    assert by["M8"]["start"] == 20.0  # wav 밖 구간 대신 다음 구간
    assert by["M9"]["rough_hint"] is True and by["F1"]["rough_hint"] is False
    assert by["F3"]["chan"] == 1 and by["F3"]["sep_db"] > 15  # 뒤바뀐 채널
    assert by["F3"]["snr_db"] > by["F9"]["snr_db"] >= pick_aihub.MIN_SNR_DB
    for r in rows:
        with wave.open(str(out / f"{r['name']}.wav"), "rb") as w:
            assert (w.getnchannels(), w.getframerate()) == (1, 16000)
            x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
        assert len(x) / 16000 == pytest.approx(r["speech_s"] + 2 * pick_aihub.PAD_S, abs=0.01)
        pad = int((pick_aihub.PAD_S - pick_aihub.CUT_MARGIN_S) * 16000)
        assert not x[:pad].any() and not x[-pad:].any()  # 전사 밖 말이 섞이지 않게 앞뒤는 디지털 무음
        assert 20 * np.log10(np.abs(x).max()) > -12  # 상대가 샌 -20dB 채널이 아니라 그 화자 채널
    cands = json.loads((out / "candidates.json").read_text("utf-8"))
    assert set(cands) == {r["name"] for r in rows} and all(Path(v["ref_audio"]).exists() for v in cands.values())
    page_html = (out / "listen.html").read_text("utf-8")
    assert "id='male_old'" in page_html and "거친 후보" in page_html and "후보 없음" in page_html
    # 톤은 귀로 정해 칸을 직접 지정한다 — 칸 없는 이름만 넘기면 거부
    name = by["M9"]["name"]
    with pytest.raises(ValueError):
        prep_ref.run(out / "candidates.json", [name], "/home/u/refs", "v1", tmp_path / "r", -21.0)
    report = prep_ref.run(out / "candidates.json", [f"male_old_rough={name}"], "/home/u/refs", "v1", tmp_path / "r",
                          -21.0)
    assert report["male_old_rough"]["session"] == "A" and report["male_old_rough"]["spk_id"] == "M9"
    part = json.loads((tmp_path / "r" / "voices.part.json").read_text("utf-8"))
    assert part["male_old_rough"]["ref_text"] == " ".join(["네 그럼 그렇게 할게요."] * 3)
    with pytest.raises(ValueError):
        prep_ref.run(out / "candidates.json", [f"male_old_harsh={name}"], "/home/u/refs", "v1", tmp_path / "r", -21.0)


def test_aihub_speakers_filter_and_non_overlapping_windows_per_speaker(tmp_path, monkeypatch):
    f1 = _sp("F1", "여성", "20대")
    conv = ([_u(1, 2.0, 5.0), _u(1, 5.3, 8.0), _u(1, 8.3, 11.0), _u(1, 11.3, 15.5)]  # 2~15.5 · 5.3~15.5 (겹침)
            + [_u(2, 17.0, 17.8, "네.")] + _block(2, 20.0) + _block(1, 40.0))
    d = {"Speaker1": f1, "Speaker2": _sp("M9", "남성", "60대 이상"), "Noise": {}, "Conversation": conv}
    other = _sess(_sp("F2", "여성", "20대"), _sp("M3", "남성", "40대"))
    lz, wz = tmp_path / "TL.zip", tmp_path / "TS.zip"
    with zipfile.ZipFile(lz, "w") as zl, zipfile.ZipFile(wz, "w") as zw:
        for name, sess in (("A", d), ("B", other)):
            zl.writestr(f"{name}.json", json.dumps(sess, ensure_ascii=False).encode("utf-8-sig"))
            zw.writestr(f"{name}.wav", _stereo(sess))
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["p.py", "--label", str(lz), "--wav", str(wz), "--out", str(out),
                                      "--speakers", "F1,M9,F2", "--windows-per-speaker", "2", "--per-cell", "0"])
    pick_aihub.main()
    rows = [json.loads(x) for x in (out / "picks.jsonl").read_text("utf-8").splitlines()]
    assert sorted(r["spk_id"] for r in rows) == ["F1", "F1", "F2", "M9"]  # M3 는 빠지고 칸 상한(0)은 무시
    f1_rows = [r for r in rows if r["spk_id"] == "F1"]
    assert not pick_aihub.overlaps(*f1_rows) and {r["start"] for r in f1_rows} >= {40.0}
    assert len({r["name"] for r in rows}) == 4 and all("_t" in r["name"] for r in rows)


def test_aihub_top_windows_skip_overlaps_in_the_same_session():
    def w(session, start, end, snr):
        return {"session": session, "start": start, "end": end, "snr_db": snr}

    a, b, c, d = w("A", 2.0, 15.5, 60), w("A", 5.3, 15.5, 59), w("A", 40.0, 51.0, 50), w("B", 2.0, 15.5, 55)
    assert pick_aihub.top_windows([c, b, a], 2) == [a, c]
    assert pick_aihub.top_windows([c, b, a, d], 2) == [a, d]  # 다른 세션은 같은 시각이어도 겹침 아님
    assert pick_aihub.top_windows([a, b], 3) == [a]


def test_aihub_speaker_counts_are_distinct_ids(tmp_path):
    labels, wavs = _aihub(tmp_path)
    loaded = pick_aihub.label_index(labels, pick_aihub.wav_sessions(wavs))
    assert set(loaded) == set("ABCDEFGH")
    counts = pick_aihub.speaker_counts(loaded)
    assert counts["female_young"] == 2 and counts["male_old"] == 2  # F1 은 두 세션이어도 한 명
