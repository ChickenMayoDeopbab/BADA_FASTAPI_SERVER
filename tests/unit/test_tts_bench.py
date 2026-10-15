import argparse
import asyncio
import base64
import importlib.util
import json
import re
import sys
import types
import wave
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
