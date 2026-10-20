import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from scipy.signal import resample_poly

from app.services.stt import AUDIO_EOS, STTEvent, STTEventType
from scripts.stt_compare import normalize
from scripts.stt_noise import mix_noise, real_manifest, run_chirp, score

SR = mix_noise.SR
BPS = mix_noise.BYTES_PER_S
SENTENCES = {"u01": "안녕하세요, 예약 변경하려고 전화드렸는데요.", "u02": "다음 주 화요일 오후 3시로 바꿀 수 있을까요?"}


def _syllable(rng: np.random.Generator, dur_s: float, f0: float) -> np.ndarray:
    """배음 + 떨림, 빠르게 오르고 천천히 내린다"""
    n = int(dur_s * SR)
    i = np.arange(n)
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.05 * np.sin(2 * np.pi * rng.uniform(2, 5) * i / SR))) / SR
    tone = sum(np.sin(k * phase + rng.uniform(0, 6.28)) / k for k in range(1, 20))
    return tone * np.sin(0.5 * np.pi * np.minimum(1, i / (0.3 * n))) * np.minimum(1, (n - i) / (0.4 * n))


def _speech(seed: int = 1, groups: tuple[int, ...] = (4, 5), pause_s: float = 0.3, *, filler_db: float | None = None,
            tail_db: tuple[float, ...] = ()) -> np.ndarray:
    """음절 레벨이 −9~+3dB(12dB) 로 퍼진 낭독. 음절 뒤 30% 로 단어 틈 0.1~0.2초, 구 사이 pause_s.
    filler_db 면 앞에 약한 군말(0.25초 + 0.25초 쉼), tail_db 는 끝에 잦아드는 음절들. 앞뒤 무음 50ms"""
    rng = np.random.default_rng(seed)
    f0 = 130 * (1 + 0.1 * rng.standard_normal())
    gap = np.zeros(int(0.05 * SR))
    parts = [gap]
    if filler_db is not None:
        parts += [_syllable(rng, 0.25, f0) * 10 ** (filler_db / 20), np.zeros(int(0.25 * SR))]
    for g, n in enumerate(groups):
        for i in range(n):
            parts.append(_syllable(rng, rng.uniform(0.12, 0.22), f0) * 10 ** (rng.uniform(-9, 3) / 20))
            if i < n - 1 and rng.random() < 0.3:
                parts.append(np.zeros(int(rng.uniform(0.1, 0.2) * SR)))
        if g < len(groups) - 1:
            parts.append(np.zeros(int(pause_s * SR)))
    for db in tail_db:
        parts.append(_syllable(rng, rng.uniform(0.14, 0.2), 0.9 * f0) * 10 ** (db / 20))
    return np.concatenate([*parts, gap])


def _talk(seed: int = 1) -> np.ndarray:
    """실제 말에 가깝게: 약한 군말(−8dB) + 퍼진 음절 + 잦아드는 끝 구절(−8·−11·−14dB)"""
    return _speech(seed, groups=(3, 4, 3), pause_s=0.15, filler_db=-8.0, tail_db=(-8.0, -11.0, -14.0))


_OTHERS = [mix_noise.to_level(_speech(s), -26.0) for s in range(20, 27)]


def _mix(kind: str, snr: float, *, lead_s: float = 1.0, tail_s: float = 2.0, seed: int = 0,
         speech: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """(녹음, 정답 구간). quiet = 백색 SNR 40(−66dBFS 근처)"""
    u = _speech(seed + 1) if speech is None else speech
    lead, tail = int(lead_s * SR), int(tail_s * SR)
    noise_kind, level = ("white", 40.0) if kind == "quiet" else (kind, snr)
    bed = mix_noise.make_noise(noise_kind, np.random.default_rng([seed, int(level)]), lead + len(u) + tail, _OTHERS, 6)
    mixed, _, info = mix_noise.mix_clip(u, bed, level, lead=lead, tail=tail, target_db=-26.0)
    return mixed, info


def _tv_noise(rng: np.random.Generator, n: int, swing_db: float = 8.0, seg_s: float = 0.3) -> np.ndarray:
    """출렁이는 잡음: 분홍 × 0.3초마다 ±swing/2 dB 로 바뀌는 이득(20ms 로 이음)"""
    seg = int(seg_s * SR)
    gain = np.repeat(10 ** (rng.uniform(-swing_db / 2, swing_db / 2, n // seg + 2) / 20), seg)[:n]
    ker = np.ones(SR // 50) / (SR // 50)
    return mix_noise.pink(rng, n) * np.convolve(gain, ker, "same")


def _write(path: Path, x: np.ndarray, sr: int = SR, channels: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if sr != SR:
        x = resample_poly(x, sr // 100, SR // 100)
    data = x if channels == 1 else np.stack([x * 1.2, x * 0.8], axis=1)  # 평균 = 원 신호
    sf.write(str(path), data, sr, subtype="PCM_16")


def _tsv(tmp_path: Path) -> Path:
    path = tmp_path / "sentences.tsv"
    path.write_text("# 주석\n\n" + "".join(f"{k}\t{v}\n" for k, v in SENTENCES.items()), encoding="utf-8")
    return path


def _read16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as f:
        assert (f.getframerate(), f.getnchannels(), f.getsampwidth()) == (SR, 1, 2)
        return np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float64) / 32768.0


def _within(d: dict, truth: dict, start: tuple[float, float], end: tuple[float, float]) -> None:
    """추정 − 정답 이 [아래, 위] 안"""
    assert start[0] <= d["speech_start_s"] - truth["speech_start_s"] <= start[1]
    assert end[0] <= d["speech_end_s"] - truth["speech_end_s"] <= end[1]


TIGHT = (-0.04, 0.06)


@pytest.mark.parametrize(
    ("kind", "snr", "seed", "end_lo"),
    [("quiet", 40.0, 1, -0.04), ("pink", 20.0, 1, -0.08), ("white", 20.0, 2, -0.08), ("babble", 20.0, 3, -0.08),
     ("white", 10.0, 1, -0.25), ("white", 10.0, 2, -0.25), ("pink", 10.0, 3, -0.25), ("pink", 10.0, 4, -0.25),
     ("babble", 10.0, 1, -0.45), ("babble", 10.0, 5, -0.45)],
)
def test_detect_follows_weak_filler_and_fading_last_phrase(kind, snr, seed, end_lo) -> None:
    """음절 레벨이 12dB 퍼지고 끝 구절이 잦아드는 말 — 문턱 하나로 자르면 끝이 0.4~1.2초 일찍 잡힌다"""
    x, truth = _mix(kind, snr, seed=seed, speech=_talk(seed))
    d = real_manifest.detect(x)
    _within(d, truth, TIGHT, (end_lo, 0.04))
    assert d["snr_est_db"] == pytest.approx(snr, abs=1.0)  # 문턱 넘는 큰 프레임만 평균하면 10dB 에서 +2dB
    assert d["noise_floor_db"] <= d["noise_db"] < d["edge_db"] <= d["threshold_db"] < d["speech_level_db"]
    assert (d["parts"], d["other_sounds"], d["unsure_sounds"], d["edge_sounds"]) == (1, 0, 0, 0)
    assert real_manifest.margin_warnings(d, len(x) / SR) == []


@pytest.mark.parametrize("seed", [0, 3, 4])
@pytest.mark.parametrize(
    ("snr", "start", "end"), [(20.0, (-0.2, 0.06), (-0.08, 0.15)), (10.0, (-0.04, 0.6), (-0.7, 0.15))]
)
def test_detect_follows_speech_not_fluctuating_noise(seed, snr, start, end) -> None:
    u = _talk(seed + 1)
    bed = _tv_noise(np.random.default_rng([seed, int(snr)]), 3 * SR + len(u))
    x, _, truth = mix_noise.mix_clip(u, bed, snr, lead=SR, tail=2 * SR, target_db=-26.0)
    d = real_manifest.detect(x)
    _within(d, truth, start, end)  # 고정 4dB 문턱이면 1초까지 번진다
    assert d["snr_est_db"] == pytest.approx(snr, abs=2.0)


def test_detect_at_low_snr_shrinks_span_and_warns() -> None:
    x, truth = _mix("pink", 5.0, seed=2, speech=_talk(2))
    d = real_manifest.detect(x)
    _within(d, truth, (-0.04, 0.7), (-1.0, 0.04))  # 잡음에 묻힌 군말·끝 음절만큼 줄고, 늘지는 않는다
    assert 5.0 <= d["snr_est_db"] < real_manifest.LOW_SNR_DB  # 묻힌 음절이 빠져 실제보다 1~2dB 높다
    assert any(w.startswith("추정 SNR") for w in real_manifest.margin_warnings(d, len(x) / SR))


def test_detect_ignores_clicks_and_flags_a_separate_sound_without_moving_the_span() -> None:
    x, truth = _mix("pink", 20.0, lead_s=2.5)
    click = np.zeros_like(x)
    c0 = int(6.0 * SR)  # 꼬리 잡음 안
    click[c0 : c0 + 160] = 0.5  # 10ms 튐, 말소리보다 훨씬 크다
    near = np.zeros_like(x)
    n0 = int((truth["speech_end_s"] + 0.3) * SR)
    near[n0 : n0 + 160] = 0.5  # 말소리 0.3초 뒤 튐 — 가까워도 음절로 흡수하지 않는다
    bang = np.zeros_like(x)
    bang[int(0.6 * SR) : int(1.0 * SR)] = np.random.default_rng(5).standard_normal(int(0.4 * SR)) * 0.03
    d = real_manifest.detect(x + click + near + bang)
    _within(d, truth, TIGHT, (-0.08, 0.04))
    assert (d["other_sounds"], d["unsure_sounds"], d["edge_sounds"]) == (1, 0, 0)
    assert "말소리 밖 소리 1곳" in real_manifest.margin_warnings(d, len(x) / SR)
    assert real_manifest.detect(x + click)["snr_est_db"] == pytest.approx(20.0, abs=1.0)  # 튐은 잡음 평균에서 뺀다


def _burst(n: int, at_s: float, dur_s: float, level_db: float, seed: int = 5) -> np.ndarray:
    out = np.zeros(n)
    a = int(at_s * SR)
    out[a : a + int(dur_s * SR)] = np.random.default_rng(seed).standard_normal(int(dur_s * SR)) * 10 ** (level_db / 20)
    return out


@pytest.mark.parametrize("louder_db", [10.0, 16.0])
@pytest.mark.parametrize("where", ["before", "after"])
def test_detect_keeps_a_short_sentence_when_a_louder_short_sound_is_far_away(where, louder_db) -> None:
    """에너지로 덩어리를 고르면 말소리보다 10dB 넘게 큰 0.3초 소리가 짧은 문장을 밀어내고 구간이 된다"""
    x, truth = _mix("pink", 20.0, lead_s=2.5, tail_s=3.0, speech=_speech(2, groups=(5,)))
    at = truth["speech_start_s"] - 1.3 if where == "before" else truth["speech_end_s"] + 1.0
    d = real_manifest.detect(x + _burst(len(x), at, 0.3, -26.0 + louder_db))
    _within(d, truth, TIGHT, (-0.08, 0.04))
    assert (d["parts"], d["other_sounds"], d["unsure_sounds"]) == (1, 1, 0)
    assert real_manifest.margin_warnings(d, len(x) / SR) == ["말소리 밖 소리 1곳"]


def test_detect_flags_a_long_louder_sound_as_ambiguous(tmp_path) -> None:
    x, truth = _mix("pink", 20.0, tail_s=3.5)
    y = x + _burst(len(x), truth["speech_end_s"] + 1.0, 0.8, -15.0)  # 말소리보다 11dB 큰 0.8초
    d = real_manifest.detect(y)
    _within(d, truth, TIGHT, (-0.08, 0.04))
    assert (d["other_sounds"], d["unsure_sounds"]) == (1, 1)
    warns = real_manifest.margin_warnings(d, len(y) / SR)
    assert warns == ["말소리 밖 소리 1곳", f"그중 말소리보다 큰 긴 소리 1곳 — {real_manifest.AMBIGUOUS}"]
    rec = tmp_path / "rec"
    _write(rec / "cafe" / "u01.wav", y)
    _write(rec / "cafe" / "u02.wav", x)
    manifest, rows = real_manifest.build(rec, SENTENCES, tmp_path / "out", clean_label="quiet")
    assert [c["file"] for c in manifest["clips"]] == ["cafe/u02.wav"]
    assert any(w.endswith(real_manifest.AMBIGUOUS) for w in rows[0]["warnings"]) and not rows[0]["included"]
    assert (tmp_path / "out" / "cafe" / "u01.wav").exists()  # 들어 보고 spans 로 고치라고 남긴다


def test_detect_keeps_phrases_split_by_a_long_pause_as_one_span() -> None:
    x, truth = _mix("pink", 20.0, speech=_speech(3, groups=(4, 5), pause_s=1.2))
    d = real_manifest.detect(x)
    _within(d, truth, TIGHT, (-0.08, 0.04))
    assert (d["parts"], d["other_sounds"]) == (2, 0)
    assert "말소리 덩어리 2개 — 구간 확인" in real_manifest.margin_warnings(d, len(x) / SR)


def test_detect_warns_about_sound_in_the_first_and_last_part_of_the_recording() -> None:
    x, truth = _mix("pink", 20.0, lead_s=2.0, tail_s=2.5)
    other = mix_noise.to_level(_speech(30, groups=(3,)), -32.0)  # 앞 0.5초 안에서 시작하는 TV 대사
    y = x.copy()
    y[int(0.1 * SR) : int(0.1 * SR) + len(other)] += other
    d = real_manifest.detect(y)
    _within(d, truth, TIGHT, (-0.08, 0.04))
    assert (d["edge_sounds"], d["other_sounds"]) == (1, 1)
    assert any(w.startswith("녹음 처음 0.5초·끝 1초 안 소리") for w in real_manifest.margin_warnings(d, len(y) / SR))
    assert real_manifest.detect(x)["edge_sounds"] == 0
    z = x.copy()
    z[len(z) - len(other) - int(0.1 * SR) : len(z) - int(0.1 * SR)] += other  # 끝 1초 안에서 끝나는 대사
    d = real_manifest.detect(z)
    _within(d, truth, TIGHT, (-0.08, 0.04))
    assert (d["edge_sounds"], d["other_sounds"]) == (1, 1)


def test_detect_takes_in_a_short_syllable_after_a_pause_within_merge_gap() -> None:
    """쉼(0.5초) 뒤 짧은 끝 음절('요')은 0.15초 미만 run 이라도 0.8초 안이면 말소리 — 넓히기(0.3초 틈)로는 못 닿는다"""
    rng = np.random.default_rng(8)
    tail = _syllable(rng, 0.09, 140.0)
    u = np.concatenate([_speech(4, groups=(4,)), np.zeros(int(0.5 * SR)), tail, np.zeros(int(0.05 * SR))])
    x, truth = _mix("white", 20.0, speech=u)
    d = real_manifest.detect(x)
    _within(d, truth, TIGHT, (-0.08, 0.04))
    assert (d["parts"], d["other_sounds"]) == (1, 0)


def test_detect_returns_none_without_speech() -> None:
    noise = mix_noise.white(np.random.default_rng(0), SR * 4) * 0.01
    assert real_manifest.detect(noise) is None
    bump = noise.copy()
    bump[SR : SR + SR // 2] *= 10 ** (5 / 20)  # 0.5초 +5dB — 바닥~피크 폭이 6dB 안 된다
    assert real_manifest.detect(bump) is None
    short = np.concatenate([np.zeros(SR // 10), _speech(1)[: SR // 5]])  # 0.3초: 앞뒤 여유를 둘 수 없는 길이
    assert real_manifest.detect(short) is None


def _info(**kw) -> dict:
    base = {"speech_start_s": 1.0, "speech_end_s": 3.0, "parts": 1, "other_sounds": 0, "unsure_sounds": 0,
            "edge_sounds": 0, "snr_est_db": 20.0}
    return {**base, **kw}


def test_margin_warnings_in_order() -> None:
    assert real_manifest.margin_warnings(_info(), 5.0) == []
    assert real_manifest.margin_warnings(_info(speech_start_s=0.2, snr_est_db=8.5), 3.5) == [
        "앞 여유 0.2초<0.5", "뒤 여유 0.5초<1.5", "추정 SNR 8.5dB<9 — 구간 추정 불확실",
    ]
    assert real_manifest.margin_warnings(_info(snr_est_db=9.0), 5.0) == []
    assert real_manifest.margin_warnings(_info(parts=2, other_sounds=2, unsure_sounds=1, edge_sounds=1), 5.0) == [
        "녹음 처음 0.5초·끝 1초 안 소리 — 대사 같은 잡음이면 구간이 끌린다", "말소리 덩어리 2개 — 구간 확인",
        "말소리 밖 소리 2곳", "그중 말소리보다 큰 긴 소리 1곳 — 구간 모호(뺌)",
    ]


def test_parse_name_and_sentences(tmp_path) -> None:
    assert real_manifest.parse_name("u01") == ("u01", 1)
    assert real_manifest.parse_name("u01_2") == ("u01", 2)
    assert real_manifest.parse_name("u01_greeting") == ("u01_greeting", 1)
    assert real_manifest.parse_name("noise_3") == ("noise", 3)
    assert real_manifest.load_sentences(_tsv(tmp_path)) == SENTENCES
    bad = tmp_path / "bad.tsv"
    bad.write_text("u01\t가\nu01\t나\n", encoding="utf-8")
    with pytest.raises(ValueError, match="중복"):
        real_manifest.load_sentences(bad)
    bad.write_text("u01 가\n", encoding="utf-8")
    with pytest.raises(ValueError, match="TAB"):
        real_manifest.load_sentences(bad)


def test_repo_sentence_table_has_14_ids_in_arabic_digits() -> None:
    rows = real_manifest.load_sentences(real_manifest.HERE / "real_sentences.tsv")
    assert list(rows) == [f"u{i:02d}" for i in range(1, 15)]
    assert "3시" in rows["u03"] and "10시 반" in rows["u05"]  # chirp 가 쓰는 표기
    assert not any(w in t for t in rows.values() for w in ("세 시", "열 시"))
    assert normalize(rows["u06"]) == "전화번호는01023456789예요"


def test_to_mono16k_averages_channels_and_resamples() -> None:
    t = np.arange(44100) / 44100
    s = np.sin(2 * np.pi * 1000 * t)
    assert np.max(np.abs(real_manifest.to_mono16k(np.stack([s, -s]), 44100))) < 1e-9
    y = real_manifest.to_mono16k(np.stack([2 * s, 0 * s]), 44100)
    assert len(y) == SR
    spec = np.abs(np.fft.rfft(y))
    assert np.fft.rfftfreq(len(y), 1 / SR)[np.argmax(spec)] == pytest.approx(1000, abs=2)
    assert np.sqrt(np.mean(y[1000:-1000] ** 2)) == pytest.approx(np.sqrt(0.5), rel=0.01)


def _rec_dir(tmp_path: Path) -> tuple[Path, dict]:
    """quiet(16k mono) · cafe(44.1k 스테레오, 테이크 2 + 잡음만) · tv(여유 부족·클리핑) · 없는 id · 잡동사니.
    u01 은 환경마다 같은 사람이 같은 문장을 읽은 것처럼 같은 말소리"""
    rec = tmp_path / "rec"
    truth = {}
    u01 = _speech(1)
    x, truth["quiet/u01.wav"] = _mix("quiet", 40.0, seed=1, speech=u01)
    _write(rec / "quiet" / "u01.wav", x)
    for take, seed in ((1, 2), (2, 3)):
        x, truth[f"cafe/u01_{take}.wav"] = _mix("pink", 10.0, seed=seed, speech=u01)
        _write(rec / "cafe" / f"u01_{take}.wav", x, sr=44100, channels=2)
    _write(rec / "cafe" / "noise.wav", mix_noise.pink(np.random.default_rng(9), SR * 3) * 0.01, sr=44100, channels=2)
    x, truth["tv/u02.wav"] = _mix("babble", 20.0, lead_s=0.2, tail_s=0.5, seed=4)
    x[int(1.0 * SR) : int(1.0 * SR) + 5] = 1.0
    _write(rec / "tv" / "u02.wav", x)
    x, _ = _mix("quiet", 40.0, seed=5)
    _write(rec / "tv" / "x99.wav", x)
    (rec / "tv" / "notes.txt").write_text("메모", encoding="utf-8")
    _write(rec / "loose.wav", x)
    _write(rec / "cafe" / "old" / "u01.wav", x)
    return rec, truth


def test_build_writes_16k_mono_clips_and_manifest_grouped_by_environment(tmp_path) -> None:
    rec, truth = _rec_dir(tmp_path)
    out = tmp_path / "out"
    manifest, rows = real_manifest.build(rec, SENTENCES, out, clean_label="quiet")
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8")) == manifest
    by_file = {c["file"]: c for c in manifest["clips"]}
    assert sorted(by_file) == ["cafe/noise.wav", "cafe/u01_1.wav", "cafe/u01_2.wav", "quiet/u01.wav", "tv/u02.wav"]
    for rel, want in truth.items():
        c = by_file[rel]
        _within(c, want, TIGHT, (-0.25, 0.04))
        assert c["dur_s"] == pytest.approx(want["dur_s"], abs=1e-3)
        assert len(_read16(out / rel)) / SR == pytest.approx(want["dur_s"], abs=1e-3)
    q, c1, c2 = by_file["quiet/u01.wav"], by_file["cafe/u01_1.wav"], by_file["cafe/u01_2.wav"]
    n = by_file["cafe/noise.wav"]
    assert (q["noise"], q["snr_db"], q["text"], q["utt"], q["variant"], q["noise_only"]) == (
        "clean", None, SENTENCES["u01"], "u01", None, False)
    assert (c1["noise"], c1["take"], c2["take"], c1["env"]) == ("cafe", 1, 2, "cafe")
    assert c1["snr_est_db"] == pytest.approx(10.0, abs=1.0)
    assert c1["snr_db"] == c2["snr_db"] == n["snr_db"] == round(float(np.median([c1["snr_est_db"], c2["snr_est_db"]])))
    assert manifest["env_snr_db"] == {"cafe": c1["snr_db"], "quiet": None, "tv": by_file["tv/u02.wav"]["snr_db"]}
    assert (n["noise_only"], n["utt"], n["text"]) == (True, None, None) and n["dur_s"] == pytest.approx(3.0, abs=1e-3)
    assert all(set(run_chirp.CLIP_KEYS) <= set(c) for c in manifest["clips"] if not c["noise_only"])
    assert all((c["lead_s"], c["utt_end_s"]) == (c["speech_start_s"], c["speech_end_s"])
               for c in manifest["clips"] if not c["noise_only"])  # ws_bench 는 utt_end_s 에서 발화를 자른다
    assert manifest["span_ref_s"] == {"u01": pytest.approx(q["speech_end_s"] - q["speech_start_s"], abs=0.01)}
    assert abs(c1["span_vs_quiet_s"]) < 0.3 and q["span_vs_quiet_s"] is None
    assert by_file["tv/u02.wav"]["span_vs_quiet_s"] is None  # quiet 테이크가 없는 문장은 대조 안 함
    assert not any(c["manual_span"] for c in manifest["clips"]) and manifest["spans_unused"] == []
    warns = {r["source"]: r["warnings"] for r in rows}
    assert any(w.startswith("앞 여유") for w in warns["tv/u02.wav"])
    assert any(w.startswith("뒤 여유") for w in warns["tv/u02.wav"])
    assert any(w.startswith("녹음 처음") for w in warns["tv/u02.wav"])
    assert any(w.startswith("클리핑") for w in warns["tv/u02.wav"])
    assert warns["tv/x99.wav"] == ["문장표에 없는 id(뺌)"]
    assert not (out / "tv" / "x99.wav").exists()
    assert sorted(manifest["skipped"]) == [
        "cafe/old/u01.wav (환경 폴더 바로 아래가 아님)", "loose.wav (환경 폴더 바로 아래가 아님)",
        "tv/notes.txt (오디오 확장자 아님)",
    ]
    assert warns["cafe/u01_1.wav"] == [] and warns["quiet/u01.wav"] == []


def test_build_without_clean_label_keeps_quiet_as_its_own_environment(tmp_path) -> None:
    rec = tmp_path / "rec"
    x, _ = _mix("quiet", 40.0)
    _write(rec / "quiet" / "u01.wav", x)
    manifest, _ = real_manifest.build(rec, SENTENCES, tmp_path / "out", clean_label=None)
    (clip,) = manifest["clips"]
    assert clip["noise"] == "quiet" and clip["snr_db"] == pytest.approx(40, abs=1)
    assert manifest["span_ref_s"] == {}


def test_build_drops_clips_without_speech_and_environments_without_snr(tmp_path) -> None:
    rec = tmp_path / "rec"
    _write(rec / "fan" / "u01.wav", mix_noise.white(np.random.default_rng(1), SR * 4) * 0.01)
    _write(rec / "fan" / "noise.wav", mix_noise.white(np.random.default_rng(2), SR * 4) * 0.01)
    manifest, rows = real_manifest.build(rec, SENTENCES, tmp_path / "out", clean_label="quiet")
    assert manifest["clips"] == []
    warns = {r["source"]: r["warnings"] for r in rows}
    assert warns["fan/u01.wav"] == ["말소리 못 찾음(뺌)"]
    assert warns["fan/noise.wav"] == ["이 환경 SNR 을 못 잼(뺌) — 문장 녹음이 하나는 있어야 한다"]


def test_build_env_snr_is_the_median_of_its_takes(tmp_path) -> None:
    rec = tmp_path / "rec"
    for take, snr in ((1, 10.0), (2, 12.0), (3, 20.0)):
        x, _ = _mix("pink", snr, seed=take, speech=_speech(1))
        _write(rec / "cafe" / f"u01_{take}.wav", x)
    manifest, _ = real_manifest.build(rec, SENTENCES, tmp_path / "out", clean_label="quiet")
    est = sorted(c["snr_est_db"] for c in manifest["clips"])
    assert manifest["env_snr_db"]["cafe"] == round(est[1]) != round(float(np.mean(est)))  # 중앙값: 한 테이크에 덜 끌림


def _dragged_kit(tmp_path: Path) -> tuple[Path, dict]:
    """같은 말소리: quiet · fan(백색 15) · tv(분홍 20 + 말소리 끝 0.4초 뒤 1초 TV 대사 −6dB, 구간이 대사로 끌린다)"""
    rec = tmp_path / "rec"
    u = _speech(7, groups=(4, 4))
    truth = {}
    x, truth["quiet"] = _mix("quiet", 40.0, seed=1, speech=u)
    _write(rec / "quiet" / "u01.wav", x)
    x, truth["fan"] = _mix("white", 15.0, seed=2, speech=u)
    _write(rec / "fan" / "u01.wav", x)
    x, truth["tv"] = _mix("pink", 20.0, seed=3, tail_s=3.0, speech=u)
    other = mix_noise.to_level(_speech(31, groups=(5,)), -32.0)
    a = int((truth["tv"]["speech_end_s"] + 0.4) * SR)
    x[a : a + len(other)] += other
    _write(rec / "tv" / "u01.wav", x)
    return rec, truth


def test_build_drops_noisy_takes_whose_span_is_dragged_away_from_the_quiet_take(tmp_path) -> None:
    rec, truth = _dragged_kit(tmp_path)
    assert real_manifest.detect(real_manifest.decode(rec / "tv" / "u01.wav")[0][0])["speech_end_s"] > (
        truth["tv"]["speech_end_s"] + 1.0)  # 대사가 0.8초 안이라 말소리에 붙는다
    manifest, rows = real_manifest.build(rec, SENTENCES, tmp_path / "out", clean_label="quiet")
    by = {r["file"]: r for r in rows}
    assert [c["file"] for c in manifest["clips"]] == ["fan/u01.wav", "quiet/u01.wav"]
    tv = by["tv/u01.wav"]
    assert not tv["included"] and tv["span_vs_quiet_s"] > 1.0
    assert tv["warnings"][-1] == f"구간 길이가 quiet 보다 {tv['span_vs_quiet_s']:+.1f}초 — 잡음에 끌림?(뺌)"
    assert abs(by["fan/u01.wav"]["span_vs_quiet_s"]) < 0.3
    assert manifest["env_snr_db"]["tv"] is None  # 뺀 클립뿐이면 그 환경은 없다
    kept, rows = real_manifest.build(rec, SENTENCES, tmp_path / "out2", clean_label="quiet", keep_mismatch=True)
    assert [c["file"] for c in kept["clips"]] == ["fan/u01.wav", "quiet/u01.wav", "tv/u01.wav"]
    assert kept["clips"][2]["warnings"][-1].endswith("잡음에 끌림?") and kept["keep_mismatch"] is True


def test_build_tolerates_take_to_take_length_differences(tmp_path) -> None:
    rec, _ = _dragged_kit(tmp_path)
    slow = _speech(7, groups=(4, 4), pause_s=0.7)  # 같은 문장을 0.4초 느리게
    x, _ = _mix("white", 15.0, seed=4, speech=slow)
    _write(rec / "fan" / "u01_2.wav", x)
    manifest, rows = real_manifest.build(rec, SENTENCES, tmp_path / "out", clean_label="quiet")
    slow_row = next(r for r in rows if r["file"] == "fan/u01_2.wav")
    assert slow_row["included"] and 0.3 < slow_row["span_vs_quiet_s"] < real_manifest.SPAN_DIFF_S


def test_spans_override_the_estimate_and_skip_the_length_check(tmp_path) -> None:
    rec, truth = _dragged_kit(tmp_path)
    t = truth["tv"]
    spans = tmp_path / "spans.tsv"
    spans.write_text(f"# 들어 보고 고친 구간\n\ntv/u01.wav\t{t['speech_start_s']}\t{t['speech_end_s']}\n"
                     "fan/u01.wav\t1.0\t99\nquiet/u07.wav\t1\t2\n", encoding="utf-8")
    manifest, rows = real_manifest.build(rec, SENTENCES, tmp_path / "out", clean_label="quiet",
                                         spans=real_manifest.load_spans(spans))
    by = {c["file"]: c for c in manifest["clips"]}
    tv = by["tv/u01.wav"]
    assert (tv["speech_start_s"], tv["speech_end_s"]) == (round(t["speech_start_s"], 3), round(t["speech_end_s"], 3))
    assert tv["manual_span"] and tv["span_vs_quiet_s"] is None and "구간 수동(spans)" in tv["warnings"]
    assert 15.0 < tv["snr_est_db"] < 21.0  # 대사도 잡음이라 분홍 20dB 보다 조금 낮다
    assert "fan/u01.wav" not in by
    fan = next(r for r in rows if r["file"] == "fan/u01.wav")
    assert fan["warnings"] == [f"spans 끝 99초가 녹음({fan['dur_s']:.1f}초) 밖(뺌)"] and not fan["included"]
    assert manifest["spans_unused"] == ["quiet/u07.wav"]


def test_load_spans_accepts_tabs_and_rejects_bad_rows(tmp_path) -> None:
    path = tmp_path / "s.tsv"
    path.write_text("# 파일\t시작\t끝\n\ntv/u01.wav\t1.02\t3.5\ntv/u02.m4a\t0\t2\n", encoding="utf-8")
    assert real_manifest.load_spans(path) == {"tv/u01.wav": (1.02, 3.5), "tv/u02.m4a": (0.0, 2.0)}
    for bad, match in (("tv/u01.wav 1.0 3.0\n", "TAB"), ("tv/u01.wav\t1.0\n", "TAB"),
                       ("tv/u01.wav\t3.0\t1.0\n", "시작 < 끝"), ("tv/u01.wav\t-1\t1.0\n", "시작 < 끝")):
        path.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            real_manifest.load_spans(path)


def test_spans_can_name_the_original_recording(tmp_path) -> None:
    rec, truth = _dragged_kit(tmp_path)
    (rec / "tv" / "u01.wav").rename(rec / "tv" / "u01.WAV")
    t = truth["tv"]
    manifest, _ = real_manifest.build(rec, SENTENCES, tmp_path / "out", clean_label="quiet",
                                      spans={"tv/u01.WAV": (t["speech_start_s"], t["speech_end_s"])})
    tv = next(c for c in manifest["clips"] if c["env"] == "tv")
    assert tv["file"] == "tv/u01.wav" and tv["manual_span"] and manifest["spans_unused"] == []


def test_decode_non_wav_goes_through_converter_and_falls_back_to_ffmpeg(tmp_path, monkeypatch) -> None:
    src = tmp_path / "u01.m4a"
    src.write_bytes(b"not really aac")
    stereo = np.stack([np.full(441, 0.25), np.full(441, -0.25)])
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if cmd[0] == "afconvert":
            return subprocess.CompletedProcess(cmd, 1, "", "unsupported")
        sf.write(cmd[-1], stereo.T, 44100, subtype="PCM_16")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(real_manifest, "converters", lambda: ["afconvert", "ffmpeg"])
    monkeypatch.setattr(real_manifest.subprocess, "run", fake_run)
    x, sr = real_manifest.decode(src)
    assert [c[0] for c in calls] == ["afconvert", "ffmpeg"]
    assert all(str(src) in c for c in calls)
    assert sr == 44100 and x.shape == (2, 441) and x[0, 0] == pytest.approx(0.25, abs=1e-4)
    monkeypatch.setattr(real_manifest.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "x"))
    with pytest.raises(ValueError, match="못 풀었다"):
        real_manifest.decode(src)


def test_decode_without_converter_explains_what_to_install(tmp_path, monkeypatch) -> None:
    src = tmp_path / "u01.m4a"
    src.write_bytes(b"x")
    monkeypatch.setattr(real_manifest.shutil, "which", lambda tool: None)
    with pytest.raises(SystemExit, match="afconvert.*ffmpeg"):
        real_manifest.decode(src)


@pytest.mark.skipif(shutil.which("afconvert") is None, reason="맥 afconvert 가 있을 때만")
def test_real_afconvert_reads_phone_style_m4a(tmp_path) -> None:
    x, truth = _mix("pink", 20.0)
    wav = tmp_path / "src.wav"
    _write(wav, x, sr=44100, channels=2)
    rec = tmp_path / "rec" / "cafe"
    rec.mkdir(parents=True)
    subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", str(wav), str(rec / "u01.m4a")], check=True)
    manifest, _ = real_manifest.build(tmp_path / "rec", SENTENCES, tmp_path / "out", clean_label=None)
    (clip,) = manifest["clips"]
    assert clip["dur_s"] == pytest.approx(truth["dur_s"], abs=0.1)
    assert clip["speech_start_s"] == pytest.approx(truth["speech_start_s"], abs=0.06)
    assert clip["speech_end_s"] == pytest.approx(truth["speech_end_s"], abs=0.06)


class _ScriptedSTT:
    """첫 청크로 클립을 알아보고 말소리 끝 + 0.1초를 받으면 그 문장을 FINAL 로 낸다(모르는 클립은 아무것도)"""

    multi_utterance = False

    def __init__(self, by_first: dict[bytes, tuple[str, float]]) -> None:
        self.by_first = by_first
        self.audio_bytes_sent = 0

    async def stream(self, queue, first_chunk=None):
        hit = self.by_first.get(first_chunk)
        got = 0
        chunk = first_chunk
        while chunk is not AUDIO_EOS:
            got += len(chunk)
            self.audio_bytes_sent += len(chunk)
            if hit and got / BPS >= hit[1] + 0.1:
                yield STTEvent(type=STTEventType.FINAL, text=hit[0], end_offset_sec=hit[1])
                hit = None
            chunk = await queue.get()


def _short_kit(tmp_path: Path, chunk_ms: int) -> tuple[Path, dict, dict]:
    """1초 남짓 녹음 3환경(실시간으로 흘리니 짧게) → (manifest 경로, manifest, 첫 청크 → (정답, 말소리 끝))"""
    rec = tmp_path / "rec"
    for env, kind, snr, seed in (("quiet", "quiet", 40.0, 1), ("cafe", "pink", 15.0, 2), ("fan", "white", 20.0, 3)):
        x, _ = _mix(kind, snr, lead_s=0.3, tail_s=0.6, seed=seed, speech=_speech(1, groups=(3,)))
        _write(rec / env / "u01.wav", x)
    _write(rec / "cafe" / "noise.wav", mix_noise.pink(np.random.default_rng(7), SR // 2) * 0.01)
    out = tmp_path / "audio"
    manifest, _ = real_manifest.build(rec, SENTENCES, out, clean_label="quiet")
    chunk = max(2, BPS * chunk_ms // 1000 // 2 * 2)
    by_first = {
        run_chirp.read_pcm(out / c["file"])[:chunk]: (c["text"], c["speech_end_s"])
        for c in manifest["clips"] if not c["noise_only"]
    }
    return out / "manifest.json", manifest, by_first


async def test_run_chirp_and_score_read_the_real_manifest_unchanged(tmp_path) -> None:
    path, manifest, by_first = _short_kit(tmp_path, 20)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    jobs = run_chirp.plan_jobs(manifest, reps=1, modes=[False, True])
    assert len(jobs) == 2 * len(manifest["clips"]) == 8
    assert run_chirp.warmup_clip(manifest)["file"] == "quiet/u01.wav"
    assert run_chirp.estimate_s(jobs, None, 0, [False, True]) == pytest.approx(
        2 * sum(c["dur_s"] for c in manifest["clips"]))
    assert run_chirp.plan_jobs(manifest, reps=2, modes=[False], noises=["cafe"]) == [
        run_chirp.Job(c, False, r) for r in (1, 2) for c in manifest["clips"] if c["noise"] == "cafe"
    ]
    clients = {False: _ScriptedSTT(by_first), True: _ScriptedSTT(by_first)}
    clients[True].snr_threshold = 0.0
    rows = await asyncio.wait_for(
        run_chirp.run_jobs(jobs, clients, path.parent, chunk_ms=20, drain_s=0.3, concurrency=8), 10.0)
    utterances, noise_only = score.group(rows)
    assert {k[:3] for k in utterances} == {
        ("clean", None, False), ("clean", None, True),
        ("cafe", manifest["env_snr_db"]["cafe"], False), ("cafe", manifest["env_snr_db"]["cafe"], True),
        ("fan", manifest["env_snr_db"]["fan"], False), ("fan", manifest["env_snr_db"]["fan"], True),
    }
    for clips in utterances.values():
        s = score.summarize_utterances(clips)
        assert (s["n"], s["finals"], s["mismatches"], s["early_turns"], s["noise_turns"]) == (1, 1, 0, 0, 0)
        assert s["mean_cer"] == 0.0 and s["final_p50"] is not None
    assert {k[:3] for k in noise_only} == {("cafe", manifest["env_snr_db"]["cafe"], d) for d in (False, True)}
    assert all(score.summarize_noise_only(v)["false_finals"] == 0 for v in noise_only.values())


async def test_run_chirp_full_run_prints_environment_conditions(tmp_path, monkeypatch, capsys) -> None:
    path, manifest, by_first = _short_kit(tmp_path, 20)
    op, dn = _ScriptedSTT(by_first), _ScriptedSTT(by_first)
    dn.snr_threshold = 0.0
    settings = SimpleNamespace(stt_engine="chirp", google_project_id="p", google_stt_location="asia-northeast1",
                               google_stt_model="chirp_3", google_stt_language="ko-KR")
    monkeypatch.setattr(run_chirp, "get_settings", lambda: settings)
    monkeypatch.setattr(run_chirp, "make_clients", lambda s, modes, thr: {False: op, True: dn})
    monkeypatch.setattr(run_chirp, "config_dict", lambda c: {})
    args = argparse.Namespace(manifest=str(path), out=str(tmp_path / "runs" / "chirp.json"), denoise_snr_threshold=0.0,
                              chunk_ms=20, reps=1, drain_s=0.3, full_tail=False, concurrency=8, warmup=1)
    jobs = run_chirp.plan_jobs(manifest, reps=1, modes=[False, True])
    await asyncio.wait_for(run_chirp._run(args, manifest, jobs, [False, True]), 15.0)
    payload = json.loads(Path(args.out).read_text(encoding="utf-8"))
    assert [r["file"] for r in payload["warmup"]] == ["quiet/u01.wav"] * 2
    assert payload["audio"]["kind"] == "real" and len(payload["clips"]) == 8
    out = capsys.readouterr().out
    assert f"cafe_snr{manifest['env_snr_db']['cafe']} u01" in out and "noise_only/cafe_snr" in out
    monkeypatch.setattr(sys, "argv", ["score.py", args.out])
    score.main()
    table = capsys.readouterr().out
    assert "clean" in table and "fan" in table and "cafe" in table


def test_main_prints_table_and_cost(tmp_path, monkeypatch, capsys) -> None:
    rec, _ = _rec_dir(tmp_path)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["real_manifest.py", "--rec-dir", str(rec), "--sentences", str(_tsv(tmp_path)),
                                      "--out", str(out)])
    real_manifest.main()
    text = capsys.readouterr().out
    assert "✗ tv/x99.wav" in text and "문장표에 없는 id(뺌)" in text and "$" in text
    assert "구간 길이 대조: quiet 테이크가 있는 1문장 — 다른 환경에서 0.5초(15%) 넘게 다르면 뺌" in text
    assert (out / "manifest.json").exists()
    empty = tmp_path / "empty"
    (empty / "quiet").mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", ["real_manifest.py", "--rec-dir", str(empty), "--sentences", str(_tsv(tmp_path)),
                                      "--out", str(tmp_path / "o2")])
    with pytest.raises(SystemExit, match="클립이 없다"):
        real_manifest.main()
    assert "구간 길이 대조 안 함" in capsys.readouterr().out


def test_main_reads_spans_and_keep_mismatch(tmp_path, monkeypatch, capsys) -> None:
    rec, truth = _dragged_kit(tmp_path)
    argv = ["real_manifest.py", "--rec-dir", str(rec), "--sentences", str(_tsv(tmp_path)), "--out", str(tmp_path / "o")]
    monkeypatch.setattr(sys, "argv", argv)
    real_manifest.main()
    text = capsys.readouterr().out
    assert "✗ tv/u01.wav" in text and "잡음에 끌림?(뺌)" in text
    spans = tmp_path / "spans.tsv"
    spans.write_text("quiet/u09.wav\t1\t2\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [*argv, "--keep-mismatch", "--spans", str(spans)])
    real_manifest.main()
    text = capsys.readouterr().out
    assert "✗ tv/u01.wav" not in text and "다르면 경고만" in text
    assert "spans 에 있지만 녹음에 없는 파일: quiet/u09.wav" in text
    t = truth["tv"]
    spans.write_text(f"tv/u01.wav\t{t['speech_start_s']}\t{t['speech_end_s']}\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [*argv, "--spans", str(spans)])
    real_manifest.main()
    manifest = json.loads((tmp_path / "o" / "manifest.json").read_text(encoding="utf-8"))
    assert [c["file"] for c in manifest["clips"] if c["manual_span"]] == ["tv/u01.wav"]
