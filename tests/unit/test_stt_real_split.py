import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.stt_noise import mix_noise, real_manifest, run_chirp
from tests.unit.test_stt_real_kit import _burst, _read16, _speech, _syllable, _write

SR = mix_noise.SR
SENT = real_manifest.load_sentences(real_manifest.HERE / "real_sentences.tsv")
IDS = list(SENT)
CHARS = [len(re.sub(r"[\W_]", "", t)) for t in SENT.values()]
LEVEL = -26.0


def _sentence(seed: int, n: int, pause_s: float = 0.15) -> np.ndarray:
    """_talk 처럼(약한 군말 + 퍼진 음절 + 잦아드는 끝 3음절) 음절 수 = 글자 수, 구 사이 pause_s"""
    body = n - 4
    groups = (body // 3, body - 2 * (body // 3), body // 3)
    return _speech(seed, groups=groups, pause_s=pause_s, filler_db=-8.0, tail_db=(-8.0, -11.0, -14.0))


def _hesitant(seed: int, n: int, pause_s: float) -> np.ndarray:
    """첫머리 '음,'(0.25초, −8dB) 뒤 pause_s 머뭇거린 문장"""
    rng = np.random.default_rng(seed + 999)
    body = n - 4
    rest = _speech(seed, groups=(body // 3, body - 2 * (body // 3), body // 3), pause_s=0.15,
                   tail_db=(-8.0, -11.0, -14.0))
    um = _syllable(rng, 0.25, 130.0) * 10 ** (-8 / 20)
    return np.concatenate([np.zeros(int(0.05 * SR)), um, np.zeros(int(pause_s * SR)), rest])


def _utts(seed: int) -> list[np.ndarray]:
    """같은 사람이 문장표를 읽은 14문장(환경마다 같은 말소리)"""
    return [_sentence(100 * seed + i, n) for i, n in enumerate(CHARS)]


def _session(kind: str, snr: float, utts: list[np.ndarray], *, seed: int = 0, pauses: list[float] | None = None,
             breath_db: float | None = None) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """(연속 녹음, 문장마다 정답 구간). 앞 1초 · 문장 사이 3~4초 · 뒤 2초, 잡음은 끊김 없이. quiet = 백색 SNR 40.
    breath_db 면 문장마다 0.3초 앞에서 끝나는 0.4초 들숨(분홍, 그 dBFS)"""
    rng = np.random.default_rng([seed, 7])
    pauses = list(rng.uniform(3.0, 4.0, len(utts) - 1)) if pauses is None else pauses
    parts, truth, t = [np.zeros(SR)], [], SR
    for i, u in enumerate(utts):
        u = mix_noise.to_level(u, LEVEL)
        a, b = mix_noise.speech_span_s(mix_noise.active_frames(u))
        truth.append((t / SR + a, t / SR + b))
        gap = np.zeros(int((pauses[i] if i < len(utts) - 1 else 2.0) * SR))
        parts += [u, gap]
        t += len(u) + len(gap)
    clean = np.concatenate(parts)
    if breath_db is not None:
        n = int(0.4 * SR)
        for s, _ in truth:
            b = mix_noise.pink(rng, n) * np.sin(np.pi * np.arange(n) / n)
            at = int((s - 0.3) * SR) - n
            clean[at : at + n] += b * 10 ** (breath_db / 20) / np.sqrt(np.mean(b**2))
    noise_kind, level = ("white", 40.0) if kind == "quiet" else (kind, snr)
    bed = mix_noise.make_noise(noise_kind, np.random.default_rng([seed, int(level)]), len(clean), [], 0)
    return clean + bed * np.sqrt(10 ** ((LEVEL - level) / 10) / np.mean(bed**2)), truth


def _abs(clip: dict) -> tuple[float, float]:
    return clip["source_start_s"] + clip["speech_start_s"], clip["source_start_s"] + clip["speech_end_s"]


def _window(clip: dict) -> tuple[float, float]:
    return clip["source_start_s"], clip["source_start_s"] + clip["dur_s"]


def _main(monkeypatch, capsys, *argv: str) -> str:
    monkeypatch.setattr(sys, "argv", ["real_manifest.py", *argv])
    real_manifest.main()
    return capsys.readouterr().out


ENVS = (("quiet", "quiet", 40.0, (-0.04, 0.06)), ("cafe", "pink", 20.0, (-0.08, 0.04)),
        ("fan", "white", 10.0, (-0.25, 0.04)))


@pytest.mark.parametrize("useed", [0, 1, 2])
def test_continuous_recording_is_cut_into_sentences_in_table_order(tmp_path, useed) -> None:
    rec, out = tmp_path / "rec", tmp_path / "out"
    utts = _utts(useed)
    truth = {}
    for i, (env, kind, snr, _) in enumerate(ENVS):
        x, truth[env] = _session(kind, snr, utts, seed=10 * useed + i)
        _write(rec / env / "all.wav", x)
    manifest, rows = real_manifest.build(rec, SENT, out, clean_label="quiet")
    assert len(manifest["clips"]) == 3 * len(IDS) and all(r["included"] for r in rows)  # quiet 길이 대조도 통과
    for env, _, _, end in ENVS:
        clips = [c for c in manifest["clips"] if c["env"] == env]
        assert [c["utt"] for c in clips] == IDS and [c["file"] for c in clips] == [f"{env}/{u}.wav" for u in IDS]
        for c, (s, e) in zip(clips, truth[env], strict=True):
            a, b = _abs(c)
            assert -0.04 <= a - s <= 0.06 and end[0] <= b - e <= end[1]
            assert (c["source"], c["split"], c["take"], c["manual_span"]) == (f"{env}/all.wav", "auto", 1, False)
            assert c["speech_start_s"] == pytest.approx(min(real_manifest.CLIP_LEAD_S, s), abs=0.06)
            assert c["dur_s"] - c["speech_end_s"] == pytest.approx(real_manifest.CLIP_TAIL_S, abs=0.3)
            assert len(_read16(out / c["file"])) / SR == pytest.approx(c["dur_s"], abs=1e-3)
            assert not any(w.startswith(("앞 여유", "뒤 여유")) or "순서" in w or "읽음" in w for w in c["warnings"])
    for s in manifest["splits"]:
        assert (s["found"], s["expected"], s["included"], s["cuts"], s["warnings"]) == (14, 14, True, False, [])
        assert s["order_r"] > 0.8
    assert manifest["env_snr_db"] == {"cafe": 20, "fan": 10, "quiet": None}


def test_weak_syllables_inside_a_sentence_do_not_split_it() -> None:
    """10dB 에선 잡음에 묻힌 약한 음절 때문에 문장 안 핵 틈이 1.5초를 넘는다 — 잰 구간 사이로 다시 잇는다"""
    x, truth = _session("white", 10.0, _utts(1), seed=0)
    _, db, raw, floor, peak = real_manifest._frames(x)
    thr = real_manifest._threshold(floor, peak)
    long = [r for r in real_manifest._runs(db >= thr) if (r[1] - r[0]) * real_manifest.FRAME_S >= real_manifest.BLIP_S]
    cores = real_manifest._clusters(long, round(real_manifest.SPLIT_PAUSE_S / real_manifest.FRAME_S))
    big = [c for c in cores if np.count_nonzero(raw[c[0] : c[1]] >= thr) >= 20]
    assert len(big) == 15  # 핵 틈만 보면 한 문장이 둘로 쪼개진다
    spans = real_manifest.find_sentences(x)
    assert len(spans) == 14
    for (a, b), (s, e) in zip(spans, truth, strict=True):
        assert abs(a - s) < 0.1 and -0.3 < b - e < 0.05


@pytest.mark.parametrize(("kind", "snr", "pause"), [("quiet", 40.0, 1.0), ("quiet", 40.0, 1.2), ("pink", 20.0, 1.0),
                                                    ("pink", 20.0, 1.2), ("white", 10.0, 0.9), ("white", 10.0, 1.1)])
def test_a_short_um_before_a_hesitation_stays_in_the_clip(tmp_path, kind, snr, pause) -> None:
    """detect 는 0.8초 넘게 떨어진 0.4초 미만 덩어리를 뺀다 — 핵 덩어리와 합치고 앞 여유가 1.5초라 '음,' 이 남는다"""
    utts = _utts(0)
    utts[4] = _hesitant(404, CHARS[4], pause)
    x, truth = _session(kind, snr, utts, seed=3)
    _write(tmp_path / "rec" / "cafe" / "all.wav", x)
    manifest, _ = real_manifest.build(tmp_path / "rec", SENT, tmp_path / "o", clean_label="quiet")
    assert manifest["splits"][0]["warnings"] == []
    for c, (s, e) in zip(manifest["clips"], truth, strict=True):
        lo, hi = _window(c)
        assert lo <= s and hi >= e
    if kind != "white":  # 10dB 에선 '음' 이 잡음에 묻혀 구간·덩어리에 안 잡히고 앞 여유 안에만 든다
        seg = manifest["splits"][0]["segments"][4]
        assert seg[0] == pytest.approx(truth[4][0], abs=0.06)


@pytest.mark.parametrize("kind", ["white", "pink"])
def test_pauses_at_commas_do_not_raise_order_warnings_at_10db(kind) -> None:
    """쉼표·마침표 자리 0.7초 쉼(u01·u02·u08) — detect 가 한 구절만 남기던 구간을 핵 덩어리와 합친다"""
    for useed in range(3):
        utts = _utts(useed)
        for k in (0, 1, 7):
            utts[k] = _sentence(100 * useed + k, CHARS[k], pause_s=0.7)
        x, truth = _session(kind, 10.0, utts, seed=useed)
        spans = real_manifest.find_sentences(x)
        assert len(spans) == 14
        r, order, long = real_manifest.order_check([e - s for s, e in spans], IDS, SENT)
        assert r > 0.8 and order == [] and long == {}
        for i, ((a, b), (s, e)) in enumerate(zip(spans, truth, strict=True)):
            lo, hi = real_manifest.clip_window((a, b), [p for j, p in enumerate(spans) if j != i], len(x) / SR)
            assert hi >= e - 0.05  # 끝(지연 기준)은 덮는다 — 첫 구절이 통째로 묻히면 앞은 못 덮을 수 있다(README §5-4)
            if i in (0, 1):
                assert lo <= s


@pytest.mark.parametrize(("pause", "useed", "kind"), [(1.2, 2, "white"), (1.2, 2, "pink"), (1.3, 1, "white")])
def test_short_phrases_split_off_by_a_long_pause_at_10db_stay_with_their_sentence(pause, useed, kind) -> None:
    """10dB 에서 구 사이 1.2~1.3초 쉼이면 묻힌 약한 음절 때문에 0.4초 안 되는 구절 덩어리가 생긴다 — 버리지 않고 잰 틈이
    1.5초 안이면 잇고, 0.4초는 이은 덩어리 전체로 센다(u08 처럼 모든 구절이 짧으면 문장이 통째로 사라졌다)"""
    utts = _utts(useed)
    for k in (0, 1, 7):
        utts[k] = _sentence(100 * useed + k, CHARS[k], pause_s=pause)
    x, truth = _session(kind, 10.0, utts, seed=useed)
    spans = real_manifest.find_sentences(x)
    assert len(spans) == 14
    for (a, b), (s, e) in zip(spans, truth, strict=True):
        assert max(a, s) < min(b, e)


def test_clip_window_keeps_away_from_neighbours() -> None:
    assert real_manifest.clip_window((10.0, 12.0), [(5.0, 7.0), (16.0, 18.0)], 30.0) == (8.5, 14.0)  # 앞 1.5초·뒤 2초
    assert real_manifest.clip_window((10.0, 12.0), [(7.0, 9.5), (13.5, 16.0)], 30.0) == pytest.approx((9.8, 13.2))
    assert real_manifest.clip_window((10.0, 12.0), [(8.0, 9.9), (12.1, 13.0)], 30.0) == (10.0, 12.0)  # 문장은 안 자른다
    assert real_manifest.clip_window((0.4, 2.0), [], 3.0) == (0.0, 3.0)


def test_short_pause_shrinks_margins_and_warns_about_the_tail(tmp_path) -> None:
    utts = _utts(1)
    pauses = [3.5] * 13
    pauses[2] = 1.55  # u03 → u04 쉼이 짧다(말소리 사이 1.7초 안팎)
    x, truth = _session("pink", 20.0, utts, seed=3, pauses=pauses)
    _write(tmp_path / "rec" / "cafe" / "all.wav", x)
    manifest, _ = real_manifest.build(tmp_path / "rec", SENT, tmp_path / "out", clean_label="quiet")
    clips = manifest["clips"]
    assert [c["utt"] for c in clips] == IDS
    for i, c in enumerate(clips):
        lo, hi = _window(c)
        if i:
            assert lo >= truth[i - 1][1] + real_manifest.NEIGHBOR_GAP_S - 0.05
        if i < len(clips) - 1:
            assert hi <= truth[i + 1][0] - real_manifest.NEIGHBOR_GAP_S + 0.05
    tail = [w for w in clips[2]["warnings"] if w.startswith("뒤 여유")]
    want = truth[3][0] - real_manifest.NEIGHBOR_GAP_S - truth[2][1]
    assert want < real_manifest.TAIL_MIN_S
    assert tail and float(tail[0].split()[2].removesuffix("초<1.5")) == pytest.approx(want, abs=0.1)
    assert all(not any(w.startswith(("앞 여유", "뒤 여유")) for w in c["warnings"]) for c in clips if c["utt"] != "u03")


@pytest.mark.parametrize("breath_db", [-56.0, -50.0])
@pytest.mark.parametrize(("lo", "hi"), [(1.7, 2.0), (2.0, 2.5)])
def test_breaths_in_quiet_do_not_fill_short_pauses(lo, hi, breath_db) -> None:
    """조용한 곳에선 문장 0.3초 앞 들숨(말소리보다 24~30dB 작음)도 소리라 쉼이 0.7초 줄어든다 — 나누기 핵은 피크 25dB
    안, 잇기는 강한 시작으로 재 무음 쉼 1.7초에도 14개. 숨 없는 신호에선 원래 그렇다"""
    for useed in range(3):
        pauses = list(np.random.default_rng(useed).uniform(lo, hi, 13))
        x, truth = _session("quiet", 40.0, _utts(useed), seed=useed, pauses=pauses, breath_db=breath_db)
        spans = real_manifest.find_sentences(x)
        assert len(spans) == 14
        for (a, b), (s, e) in zip(spans, truth, strict=True):
            assert -0.8 < a - s < 0.06 and abs(b - e) < 0.06  # 시작은 숨까지 갈 수 있다(클립에 숨이 든다)


def test_breaths_before_sentences_do_not_drop_noisy_clips_on_the_quiet_length_check(tmp_path) -> None:
    """quiet 에서만 들리는 들숨이 구간 앞에 붙어도 길이 대조는 강한 시작(말소리 −15dB)부터 잰다"""
    rec = tmp_path / "rec"
    utts = _utts(2)
    for i, (env, kind, snr, _) in enumerate(ENVS):
        x, _ = _session(kind, snr, utts, seed=20 + i, breath_db=-52.0)
        _write(rec / env / "all.wav", x)
    manifest, rows = real_manifest.build(rec, SENT, tmp_path / "o", clean_label="quiet")
    assert all(s["found"] == 14 for s in manifest["splits"])
    quiet = [r for r in rows if r["env"] == "quiet"]
    assert sum(r["strong_start_s"] - r["speech_start_s"] > 0.5 for r in quiet) >= 10  # 숨이 quiet 구간에 붙는다
    assert len(manifest["clips"]) == 42 and all(abs(c["span_vs_quiet_s"]) < 0.5 for c in manifest["clips"]
                                                if c["env"] != "quiet")
    assert all(c["strong_start_s"] >= c["speech_start_s"] for c in manifest["clips"])


def _misread_kit(tmp_path: Path, pause_s: float | None = None,
                 wrong: np.ndarray | None = None) -> tuple[Path, list[tuple[float, float]]]:
    """fan: u03 을 읽다 틀려(앞 60%, 또는 wrong) pause_s 쉬고 다시 읽었다(없으면 3~4초 → 덩어리 15개). quiet 는 정상"""
    rec = tmp_path / "rec"
    utts = _utts(0)
    x, _ = _session("quiet", 40.0, utts, seed=1)
    _write(rec / "quiet" / "all.wav", x)
    wrong = _sentence(77, round(0.6 * CHARS[2])) if wrong is None else wrong
    pauses = list(np.random.default_rng(8).uniform(3.0, 4.0, 14))
    if pause_s is not None:
        pauses[2] = pause_s
    x, truth = _session("white", 20.0, [*utts[:2], wrong, *utts[2:]], seed=2, pauses=pauses)
    _write(rec / "fan" / "all.wav", x)
    return rec, truth


def test_count_mismatch_prints_the_segments_and_drops_the_file(tmp_path, monkeypatch, capsys) -> None:
    rec, truth = _misread_kit(tmp_path)
    manifest, _ = real_manifest.build(rec, SENT, tmp_path / "o", clean_label="quiet")
    fan = next(s for s in manifest["splits"] if s["env"] == "fan")
    assert (fan["found"], fan["expected"], fan["included"]) == (15, 14, False)
    assert fan["warnings"] == ["문장 15개를 찾음, 표는 14개 — --cuts 로 지정하거나 다시 녹음(뺌)"]
    for (a, b), (s, e) in zip(fan["segments"], truth, strict=True):
        assert abs(a - s) < 0.1 and abs(b - e) < 0.15
    assert {c["env"] for c in manifest["clips"]} == {"quiet"}
    text = _main(monkeypatch, capsys, "--rec-dir", str(rec), "--out", str(tmp_path / "o2"))
    assert "✗ fan/all.wav" in text and "문장 15개를 찾음, 표는 14개 — --cuts 로 지정하거나 다시 녹음(뺌)" in text
    seg = text.split("✗ fan/all.wav", 1)[1].splitlines()[1:17]
    assert seg[0].split() == ["번호", "시작", "끝", "길이"]
    assert [ln.split()[0] for ln in seg[1:]] == [str(i) for i in range(1, 16)]
    a, b, n = (float(v) for v in seg[3].split()[1:])  # 3번 = 틀린 u03
    assert (a, b, n) == pytest.approx((truth[2][0], truth[2][1], truth[2][1] - truth[2][0]), abs=0.15)


def test_cuts_override_the_split_and_make_only_the_given_ids(tmp_path, monkeypatch, capsys) -> None:
    """틀린 시도 뒤 1초만 쉬고 다시 읽으면 한 덩어리라 개수는 맞지만 u03 클립에 둘 다 든다 → cuts 로 고친다"""
    rec, truth = _misread_kit(tmp_path, 1.0)
    m_auto, auto = real_manifest.build(rec, SENT, tmp_path / "a", clean_label="quiet")
    merged = next(r for r in auto if r["file"] == "fan/u03.wav")
    assert _abs(merged)[0] == pytest.approx(truth[2][0], abs=0.06) and not merged["included"]
    assert merged["warnings"] == ["글자 수에 비해 긴 구간 — 다시 읽음?", "말소리 덩어리 2개 — 구간 확인",
                                  f"구간 길이가 quiet 보다 {merged['span_vs_quiet_s']:+.1f}초 — 잡음에 끌림?(뺌)"]
    split = next(s for s in m_auto["splits"] if s["env"] == "fan")
    x = split["long"]["u03"]
    assert split["warnings"] == [f"u03 자리 구간이 글자 수에 비해 길다(중앙값의 {x:.2f}배, 다시 읽음?)"]
    assert all("순서" not in w for r in auto for w in r["warnings"])  # 다시 읽음은 순서 어긋남이 아니다
    keep = [t for i, t in enumerate(truth) if i != 2]  # 틀린 첫 시도는 뺀다
    cuts = tmp_path / "cuts.tsv"
    cuts.write_text("# 파일\t문장\t시작\t끝\n\n" + "".join(
        f"fan/all.wav\t{sid}\t{s:.2f}\t{e:.2f}\n" for sid, (s, e) in zip(IDS, keep, strict=True) if sid != "u09"
    ) + "fan/all.wav\tx99\t1\t2\nfan/all.wav\tu09\t1\t999\ntv/all.wav\tu01\t1\t2\n", encoding="utf-8")
    loaded = real_manifest.load_cuts(cuts)
    assert loaded["fan/all.wav"]["u03"] == (round(keep[2][0], 2), round(keep[2][1], 2))
    spans = {"fan/u02.wav": (1.0, 2.0)}
    manifest, rows = real_manifest.build(rec, SENT, tmp_path / "o", clean_label="quiet", cuts=loaded, spans=spans)
    fan = [c for c in manifest["clips"] if c["env"] == "fan"]
    assert [c["utt"] for c in fan] == [u for u in IDS if u != "u09"]
    for c in fan:
        s, e = loaded["fan/all.wav"][c["utt"]]
        assert _abs(c) == pytest.approx((s, e), abs=2e-3)
        assert (c["split"], c["manual_span"], c["span_vs_quiet_s"]) == ("cuts", True, None)
        ignored = ["spans 는 문장별 파일만 — 연속 녹음은 --cuts(무시)"] if c["utt"] == "u02" else []
        assert c["warnings"] == [*ignored, "구간 수동(cuts)"]
    u03 = next(c for c in fan if c["utt"] == "u03")
    assert u03["source_start_s"] >= truth[2][1] + real_manifest.NEIGHBOR_GAP_S - 0.05  # 틀린 시도는 클립 밖
    lead = round(keep[2][0], 2) - truth[2][1] - real_manifest.NEIGHBOR_GAP_S  # 1초 쉼이라 앞 여유가 줄어든다
    assert lead < real_manifest.CLIP_LEAD_S and u03["speech_start_s"] == pytest.approx(lead, abs=0.05)
    split = next(s for s in manifest["splits"] if s["env"] == "fan")
    assert split["cuts"] and split["included"] and split["warnings"] == [
        "cuts 의 x99 는 문장표에 없음(뺌)", f"cuts u09 끝 999초가 녹음({split['dur_s']:.1f}초) 밖(뺌)"]
    assert manifest["cuts_unused"] == ["tv/all.wav"] and manifest["spans_unused"] == []
    text = _main(monkeypatch, capsys, "--rec-dir", str(rec), "--out", str(tmp_path / "o2"), "--cuts", str(cuts))
    assert "cuts 에 있지만 녹음에 없는 연속 녹음: tv/all.wav" in text and "fan/u03.wav" in text


def test_cuts_with_no_usable_entry_drop_the_file_with_a_mark(tmp_path, monkeypatch, capsys) -> None:
    """cuts 가 전부 무효면 클립이 하나도 안 나오므로 ✗ 로 뺀다(경고만 늘어놓지 않는다)"""
    rec, _ = _misread_kit(tmp_path)
    cuts = tmp_path / "cuts.tsv"
    cuts.write_text("fan/all.wav\tx99\t1\t2\nfan/all.wav\tu01\t1\t999\n", encoding="utf-8")
    loaded = real_manifest.load_cuts(cuts)
    manifest, _ = real_manifest.build(rec, SENT, tmp_path / "o", clean_label="quiet", cuts=loaded)
    fan = next(s for s in manifest["splits"] if s["env"] == "fan")
    assert fan["included"] is False
    assert fan["warnings"] == [
        "cuts 의 x99 는 문장표에 없음(뺌)",
        f"cuts u01 끝 999초가 녹음({fan['dur_s']:.1f}초) 밖(뺌)",
        "cuts 에 쓸 수 있는 항목이 없음 — 클립 없음(뺌)",
    ]
    assert {c["env"] for c in manifest["clips"]} == {"quiet"}
    text = _main(monkeypatch, capsys, "--rec-dir", str(rec), "--out", str(tmp_path / "o2"), "--cuts", str(cuts))
    assert "✗ fan/all.wav" in text and "cuts 에 쓸 수 있는 항목이 없음" in text


@pytest.mark.parametrize("pause", [0.3, 0.5, 0.6])
def test_cuts_keep_a_quick_reread_out_of_the_clip(tmp_path, pause) -> None:
    """틀린 시도 뒤 0.8초 안에 바로 다시 읽어도 cuts 클립 앞 여유에 틀린 시도가 안 든다(구절·핵 run 단위로 피한다)"""
    rec, truth = _misread_kit(tmp_path, pause)
    s, e = truth[3]
    manifest, _ = real_manifest.build(rec, SENT, tmp_path / "o", clean_label="quiet",
                                      cuts={"fan/all.wav": {"u03": (round(s, 2), round(e, 2))}})
    (u03,) = [c for c in manifest["clips"] if c["env"] == "fan"]
    lo, hi = _window(u03)
    assert truth[2][1] <= lo < s and hi >= e


def test_cuts_keep_a_one_syllable_false_start_out_of_the_clip(tmp_path) -> None:
    """'다,' 하고 0.5초 뒤 처음부터 — 0.4초 안 되는 덩어리는 구절로 안 남으니 핵 run 으로 피한다"""
    gap = np.zeros(int(0.05 * SR))
    da = np.concatenate([gap, _syllable(np.random.default_rng(3), 0.22, 130.0), gap])
    rec, truth = _misread_kit(tmp_path, 0.5, wrong=da)
    s, e = truth[3]
    manifest, _ = real_manifest.build(rec, SENT, tmp_path / "o", clean_label="quiet",
                                      cuts={"fan/all.wav": {"u03": (round(s, 2), round(e, 2))}})
    (u03,) = [c for c in manifest["clips"] if c["env"] == "fan"]
    assert truth[2][1] <= u03["source_start_s"] < s


def test_load_cuts_rejects_bad_rows(tmp_path) -> None:
    path = tmp_path / "c.tsv"
    for bad, match in (("fan/all.wav\tu01\t1.0\n", "TAB"), ("fan/all.wav u01 1 2\n", "TAB"),
                       ("fan/all.wav\tu01\t2\t1\n", "시작 < 끝"),
                       ("fan/all.wav\tu01\t1\t2\nfan/all.wav\tu01\t3\t4\n", "중복")):
        path.write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            real_manifest.load_cuts(path)


def test_sentence_file_takes_precedence_over_the_continuous_one(tmp_path, monkeypatch, capsys) -> None:
    rec = tmp_path / "rec"
    utts = _utts(0)
    x, _ = _session("quiet", 40.0, utts, seed=3)
    _write(rec / "quiet" / "all.wav", x)
    x, truth = _session("pink", 20.0, utts, seed=4)
    _write(rec / "cafe" / "all.wav", x)
    again = np.concatenate([np.zeros(SR), mix_noise.to_level(utts[4], LEVEL), np.zeros(2 * SR)])
    _write(rec / "cafe" / "u05.wav", again + np.random.default_rng(5).standard_normal(len(again)) * 10 ** (-46 / 20))
    manifest, _ = real_manifest.build(rec, SENT, tmp_path / "o", clean_label="quiet")
    by = {c["file"]: c for c in manifest["clips"]}
    assert len(by) == 28
    assert (by["cafe/u05.wav"]["source"], by["cafe/u05.wav"]["split"], by["cafe/u05.wav"]["source_start_s"]) == (
        "cafe/u05.wav", None, None)
    assert all(by[f"cafe/{u}.wav"]["source"] == "cafe/all.wav" and by[f"cafe/{u}.wav"]["split"] == "auto"
               for u in IDS if u != "u05")
    assert all(by[f"quiet/{u}.wav"]["source"] == "quiet/all.wav" and by[f"quiet/{u}.wav"]["split"] == "auto"
               for u in IDS)  # 다른 환경의 u05 파일은 이 환경과 상관없다
    assert {s["env"]: s["single"] for s in manifest["splits"]} == {"cafe": ["u05"], "quiet": []}
    assert abs(by["cafe/u05.wav"]["span_vs_quiet_s"]) < 0.3  # 문장별 파일도 연속 녹음 quiet 와 대조
    text = _main(monkeypatch, capsys, "--rec-dir", str(rec), "--out", str(tmp_path / "o2"))
    assert "문장별 파일 우선: u05" in text
    line = {ln.split()[0]: ln for ln in text.splitlines() if ln.startswith("cafe/u0")}
    assert line["cafe/u05.wav"].split()[1] == "u05.wav" and line["cafe/u04.wav"].split()[1].startswith("all.wav@")


def test_clipping_is_counted_inside_each_clip(tmp_path) -> None:
    x, truth = _session("pink", 20.0, _utts(0), seed=4)
    at = int((truth[6][0] + 0.5) * SR)
    x[at : at + 5] = 0.995  # u07 말소리 안에서만 찌그러짐
    _write(tmp_path / "rec" / "cafe" / "all.wav", x)
    manifest, _ = real_manifest.build(tmp_path / "rec", SENT, tmp_path / "o", clean_label="quiet")
    clipped = {c["utt"]: c["clipped"] for c in manifest["clips"]}
    assert clipped["u07"] == 5 and sum(clipped.values()) == 5
    assert [c["utt"] for c in manifest["clips"] if any(w.startswith("클리핑") for w in c["warnings"])] == ["u07"]


def test_skipped_sentence_plus_split_sentence_warns_about_order(tmp_path, monkeypatch, capsys) -> None:
    """u05 를 빼먹고 u11 을 두 번에 나눠 읽으면 개수는 14 로 같다 — u05~u10 자리에 한 칸씩 밀린 문장이 붙는다"""
    utts = _utts(0)
    k = round(CHARS[10] / 3)
    said = [*utts[:4], *utts[5:10], _sentence(60, k), _sentence(61, CHARS[10] - k), *utts[11:]]
    x, _ = _session("pink", 20.0, said, seed=5)
    rec = tmp_path / "rec"
    _write(rec / "cafe" / "all.wav", x)
    manifest, _ = real_manifest.build(rec, SENT, tmp_path / "o", clean_label="quiet")
    split = manifest["splits"][0]
    assert split["found"] == 14 and split["included"] and split["order_r"] < real_manifest.ORDER_MIN_R
    assert split["warnings"][0] == f"순서가 어긋났을 수 있음 — 구간 길이·글자 수 상관 r={split['order_r']:.2f}<0.5"
    assert split["warnings"][1].startswith("순서가 어긋났을 수 있음 — u10 자리 구간이 글자 수에 비해 짧다")
    assert len(manifest["clips"]) == 14 and all("연속 녹음 순서 의심" in c["warnings"] for c in manifest["clips"])
    text = _main(monkeypatch, capsys, "--rec-dir", str(rec), "--out", str(tmp_path / "o2"))
    assert "순서가 어긋났을 수 있음" in text and "번호" in text


def test_order_check_criteria() -> None:
    lengths = [0.2 * n for n in CHARS]
    assert real_manifest.order_check(lengths, IDS, SENT) == (1.0, [], {})
    rev = [0.2 * n for n in sorted(CHARS, reverse=True)]
    r, warns, long = real_manifest.order_check(rev, IDS, SENT)
    assert r < 0 and warns == [f"순서가 어긋났을 수 있음 — 구간 길이·글자 수 상관 r={r:.2f}<0.5"] and long == {}
    half = list(lengths)
    half[3] *= 0.45
    r, warns, long = real_manifest.order_check(half, IDS, SENT)
    assert r > real_manifest.ORDER_MIN_R and long == {}
    assert warns == ["순서가 어긋났을 수 있음 — u04 자리 구간이 글자 수에 비해 짧다(중앙값의 0.45배, 쪼개짐?)"]
    for f, want in ((1.55, {}), (1.65, {"u06": 1.65})):
        again = list(lengths)
        again[5] *= f
        assert real_manifest.order_check(again, IDS, SENT)[1:] == ([], want)  # 다시 읽음은 순서 경고가 아니다
    assert real_manifest.order_check([1.0, 2.0], IDS[:2], SENT) == (None, [], {})


def _reread_quiet(tmp_path: Path, k: int, frac: float, pause_s: float) -> tuple[dict, list[dict]]:
    """quiet 에서 u_k 를 frac 만큼 읽다 pause_s 쉬고 처음부터 다시 — cafe(분홍 20)·fan(백색 20)은 정상"""
    rec = tmp_path / "rec"
    utts = _utts(0)
    wrong = _sentence(77, max(5, round(frac * CHARS[k])))
    pauses = list(np.random.default_rng(8).uniform(3.0, 4.0, 13))
    pauses.insert(k, pause_s)
    x, _ = _session("quiet", 40.0, [*utts[:k], wrong, *utts[k:]], seed=1, pauses=pauses)
    _write(rec / "quiet" / "all.wav", x)
    for env, kind, seed in (("cafe", "pink", 2), ("fan", "white", 3)):
        x, _ = _session(kind, 20.0, utts, seed=seed)
        _write(rec / env / "all.wav", x)
    return real_manifest.build(rec, SENT, tmp_path / "o", clean_label="quiet")


def test_a_quiet_reread_in_two_parts_is_not_the_length_reference(tmp_path) -> None:
    """1초 쉬고 다시 읽으면 덩어리 2개 — 글자 수 기준(1.6배)엔 안 걸려도 quiet 기준에서 뺀다"""
    manifest, rows = _reread_quiet(tmp_path, 0, 0.3, 1.0)
    q = next(r for r in rows if r["file"] == "quiet/u01.wav")
    assert q["parts"] == 2 and "long_span" not in q and q["warnings"][-1] == "quiet 길이 기준에서 뺌"
    others = [r for r in rows if r["utt"] == "u01" and r["env"] != "quiet"]
    assert len(others) == 2 and all(r["included"] and r.get("span_vs_quiet_s") is None for r in others)


def test_noisy_takes_that_disagree_with_each_other_are_still_blamed_on_noise(tmp_path) -> None:
    """두 환경 다 quiet 보다 길어도 서로 길이가 다르면(대사에 다르게 끌림) quiet 탓이 아니다"""
    rec = tmp_path / "rec"
    utts = _utts(0)
    x, _ = _session("quiet", 40.0, utts, seed=1)
    _write(rec / "quiet" / "all.wav", x)
    pauses = [3.5] * 13
    pauses[4] = 5.0  # 대사가 다음 문장과 이어지지 않게
    for env, seed, n in (("tv1", 2, 3), ("tv2", 3, 9)):
        x, truth = _session("pink", 20.0, utts, seed=seed, pauses=pauses)
        talk = mix_noise.to_level(_speech(31, groups=(n,)), -32.0)  # 말소리 끝 0.4초 뒤 대사(길이 다름)
        at = int((truth[4][1] + 0.4) * SR)
        x[at : at + len(talk)] += talk
        _write(rec / env / "all.wav", x)
    _, rows = real_manifest.build(rec, SENT, tmp_path / "o", clean_label="quiet")
    tv = [r for r in rows if r["utt"] == "u05" and r["env"] != "quiet"]
    assert len(tv) == 2 and abs(tv[0]["span_vs_quiet_s"] - tv[1]["span_vs_quiet_s"]) > 0.8
    for r in tv:
        assert not r["included"] and r["warnings"][-1].endswith("잡음에 끌림?(뺌)") and r["span_vs_quiet_s"] > 0.5
    q = next(r for r in rows if r["file"] == "quiet/u05.wav")
    assert not any("다시 읽음" in w for w in q["warnings"])


@pytest.mark.parametrize(("k", "pause"), [(2, 0.5), (8, 0.5), (12, 1.0)])
def test_a_quiet_reread_is_not_the_length_reference(tmp_path, k, pause) -> None:
    """quiet 에서 그 자리에서 다시 읽은 구간은 글자 수에 비해 길다 — 경고하고 quiet 기준에서 빼 다른 환경을 살린다"""
    manifest, rows = _reread_quiet(tmp_path, k, 0.6, pause)
    sid = IDS[k]
    q = next(r for r in rows if r["file"] == f"quiet/{sid}.wav")
    assert q["included"] and q["long_span"] and "글자 수에 비해 긴 구간 — 다시 읽음?" in q["warnings"]
    assert q["warnings"][-1] == "quiet 길이 기준에서 뺌" and sid not in manifest["span_ref_s"]
    assert len(manifest["clips"]) == 42 and all(c["span_vs_quiet_s"] is None for c in manifest["clips"]
                                                if c["utt"] == sid)
    split = next(s for s in manifest["splits"] if s["env"] == "quiet")
    assert set(split["long"]) == {sid} and not any("순서" in w for w in split["warnings"])


def test_a_short_quiet_reread_is_blamed_on_quiet_when_other_environments_agree(tmp_path) -> None:
    """짧게 틀리고 바로 다시 읽으면 글자 수 기준(1.6배)엔 안 걸린다 — 다른 환경끼리 맞으면 사유를 quiet 쪽으로"""
    manifest, rows = _reread_quiet(tmp_path, 5, 0.3, 0.4)
    q = next(r for r in rows if r["file"] == "quiet/u06.wav")
    assert "long_span" not in q and q["parts"] == 1
    others = [r for r in rows if r["utt"] == "u06" and r["env"] != "quiet"]
    assert len(others) == 2 and not any(r["included"] for r in others)
    for r in others:
        why = "다른 환경끼리는 맞음, quiet 쪽 확인(뺌)"
        assert r["warnings"][-1] == f"구간 길이가 quiet 보다 {r['span_vs_quiet_s']:+.1f}초 — {why}"
    mean = -float(np.mean([r["span_vs_quiet_s"] for r in others]))
    assert q["warnings"][-1] == f"다른 환경 2곳보다 {mean:+.1f}초 — 다시 읽음?" and mean > 1.0
    assert len(manifest["clips"]) == 40


@pytest.mark.parametrize("where", ["pause", "after"])
def test_a_cough_or_phone_beep_is_not_counted_as_a_sentence(tmp_path, where) -> None:
    utts = _utts(1)
    x, truth = _session("white", 20.0, utts, seed=6)
    at = (truth[5][1] + truth[6][0]) / 2 - 0.15 if where == "pause" else truth[3][1] + 1.0  # 쉼 가운데 · 문장 1초 뒤
    x = x + _burst(len(x), at, 0.3, LEVEL + 6)  # 헛기침: 말소리보다 6dB 큰 0.3초
    t = np.arange(int(0.25 * SR)) / SR
    b0 = int((truth[9][1] + 1.2) * SR)
    x[b0 : b0 + len(t)] += 0.3 * np.sin(2 * np.pi * 1000 * t)  # 폰 알림음 0.25초
    _write(tmp_path / "rec" / "cafe" / "all.wav", x)
    manifest, _ = real_manifest.build(tmp_path / "rec", SENT, tmp_path / "o", clean_label="quiet")
    assert manifest["splits"][0]["found"] == 14 and manifest["splits"][0]["warnings"] == []
    for c, (s, e) in zip(manifest["clips"], truth, strict=True):
        a, b = _abs(c)
        assert -0.04 <= a - s <= 0.06 and -0.08 <= b - e <= 0.04


def test_take_names_follow_the_continuous_file(tmp_path) -> None:
    rec = tmp_path / "rec"
    utts = _utts(0)
    for name, seed in (("all", 1), ("all_2", 2)):
        x, _ = _session("pink", 20.0, utts, seed=seed)
        _write(rec / "cafe" / f"{name}.wav", x)
    manifest, _ = real_manifest.build(rec, SENT, tmp_path / "o", clean_label="quiet")
    files = [c["file"] for c in manifest["clips"]]
    assert sorted(files) == sorted([f"cafe/{u}.wav" for u in IDS] + [f"cafe/{u}_2.wav" for u in IDS])
    take2 = [c for c in manifest["clips"] if c["source"] == "cafe/all_2.wav"]
    assert len(take2) == 14 and {c["take"] for c in take2} == {2}
    assert [c["file"] for c in take2] == [f"cafe/{u}_2.wav" for u in IDS]
    assert [s["take"] for s in manifest["splits"]] == [1, 2]
    assert all((tmp_path / "o" / f).exists() for f in files)


def test_manifest_from_continuous_recordings_feeds_run_chirp(tmp_path) -> None:
    rec = tmp_path / "rec"
    utts = _utts(0)
    for env, kind, snr, seed in (("quiet", "quiet", 40.0, 1), ("cafe", "pink", 20.0, 2)):
        x, _ = _session(kind, snr, utts, seed=seed)
        _write(rec / env / "all.wav", x)
    _write(rec / "cafe" / "noise.wav", mix_noise.pink(np.random.default_rng(3), 3 * SR) * 0.01)
    out = tmp_path / "audio"
    real_manifest.build(rec, SENT, out, clean_label="quiet")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    jobs = run_chirp.plan_jobs(manifest, reps=1, modes=[False, True])
    assert len(jobs) == 2 * len(manifest["clips"]) == 2 * 29
    assert run_chirp.warmup_clip(manifest)["file"] == "quiet/u01.wav"
    assert [j.clip["utt"] for j in jobs if j.clip["noise"] == "cafe" and not j.denoise] == [None, *IDS]
    assert all(set(run_chirp.CLIP_KEYS) <= set(c) for c in manifest["clips"] if not c["noise_only"])
    src = real_manifest.to_mono16k(*real_manifest.decode(rec / "cafe" / "all.wav"))
    for c in manifest["clips"]:
        y = run_chirp.read_pcm(out / c["file"])
        assert len(y) / mix_noise.BYTES_PER_S == pytest.approx(c["dur_s"], abs=1e-3)
        if not c["noise_only"]:
            assert (c["lead_s"], c["utt_end_s"]) == (c["speech_start_s"], c["speech_end_s"])
        if c["source"] == "cafe/all.wav":  # 클립 = 원본의 source_start_s 부터
            a = round(c["source_start_s"] * SR)
            pcm = np.frombuffer(y, dtype="<i2").astype(np.float64) / 32768.0
            assert np.max(np.abs(pcm - src[a : a + len(pcm)])) < 2e-4


def test_print_script_lists_the_sentences_without_a_recording_folder(monkeypatch, capsys) -> None:
    text = _main(monkeypatch, capsys, "--print-script")
    lines = [ln for ln in text.splitlines() if re.match(r"^\s*\d+\. ", ln)]
    assert len(lines) == 14 and lines[0].strip() == f"1. {SENT['u01']}   [u01]" and lines[-1].endswith("[u14]")
    assert "3~4초" in text and "1초 안" in text
    monkeypatch.setattr(sys, "argv", ["real_manifest.py"])
    with pytest.raises(SystemExit):
        real_manifest.main()
    assert "--print-script" in capsys.readouterr().err
