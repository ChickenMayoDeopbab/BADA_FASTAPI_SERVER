"""폰으로 녹음한 실발화 폴더를 run_chirp·score 가 그대로 읽는 manifest 로 바꾼다 (네트워크 없음).

  .venv/bin/python scripts/stt_noise/real_manifest.py --rec-dir ~/Desktop/bada-rec
  (기본: 문장표 scripts/stt_noise/real_sentences.tsv → .harness/stt-noise-2026-10/real/audio)

녹음 폴더: <환경 라벨>/<문장 id>.<확장자> 또는 <문장 id>_<테이크>.<확장자>. noise[_<테이크>] 는 잡음만 클립.
wav 는 soundfile, 그 밖(m4a 등)은 afconvert(없으면 ffmpeg)로 PCM 을 꺼낸 뒤
스테레오는 평균, 16kHz 로 리샘플 → 16k mono PCM16.
말소리 구간: 20ms 프레임 전력을 100ms 로 고르고, 잡음 바닥(하위 10%) 위 4~12dB(바닥~피크 폭의 절반) 문턱으로
핵을 찾는다. 0.15초 미만 튐은 버리고 0.8초 안 틈은 잇는다. 가장 긴 덩어리가 문장이다. 떨어진 덩어리는
0.4초 이상·에너지가 문장의 20% 이상이고 문장보다 6dB 넘게 크지 않을 때만 붙인다(크기만 넘으면 구간 모호로 뺀다).
0.8초 안의 짧은 run 은 흡수하고, 가장자리는 고른 레벨이 잡음 중앙값 + max(3dB, 잡음 출렁임) 이상인 동안
넓힌다(0.3초 이하 틈은 잇는다).
추정 SNR = (말소리 프레임 평균 전력 − 잡음 평균 전력) ÷ 잡음 평균 전력. 말소리 프레임 = 구간 안 고른 레벨이 잡음
+3dB 이상, 잡음 = 구간 앞뒤 0.1초 밖(다른 소리·튐 뺌). 잡음에 묻힌 약한 음절은 빠져 실제보다 조금 높게 나온다.
--clean-label 테이크가 있으면 같은 문장의 말소리 길이와 견줘 0.5초(또는 15%) 넘게 다른 클립은 뺀다(--keep-mismatch).
--spans TSV(파일<TAB>시작초<TAB>끝초)로 들어 보고 고친 구간을 덮어쓴다.
조건 묶음은 환경 라벨: noise = 라벨(--clean-label 은 clean), snr_db = 그 환경 추정 SNR 중앙값(정수, clean 은 null).
클립마다의 추정은 snr_est_db. 클립은 variant=null 이라 run_chirp --reps 1 이면 파일마다 한 번씩 간다.
"""

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.stt_noise.mix_noise import FRAME, OUT_DIR, SR, frame_power, stt_usd, write_pcm16  # noqa: E402

HERE = Path(__file__).resolve().parent
FRAME_S = FRAME / SR
SMOOTH_FRAMES = 5  # 100ms
FLOOR_PCT, PEAK_PCT = 10, 98
RISE_DB = (4.0, 12.0)
MIN_RANGE_DB = 6.0
BLIP_S, MERGE_GAP_S = 0.15, 0.8
MIN_BODY_FRAMES = 3  # 구간 가까이의 짧은 run 도 원래 프레임 60ms 는 넘어야 음절(아니면 탁 소리)
MIN_PHRASE_S, KEEP_ENERGY, LOUD_DB = 0.4, 0.2, 6.0  # 떨어진 덩어리를 붙일 조건: 길이·문장 대비 에너지, 이만큼 크면 아님
EDGE_MIN_DB, EDGE_SPREAD, EDGE_PAD_DB, EDGE_GAP_S = 3.0, 1.3, 2.0, 0.3  # 출렁임 = 1.3 × (중앙값 − 하위 10%) + 2dB
SPEECH_OVER_DB = 3.0  # 추정 SNR 의 말소리 프레임 = 고른 레벨이 잡음 평균 +3dB 이상
EDGE_ZONE_S = (0.5, 1.0)  # 녹음 처음·끝 이 안의 소리는 따로 경고
LOW_SNR_DB = 9.0  # 추정치가 이 아래(실제 8dB 안팎 아래)면 끝 음절을 놓쳐 구간이 짧아진다
SPAN_DIFF_S, SPAN_DIFF_FRAC = 0.5, 0.15  # quiet 테이크와 말소리 길이 차 허용(큰 쪽)
AMBIGUOUS = "구간 모호(뺌)"
GUARD_S, MIN_NOISE_S = 0.1, 0.2
LEAD_MIN_S, TAIL_MIN_S = 0.5, 1.5
CLIP_LEVEL, CLIP_MIN_SAMPLES = 0.99, 3
NOISE_ID = "noise"
AUDIO_EXTS = {".wav", ".m4a", ".mp3", ".aac", ".caf", ".aif", ".aiff", ".flac", ".ogg", ".opus", ".3gp", ".amr",
              ".mp4", ".webm"}
TAKE = re.compile(r"^(?P<id>.+?)_(?P<take>\d+)$")


def load_sentences(path: Path) -> dict[str, str]:
    """id<TAB>정답, 빈 줄·# 줄은 건너뛴다"""
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        key, sep, text = line.partition("\t")
        if not sep or not text.strip():
            raise ValueError(f"id<TAB>문장 이 아니다: {line!r}")
        if key.strip() in rows:
            raise ValueError(f"문장 id 중복: {key.strip()}")
        rows[key.strip()] = text.strip()
    return rows


def parse_name(stem: str) -> tuple[str, int]:
    """'u01' → (u01, 1), 'u01_2' → (u01, 2)"""
    m = TAKE.match(stem)
    return (m["id"], int(m["take"])) if m else (stem, 1)


def converters() -> list[str]:
    return [tool for tool in ("afconvert", "ffmpeg") if shutil.which(tool)]


def _convert_cmd(tool: str, src: Path, dst: Path) -> list[str]:
    """채널·샘플레이트는 그대로 두고 PCM16 wav 로만 푼다(섞기·리샘플은 여기서 똑같이)"""
    if tool == "afconvert":
        return ["afconvert", "-f", "WAVE", "-d", "LEI16", str(src), str(dst)]
    return ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-acodec", "pcm_s16le",
            "-f", "wav", str(dst)]


def decode(path: Path) -> tuple[np.ndarray, int]:
    """(채널 × 샘플 float, 샘플레이트). wav 가 아니면 afconvert → ffmpeg 순으로 푼다"""
    if path.suffix.lower() == ".wav":
        x, sr = sf.read(str(path), dtype="float64", always_2d=True)
        return x.T, sr
    tools = converters()
    if not tools:
        raise SystemExit(
            f"{path.name}: wav 가 아닌 파일은 맥 내장 afconvert 나 ffmpeg 로 푼다 — 둘 다 없다. "
            "맥에서 돌리거나 `brew install ffmpeg`, 아니면 녹음 앱에서 wav 로 내보낼 것"
        )
    errors = []
    with tempfile.TemporaryDirectory() as tmp:
        dst = Path(tmp) / "decoded.wav"
        for tool in tools:
            r = subprocess.run(_convert_cmd(tool, path, dst), capture_output=True, text=True, check=False)
            if r.returncode == 0 and dst.exists():
                x, sr = sf.read(str(dst), dtype="float64", always_2d=True)
                return x.T, sr
            errors.append(f"{tool}: {(r.stderr or r.stdout).strip()[:200]}")
    raise ValueError(f"{path.name} 을 못 풀었다 — " + " / ".join(errors))


def to_mono16k(x: np.ndarray, sr: int) -> np.ndarray:
    mono = np.asarray(x, dtype=np.float64).mean(axis=0)
    if sr == SR:
        return mono
    g = math.gcd(SR, sr)
    return resample_poly(mono, SR // g, sr // g)


def clipped_samples(x: np.ndarray) -> int:
    return int(np.count_nonzero(np.abs(x) >= CLIP_LEVEL))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """True 가 이어진 [시작, 끝) 프레임 목록"""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist(), strict=True))


def _db(p: np.ndarray | float) -> np.ndarray | float:
    return 10 * np.log10(np.asarray(p) + 1e-10)


def _clusters(runs: list[tuple[int, int]], gap: int) -> list[list[int]]:
    out: list[list[int]] = []
    for a, b in runs:
        if out and a - out[-1][1] <= gap:
            out[-1][1] = b
        else:
            out.append([a, b])
    return out


def _extend(db: np.ndarray, a: int, b: int, edge: float, gap: int) -> tuple[int, int]:
    """[a, b) 를 고른 레벨이 edge 이상인 동안 바깥으로 넓힌다(gap 프레임 이하 틈은 잇는다)"""
    last = b - 1
    for j in range(b, len(db)):
        if db[j] >= edge:
            last = j
        elif j - last > gap:
            break
    first = a
    for j in range(a - 1, -1, -1):
        if db[j] >= edge:
            first = j
        elif first - j > gap:
            break
    return first, last + 1


def _frames(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """(프레임 전력, 고른 dB, 원래 dB, 잡음 바닥, 피크)"""
    p = frame_power(x)
    win = np.ones(SMOOTH_FRAMES)
    smooth = np.convolve(p, win, "same") / np.convolve(np.ones_like(p), win, "same")
    db = _db(smooth)
    return p, db, _db(p), float(np.percentile(db, FLOOR_PCT)), float(np.percentile(db, PEAK_PCT))


def _threshold(floor: float, peak: float) -> float:
    return floor + min(max(0.5 * (peak - floor), RISE_DB[0]), RISE_DB[1])


def _mask(n: int, runs: list[tuple[int, int]]) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    for s, e in runs:
        m[s:e] = True
    return m


def _levels(p: np.ndarray, db: np.ndarray, in_runs: np.ndarray, a: int, b: int, thr: float) -> dict:
    """잡음 = 구간 앞뒤 0.1초 밖(다른 소리·튐 뺌) 평균, 말소리 = 구간 안 고른 레벨이 잡음 +3dB 이상인 프레임"""
    guard = round(GUARD_S / FRAME_S)
    outside = ~in_runs
    outside[max(0, a - guard) : b + guard] = False
    noise = float(np.mean(p[outside])) if outside.sum() * FRAME_S >= MIN_NOISE_S else None
    voiced = db[a:b] >= (thr if noise is None else _db(noise) + SPEECH_OVER_DB)
    mix = float(np.mean(p[a:b][voiced])) if voiced.any() else float(np.mean(p[a:b]))
    speech = mix - noise if noise is not None else mix
    snr = float(_db(speech) - _db(noise)) if noise is not None and speech > 0 else None
    return {
        "noise_db": None if noise is None else round(float(_db(noise)), 1),
        "speech_level_db": round(float(_db(speech if speech > 0 else mix)), 1),
        "snr_est_db": None if snr is None else round(snr, 1),
    }


def detect(x: np.ndarray) -> dict | None:
    """말소리 구간·잡음 바닥·추정 SNR. 말소리를 못 찾으면 None"""
    if len(x) // FRAME < 4 * SMOOTH_FRAMES:
        return None
    p, db, raw, floor, peak = _frames(x)
    if peak - floor < MIN_RANGE_DB:
        return None
    thr = _threshold(floor, peak)
    runs = _runs(db >= thr)
    long = [r for r in runs if (r[1] - r[0]) * FRAME_S >= BLIP_S]
    if not long:
        return None
    gap = round(MERGE_GAP_S / FRAME_S)
    clusters = _clusters(long, gap)
    body = [int(np.count_nonzero(raw[a:b] >= thr)) for a, b in clusters]  # 원래 프레임 기준 길이
    energy = [float(np.sum(p[a:b][raw[a:b] >= thr])) for a, b in clusters]
    main = int(np.argmax(body))  # 가장 긴 덩어리 = 문장
    big = {i for i in range(len(clusters))
           if body[i] >= round(MIN_PHRASE_S / FRAME_S) and energy[i] >= KEEP_ENERGY * energy[main]}
    loud_ratio = 10 ** (LOUD_DB / 10) * energy[main] / max(body[main], 1)
    kept = [i for i in range(len(clusters)) if i == main or (i in big and energy[i] / body[i] <= loud_ratio)]
    a, b = clusters[kept[0]][0], clusters[kept[-1]][1]
    near = [r for r in runs if r not in long and np.count_nonzero(raw[r[0] : r[1]] >= thr) >= MIN_BODY_FRAMES]
    grew = True
    while grew:  # 구간 가까이의 짧은 run(약한 음절)은 흡수
        grew = False
        for s, e in near:
            if (s < a and a - e <= gap) or (e > b and s - b <= gap):
                a, b, grew = min(a, s), max(b, e), True
    in_runs = _mask(len(p), runs)
    cand = ~in_runs
    cand[a:b] = False
    edge = thr
    if cand.sum() * FRAME_S >= MIN_NOISE_S:  # 가장자리 문턱 = 잡음 중앙값 + 잡음 출렁임
        lo, mid = np.percentile(db[cand], [FLOOR_PCT, 50])
        edge = min(thr, float(mid + max(EDGE_MIN_DB, EDGE_SPREAD * (mid - lo) + EDGE_PAD_DB)))
        clicks = _mask(len(p), [(s, e) for s, e in runs if (e <= a or s >= b) and (s, e) not in long
                                and np.count_nonzero(raw[s:e] >= edge) < MIN_BODY_FRAMES])
        a, b = _extend(np.where(clicks, -np.inf, db), a, b, edge, round(EDGE_GAP_S / FRAME_S))  # 탁 소리는 못 밟는다
    loud = np.flatnonzero(raw[a:b] >= edge)
    if loud.size:  # 고르기로 번진 가장자리를 원래 프레임으로 다시 자른다
        a, b = a + int(loud[0]), a + int(loud[-1]) + 1
    away = [i for i, c in enumerate(clusters) if c[1] <= a or c[0] >= b]
    dur = len(p) * FRAME_S
    return {
        "speech_start_s": round(a * FRAME_S, 3),
        "speech_end_s": round(b * FRAME_S, 3),
        "noise_floor_db": round(floor, 1),
        **_levels(p, db, in_runs, a, b, thr),
        "threshold_db": round(thr, 1),
        "edge_db": round(edge, 1),
        "other_sounds": len(away),
        "unsure_sounds": sum(1 for i in away if i in big),  # 문장만큼 큰데 문장보다 6dB 넘게 크다
        "edge_sounds": sum(1 for s, e in long if s * FRAME_S < EDGE_ZONE_S[0] or e * FRAME_S > dur - EDGE_ZONE_S[1]),
        "parts": len(kept),
    }


def measure_span(x: np.ndarray, start_s: float, end_s: float) -> dict:
    """손으로 정한 구간으로 잡음·추정 SNR 을 잰다"""
    p, db, _, floor, peak = _frames(x)
    thr = _threshold(floor, peak)
    a, b = round(start_s / FRAME_S), round(end_s / FRAME_S)
    return {
        "speech_start_s": round(start_s, 3), "speech_end_s": round(end_s, 3), "noise_floor_db": round(floor, 1),
        **_levels(p, db, _mask(len(p), _runs(db >= thr)), a, b, thr), "threshold_db": round(thr, 1),
        "edge_db": None, "other_sounds": 0, "unsure_sounds": 0, "edge_sounds": 0, "parts": 1,
    }


def margin_warnings(info: dict, dur_s: float) -> list[str]:
    out = []
    lead, tail = info["speech_start_s"], dur_s - info["speech_end_s"]
    if lead < LEAD_MIN_S:
        out.append(f"앞 여유 {lead:.1f}초<{LEAD_MIN_S:g}")
    if tail < TAIL_MIN_S:
        out.append(f"뒤 여유 {tail:.1f}초<{TAIL_MIN_S:g}")
    if info["edge_sounds"]:
        out.append(f"녹음 처음 {EDGE_ZONE_S[0]:g}초·끝 {EDGE_ZONE_S[1]:g}초 안 소리 — 대사 같은 잡음이면 구간이 끌린다")
    if info["parts"] > 1:
        out.append(f"말소리 덩어리 {info['parts']}개 — 구간 확인")
    if info["other_sounds"]:
        out.append(f"말소리 밖 소리 {info['other_sounds']}곳")
    if info["unsure_sounds"]:
        out.append(f"그중 말소리보다 큰 긴 소리 {info['unsure_sounds']}곳 — {AMBIGUOUS}")
    if info["snr_est_db"] is not None and info["snr_est_db"] < LOW_SNR_DB:
        out.append(f"추정 SNR {info['snr_est_db']:g}dB<{LOW_SNR_DB:g} — 구간 추정 불확실")
    return out


def load_spans(path: Path) -> dict[str, tuple[float, float]]:
    """파일<TAB>시작초<TAB>끝초 (파일 = 표의 파일 열 또는 원본 경로), 빈 줄·# 줄은 건너뛴다"""
    rows: dict[str, tuple[float, float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        try:
            key, a, b = parts[0].strip(), float(parts[1]), float(parts[2])
        except (IndexError, ValueError):
            raise ValueError(f"파일<TAB>시작초<TAB>끝초 가 아니다: {line!r}") from None
        if not 0 <= a < b:
            raise ValueError(f"시작 < 끝 이 아니다: {line!r}")
        rows[key] = (a, b)
    return rows


def scan(rec_dir: Path) -> tuple[list[tuple[str, Path]], list[str]]:
    """(환경 라벨, 파일) 목록과 건너뛴 파일"""
    found, skipped = [], []
    for path in sorted(rec_dir.rglob("*")):
        if path.is_dir() or path.name.startswith("."):
            continue
        rel = path.relative_to(rec_dir)
        if len(rel.parts) != 2:
            skipped.append(f"{rel} (환경 폴더 바로 아래가 아님)")
        elif path.suffix.lower() not in AUDIO_EXTS:
            skipped.append(f"{rel} (오디오 확장자 아님)")
        else:
            found.append((rel.parts[0], path))
    return found, skipped


def check_against_quiet(rows: list[dict], clean_label: str | None, *, keep: bool) -> dict[str, float]:
    """같은 문장 quiet 테이크의 말소리 길이와 견줘 잡음에 끌린 구간을 뺀다(keep 이면 경고만). 기준 길이를 돌려준다"""
    lens: dict[str, list[float]] = {}
    for r in rows:
        if r["env"] == clean_label and r["included"] and not r["noise_only"]:
            lens.setdefault(r["utt"], []).append(r["speech_end_s"] - r["speech_start_s"])
    ref = {sid: float(np.median(v)) for sid, v in lens.items()}
    for r in rows:
        if not r["included"] or r["noise_only"] or r["env"] == clean_label or r["manual"] or r["utt"] not in ref:
            continue
        diff = r["speech_end_s"] - r["speech_start_s"] - ref[r["utt"]]
        r["span_vs_quiet_s"] = round(diff, 2)
        if abs(diff) > max(SPAN_DIFF_S, SPAN_DIFF_FRAC * ref[r["utt"]]):
            tag = "" if keep else "(뺌)"
            r["warnings"].append(f"구간 길이가 {clean_label} 보다 {diff:+.1f}초 — 잡음에 끌림?{tag}")
            r["included"] = keep
    return ref


def build(
    rec_dir: Path, sentences: dict[str, str], out: Path, *, clean_label: str | None,
    spans: dict[str, tuple[float, float]] | None = None, keep_mismatch: bool = False,
) -> tuple[dict, list[dict]]:
    """(manifest, 표 행). manifest 에 못 넣는 클립도 표에는 남긴다(included=False)"""
    found, skipped = scan(rec_dir)
    spans = spans or {}
    used: set[str] = set()
    rows: list[dict] = []
    seen: set[str] = set()
    for env, path in found:
        sid, take = parse_name(path.stem)
        rel = f"{env}/{path.stem}.wav"
        row = {"file": rel, "source": str(path.relative_to(rec_dir)), "env": env, "utt": sid, "take": take,
               "noise_only": sid == NOISE_ID, "warnings": [], "included": False, "manual": False}
        rows.append(row)
        if rel in seen:
            row["warnings"].append("같은 이름 파일 중복(뺌)")
            continue
        seen.add(rel)
        if not row["noise_only"] and sid not in sentences:
            row["warnings"].append("문장표에 없는 id(뺌)")
            continue
        x, sr = decode(path)
        n_clip = clipped_samples(x)
        y = to_mono16k(x, sr)
        dur = len(y) / SR
        row.update(src_sr=sr, src_channels=int(x.shape[0]), dur_s=round(dur, 3), clipped=n_clip)
        if n_clip >= CLIP_MIN_SAMPLES:
            row["warnings"].append(f"클리핑 {n_clip}샘플")
        if not row["noise_only"]:
            key = rel if rel in spans else row["source"]
            fixed = spans.get(key)
            used.add(key)
            if fixed and fixed[1] > dur + FRAME_S:
                row["warnings"].append(f"spans 끝 {fixed[1]:g}초가 녹음({dur:.1f}초) 밖(뺌)")
                continue
            info = measure_span(y, *fixed) if fixed else detect(y)
            if info is None:
                row["warnings"].append("말소리 못 찾음(뺌)")
                continue
            row.update(info, manual=bool(fixed))
            row["warnings"] += margin_warnings(info, dur) + (["구간 수동(spans)"] if fixed else [])
            if info["snr_est_db"] is None:
                row["warnings"].append("SNR 못 잼")
            if info["unsure_sounds"]:
                write_pcm16(out / rel, y)  # 들어 보고 spans 로 고칠 수 있게 wav 는 남긴다
                continue
        else:
            p = frame_power(y)
            row["noise_db"] = round(float(_db(np.mean(p))), 1) if len(p) else None
        write_pcm16(out / rel, y)
        row["included"] = True
    ref = check_against_quiet(rows, clean_label, keep=keep_mismatch)

    env_snr: dict[str, int | None] = {}
    for env in sorted({r["env"] for r in rows}):
        if env in (clean_label, "clean"):
            env_snr[env] = None
            continue
        snrs = [r["snr_est_db"] for r in rows if r["env"] == env and r["included"] and r.get("snr_est_db") is not None]
        env_snr[env] = round(float(np.median(snrs))) if snrs else None
        if not snrs:
            for r in rows:
                if r["env"] == env and r["included"]:
                    r["included"] = False
                    r["warnings"].append("이 환경 SNR 을 못 잼(뺌) — 문장 녹음이 하나는 있어야 한다")

    clips = []
    for r in rows:
        if not r["included"]:
            continue
        clean = r["env"] in (clean_label, "clean")
        speech = {} if r["noise_only"] else {
            "lead_s": r["speech_start_s"], "speech_start_s": r["speech_start_s"],
            "speech_end_s": r["speech_end_s"], "utt_end_s": r["speech_end_s"],
        }
        clips.append({
            "file": r["file"], "utt": None if r["noise_only"] else r["utt"],
            "text": None if r["noise_only"] else sentences[r["utt"]],
            "noise": "clean" if clean else r["env"], "snr_db": env_snr[r["env"]], "variant": None,
            "noise_only": r["noise_only"], **speech, "dur_s": r["dur_s"],
            "env": r["env"], "take": r["take"], "source": r["source"], "snr_est_db": r.get("snr_est_db"),
            "noise_floor_db": r.get("noise_floor_db"), "noise_db": r.get("noise_db"),
            "speech_level_db": r.get("speech_level_db"), "clipped": r["clipped"], "manual_span": r["manual"],
            "span_vs_quiet_s": r.get("span_vs_quiet_s"), "warnings": r["warnings"],
        })
    manifest = {
        "kind": "real",
        "sr": SR,
        "src": str(rec_dir),
        "clean_label": clean_label,
        "variants": 1,
        "env_snr_db": env_snr,
        "snr_definition": "snr_db = 환경 라벨의 추정 SNR 중앙값(조건 묶음용, clean 은 null) · "
                          "snr_est_db = 클립 추정 (말소리 프레임 평균 전력 − 잡음 평균 전력) / 잡음 평균 전력, "
                          "잡음 = 말소리 구간 앞뒤 0.1초 밖",
        "span_ref_s": {k: round(v, 2) for k, v in sorted(ref.items())},
        "keep_mismatch": keep_mismatch,
        "spans_unused": sorted(set(spans) - used),
        "skipped": skipped,
        "clips": clips,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return manifest, rows


def _f(v: float | None, fmt: str = ".1f") -> str:
    return "—" if v is None else format(v, fmt)


def print_report(manifest: dict, rows: list[dict], sentences: dict[str, str]) -> None:
    print(f"{'파일':<22} {'문장':<26} {'길이':>5} {'말소리 구간':>13} {'바닥':>6} {'잡음':>6} {'SNR':>5}  경고")
    for r in rows:
        mark = "" if r["included"] else "✗ "
        text = "(잡음만)" if r["noise_only"] else f"{r['utt']} {sentences.get(r['utt'], '?')}"
        span = "—" if "speech_start_s" not in r else f"{r['speech_start_s']:.2f}~{r['speech_end_s']:.2f}"
        print(
            f"{mark + r['file']:<22} {text[:24]:<26} {_f(r.get('dur_s')):>5} {span:>13} "
            f"{_f(r.get('noise_floor_db')):>6} {_f(r.get('noise_db')):>6} {_f(r.get('snr_est_db')):>5}  "
            f"{'; '.join(r['warnings'])}"
        )
    for s in manifest["skipped"]:
        print(f"건너뜀: {s}")
    for s in manifest["spans_unused"]:
        print(f"spans 에 있지만 녹음에 없는 파일: {s}")
    ref = manifest["span_ref_s"]
    if ref:
        act = "경고만" if manifest["keep_mismatch"] else "뺌"
        print(f"구간 길이 대조: {manifest['clean_label']} 테이크가 있는 {len(ref)}문장 — 다른 환경에서 "
              f"{SPAN_DIFF_S:g}초({SPAN_DIFF_FRAC:.0%}) 넘게 다르면 {act}")
    else:
        print("구간 길이 대조 안 함 — --clean-label 환경 녹음이 없다")
    print("✗ = manifest 에서 뺌 · 바닥 = 프레임 전력 하위 10%(dBFS) · 잡음 = 말소리 밖 평균(dBFS) · SNR = 추정(dB)")
    print(f"\n{'환경':<12} {'noise':<10} {'snr_db':>6} {'문장':>4} {'잡음만':>5} {'초':>7}")
    clips = manifest["clips"]
    for env, snr in manifest["env_snr_db"].items():
        mine = [c for c in clips if c["env"] == env]
        if not mine:
            continue
        n_utt = sum(not c["noise_only"] for c in mine)
        print(f"{env:<12} {mine[0]['noise']:<10} {_f(snr, 'g'):>6} {n_utt:>4} {len(mine) - n_utt:>5} "
              f"{sum(c['dur_s'] for c in mine):>7.1f}")
    upper = sum(c["dur_s"] for c in clips) * 2
    print(f"\n클립 {len(clips)}개 · 끔·켬 2조건 한 번씩 끝까지 보내면 {upper / 60:.1f}분 ≈ ${stt_usd(upper):.2f}"
          " (턴 FINAL 에서 멈추니 실제는 더 적다, 정확히는 run_chirp --dry-run)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rec-dir", required=True, help="<환경 라벨>/<문장 id>[_<테이크>].<확장자> 녹음 폴더")
    ap.add_argument("--sentences", default=str(HERE / "real_sentences.tsv"), help="id<TAB>정답 문장")
    ap.add_argument("--out", default=str(OUT_DIR / "real" / "audio"), help="16k wav 와 manifest.json 을 쓸 폴더")
    ap.add_argument(
        "--clean-label", default="quiet", help="clean 으로 바꿀 환경 라벨(run_chirp 워밍업·score 첫 줄), ''=없음"
    )
    ap.add_argument("--spans", default=None, help="파일<TAB>시작초<TAB>끝초 — 들어 보고 고친 말소리 구간")
    ap.add_argument("--keep-mismatch", action="store_true", help="quiet 와 구간 길이가 다른 클립도 경고만 하고 넣는다")
    a = ap.parse_args()

    rec_dir = Path(a.rec_dir).expanduser()
    if not rec_dir.is_dir():
        raise SystemExit(f"녹음 폴더가 없다: {rec_dir}")
    sentences = load_sentences(Path(a.sentences).expanduser())
    out = Path(a.out).expanduser()
    spans = load_spans(Path(a.spans).expanduser()) if a.spans else None
    manifest, rows = build(rec_dir, sentences, out, clean_label=a.clean_label or None, spans=spans,
                           keep_mismatch=a.keep_mismatch)
    print_report(manifest, rows, sentences)
    if not manifest["clips"]:
        raise SystemExit("manifest 에 넣을 클립이 없다")
    print(f"→ {out}/manifest.json")


if __name__ == "__main__":
    main()
