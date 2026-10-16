"""고른 후보를 운영 워커 참조로 정규화하고 voices.json 조각을 만든다 (맥, 레포 venv).

  .venv/bin/python scripts/tts_bench/prep_ref.py --candidates <design 결과>/candidates.json \
      --pick female_young_soft__c2 --pick male_old_rough=male_old__<세션>_2 ... \
      --dest-dir /home/<사용자>/bada-qwen3-tts/refs \
      --tag v1 --level-ref ~/school/narsha/2026/qwen-tests/ref_ai.wav --out <design 결과>/refs_v1
후보마다 24kHz mono PCM16 · 말소리 앞뒤를 잘라 무음 0.5초로 맞춤(길이 맞추기 패딩 아님) ·
말소리 레벨을 --level-ref 와 같게.
결과: out/<칸>_<tag>.wav · voices.part.json(키 = 칸, ref_audio = --dest-dir 절대경로) · prep.json(지표·출처) + 표 ·
set/(g1_score 로 참조 발화가 ref_text 와 같은지 재는 세트 — `--cond ref=<out>/set:ref=prompt`).
같은 후보·태그로 --dest-dir 만 바꿔 다시 돌리면 오디오는 같고 경로만 바뀐다(게이트용 GPU 박스 경로 → 학교 경로).
워커는 ref_audio 를 그대로 열고 cwd 가 fork/ 라 상대경로·~ 는 깨진다 → --dest-dir 은 절대경로만 받는다.
"""

import argparse
import json
import math
import os
import pathlib
import re
import shutil
import wave

import numpy as np
from scipy.signal import resample_poly

SR = 24000
FRAME_S, SOUND_DBFS, ONSET_FRAMES = 0.02, -45.0, 3  # post.py·design_voices.py 와 같은 말소리 판정
PAD_S, MARGIN_S, FADE_S = 0.5, 0.05, 0.01
DEFAULT_LEVEL_DB = -21.0  # 운영 ai·user 참조 말소리 레벨(실측 -21.3/-21.1)
PEAK_MAX_DB = -1.0
REF_RANGE_S = (10.0, 15.0)
FLOOR_MAX_DB = -60.0  # 잡음 바닥이 이보다 크면 복제 출력에 잡음이 따라온다
DIGITAL_SILENCE_DB = -100.0
SLOT = re.compile(r"(female|male)_(young|middle|old)_(soft|rough)")
PAUSE_FRAMES = 5  # 100ms


def read_wav(path: pathlib.Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as f:
        if f.getsampwidth() != 2:
            raise ValueError(f"PCM16 wav 만 읽는다: {path}")
        x = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        ch = f.getnchannels()
        return (x.reshape(-1, ch).mean(axis=1) if ch > 1 else x), f.getframerate()


def write_wav(path: pathlib.Path, w: np.ndarray) -> None:
    tmp = path.with_suffix(".part")
    with wave.open(str(tmp), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((np.clip(w, -1.0, 1.0) * 32767).astype("<i2").tobytes())
    tmp.replace(path)


def to_sr(w: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return w.astype(np.float32)
    g = math.gcd(SR, sr)
    return resample_poly(w, SR // g, sr // g).astype(np.float32)


def frames_db(w: np.ndarray) -> np.ndarray:
    n = int(SR * FRAME_S)
    k = len(w) // n
    x = w[: k * n].reshape(k, n).astype(np.float64)
    return 10 * np.log10(np.mean(x * x, axis=1) + 1e-20)


def speech_span(db: np.ndarray) -> tuple[int, int] | None:
    """(첫 말소리 프레임, 마지막 소리 프레임+1) — 60ms 이어져야 시작으로 본다"""
    loud = db >= SOUND_DBFS
    onset = next((i for i in range(len(loud) - ONSET_FRAMES + 1) if loud[i : i + ONSET_FRAMES].all()), None)
    if onset is None:
        return None
    return onset, len(loud) - int(np.argmax(loud[::-1]))


def speech_level_db(w: np.ndarray) -> float:
    db = frames_db(w)
    loud = db[db >= SOUND_DBFS]
    if not len(loud):
        raise ValueError("말소리가 없다")
    return float(10 * np.log10(np.mean(10 ** (loud / 10))))


def noise_floor_db(w: np.ndarray) -> float | None:
    """100ms 이상 이어진 쉼(말소리 문턱 아래) 프레임의 중앙값, 쉼이 없으면(잡음이 문턱보다 큼) 하위 10%"""
    db = frames_db(w)
    db = db[db > DIGITAL_SILENCE_DB]
    if not len(db):
        return None
    pause, run = [], []
    for v in np.append(db, 0.0):
        if v < SOUND_DBFS:
            run.append(v)
            continue
        if len(run) >= PAUSE_FRAMES:
            pause += run
        run = []
    return round(float(np.median(pause) if pause else np.percentile(db, 10)), 1)


def prep(w: np.ndarray, sr: int, target_db: float) -> tuple[np.ndarray, dict]:
    w = to_sr(np.asarray(w, dtype=np.float32), sr)
    span = speech_span(frames_db(w))
    if span is None:
        raise ValueError("말소리가 없다")
    hop = int(SR * FRAME_S)
    margin = int(SR * MARGIN_S)
    seg = w[max(0, span[0] * hop - margin) : min(len(w), span[1] * hop + margin)].copy()
    fade = min(int(SR * FADE_S), len(seg) // 2)
    if fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        seg[:fade] *= ramp
        seg[-fade:] *= ramp[::-1]
    gain = target_db - speech_level_db(seg)
    peak = 20 * math.log10(float(np.max(np.abs(seg))) + 1e-12)
    limited = peak + gain > PEAK_MAX_DB
    if limited:
        gain = PEAK_MAX_DB - peak
    seg *= np.float32(10 ** (gain / 20))
    pad = np.zeros(int(SR * (PAD_S - MARGIN_S)), dtype=np.float32)
    out = np.concatenate([pad, seg, pad])
    dur = round(len(out) / SR, 2)
    floor = noise_floor_db(w)
    floor = None if floor is None else round(floor + gain, 1)  # 워커가 받는 참조 기준(이득 반영)
    warnings = []
    if not REF_RANGE_S[0] <= dur <= REF_RANGE_S[1]:
        warnings.append(f"길이 {dur}s 가 {REF_RANGE_S[0]:g}~{REF_RANGE_S[1]:g}s 밖")
    if floor is not None and floor > FLOOR_MAX_DB:
        warnings.append(f"잡음 바닥 {floor}dBFS > {FLOOR_MAX_DB:g}")
    if limited:
        warnings.append(f"피크 {PEAK_MAX_DB:g}dBFS 로 제한 — 말소리 레벨이 목표보다 낮다")
    info = dict(dur_s=dur, level_db=round(speech_level_db(out), 1), gain_db=round(gain, 1),
                floor_db=floor, warnings=warnings)
    return out, info


def parse_picks(picks: list[str], cands: dict) -> dict[str, str]:
    """후보 이름 → 칸 (칸마다 하나). '<칸>=<후보>' 로 칸을 직접 정하거나, 후보 이름 앞부분이 칸이면 이름만"""
    by_slot: dict[str, str] = {}
    for spec in picks:
        for item in (x.strip() for x in spec.split(",") if x.strip()):
            slot, sep, name = item.partition("=")
            if not sep:
                name, slot = item, item.split("__", 1)[0]
            if not SLOT.fullmatch(slot):
                raise ValueError(f"칸이 {{gender}}_{{age}}_{{tone}} 형식이 아니다: {item} — '<칸>=<후보>' 로 지정")
            if name not in cands:
                raise ValueError(f"candidates.json 에 없는 후보: {name}")
            if slot in by_slot:
                raise ValueError(f"한 칸에 후보 둘: {by_slot[slot]}, {name}")
            by_slot[slot] = name
    if not by_slot:
        raise ValueError("--pick 이 필요하다")
    return by_slot


def source_wav(cands_path: pathlib.Path, ref_audio: str) -> pathlib.Path:
    """GPU 박스 절대경로 대신 candidates.json 옆의 같은 이름 파일(폴더째 복사한 경우)을 먼저 쓴다"""
    local = cands_path.parent / pathlib.Path(ref_audio).name
    return local if local.exists() else pathlib.Path(os.path.expanduser(ref_audio))


def check_dest(dest_dir: str) -> str:
    if not dest_dir.startswith("/") or "~" in dest_dir:
        raise ValueError(f"--dest-dir 은 워커 쪽 절대경로여야 한다(상대경로·~ 불가): {dest_dir}")
    return dest_dir.rstrip("/")


def write_check_set(out: pathlib.Path, rows: dict[str, tuple[str, str]]) -> None:
    """g1_score 세트: 참조 자체를 프롬프트이자 조건으로 — ref=prompt 로 채점하면 참조 발화 CER"""
    sdir = out / "set"
    sdir.mkdir(parents=True, exist_ok=True)
    lines = []
    for slot, (fname, text) in sorted(rows.items()):
        if any(c in text for c in "|\n\r"):
            raise ValueError(f"meta.lst 칸을 깨는 전사: {slot}")
        shutil.copyfile(out / fname, sdir / f"{slot}.wav")
        lines.append(f"{slot}|{text}|{slot}.wav|{text}")
    (sdir / "meta.lst").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cands_path: pathlib.Path, picks: list[str], dest_dir: str, tag: str, out: pathlib.Path,
        target_db: float) -> dict:
    dest = check_dest(dest_dir)
    if not tag or "/" in tag:
        raise ValueError(f"--tag 이 이상하다: {tag!r}")
    cands = json.loads(cands_path.read_text(encoding="utf-8"))
    chosen = parse_picks(picks, cands)
    origin = {}  # design_voices 의 design.jsonl(seed·instruct) 또는 pick_aihub_voices 의 picks.jsonl(세션·화자)
    for log_name in ("design.jsonl", "picks.jsonl"):
        log_path = cands_path.parent / log_name
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    origin[row["name"]] = row
    out.mkdir(parents=True, exist_ok=True)
    part, report, check = {}, {}, {}
    for slot, name in sorted(chosen.items()):
        w, sr = read_wav(source_wav(cands_path, cands[name]["ref_audio"]))
        ref, info = prep(w, sr, target_db)
        fname = f"{slot}_{tag}.wav"
        write_wav(out / fname, ref)
        part[slot] = {"ref_audio": f"{dest}/{fname}", "ref_text": cands[name]["ref_text"]}
        check[slot] = (fname, cands[name]["ref_text"])
        src = origin.get(name, {})
        report[slot] = dict(source=name, seed=src.get("seed"), instruct=src.get("instruct"), session=src.get("session"),
                            spk_id=src.get("spk_id"), file=fname, **info)
    write_check_set(out, check)
    (out / "voices.part.json").write_text(json.dumps(part, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "prep.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="고른 후보를 운영 참조로 정규화")
    ap.add_argument("--candidates", required=True, help="design_voices.py 의 candidates.json")
    ap.add_argument("--pick", action="append", default=[],
                    help="<칸>=<후보> 또는 칸으로 시작하는 후보 이름(<칸>__c<i>), 여러 번 또는 쉼표")
    ap.add_argument("--dest-dir", required=True, help="워커에서 참조를 둘 절대경로")
    ap.add_argument("--tag", default="v1", help="파일 이름 꼬리 — 참조를 바꿀 땐 올린다(같은 이름 덮어쓰기 금지)")
    ap.add_argument("--level-ref", help="말소리 레벨을 맞출 기준 wav(운영 ref_ai.wav), 없으면 -21dBFS")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    target = DEFAULT_LEVEL_DB
    if a.level_ref:
        w, sr = read_wav(pathlib.Path(os.path.expanduser(a.level_ref)))
        target = speech_level_db(to_sr(w, sr))
    out = pathlib.Path(os.path.expanduser(a.out))
    report = run(pathlib.Path(os.path.expanduser(a.candidates)), a.pick, a.dest_dir, a.tag, out, target)
    print(f"목표 말소리 레벨 {target:.1f}dBFS")
    print(f"{'칸':<20} {'후보':<24} {'길이':>6} {'레벨':>6} {'이득':>6} {'바닥':>7}  경고")
    for slot, r in report.items():
        floor = "-" if r["floor_db"] is None else f"{r['floor_db']:.1f}"
        print(f"{slot:<20} {r['source']:<24} {r['dur_s']:>5.1f}s {r['level_db']:>6.1f} {r['gain_db']:>+6.1f} "
              f"{floor:>7}  {'; '.join(r['warnings'])}")
    print(f"끝 → {out}/voices.part.json (워커 voices.json 에 합칠 것, ai·user 는 그대로 둔다)")
    print(f"전사 대조: g1_score.py --set {out}/set --cond ref={out}/set:ref=prompt --out {out}/score")


if __name__ == "__main__":
    main()
