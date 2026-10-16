"""AIHub 71631(감정이 태깅된 자유대화, 성인)에서 화자 칸별 실제 사람 참조 후보를 뽑는다 (집 PC, numpy 만).

  <포크 venv>/bin/python pick_aihub_voices.py --label <라벨 zip 들> --wav <원천 zip 들> \
      --out /mnt/c/qwen-out/aihub_voices [--per-cell 10] [--speakers 0743,1803 --windows-per-speaker 3]
1단계(라벨, wav 가 있는 세션만): 화자마다 연속 발화 구간(말소리 9~14초, 발화 사이 1.5초 이하, 전사에 마스킹·주석 기호
  없음)을 찾는다. 라벨은 긴 말을 잘게 나눠 같은 화자 발화가 붙어 있으므로, 구간 앞뒤에 같은 화자의 쉼이 0.3초 이상
  있는 곳만 경계로 삼고 상대 발화는 앞뒤 0.5초 안에 없어야 한다. 문장 중간에서 끝난 참조는 클론 끝을 자르므로
  마지막 발화는 문장 끝 부호(. ? !)로 끝나고, 앞 같은 화자 발화가 문장 끝이거나 1초 이상 쉰 뒤에 시작해야 한다.
  성별 × 나이(young 20~30대 · middle 40~50대 · old 60대 이상) 칸, 마스크 착용·목 이상 상태 화자는 뺀다.
2단계(필요한 wav 만): 화자 채널을 에너지로 고르고 구간 ±0.15초로 바짝 잘라 앞뒤를 무음으로 0.5초 채운다(전사 밖 말이
  섞이지 않게). 말소리 − 잡음 바닥(둘 다 조용한 쉼에서), 채널 섞임, 내 채널에 들릴 만큼 샌 상대 소리를 검사한다.
  화자마다 통과한 구간 중 깨끗한 순으로 겹치지 않는 --windows-per-speaker 개, 칸마다 깨끗한 순으로 --per-cell 명.
  --speakers 를 주면 그 화자 ID 만(칸 상한 없이) — 귀로 고른 화자의 다른 구간을 다시 뽑을 때.
결과: out/<칸>__<세션>_<화자>_t<시작 초>.wav(16kHz mono) · candidates.json(gen_qwen·prep_ref 형식) ·
  picks.jsonl · listen.html(칸별). 톤(soft/rough)은 귀로 정해 prep_ref --pick <칸>=<후보>.
"""

import argparse
import collections
import html
import io
import json
import os
import pathlib
import re
import wave
import zipfile

import numpy as np

SR = 16000
CHANNEL = {"Speaker1": 0, "Speaker2": 1}
ODD = re.compile(r"[^가-힣0-9a-zA-Z\s.,?!~'\"·%-]")  # 마스킹(#@이름#)·주석 기호((), /, + 등)가 든 전사는 안 쓴다
ROUGH_HINT = ("화남", "분노", "짜증", "혐오", "angry", "anger", "disgust", "annoy")
CELLS = [f"{g}_{a}" for g in ("female", "male") for a in ("young", "middle", "old")]
FRAME_S, SOUND_DBFS, PAUSE_FRAMES = 0.02, -45.0, 5
PAD_S = 0.5  # 클립 앞뒤 무음(운영 참조와 같게)
CUT_MARGIN_S = 0.15  # 라벨 경계 밖으로 더 자르는 원본 — 나머지 패딩은 디지털 무음
EDGE_S = 0.3  # 구간 앞뒤에 같은 화자의 쉼이 이만큼은 있어야 경계로 쓴다
OTHER_GAP_S = 0.5  # 구간 앞뒤로 상대 발화가 없어야 하는 시간
INNER_GAP_MAX_S = 1.5  # 구간 안 같은 화자 발화 사이 최대 쉼
TURN_GAP_S = 1.0  # 앞 같은 화자 발화가 문장 끝이 아니어도 이만큼 쉬었으면 새로 시작한 말로 본다
SPEECH_RANGE_S = (9.0, 14.0)
TARGET_S = 12.0  # 앞뒤 0.5초를 붙여 13초 — 운영 참조 10~15초 가운데
MIN_SNR_DB = 30.0  # 말소리 − 잡음 바닥이 이보다 작으면 뺀다
PREP_SNR_DB = 39.0  # 이보다 작으면 prep_ref 가 잡음 바닥 경고를 낸다(-21dBFS 로 맞춘 뒤 -60 초과) — 페이지에 표시만
MIN_SEP_DB = 3.0  # 화자 채널 분리가 이보다 작으면 두 채널에 같은 소리(섞임)
LEAK_MARGIN_DB = 3.0  # 상대 채널이 이만큼 더 크고
LEAK_AUDIBLE_DB = 10.0  # 그때 내 채널이 잡음 바닥보다 이만큼 크고
LEAK_MIN_FRAMES = 3  # 60ms 이상 이어지면 상대 소리가 들리는 것(숨·딸깍 한두 프레임은 둔다)
WINDOWS_PER_SPEAKER = 4  # 화자당 2단계에서 잘라 볼 구간 수(최소)


def fl(s) -> float:
    """라벨 시간 문자열 → float('1,000.73' 처럼 천 단위 쉼표가 있다)"""
    return float(str(s).replace(",", "").strip())


def parse_gender(raw) -> str | None:
    s = str(raw or "").strip().lower()
    if s in ("남", "남자", "남성", "m", "male"):
        return "male"
    if s in ("여", "여자", "여성", "f", "female"):
        return "female"
    return None


def parse_age(raw) -> str | None:
    """나이 칸 — 숫자면 그 나이, '20대'·'60대 이상' 이면 그 십 년대 시작"""
    m = re.search(r"\d+", str(raw or ""))
    if not m:
        return None
    age = int(m.group())
    if not 15 <= age <= 100:  # 출생연도 등 다른 표기는 분포를 보고 정한다
        return None
    return "young" if age < 40 else "middle" if age < 60 else "old"


def parse_snr(raw) -> float | None:
    """'+31dB'·'0dB' → 숫자"""
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(raw or ""))
    return float(m.group()) if m else None


def usable_speaker(sp: dict) -> bool:
    mask = str(sp.get("Mask") or "").strip()
    cond = str(sp.get("Condition") or "").strip()
    return mask != "착용" and (not cond or cond == "정상")


def utterances(d: dict) -> list[tuple[str, float, float, str, str]]:
    us = []
    for u in d.get("Conversation", []):
        try:
            s, e = fl(u["StartTime"]), fl(u["EndTime"])
        except (KeyError, TypeError, ValueError):
            continue
        if u.get("SpeakerNo") not in CHANNEL or e <= s:
            continue
        emo = u.get("VerifyEmotionTarget") or u.get("SpeakerEmotionTarget") or ""
        us.append((u["SpeakerNo"], s, e, (u.get("Text") or "").strip(), str(emo)))
    us.sort(key=lambda x: (x[1], x[2]))
    return us


def clean_ok(text: str) -> bool:
    return bool(text) and not ODD.search(text)


FINAL_LAST = set("요다죠네어아야지래냐")  # 마지막 어절이 이 글자로 끝나면 종결(71631 화자 79% 는 문장부호를 안 쓴다)
RIEUL = 8  # 받침 ㄹ — -ㄹ게·-ㄹ까
BIEUP = 17  # 받침 ㅂ — -ㅂ니까·-습니까


def jong(ch: str) -> int:
    """한글 음절의 받침 번호(없으면 0)"""
    code = ord(ch) - 0xAC00
    return code % 28 if 0 <= code < 11172 else 0


def sentence_end(text: str) -> bool:
    """문장 끝 부호, 없으면 마지막 어절의 종결어미로 판정"""
    t = text.rstrip().rstrip("~")
    if t.endswith((".", "?", "!")):
        return True
    if not t:
        return False
    word = t.split()[-1]
    if len(word) < 2:
        return word == "네"  # 어·아·음·응 같은 한 글자 추임새는 끝이 아니다
    if word[-1] == "게":
        return jong(word[-2]) == RIEUL
    if word[-1] == "까":
        return jong(word[-2]) == RIEUL or (word.endswith("니까") and len(word) > 2 and jong(word[-3]) == BIEUP)
    return word[-1] in FINAL_LAST


def windows(us: list, spk: str) -> list[dict]:
    """그 화자의 연속 발화 구간 — 시작 발화마다 TARGET_S 에 가장 가까운 하나"""
    other = [(s, e) for sp, s, e, _, _ in us if sp != spk]
    mine = [u for u in us if u[0] == spk]
    lo, hi = SPEECH_RANGE_S
    out = []
    for i in range(len(mine)):
        if i and not sentence_end(mine[i - 1][3]) and mine[i][1] - mine[i - 1][2] < TURN_GAP_S:
            continue  # 문장 중간에서 시작
        best, end = None, mine[i][2]
        for j in range(i, len(mine)):
            if j > i and mine[j][1] - end > INNER_GAP_MAX_S:
                break
            start, end = mine[i][1], max(end, mine[j][2])
            if end - start > hi or not clean_ok(mine[j][3]):
                break
            if any(s < end + OTHER_GAP_S and e > start - OTHER_GAP_S for s, e in other):
                break
            if end - start < lo or not sentence_end(mine[j][3]):
                continue
            outside = mine[:i] + mine[j + 1 :]  # 경계 앞뒤에 같은 화자의 쉼이 있어야 전사 밖 말이 안 섞인다
            if any(s < end + EDGE_S and e > start - EDGE_S for _, s, e, _, _ in outside):
                continue
            if best is None or abs(end - start - TARGET_S) < abs(best[1] - best[0] - TARGET_S):
                best = (start, end, mine[i : j + 1])
        if best:
            start, end, seg = best
            out.append(dict(start=start, end=end, text=" ".join(u[3] for u in seg),
                            voiced_s=round(sum(u[2] - u[1] for u in seg), 2),
                            emotions=collections.Counter(u[4] for u in seg if u[4] and u[4] != "없음")))
    return out


def rough_hint(emotions: dict) -> bool:
    return any(h in e.lower() for e in emotions for h in ROUGH_HINT)


def wav_sessions(paths: list[str]) -> set[str]:
    """원천 zip 의 목록만 읽는다(내용은 안 읽음)"""
    names = set()
    for p in paths:
        with zipfile.ZipFile(p) as z:
            names |= {pathlib.Path(n).stem for n in z.namelist() if n.lower().endswith(".wav")}
    return names


def label_index(paths: list[str], keep: set[str] | None = None) -> dict[str, dict]:
    idx = {}
    for p in paths:
        with zipfile.ZipFile(p) as z:
            for info in z.infolist():
                stem = pathlib.Path(info.filename).stem
                if not info.filename.lower().endswith(".json") or (keep is not None and stem not in keep):
                    continue
                try:
                    idx[stem] = json.loads(z.read(info).decode("utf-8-sig"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
    return idx


def speaker_counts(labels: dict[str, dict]) -> dict[str, int]:
    """칸별 서로 다른 화자 ID 수"""
    ids = collections.defaultdict(set)
    for d in labels.values():
        for spk in CHANNEL:
            sp = d.get(spk) or {}
            g, a = parse_gender(sp.get("Gender")), parse_age(sp.get("Age"))
            if g and a:
                ids[f"{g}_{a}"].add(str(sp.get("ID")))
    return {c: len(ids[c]) for c in CELLS}


def shortlist(labels: dict[str, dict], per_cell: int, speakers: set[str] | None = None,
              per_speaker: int = 1) -> dict[str, list[dict]]:
    """칸 → 화자(구간 후보 여러 개)들, SNR 라벨 높은 순 per_cell×4 명(speakers 를 주면 그 화자만, 상한 없이)"""
    per_session, cap = 2 * per_speaker, max(WINDOWS_PER_SPEAKER, 3 * per_speaker)
    by_speaker: dict[tuple[str, str], dict] = {}
    for session, d in sorted(labels.items()):
        us = utterances(d)
        noise = d.get("Noise") or {}
        for spk in CHANNEL:
            sp = d.get(spk) or {}
            g, a = parse_gender(sp.get("Gender")), parse_age(sp.get("Age"))
            if not g or not a or not usable_speaker(sp):
                continue
            k = spk[-1]
            key = (f"{g}_{a}", str(sp.get("ID") or f"{session}/{spk}"))
            if speakers is not None and key[1] not in speakers:
                continue
            entry = by_speaker.setdefault(key, dict(cell=key[0], spk_id=key[1], age=sp.get("Age"),
                                                    gender=sp.get("Gender"), snr=None, windows=[]))
            label_snr = parse_snr(noise.get(f"Speaker{k}SNR"))
            if label_snr is not None:
                entry["snr"] = max(entry["snr"] or label_snr, label_snr)
            ws = sorted(windows(us, spk), key=lambda w: abs(w["end"] - w["start"] - TARGET_S))
            for w in ws[:per_session]:
                entry["windows"].append(dict(w, session=session, spk=spk, device=sp.get("RecDevice"),
                                             condition=sp.get("Condition"), snr_label=noise.get(f"Speaker{k}SNR"),
                                             noise_cat=noise.get(f"Speaker{k}NoiseCategory")))
    cells: dict[str, list[dict]] = {c: [] for c in CELLS}
    ranked = sorted(by_speaker.values(), key=lambda e: e["snr"] if e["snr"] is not None else -99.0, reverse=True)
    for e in ranked:
        if e["windows"] and (speakers is not None or len(cells[e["cell"]]) < per_cell * 4):
            e["windows"] = sorted(e["windows"], key=lambda w: abs(w["end"] - w["start"] - TARGET_S))[:cap]
            cells[e["cell"]].append(e)
    return cells


def frames_db(w: np.ndarray) -> np.ndarray:
    n = int(SR * FRAME_S)
    k = len(w) // n
    x = w[: k * n].reshape(k, n).astype(np.float64)
    return 10 * np.log10(np.mean(x * x, axis=1) + 1e-20)


def speech_level_db(db: np.ndarray) -> float | None:
    loud = db[db >= SOUND_DBFS]
    return float(10 * np.log10(np.mean(10 ** (loud / 10)))) if len(loud) else None


def noise_floor_db(db: np.ndarray, quiet: np.ndarray | None = None) -> float | None:
    """100ms 이상 쉼 프레임의 중앙값, 쉼이 없으면 하위 10% (prep_ref.noise_floor_db 와 같은 계산).
    quiet 이 있으면 그 프레임(상대도 조용한 때)만 쉼으로 센다"""
    ok = db > -100.0
    if not ok.any():
        return None
    still = (db < SOUND_DBFS) & ok & (quiet if quiet is not None else True)
    pause, run = [], []
    for v, q in zip(np.append(db, 0.0), np.append(still, False), strict=True):
        if q:
            run.append(v)
            continue
        if len(run) >= PAUSE_FRAMES:
            pause += run
        run = []
    return round(float(np.median(pause) if pause else np.percentile(db[ok], 10)), 1)


def read_stereo(b: bytes) -> np.ndarray | None:
    with wave.open(io.BytesIO(b)) as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (2, 2, SR):
            return None
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, 2)


def channel_of(x: np.ndarray, us: list, spk: str) -> tuple[int, float]:
    """그 화자 소리가 큰 채널과 분리 dB — 상대와 겹치지 않는 발화만으로 잰다(tok_emo.channel_map)"""
    e, n = np.zeros(2), 0
    for sp, s, t, _, _ in us:
        if sp != spk or any(o != spk and os_ < t and oe > s for o, os_, oe, _, _ in us):
            continue
        a, b = int(s * SR), int(t * SR)
        if 0 <= a < b <= len(x):
            e += (x[a:b].astype(np.float64) ** 2).mean(0)
            n += 1
    if not n or not e.sum():
        return CHANNEL[spk], 0.0
    c = int(np.argmax(e))
    return c, round(float(10 * np.log10((e[c] + 1e-12) / (e[1 - c] + 1e-12))), 1)


def longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def judge_clip(x: np.ndarray, w: dict, ch: int, sep: float) -> dict:
    """클립(구간 ±CUT_MARGIN_S 원본 + 디지털 무음 패딩)과 지표, 탈락 이유(없으면 None)"""
    a, b = int(round((w["start"] - PAD_S) * SR)), int(round((w["end"] + PAD_S) * SR))
    if a < 0 or b > len(x):
        return dict(reason="wav 밖")
    wide = x[a:b, ch].astype(np.float32) / 32768.0
    db, odb = frames_db(wide), frames_db(x[a:b, 1 - ch].astype(np.float32) / 32768.0)
    floor = noise_floor_db(db, odb < SOUND_DBFS)
    f0, f1 = int((PAD_S - CUT_MARGIN_S) / FRAME_S), len(db) - int((PAD_S - CUT_MARGIN_S) / FRAME_S)
    level = speech_level_db(db[f0:f1])
    m = dict(sep_db=sep, level_db=None if level is None else round(level, 1), floor_db=floor)
    if level is None or floor is None:
        return dict(m, reason="말소리 없음")
    m["snr_db"] = round(level - floor, 1)
    if sep < MIN_SEP_DB:
        return dict(m, reason="채널 섞임")
    tight_db, tight_odb = db[f0:f1], odb[f0:f1]
    leak = (tight_odb >= SOUND_DBFS) & (tight_odb > tight_db + LEAK_MARGIN_DB) & (tight_db > floor + LEAK_AUDIBLE_DB)
    if longest_run(leak) >= LEAK_MIN_FRAMES:
        return dict(m, reason="상대 소리")
    if m["snr_db"] < MIN_SNR_DB:
        return dict(m, reason="잡음")
    pad = int(round((PAD_S - CUT_MARGIN_S) * SR))
    tight = wide[pad : len(wide) - pad].copy()
    fade = min(int(0.01 * SR), len(tight) // 2)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    tight[:fade] *= ramp
    tight[-fade:] *= ramp[::-1]
    zeros = np.zeros(pad, dtype=np.float32)
    return dict(m, clip=np.concatenate([zeros, tight, zeros]), reason=None)


def write_wav(path: pathlib.Path, w: np.ndarray) -> None:
    tmp = path.with_suffix(".part")
    with wave.open(str(tmp), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((np.clip(w, -1.0, 1.0) * 32767).astype("<i2").tobytes())
    tmp.replace(path)


def overlaps(a: dict, b: dict) -> bool:
    return a["session"] == b["session"] and a["start"] < b["end"] and b["start"] < a["end"]


def top_windows(rows: list[dict], n: int) -> list[dict]:
    """깨끗한 순으로 서로 겹치지 않는 n 개"""
    keep: list[dict] = []
    for r in sorted(rows, key=lambda r: r["snr_db"], reverse=True):
        if len(keep) < n and not any(overlaps(r, k) for k in keep):
            keep.append(r)
    return keep


def cut_clips(cells: dict[str, list[dict]], labels: dict[str, dict], wav_zips: list[str], out: pathlib.Path,
              per_cell: int | None, per_speaker: int = 1) -> tuple[list[dict], collections.Counter]:
    """필요한 세션 wav 만 읽어 화자마다 통과한 구간 중 깨끗한 순으로 겹치지 않는 per_speaker 개를 남긴다"""
    want = collections.defaultdict(list)
    for speakers in cells.values():
        for e in speakers:
            for w in e["windows"]:
                want[w["session"]].append((e, w))
    passed: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    rejected = collections.Counter()
    for p in wav_zips:
        with zipfile.ZipFile(p) as z:
            for info in z.infolist():
                session = pathlib.Path(info.filename).stem
                if not info.filename.lower().endswith(".wav") or session not in want:
                    continue
                x = read_stereo(z.read(info))
                if x is None:
                    rejected["wav 형식 아님"] += len(want.pop(session))
                    continue
                us = utterances(labels[session])
                for e, w in want.pop(session):
                    ch, sep = channel_of(x, us, w["spk"])
                    m = judge_clip(x, w, ch, sep)
                    if m["reason"]:
                        rejected[m["reason"]] += 1
                        continue
                    passed[(e["cell"], e["spk_id"])].append(
                        dict({k: v for k, v in e.items() if k != "windows"}, **w, **m, chan=ch))
    chosen = {key: top_windows(rs, per_speaker) for key, rs in passed.items()}
    rows = []
    for cell in CELLS:
        speakers = sorted((v for k, v in chosen.items() if k[0] == cell), key=lambda v: v[0]["snr_db"], reverse=True)
        for r in [r for v in speakers[:per_cell] for r in v]:
            r["name"] = f"{cell}__{r['session']}_{r['spk'][-1]}_t{int(r['start'])}"
            write_wav(out / f"{r['name']}.wav", r.pop("clip"))
            r.pop("reason", None)
            r["speech_s"] = round(r["end"] - r["start"], 2)
            r["emotions"] = dict(r["emotions"])
            r["rough_hint"] = rough_hint(r["emotions"])
            rows.append(r)
    return rows, rejected


def page(rows: list[dict], counts: dict[str, int]) -> str:
    parts = ["<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width'>",
             "<title>AIHub 화자 후보</title><style>body{font-family:sans-serif;max-width:1100px;margin:2em auto;"
             "padding:0 16px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;"
             "padding:4px 6px;text-align:left;vertical-align:top}.r{color:#b45309}small{color:#555}</style>",
             "<h1>AIHub 71631 실제 사람 화자 후보</h1>",
             "<p>칸(성별·나이)마다 부드러운 목소리 1개, 거친 목소리 1개를 고른다. "
             "'거친 후보' 는 감정 라벨에 화남이 있는 구간일 뿐이니 귀로 정한다. "
             "고른 이름은 <code>prep_ref.py --pick female_young_soft=&lt;이름&gt;</code> 처럼 넘긴다. "
             "깨끗한 순(말소리 − 잡음 바닥)으로 놓았다. 앞뒤 0.35초는 디지털 무음이다. "
             "원본이 16kHz 라 24kHz 운영 참조보다 고음이 적다.</p>"]
    by = collections.defaultdict(list)
    for r in rows:
        by[r["cell"]].append(r)
    for cell in CELLS:
        parts.append(f"<h2 id='{cell}'>{cell} <small>(후보 {len(by[cell])} · wav 있는 화자 {counts.get(cell, 0)}명)"
                     "</small></h2>")
        if not by[cell]:
            parts.append("<p>후보 없음 — 이 칸 화자가 받은 wav 에 없거나 조건에 맞는 깨끗한 구간이 없다.</p>")
            continue
        parts.append("<table><tr><th>이름</th><th>소리</th><th>화자</th><th>감정</th><th>깨끗함</th><th>전사</th></tr>")
        for r in by[cell]:
            emo = ", ".join(f"{k} {v}" for k, v in sorted(r["emotions"].items(), key=lambda kv: -kv[1])) or "-"
            rough = " <span class='r'>거친 후보</span>" if r["rough_hint"] else ""
            name = html.escape(r["name"])
            who = f"{r['age']} · {r['gender']} · {r['device']} · ID {r['spk_id']}"
            warn = "" if r["snr_db"] >= PREP_SNR_DB else " <span class='r'>prep_ref 잡음 경고 예상</span>"
            parts.append(f"<tr><td><code>{name}</code></td>"
                         f"<td><audio controls preload='none' src='{name}.wav'></audio>"
                         f"<br><small>{r['speech_s']}s (말한 시간 {r['voiced_s']}s)</small></td>"
                         f"<td><small>{html.escape(who)}</small></td><td>{html.escape(emo)}{rough}</td>"
                         f"<td>{r['snr_db']} dB{warn}<br><small>말소리 {r['level_db']} · 바닥 {r['floor_db']} dBFS"
                         f"<br>SNR 라벨 {html.escape(str(r['snr_label']))} · 채널 분리 {r['sep_db']} dB</small></td>"
                         f"<td><small>{html.escape(r['text'])}</small></td></tr>")
        parts.append("</table>")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="AIHub 71631 에서 화자 칸별 실제 사람 참조 후보 뽑기")
    ap.add_argument("--label", nargs="+", required=True, help="라벨 zip(TL_*/VL_*)")
    ap.add_argument("--wav", nargs="+", required=True, help="원천 zip(TS_*/VS_*)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-cell", type=int, default=10, help="칸마다 들을 화자 수")
    ap.add_argument("--speakers", help="이 화자 ID 만(쉼표, 칸 상한 없음) — 고른 화자의 다른 구간을 볼 때")
    ap.add_argument("--windows-per-speaker", type=int, default=1, help="화자마다 겹치지 않는 구간 수")
    a = ap.parse_args()
    out = pathlib.Path(os.path.expanduser(a.out))
    out.mkdir(parents=True, exist_ok=True)

    have = wav_sessions(a.wav)
    labels = label_index(a.label, have)
    counts = speaker_counts(labels)
    print(f"wav 세션 {len(have):,}개 중 라벨 있는 것 {len(labels):,}개", flush=True)
    print("칸별 화자 수(wav 있는 세션 기준):", counts, flush=True)
    speakers = {x.strip() for x in a.speakers.split(",") if x.strip()} if a.speakers else None
    n = max(1, a.windows_per_speaker)
    cells = shortlist(labels, a.per_cell, speakers, n)
    print("1단계 화자:", {c: len(v) for c, v in cells.items()}, flush=True)
    rows, rejected = cut_clips(cells, labels, a.wav, out, None if speakers else a.per_cell, n)
    print("2단계 탈락 이유(구간 수):", dict(rejected), flush=True)
    with open(out / "picks.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    cands = {r["name"]: {"ref_audio": str((out / f"{r['name']}.wav").resolve()), "ref_text": r["text"]} for r in rows}
    (out / "candidates.json").write_text(json.dumps(cands, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "listen.html").write_text(page(rows, counts), encoding="utf-8")
    print("칸별 후보:", dict(collections.Counter(r["cell"] for r in rows)), flush=True)
    print(f"끝 → {out}/listen.html", flush=True)


if __name__ == "__main__":
    main()
