"""Qwen3-TTS VoiceDesign 으로 화자 12칸의 후보 참조 음성을 만든다 (GPU, 포크 venv).

  <포크 venv>/bin/python design_voices.py --out /mnt/c/qwen-out/voice_design --n 5 \
      [--slots male_old_rough,female_old_soft] [--instruct 'male_old_rough=...'] \
      [--ref 'female_young_soft__ai=<wav>:<전사>']
칸 = 성별(female/male) × 나이(young/middle/old) × 톤(soft/rough).
이름은 운영 voices.json 키와 같은 {gender}_{age}_{tone}.
결과: out/<칸>__c<i>.wav(모델 SR PCM16) · candidates.json(gen_qwen.py --voices 형식, 길이로 빠진 후보 제외) ·
design.jsonl · listen.html(칸마다 후보를 나란히, 라벨 공개). 이미 만든 후보는 건너뛰어 --n 을 늘려 다시 돌려도 된다.
--instruct 로 바꾼 설명은 이름에 꼬리표(<칸>__<4자>c<i>)가 붙어 같은 seed 로 따로 만들어진다 → 옛 후보와 나란히.
같은 GPU 에 qwen-tts 서비스가 떠 있으면 먼저 내릴 것(8GB 에 모델 두 벌은 안 들어간다).
"""

import argparse
import hashlib
import html
import json
import os
import pathlib
import re
import shutil
import time
import wave

import numpy as np

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
GENDERS = ("female", "male")
AGES = ("young", "middle", "old")
TONES = ("soft", "rough")
SLOTS = [f"{g}_{a}_{t}" for g in GENDERS for a in AGES for t in TONES]

_WHO = {
    "female_young": "A Korean woman in her mid 20s",
    "female_middle": "A Korean woman in her late 40s",
    "female_old": "An elderly Korean woman in her early 70s",
    "male_young": "A Korean man in his mid 20s",
    "male_middle": "A Korean man in his late 40s",
    "male_old": "An elderly Korean man in his early 70s",
}
_VOICE = {
    "female_young_soft": "a soft, bright, warm and gentle voice, polite and kind",
    "female_young_rough": "a husky, slightly raspy low voice, blunt, curt and unfriendly",
    "female_middle_soft": "a soft, calm, warm and mature voice, polite and patient",
    "female_middle_rough": "a hoarse, raspy, harsh voice, impatient, blunt and stern",
    "female_old_soft": "a soft, gentle, warm and slightly thin aged voice, kind",
    "female_old_rough": "a hoarse, croaky, gravelly aged voice, grumpy and curt",
    "male_young_soft": "a soft, clear, warm and gentle voice, polite and friendly",
    "male_young_rough": "a rough, husky, low voice, blunt, curt and indifferent",
    "male_middle_soft": "a soft, deep, calm and warm voice, polite and reassuring",
    "male_middle_rough": "a deep, gravelly, gruff and harsh voice, stern and impatient",
    "male_old_soft": "a soft, low, warm and slightly thin aged voice, kind",
    "male_old_rough": "a hoarse, gravelly, gruff aged voice, grumpy and terse",
}
_STYLE = "Natural Korean phone call at a normal pace, clear pronunciation, quiet room, no background noise."
INSTRUCTS = {s: f"{_WHO[s.rsplit('_', 1)[0]]} with {_VOICE[s]}. {_STYLE}" for s in SLOTS}
# 참조 대본 — ICL 은 참조의 말투를 따라 하므로 톤마다 따로, 노년은 말이 느려 짧게(앞뒤 무음 붙여 10~15초 목표).
# 끝인사·통화 종료 말은 넣지 않는다(참조 문장이 출력에 새는 사고가 있었다). ref_text 로 그대로 쓴다.
SCRIPTS = {
    "soft": "네, 안녕하세요. 문의 주셔서 감사합니다. 말씀하신 내용은 확인해 볼게요. "
    "잠시만 기다려 주시겠어요? 확인됐어요. 그대로 처리해 드릴게요. 더 궁금하신 점 있으세요?",
    "rough": "네, 말씀하세요. 그건 지금 바로 확인이 안 돼요. 예약 번호부터 불러 주세요. "
    "그 번호로는 조회가 안 되는데요. 기다리셔야 하고요, 처리되면 따로 연락 드릴게요.",
    "soft_old": "네, 안녕하세요. 문의 주셔서 감사합니다. 말씀하신 내용은 확인해 볼게요. "
    "잠시만 기다려 주시겠어요? 네, 그렇게 해 드릴게요.",
    "rough_old": "네, 말씀하세요. 그건 지금 바로 확인이 안 돼요. 예약 번호부터 불러 주세요. "
    "기다리셔야 하고요, 처리되면 따로 연락 드릴게요.",
}
LANGUAGE = "Korean"
SEED_BASE = 2670
FRAME_S, SOUND_DBFS, ONSET_FRAMES = 0.02, -45.0, 3  # post.py·guard_fp/measure.py 와 같은 말소리 판정
PAD_S = 0.5  # 운영 참조 앞뒤 무음
REF_RANGE_S = (10.0, 15.0)  # 운영 참조 규격(앞뒤 무음 붙인 길이) — 밖이면 페이지에 경고만
RATE_RANGE = (2.5, 9.0)  # 음절/s — 밖이면 EOS 누락(너무 김)·끝 잘림(너무 짧음)으로 보고 후보에서 뺀다
MAX_NEW_TOKENS = 375  # 12.5Hz × 30초
HANGUL = re.compile("[가-힣]")


def slot_of(name: str) -> str:
    """후보 이름(<칸>__<꼬리>)의 칸"""
    slot, sep, tail = name.partition("__")
    if slot not in SLOTS or not sep or not tail:
        raise ValueError(f"후보 이름은 <칸>__<꼬리> 형식이어야 한다: {name}")
    return slot


def script_for(slot: str) -> str:
    _, age, tone = slot.split("_")
    return SCRIPTS[f"{tone}_old" if age == "old" else tone]


def instruct_tag(slot: str, instruct: str) -> str:
    """기본 설명은 빈 꼬리표, 바꾼 설명은 sha1 앞 4자"""
    return "" if instruct == INSTRUCTS[slot] else hashlib.sha1(instruct.encode()).hexdigest()[:4]


def name_for(slot: str, instruct: str, i: int) -> str:
    return f"{slot}__{instruct_tag(slot, instruct)}c{i}"


def parse_slots(spec: str | None) -> list[str]:
    if not spec:
        return list(SLOTS)
    slots = [s.strip() for s in spec.split(",") if s.strip()]
    bad = [s for s in slots if s not in SLOTS]
    if bad or not slots:
        raise ValueError(f"모르는 칸: {bad} (가능: {SLOTS})")
    return slots


def parse_instructs(specs: list[str]) -> dict[str, str]:
    out, seen = dict(INSTRUCTS), set()
    for spec in specs:
        slot, _, text = spec.partition("=")
        if slot not in SLOTS or not text.strip():
            raise ValueError(f"--instruct 는 <칸>=<설명>: {spec}")
        if slot in seen:
            raise ValueError(f"--instruct 에 같은 칸이 두 번: {slot}")
        seen.add(slot)
        out[slot] = text.strip()
    return out


def parse_refs(specs: list[str]) -> dict[str, tuple[str, str]]:
    refs: dict[str, tuple[str, str]] = {}
    for spec in specs:
        name, _, rest = spec.partition("=")
        wav, _, text = rest.partition(":")
        slot_of(name)
        if not wav or not text.strip():
            raise ValueError(f"--ref 는 <칸>__<꼬리>=<wav>:<전사>: {spec}")
        if name in refs:
            raise ValueError(f"--ref 에 같은 이름이 두 번: {name}")
        refs[name] = (os.path.expanduser(wav), text.strip())
    return refs


def frames_db(w: np.ndarray, sr: int) -> np.ndarray:
    n = int(sr * FRAME_S)
    k = len(w) // n
    x = w[: k * n].reshape(k, n).astype(np.float64)
    return 10 * np.log10(np.mean(x * x, axis=1) + 1e-20)


def measure(w: np.ndarray, sr: int) -> dict:
    """말소리 구간(20ms RMS ≥ -45dBFS 가 60ms 이어진 첫 지점 ~ 마지막 소리)과 앞뒤 무음 붙인 길이"""
    db = frames_db(w, sr)
    loud = db >= SOUND_DBFS
    dur = round(len(w) / sr, 2)
    onset = next((i for i in range(len(loud) - ONSET_FRAMES + 1) if loud[i : i + ONSET_FRAMES].all()), None)
    if onset is None:
        return dict(dur_s=dur, speech_s=0.0, lead_s=dur, tail_s=dur, ref_s=0.0)
    last = len(loud) - 1 - int(np.argmax(loud[::-1]))
    speech = (last + 1 - onset) * FRAME_S
    return dict(dur_s=dur, speech_s=round(speech, 2), lead_s=round(onset * FRAME_S, 2),
                tail_s=round(len(w) / sr - (last + 1) * FRAME_S, 2), ref_s=round(speech + 2 * PAD_S, 2))


def judge(m: dict, text: str) -> tuple[bool, str | None]:
    """대본 음절 수 대비 말소리 길이로 끝 잘림·EOS 누락만 거른다(10~15초 규격은 경고로만)"""
    syllables = len(HANGUL.findall(text))
    lo, hi = syllables / RATE_RANGE[1], syllables / RATE_RANGE[0]
    if m["speech_s"] < lo:
        return False, f"말소리 {m['speech_s']}s < {lo:.1f}s (끝 잘림·빈 출력 의심)"
    if m["speech_s"] > hi:
        return False, f"말소리 {m['speech_s']}s > {hi:.1f}s (EOS 누락 의심)"
    return True, None


def write_wav(path: pathlib.Path, w: np.ndarray, sr: int) -> None:
    tmp = path.with_suffix(".part")
    with wave.open(str(tmp), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes((np.clip(w, -1.0, 1.0) * 32767).astype("<i2").tobytes())
    tmp.replace(path)


def read_wav(path: pathlib.Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as f:
        if f.getsampwidth() != 2:
            raise ValueError(f"PCM16 wav 만 읽는다: {path}")
        x = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        ch = f.getnchannels()
        return (x.reshape(-1, ch).mean(axis=1) if ch > 1 else x), f.getframerate()


def load_log(out: pathlib.Path) -> dict[str, dict]:
    """design.jsonl — 같은 이름은 마지막 줄이 이긴다"""
    path = out / "design.jsonl"
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["name"]: r for r in rows}


def append_log(out: pathlib.Path, row: dict) -> None:
    with open(out / "design.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pending(out: pathlib.Path, slots: list[str], n: int, instructs: dict[str, str]) -> list[str]:
    """아직 없는 후보 — 같은 이름이 다른 설명(꼬리표 충돌·--ref 이름 겹침)으로 있으면 덮어쓰지 않고 멈춘다"""
    done = load_log(out)
    names = [(name_for(s, instructs[s], i), instructs[s]) for s in slots for i in range(1, n + 1)]
    for name, instruct in names:
        if name in done and done[name]["instruct"] != instruct:
            raise ValueError(f"{name} 은 이미 다른 설명으로 만든 후보다 — --out 폴더를 바꾸거나 설명을 고칠 것")
    return [x for x, _ in names if x not in done or not (out / f"{x}.wav").exists()]


def design(model, seed_fn, out: pathlib.Path, names: list[str], instructs: dict[str, str]) -> None:
    """후보마다 seed 를 고정해 생성 — 같은 seed·설명으로 같은 후보를 다시 뽑을 수 있다"""
    for name in names:
        slot = slot_of(name)
        i = int(re.search(r"c(\d+)$", name).group(1))
        instruct, text, seed = instructs[slot], script_for(slot), SEED_BASE + i
        if name != name_for(slot, instruct, i):
            raise ValueError(f"이름과 설명이 맞지 않는다: {name}")
        seed_fn(seed)
        t = time.perf_counter()
        wavs, sr = model.generate_voice_design(text=text, instruct=instruct, language=LANGUAGE,
                                               temperature=0.7, top_k=20, max_new_tokens=MAX_NEW_TOKENS)
        w = np.asarray(wavs[0], dtype=np.float32)
        write_wav(out / f"{name}.wav", w, sr)
        m = measure(w, sr)
        ok, reason = judge(m, text)
        append_log(out, dict(name=name, slot=slot, seed=seed, instruct=instruct, text=text, sr=sr, **m,
                             ok=ok, reason=reason, gen_s=round(time.perf_counter() - t, 1)))
        print(f"[{slot}] {name.split('__', 1)[1]} 말소리 {m['speech_s']:5.2f}s" + ("" if ok else f" ⚠ {reason}"),
              flush=True)


def add_refs(out: pathlib.Path, refs: dict[str, tuple[str, str]]) -> None:
    """기존 참조(예: 운영 ai)를 같은 페이지에서 비교하도록 후보로 복사"""
    done = load_log(out)
    for name, (src, text) in refs.items():
        dst = out / f"{name}.wav"
        shutil.copyfile(src, dst)
        w, sr = read_wav(dst)
        m = measure(w, sr)
        ok, reason = judge(m, text)
        row = dict(name=name, slot=slot_of(name), seed=None, instruct="(기존 참조)", text=text, sr=sr, **m,
                   ok=ok, reason=reason, gen_s=None)
        if done.get(name) != row:
            append_log(out, row)


def write_candidates(out: pathlib.Path) -> dict[str, dict]:
    rows = load_log(out)
    cands = {name: {"ref_audio": str((out / f"{name}.wav").resolve()), "ref_text": r["text"]}
             for name, r in sorted(rows.items()) if r["ok"] and (out / f"{name}.wav").exists()}
    (out / "candidates.json").write_text(json.dumps(cands, ensure_ascii=False, indent=1), encoding="utf-8")
    return cands


def page(rows: dict[str, dict]) -> str:
    """칸마다 후보를 나란히 — 성별·나이·톤이 맞게 들리는지, 자연스러운지 고른다"""
    lo, hi = REF_RANGE_S
    parts = ["<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width'>",
             "<title>화자 후보</title><style>body{font-family:sans-serif;max-width:960px;margin:2em auto;"
             "padding:0 16px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;"
             "padding:4px 6px;text-align:left}.bad{opacity:.45}.warn{color:#b45309}</style>",
             "<h1>화자 후보 — 칸마다 1~2개 고르기</h1>",
             "<p>성별·나이·톤이 칸 이름대로 들리는가, 사람처럼 자연스러운가, 끝음절이 잘리지 않았는가로 고른다. "
             "UTMOS 로 고르지 않는다. 흐린 줄은 길이로 빠진 후보(candidates.json 에 없음).</p>"]
    by_slot: dict[str, list[dict]] = {}
    for r in rows.values():
        by_slot.setdefault(r["slot"], []).append(r)
    for slot in [s for s in SLOTS if s in by_slot]:
        items = sorted(by_slot[slot], key=lambda r: r["name"])
        parts.append(f"<h2 id='{slot}'>{slot}</h2><ul>")
        for instruct in dict.fromkeys(r["instruct"] for r in items if r["seed"] is not None):
            label = instruct_tag(slot, instruct) or "기본"
            parts.append(f"<li><small><b>설명 {label}</b>: {html.escape(instruct)}</small></li>")
        parts.append("</ul><table><tr><th>후보</th><th>소리</th><th>앞뒤 무음 붙인 길이</th><th>앞/뒤 무음</th>"
                     "<th>비고</th></tr>")
        for r in items:
            warn = "" if lo <= r["ref_s"] <= hi else f" <span class='warn'>({lo:g}~{hi:g}s 밖)</span>"
            if r["seed"] is None:
                note = "기존 참조"
            else:
                note = f"설명 {instruct_tag(slot, r['instruct']) or '기본'} · seed {r['seed']}"
            if not r["ok"]:
                note = f"{r['reason']} — {note}"
            name = html.escape(r["name"])
            parts.append(f"<tr class='{'' if r['ok'] else 'bad'}'><td>{name}</td>"
                         f"<td><audio controls preload='none' src='{name}.wav'></audio></td>"
                         f"<td>{r['ref_s']:.1f}s{warn}</td><td>{r['lead_s']:.2f} / {r['tail_s']:.2f}s</td>"
                         f"<td>{html.escape(note)}</td></tr>")
        parts.append("</table>")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="VoiceDesign 으로 화자 칸별 후보 참조 음성 만들기")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=5, help="칸마다 후보 수")
    ap.add_argument("--slots", help="이 칸들만(쉼표), 기본 12칸 전부")
    ap.add_argument("--instruct", action="append", default=[], help="<칸>=<영어 설명> 으로 기본 설명 바꾸기")
    ap.add_argument("--ref", action="append", default=[], help="<칸>__<꼬리>=<wav>:<전사> 기존 참조를 같이 비교")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    slots = parse_slots(a.slots)
    instructs = parse_instructs(a.instruct)
    refs = parse_refs(a.ref)
    for name, (wav_path, _) in refs.items():
        if not os.path.isfile(wav_path):
            raise FileNotFoundError(f"{name}: {wav_path}")
    out = pathlib.Path(os.path.expanduser(a.out))
    out.mkdir(parents=True, exist_ok=True)
    add_refs(out, refs)

    todo = pending(out, slots, a.n, instructs)
    print(f"만들 후보 {len(todo)}개 (칸 {len(slots)} × {a.n}, 이미 있는 것 건너뜀)", flush=True)
    if todo:
        import torch
        from qwen_tts import Qwen3TTSModel

        model = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map=a.device, dtype=torch.bfloat16,
                                              attn_implementation="flash_attention_2")
        design(model, torch.manual_seed, out, todo, instructs)
    cands = write_candidates(out)
    (out / "listen.html").write_text(page(load_log(out)), encoding="utf-8")
    print(f"끝 → {out}/listen.html · 후보 {len(cands)}개(candidates.json)", flush=True)


if __name__ == "__main__":
    main()
