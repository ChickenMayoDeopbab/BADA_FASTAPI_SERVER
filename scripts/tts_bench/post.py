"""벤치 결과를 묶는다 (맥, 레포 venv).

  .venv/bin/python scripts/tts_bench/post.py --out <out> --ref <ref.wav> --ref-text '녹음 전사' [--qwen-speed 1.1]
Qwen 턴 이어 붙이기(운영처럼 문장 오디오를 그대로 잇는다)·선택 속도 늘이기(운영 PcmStretcher)·g1_score 세트(set/)·
바다 지표(요청 끝 '뚝'·앞 무음·말 속도·레벨)·블라인드 청감 페이지·bench.md. g1_score 결과(out/score/result.json)가
있으면 bench.md 에 합친다. 몇 번을 다시 돌려도 된다.
"""

import argparse
import hashlib
import html
import json
import os
import random
import re
import shutil
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.services.pcm_stretch import PcmStretcher  # noqa: E402
from app.services.qwen_tts import _SPEED_WARMUP_MS  # noqa: E402

HERE = Path(__file__).resolve().parent
SR = 16000
# merge_test/count_cuts.py 와 같은 값 — 2ms 최대진폭이 피크 대비 -25dB → -45dB 로 10ms 안에 떨어지면 '뚝'
BLOCK_S, LOUD_DB, QUIET_DB, MIN_SIL_S, ABRUPT_MS = 0.002, -25.0, -45.0, 0.15, 10.0
# guard_fp/measure.py 와 같은 값 — 20ms RMS 가 -45dBFS 이상으로 60ms 이어지면 말소리 시작
FRAME_S, SOUND_DBFS, ONSET_FRAMES = 0.02, -45.0, 3
HANGUL = re.compile("[가-힣]")


# ───────── 입출력 ─────────
def load_tsv(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            key, value = line.split("\t", 1)
            rows[key] = value
    return rows


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, SR), path
        return w.readframes(w.getnframes())


def write_wav(path: Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)


def to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


# ───────── 조건 만들기 ─────────
def stretch(pcm: bytes, speed: float, warmup: bool) -> bytes:
    st = PcmStretcher(speed, warmup_ms=_SPEED_WARMUP_MS if warmup else 0)
    return st.push(pcm) + st.flush()


def turn_audio(pcms: list[bytes], speed: float | None = None) -> bytes:
    """운영 턴: 문장 오디오를 틈 없이 잇는다. 속도를 쓰면 처음 소리 난 문장에만 앞 180ms 원속도"""
    if not speed:
        return b"".join(pcms)
    out, started = [], False
    for pcm in pcms:
        y = stretch(pcm, speed, warmup=not started)
        started = started or bool(y)
        out.append(y)
    return b"".join(out)


def speed_tag(speed: float) -> str:
    return f"x{speed:g}"


def build_conditions(out: Path, sentences: dict[str, str], turns: dict[str, list[str]],
                     speed: float | None) -> None:
    """qwen_<이름> 마다 (속도면 qwen_<이름>_x<속도> 문장 폴더도) 와 그 _turns 폴더를 다시 만든다"""
    bases = sorted(d for d in out.glob("qwen_*") if d.is_dir() and not d.name.endswith("_turns")
                   and not re.search(r"_x[0-9.]+$", d.name))
    for base in bases:
        have = {sid: read_pcm(base / f"{sid}.wav") for sid in sentences if (base / f"{sid}.wav").exists()}
        variants = [(base, None)]
        if speed:
            vdir = out / f"{base.name}_{speed_tag(speed)}"
            for sid, pcm in have.items():
                write_wav(vdir / f"{sid}.wav", stretch(pcm, speed, warmup=True))
            variants.append((vdir, speed))
        for vdir, sp in variants:
            tdir = out / f"{vdir.name}_turns"
            shutil.rmtree(tdir, ignore_errors=True)
            for tid, ids in turns.items():
                if all(sid in have for sid in ids):
                    write_wav(tdir / f"{tid}.wav", turn_audio([have[sid] for sid in ids], sp))


def conditions(out: Path) -> tuple[list[str], list[str]]:
    """(문장 조건, 턴 조건) — el 먼저, 나머지는 이름순"""
    names = sorted(d.name for d in out.iterdir() if d.is_dir() and (d.name.startswith("el") or
                                                                     d.name.startswith("qwen_")))
    order = sorted(names, key=lambda n: (not n.startswith("el"), n))
    turn = [n for n in order if n.endswith("_turns")]
    return [n for n in order if not n.endswith("_turns")], turn


def write_set(out: Path, sentences: dict[str, str], ref: Path, ref_text: str) -> None:
    """g1_score 세트: id|프롬프트 전사|프롬프트 wav|읽을 글 — 프롬프트(SIM 기준)는 새 ref"""
    # 문장은 합성한 글과 같아야 채점이 맞으므로 고치지 않고 막는다
    bad = [sid for sid, text in sentences.items() if "|" in sid or "|" in text or "\n" in text]
    if bad:
        raise ValueError(f"meta.lst 는 | 로 칸을 나눈다 — | 나 줄바꿈이 든 문장: {', '.join(bad)}")
    sdir = out / "set"
    sdir.mkdir(parents=True, exist_ok=True)
    name = "ref" + ref.suffix.lower()
    shutil.copyfile(ref, sdir / name)
    clean = " ".join(ref_text.replace("|", " ").split())
    lines = [f"{sid}|{clean}|{name}|{text}" for sid, text in sentences.items()]
    (sdir / "meta.lst").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ───────── 바다 지표 ─────────
def speech_ends(w: np.ndarray) -> list[tuple[float, float, bool]]:
    """count_cuts.offsets 와 같은 계산: (시각 s, 감쇠 ms, 파일 끝인가)"""
    b = max(1, int(SR * BLOCK_S))
    n = len(w) // b
    if n == 0:
        return []
    env = np.abs(w[: n * b]).reshape(n, b).max(1)
    db = 20 * np.log10(env / (env.max() + 1e-12) + 1e-12)
    min_sil = int(MIN_SIL_S / BLOCK_S)
    res, i = [], 0
    while i < n:
        if db[i] < QUIET_DB:
            j = i
            while j < n and db[j] < QUIET_DB:
                j += 1
            if (j - i >= min_sil or j == n) and i > 0:
                k = i - 1
                while k >= 0 and db[k] < LOUD_DB:
                    k -= 1
                if k >= 0:
                    res.append((i * BLOCK_S, (i - 1 - k) * BLOCK_S * 1000, j == n))
            i = j
        else:
            i += 1
    if db[-1] >= QUIET_DB:
        res.append((n * BLOCK_S, 0.0, True))
    return res


def frames_db(w: np.ndarray) -> np.ndarray:
    n = int(SR * FRAME_S)
    k = len(w) // n
    x = w[: k * n].reshape(k, n).astype(np.float64)
    return 10 * np.log10(np.mean(x * x, axis=1) + 1e-20)


def lead_tail(db: np.ndarray) -> tuple[float, float]:
    """measure.lead_tail 과 같은 계산: (앞 무음 s, 뒤 무음 s)"""
    loud = db >= SOUND_DBFS
    run, onset = 0, None
    for i, v in enumerate(loud):
        run = run + 1 if v else 0
        if run == ONSET_FRAMES:
            onset = i - ONSET_FRAMES + 1
            break
    if onset is None:
        return len(db) * FRAME_S, len(db) * FRAME_S
    last = len(db) - 1 - int(np.argmax(loud[::-1]))
    return onset * FRAME_S, (len(db) - 1 - last) * FRAME_S


def speech_level_db(db: np.ndarray) -> float | None:
    loud = db[db >= SOUND_DBFS]
    return float(10 * np.log10(np.mean(10 ** (loud / 10)))) if len(loud) else None


def file_metrics(path: Path, text: str) -> dict:
    w = to_float(read_pcm(path))
    db = frames_db(w)
    lead, tail = lead_tail(db)
    ends = speech_ends(w)
    end_cut = any(at_end and ms <= ABRUPT_MS for _, ms, at_end in ends)
    mids = [ms for _, ms, at_end in ends if not at_end]
    dur = len(w) / SR
    spoken = dur - lead - tail
    syl = len(HANGUL.findall(text))
    return dict(audio_s=round(dur, 3), lead_s=round(lead, 2), tail_s=round(tail, 2), end_cut=end_cut,
                mid_pauses=len(mids), mid_cuts=sum(ms <= ABRUPT_MS for ms in mids),
                rate=round(syl / spoken, 2) if spoken > 0.2 and syl else None, level_db=speech_level_db(db))


def pct(values: list[float], q: float) -> float | None:
    v = sorted(x for x in values if x is not None)
    return v[min(len(v) - 1, int(q * len(v)))] if v else None


def summarize(rows: dict[str, dict]) -> dict:
    r = list(rows.values())
    lead = [x["lead_s"] for x in r]
    return dict(n=len(r), end_cut=sum(x["end_cut"] for x in r), mid_cuts=sum(x["mid_cuts"] for x in r),
                mid_pauses=sum(x["mid_pauses"] for x in r), lead_p50=pct(lead, 0.5), lead_p90=pct(lead, 0.9),
                lead_max=max(lead) if lead else None, rate_p50=pct([x["rate"] for x in r], 0.5),
                level_p50=pct([x["level_db"] for x in r], 0.5))


def metrics(out: Path, names: list[str], texts: dict[str, str]) -> dict[str, dict[str, dict]]:
    return {name: {k: file_metrics(out / name / f"{k}.wav", text) for k, text in texts.items()
                   if (out / name / f"{k}.wav").exists()} for name in names}


def gen_flags(out: Path, name: str) -> dict:
    """gen.jsonl 의 끝남 실패(eos)·가드 — Qwen 만 있다. 같은 id 는 마지막 줄"""
    path = out / name / "gen.jsonl"
    if not path.exists():
        return {}
    rows = {r["id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r}
    if not any("eos" in r for r in rows.values()):
        return {}
    return dict(eos_fail=sum(not r.get("eos", True) for r in rows.values()),
                guard=sum(bool(r.get("guard")) for r in rows.values()), gen_n=len(rows))


# ───────── g1_score 결과 ─────────
def fmt(x: float | None, spec: str = ".2f") -> str:
    return "—" if x is None else format(x, spec)


def paired_ci(a: list[float], b: list[float], seed: int = 0, n_boot: int = 1000) -> tuple[float, float, float]:
    """같은 문장끼리 a−b 평균과 95% 구간(문장 복원추출)"""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = random.Random(seed)
    means = sorted(float(np.mean([d[rng.randrange(len(d))] for _ in d])) for _ in range(n_boot))
    return float(d.mean()), means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def g1_section(result: dict, base: str = "el") -> list[str]:
    lines = ["| 조건 | n | CER % [95%] | WER % | UTMOS | SIM(새 ref) | 끝남 실패 % | 레벨 dBFS p50 |",
             "|---|---:|---|---:|---:|---:|---:|---:|"]
    for s in result["summary"]:
        cer = f"**{s['cer_mean']:.2f}** [{s['cer_ci'][0]:.1f}, {s['cer_ci'][1]:.1f}]" if "cer_mean" in s else "—"
        lines.append(f"| {s['name']} | {s['n']} | {cer} | {fmt(s.get('wer_mean'))} | {fmt(s.get('utmos'))} | "
                     f"{fmt(s.get('sim'), '.3f')} | {fmt(s.get('eos_fail'), '.1f')} | "
                     f"{fmt(s.get('level_p50'), '+.1f')} |")
    per = result["per"]
    if base in per:
        lines += ["", f"같은 문장 짝 비교(조건 − {base}, 95% 구간 — 0 을 안 품으면 차이가 있다):", ""]
        for name in per:
            if name == base:
                continue
            for key, scale, unit in (("cer", 100.0, "%p"), ("utmos", 1.0, "")):
                ids = [i for i in per[name] if per[name][i].get(key) is not None
                       and per[base].get(i, {}).get(key) is not None]
                if len(ids) < 5:
                    continue
                m, lo, hi = paired_ci([per[name][i][key] * scale for i in ids],
                                      [per[base][i][key] * scale for i in ids])
                lines.append(f"- {name} {key.upper()} {m:+.2f}{unit} [{lo:+.2f}, {hi:+.2f}] (n={len(ids)})")
    return lines


# ───────── 블라인드 청감 ─────────
def link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def blind_page(out: Path, sections: list[tuple[str, list[str], dict[str, str], dict[str, dict]]],
               seed: int) -> str:
    """행마다 조건 순서를 섞고 이름을 숨긴다(파일도 blind/ 에 무작위 이름으로). 고른 것은 브라우저에 남는다"""
    rng = random.Random(seed)
    blind = out / "blind"
    shutil.rmtree(blind, ignore_errors=True)
    blind.mkdir(parents=True)
    data, blocks = [], []
    for sec_i, (title, names, texts, mets) in enumerate(sections):
        rows = []
        for key, text in texts.items():
            have = [n for n in names if (out / n / f"{key}.wav").exists()]
            if len(have) < 2:
                continue
            order = have[:]
            rng.shuffle(order)
            levels = [mets[n][key]["level_db"] for n in order]
            floor = min((x for x in levels if x is not None), default=None)
            row_id = f"{sec_i}-{key}"
            cells = []
            for j, name in enumerate(order):
                gain = 1.0 if floor is None or levels[j] is None else min(1.0, 10 ** ((floor - levels[j]) / 20))
                label = chr(65 + j)
                link(out / name / f"{key}.wav", blind / f"{row_id}-{label}.wav")
                cells.append(
                    f"<label class=opt><input type=radio name='{row_id}' value='{label}'> <b>{label}</b> "
                    f"<audio controls preload=none data-gain='{gain:.3f}' src='blind/{row_id}-{label}.wav'>"
                    "</audio></label>")
            cells.append(f"<label class=opt><input type=radio name='{row_id}' value='='> 비슷함</label>")
            rows.append(f"<div class=row data-row='{row_id}'><p class=t>{html.escape(key)} · {html.escape(text)}</p>"
                        f"{''.join(cells)}</div>")
            data.append({"row": row_id, "section": title, "order": order})
        blocks.append(f"<h2>{html.escape(title)}</h2>{''.join(rows)}")
    payload = json.dumps(data, ensure_ascii=False)
    tag = hashlib.sha1(f"{seed}:{payload}".encode()).hexdigest()[:10]  # 행 구성이 바뀌면 고른 것을 새로
    css = (":root{--fg:#1d1d1f;--bg:#fff;--line:#ddd;--mut:#666}"
           "@media(prefers-color-scheme:dark){:root{--fg:#e6e6e6;--bg:#141414;--line:#333;--mut:#aaa}}"
           "body{font:15px/1.5 -apple-system,sans-serif;color:var(--fg);background:var(--bg);max-width:980px;"
           "margin:24px auto;padding:0 16px}.row{border-bottom:1px solid var(--line);padding:8px 0}"
           ".t{margin:0 0 4px;color:var(--mut)}"
           ".opt{display:inline-flex;align-items:center;gap:6px;margin:2px 16px 2px 0}"
           "audio{height:30px}pre{white-space:pre-wrap;background:rgba(127,127,127,.12);padding:8px}"
           ".bar{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line)}")
    js = """
const KEY='tts-bench-blind:'+location.pathname+':'+TAG, $=s=>document.querySelectorAll(s);
let saved={}; try{saved=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(saved))}catch(e){}}
for(const r of $('input[type=radio]')){ if(saved[r.name]===r.value) r.checked=true;
  r.addEventListener('change',()=>{saved[r.name]=r.value; save();}); }
function applyGain(){const on=document.getElementById('match').checked;
  for(const a of $('audio')) a.volume=on?parseFloat(a.dataset.gain):1;}
document.getElementById('match').addEventListener('change',applyGain); applyGain();
for(const a of $('audio')) a.addEventListener('play',()=>{for(const b of $('audio')) if(b!==a) b.pause();});
function reveal(){const tally={}, picks=[];
  for(const d of DATA){const v=saved[d.row]; if(!v) continue; const s=tally[d.section]||(tally[d.section]={'비슷함':0});
    const name=v==='='?'비슷함':d.order[v.charCodeAt(0)-65]; s[name]=(s[name]||0)+1; picks.push({row:d.row,pick:name});
    const el=document.querySelector(`[data-row="${d.row}"] .t`); if(el&&!el.dataset.shown){el.dataset.shown=1;
      el.textContent+=' — '+d.order.map((n,i)=>String.fromCharCode(65+i)+'='+n).join(', ');}}
  document.getElementById('res').textContent=JSON.stringify({tally,picks},null,1);}
"""
    return ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>TTS 블라인드 청감</title><style>{css}</style>"
            "<h1>TTS 블라인드 청감</h1><p>행마다 순서가 섞여 있고 이름은 숨겨져 있습니다. "
            "가장 사람처럼 자연스러운 것을 고르세요(고른 것은 이 브라우저에 남습니다). "
            "다 고른 뒤 '결과 보기' 를 누르면 행마다 정답과 조건별 집계가 나옵니다.</p>"
            "<div class=bar><label><input type=checkbox id=match checked> "
            "음량 맞추기(행 안에서 가장 작은 소리에 맞춤)</label> "
            "<button onclick='reveal()'>결과 보기</button></div>"
            f"{''.join(blocks)}<h2>결과</h2><pre id=res>아직 안 봄</pre>"
            f"<script>const TAG='{tag}', DATA={payload};{js}</script>")


# ───────── 보고서 ─────────
def bada_section(title: str, names: list[str], mets: dict[str, dict[str, dict]], out: Path) -> list[str]:
    lines = [f"### {title}", "",
             "| 조건 | 파일 | 요청 끝 '뚝' | 쉼 '뚝' / 쉼 | 앞 무음 p50 / p90 / 최대 s | "
             "말 속도 음절/s p50 | 레벨 dBFS p50 | 끝남 실패 · 가드 |",
             "|---|---:|---:|---:|---|---:|---:|---|"]
    for name in names:
        s = summarize(mets[name])
        if not s["n"]:
            continue
        g = gen_flags(out, name)
        flags = f"{g['eos_fail']} · {g['guard']} / {g['gen_n']}" if g else "—"
        lines.append(f"| {name} | {s['n']} | {s['end_cut']}/{s['n']} ({100 * s['end_cut'] / s['n']:.0f}%) | "
                     f"{s['mid_cuts']}/{s['mid_pauses']} | {fmt(s['lead_p50'])} / {fmt(s['lead_p90'])} / "
                     f"{fmt(s['lead_max'])} | {fmt(s['rate_p50'])} | {fmt(s['level_p50'], '+.1f')} | {flags} |")
    return lines + [""]


def main() -> int:
    ap = argparse.ArgumentParser(description="TTS 벤치 결과 묶기")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sentences", type=Path, default=HERE / "sentences.tsv")
    ap.add_argument("--turns", type=Path, default=HERE / "turns.tsv")
    ap.add_argument("--ref", type=Path, help="새 ref.wav (g1_score SIM 기준)")
    ap.add_argument("--ref-text", help="ref.wav 전사")
    ap.add_argument("--qwen-speed", type=float, help="운영 QWEN_TTS_SPEED 가 1.0 이 아니면 그 값")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = a.out.expanduser()
    sentences = load_tsv(a.sentences)
    turns = {k: v.split() for k, v in load_tsv(a.turns).items()}
    dropped = [tid for tid, ids in turns.items() if not all(s in sentences for s in ids)]
    if dropped:
        print(f"문장 목록에 없는 문장을 쓰는 턴은 뺀다: {', '.join(dropped)}")
    turns = {tid: ids for tid, ids in turns.items() if tid not in dropped}
    turn_texts = {tid: " ".join(sentences[s] for s in ids) for tid, ids in turns.items()}
    speed = a.qwen_speed if a.qwen_speed and abs(a.qwen_speed - 1.0) > 1e-9 else None

    build_conditions(out, sentences, turns, speed)
    if a.ref:
        assert a.ref_text, "--ref 에는 --ref-text 가 필요하다"
        write_set(out, sentences, a.ref.expanduser(), a.ref_text)
    sent_names, turn_names = conditions(out)
    sent_m, turn_m = metrics(out, sent_names, sentences), metrics(out, turn_names, turn_texts)
    (out / "bada_metrics.json").write_text(json.dumps({"sentences": sent_m, "turns": turn_m}, ensure_ascii=False,
                                                      indent=1), encoding="utf-8")
    page = blind_page(out, [("문장 52", sent_names, sentences, sent_m), ("턴 16 (운영처럼 이어 듣기)", turn_names,
                                                                         turn_texts, turn_m)], a.seed)
    (out / "listen_blind.html").write_text(page, encoding="utf-8")

    md = ["# ElevenLabs 대 Qwen 음질 비교", "",
          "계획 0067. 같은 문장 52개(운영 대사 44 + 감탄사·단문 8)와 운영 턴 16개.", ""]
    result = out / "score" / "result.json"
    if result.exists():
        md += ["## g1_score (Gemini 심판 CER · UTMOS · SIM)", ""]
        md += g1_section(json.loads(result.read_text("utf-8")))
        md += ["", "- SIM 은 새 ref 기준 — 그 ref 를 복제한 조건만 복제 충실도, "
               "나머지는 다른 사람 바닥값(대체 모델, 상대 비교만).",
               "- UTMOS 는 영어 학습 모델이라 상대값만 본다.", ""]
    else:
        md += ["g1_score 결과 없음 — README 의 채점 명령을 돌린 뒤 다시 실행.", ""]
    md += ["## 바다 지표", ""]
    md += bada_section("문장", sent_names, sent_m, out) + bada_section("턴", turn_names, turn_m, out)
    md += ["- 요청 끝 '뚝' = 파일 끝 말소리가 피크 대비 -25dB → -45dB 로 10ms 안에 떨어짐(count_cuts.py). "
           "턴은 문장 사이 쉼도 센다.",
           "- 턴 안 '쉼' 은 150ms 이상 조용한 곳만 센다(count_cuts.py 그대로). Qwen 턴은 문장 사이가 짧아 "
           "문장 끝 잘림이 여기 안 잡힐 수 있다 — 문장 끝 잘림은 문장 표로 본다.",
           "- 앞 무음 = 20ms RMS -45dBFS 가 60ms 이어지기 전까지(measure.py). "
           "말 속도 = 한글 음절 수 ÷ (길이 − 앞뒤 무음), 숫자는 안 센다.",
           f"- 블라인드 청감: `{out / 'listen_blind.html'}`", ""]
    (out / "bench.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
