"""프리셋 예시 대화를 지금 방식(plain)과 끝 잘림 피하기(clean)로 구워 비교한다 (계획 0070).

  (맥, 레포 venv)    .venv/bin/python scripts/tts_bench/bake_presets.py export --out presets.json
  (집 PC, 포크 venv) <포크 venv>/bin/python bake_presets.py gen --presets presets.json \
      --voices /mnt/c/qwen-out/voices.json --voices /mnt/c/qwen-out/aihub_voices2/refs_h3/voices.part.json \
      --ai-voice slot --out /mnt/c/qwen-out/presets
--voices 는 여러 번 줄 수 있다(뒤 파일이 이김). ai·user 와 프리셋 칸 보이스가 모두 있어야 한다.
clean 은 운영 app/services/example_audio.py 를 그대로 불러 쓴다(레포 밖이면 이 파일 옆 example_audio.py).
GPU 박스에는 이 파일·gen_qwen.py·example_audio.py 세 개를 같은 폴더에 둔다.
결과: <out>/{plain,clean}/<프리셋>.wav(턴 사이 400ms, 운영과 같음) · turns/<방식>/p<프리셋>_t<NN>.wav ·
gen.jsonl · report.md · index.html(대화 단위 청취) · set/(g1_score 받아쓰기 세트 — 꼬리말이 남거나 말이 잘렸는지).
같은 GPU 에 qwen-tts 서비스가 떠 있으면 먼저 내릴 것.
"""

import argparse
import asyncio
import html
import importlib.util
import json
import os
import pathlib
import shutil
import sys
import time
import wave

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SR = 16000
GAP_MS = 400
METHODS = ("plain", "clean")


def _load_example_audio():
    path = ROOT / "app" / "services" / "example_audio.py"
    if not path.exists():
        path = HERE / "example_audio.py"
    spec = importlib.util.spec_from_file_location("bench_example_audio", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # dataclass 가 모듈을 찾는다
    spec.loader.exec_module(mod)
    return mod


ea = _load_example_audio()


# ───────── 입출력 ─────────
def write_wav(path: pathlib.Path, pcm: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm)


def export(out: pathlib.Path) -> list[dict]:
    """레포 프리셋(대본·칸)을 GPU 박스로 가져갈 json 으로"""
    sys.path.insert(0, str(ROOT))
    from app.core.preset_scenarios import PRESET_SCENARIOS

    rows = [{"id": p["scenario_id"], "title": p["title"], "voice": p["tts_voice_id"],
             "dialogue": p["example_dialogue"]} for p in PRESET_SCENARIOS]
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    return rows


def load_voices(paths: list[str]) -> dict[str, tuple[str, str]]:
    voices: dict[str, tuple[str, str]] = {}
    for path in paths:
        with open(os.path.expanduser(path), encoding="utf-8") as f:
            for name, v in json.load(f).items():
                voices[name] = (os.path.expanduser(v["ref_audio"]), v["ref_text"])
    return voices


def needed_voices(presets: list[dict], ai_voice: str) -> set[str]:
    return {"user"} | ({p["voice"] for p in presets} if ai_voice == "slot" else {"ai"})


def check_voices(presets: list[dict], voices: dict[str, tuple[str, str]], ai_voice: str) -> None:
    missing = sorted(needed_voices(presets, ai_voice) - set(voices))
    if missing:
        raise ValueError(f"voices.json 에 없는 보이스: {', '.join(missing)}")


# ───────── 굽기 ─────────
def lead_s(pcm: bytes) -> float | None:
    onset = ea._onset(ea._samples(pcm))
    return round(onset / SR, 3) if onset is not None else None


async def bake_turn(method: str, synth, voice: str, text: str) -> tuple[bytes, dict]:
    if method == "plain":
        pcm = await synth(voice, text)
        return pcm, {"attempts": 1, "cut": ea.is_cut(pcm)}
    got = await ea.clean_turn(lambda t: synth(voice, t), text)
    return got.pcm, {"attempts": got.attempts, "cut": got.cut}


async def bake(presets: list[dict], synth, out: pathlib.Path, ai_voice: str,
               methods: tuple[str, ...] = METHODS, log=print) -> list[dict]:
    """프리셋마다 턴을 굽고 운영처럼 400ms 무음으로 잇는다"""
    gap = b"\x00" * (SR * GAP_MS // 1000 * 2)
    rows: list[dict] = []
    for method in methods:
        for p in presets:
            parts = []
            for n, turn in enumerate(p["dialogue"], 1):
                voice = (p["voice"] if ai_voice == "slot" else "ai") if turn["speaker"] == "ai" else "user"
                tid = f"p{p['id']}_t{n:02d}"
                t0 = time.perf_counter()
                pcm, info = await bake_turn(method, synth, voice, turn["text"])
                write_wav(out / "turns" / method / f"{tid}.wav", pcm)
                parts.append(pcm)
                row = {"method": method, "preset": p["id"], "turn": n, "id": tid, "speaker": turn["speaker"],
                       "voice": voice, "text": turn["text"], **info, "lead_s": lead_s(pcm),
                       "end_decay_ms": ea.end_decay_ms(pcm), "sec": round(time.perf_counter() - t0, 2)}
                rows.append(row)
                flag = " ⚠ 끝 잘림" if info["cut"] else ""
                log(f"[{method}] {tid} {voice} 시도 {info['attempts']}{flag} · {turn['text'][:30]}")
            write_wav(out / method / f"{p['id']}.wav", gap.join(parts))
    (out / "gen.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return rows


# ───────── 보고 ─────────
def summarize(rows: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for method in dict.fromkeys(r["method"] for r in rows):
        rs = [r for r in rows if r["method"] == method]
        leads = [r["lead_s"] for r in rs if r["lead_s"] is not None]
        out[method] = dict(
            turns=len(rs), cut=sum(r["cut"] for r in rs),
            attempts_mean=round(sum(r["attempts"] for r in rs) / len(rs), 2),
            attempts_max=max(r["attempts"] for r in rs),
            plain_fallback=sum(r["attempts"] > ea.MAX_ATTEMPTS for r in rs) if method == "clean" else 0,
            lead_max=max(leads) if leads else None, sec=round(sum(r["sec"] for r in rs), 1),
        )
    return out


def by_voice(rows: list[dict]) -> dict[tuple[str, str], tuple[int, int]]:
    res: dict[tuple[str, str], tuple[int, int]] = {}
    for r in rows:
        cut, n = res.get((r["method"], r["voice"]), (0, 0))
        res[(r["method"], r["voice"])] = (cut + r["cut"], n + 1)
    return res


def report(rows: list[dict], out: pathlib.Path) -> str:
    s = summarize(rows)
    md = ["# 프리셋 예시 굽기 비교 (계획 0070)", "", f"꼬리말: `{ea.TAIL_PHRASE}`", "",
          "| 방식 | 턴 | 끝 잘림 | 시도 평균 / 최대 | 꼬리말 포기(plain 대체) | 앞 무음 최대 s | 합성 시간 s |",
          "|---|---:|---:|---|---:|---:|---:|"]
    for method, m in s.items():
        md.append(f"| {method} | {m['turns']} | {m['cut']} ({m['cut'] / m['turns']:.0%}) | {m['attempts_mean']} / "
                  f"{m['attempts_max']} | {m['plain_fallback']} | {m['lead_max']} | {m['sec']} |")
    md += ["", "## 보이스별 끝 잘림", "", "| 방식 | 보이스 | 끝 잘림 / 턴 |", "|---|---|---:|"]
    for (method, voice), (cut, n) in sorted(by_voice(rows).items()):
        md.append(f"| {method} | {voice} | {cut} / {n} |")
    md += ["", "- 끝 잘림 = 턴 끝 말소리가 피크 대비 -25dB → -45dB 로 10ms 안에 떨어짐(운영 example_audio.is_cut).",
           "- 승인 기준(제안): clean 끝 잘림 ≤ 1/80 · 앞 무음 최대 < 0.3초 · 받아쓰기에 꼬리말 '네' 남은 턴 0."]
    text = "\n".join(md) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    return text


def page(presets: list[dict], rows: list[dict], out: pathlib.Path) -> None:
    """대화 단위 청취: 프리셋마다 plain·clean 두 개"""
    idx = {(r["method"], r["id"]): r for r in rows}
    methods = list(dict.fromkeys(r["method"] for r in rows))
    body = []
    for p in presets:
        body.append(f"<h2>{p['id']}. {html.escape(p['title'])} <small>{html.escape(str(p['voice']))}</small></h2>")
        body.append("".join(f"<p>{m} <audio controls preload='none' src='{m}/{p['id']}.wav'></audio></p>"
                            for m in methods))
        trs = []
        for n, turn in enumerate(p["dialogue"], 1):
            tid = f"p{p['id']}_t{n:02d}"
            cells = "".join(
                f"<td>{'⚠ 잘림' if idx[(m, tid)]['cut'] else '정상'} · {idx[(m, tid)]['attempts']}회</td>"
                for m in methods if (m, tid) in idx)
            trs.append(f"<tr><td>{tid}</td><td>{html.escape(turn['text'])}</td>{cells}</tr>")
        head = "".join(f"<th>{m}</th>" for m in methods)
        body.append(f"<table><tr><th>턴</th><th>대사</th>{head}</tr>{''.join(trs)}</table>")
    (out / "index.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>프리셋 예시 굽기 비교</title>"
        "<style>body{font-family:sans-serif;max-width:960px;margin:24px auto;padding:0 16px}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:4px;font-size:14px}</style>"
        "<h1>프리셋 예시 굽기 비교</h1>" + "".join(body), encoding="utf-8")


def check_set(rows: list[dict], out: pathlib.Path) -> str:
    """g1_score 세트: 턴 글을 정답으로 방식별 받아쓰기 CER, 명령의 $OUT 은 맥으로 가져온 폴더"""
    methods = list(dict.fromkeys(r["method"] for r in rows))
    sdir = out / "set"
    sdir.mkdir(parents=True, exist_ok=True)
    lines = []
    for r in rows:
        if r["method"] != methods[0]:
            continue
        if any(c in r["text"] for c in "|\n\r"):
            raise ValueError(f"meta.lst 칸을 깨는 대사: {r['id']}")
        shutil.copyfile(out / "turns" / methods[0] / f"{r['id']}.wav", sdir / f"{r['id']}.wav")
        lines.append(f"{r['id']}|{r['text']}|{r['id']}.wav|{r['text']}")
    (sdir / "meta.lst").write_text("\n".join(lines) + "\n", encoding="utf-8")
    conds = " ".join(f"--cond {m}=$OUT/turns/{m}" for m in methods)
    return f"python g1_score.py --set $OUT/set {conds} --out $OUT/score --metrics wer"


# ───────── 모델 ─────────
def model_synth(voices: dict[str, tuple[str, str]], device: str):
    """워커 _synth 와 같은 생성·리샘플, 상한에 닿으면 호출측처럼 다시"""
    import torch
    import torchaudio
    from qwen_tts import Qwen3TTSModel

    sys.path.insert(0, str(HERE))
    import gen_qwen as gq

    model = Qwen3TTSModel.from_pretrained(gq.MODEL_ID, device_map=device, dtype=torch.bfloat16,
                                          attn_implementation="flash_attention_2")
    model.enable_streaming_optimizations()

    async def synth(voice: str, text: str) -> bytes:
        ref, ref_text = voices[voice]
        for _ in range(3):
            kw = gq.gen_kwargs(text, ref, ref_text)
            parts, sr = [], None
            for pcm, chunk_sr in model.stream_generate_voice_clone(**kw):
                parts.append(np.asarray(pcm, dtype=np.float32))
                sr = chunk_sr
            w = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
            if len(w) and not gq.hit_cap(len(w), sr, kw["max_frames"]):
                return gq.to_pcm16(torchaudio.functional.resample(torch.from_numpy(w), sr, gq.OUT_SR).numpy())
        raise RuntimeError(f"EOS 없이 상한 3번: {voice} · {text}")

    return synth


def main() -> None:
    ap = argparse.ArgumentParser(description="프리셋 예시를 plain·clean 으로 구워 비교")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="(맥) 레포 프리셋 → json")
    ex.add_argument("--out", required=True)
    g = sub.add_parser("gen", help="(GPU) 굽기 + 보고서·청취 페이지·g1_score 세트")
    g.add_argument("--presets", required=True)
    g.add_argument("--voices", action="append", required=True)
    g.add_argument("--ai-voice", choices=("slot", "ai"), default="slot", help="slot = 스위치 켬, ai = 지금 운영")
    g.add_argument("--methods", default=",".join(METHODS))
    g.add_argument("--only", help="프리셋 번호(쉼표)")
    g.add_argument("--tail-phrase", help=f"꼬리말 바꿔 보기(운영 값 '{ea.TAIL_PHRASE}')")
    g.add_argument("--out", required=True)
    g.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    if a.cmd == "export":
        rows = export(pathlib.Path(a.out))
        print(f"프리셋 {len(rows)}개 → {a.out}")
        return
    presets = json.loads(pathlib.Path(os.path.expanduser(a.presets)).read_text(encoding="utf-8"))
    if a.only:
        keep = {int(x) for x in a.only.split(",")}
        presets = [p for p in presets if p["id"] in keep]
    voices = load_voices(a.voices)
    check_voices(presets, voices, a.ai_voice)
    for name in sorted(needed_voices(presets, a.ai_voice)):
        assert os.path.isfile(voices[name][0]), (name, voices[name][0])
    out = pathlib.Path(os.path.expanduser(a.out))
    if a.tail_phrase:
        ea.TAIL_PHRASE = a.tail_phrase
    synth = model_synth(voices, a.device)
    for name in sorted(needed_voices(presets, a.ai_voice)):
        t = time.perf_counter()
        asyncio.run(synth(name, "네, 안녕하세요."))
        print(f"[{name}] 워밍업 {time.perf_counter() - t:.0f}s", flush=True)
    rows = asyncio.run(bake(presets, synth, out, a.ai_voice, tuple(a.methods.split(",")),
                            log=lambda s: print(s, flush=True)))
    print(report(rows, out))
    page(presets, rows, out)
    print(f"청취: {out}/index.html")
    print(f"받아쓰기(맥, labs venv, OUT=맥으로 가져온 이 폴더): {check_set(rows, out)}")


if __name__ == "__main__":
    main()
