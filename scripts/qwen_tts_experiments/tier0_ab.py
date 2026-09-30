"""Tier-0 A/B — Qwen3-TTS Base 클론의 참조 클립(합성 Sohee 낭독 vs 사람 즉흥 녹음) × 호출 방식(문장별 3회 vs 3문장 통째 1회).

  같은 대사 3문장(2026-09-17 청감 페이지와 동일)을 조건마다 뽑아 wav 와 청감 페이지(index.html, 블라인드 모드)를 만든다.
  서버(운영 워커와 다른 GPU)에서:
    CUDA_VISIBLE_DEVICES=0 nohup <venv>/bin/python tier0_ab.py --voices <운영 voices.json> \
        --ref natural=/path/natural.wav:"말한 그대로의 전사" --out ~/tier0 > ~/tier0.log 2>&1 &
  --voices 의 항목(예: ai = 운영 참조)과 --ref 로 준 항목이 모두 조건이 된다. 보이스마다 첫 호출에 컴파일(≈90 s).
  생성 파라미터는 운영 server.py 와 같다(temperature 0.7, top_k 20, emit_every_frames 2).
"""
import argparse, html, json, os, pathlib, time
import numpy as np, soundfile as sf

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
LINES = [
    ("hospital1", "어... 잠시만요. 김민수 님, 아, 여기 있네요. 12일 화요일 2시 예약 맞으시고요."),
    ("hospital2", "아, 너무 늦으시면 대기가 길어지거나 진료가 밀릴 수 있어서요. 혹시 11시 20분으로 하시면 괜찮으실까요?"),
    ("pizza_rude", "하아... 반반이요? 그런 건 메뉴에 없는데요. 그냥 한 판으로 하실 거예요, 말 거예요?"),
]


def synth(m, ref, rtxt, text):
    t0 = time.perf_counter(); first = None; parts = []; sr = None
    for pcm, sr in m.stream_generate_voice_clone(text=text, language="Korean", ref_audio=ref, ref_text=rtxt,
                                                 emit_every_frames=2, temperature=0.7, top_k=20):
        if first is None: first = (time.perf_counter() - t0) * 1000
        parts.append(np.asarray(pcm, dtype=np.float32))
    return np.concatenate(parts), sr, first, time.perf_counter() - t0


def page(out, voices):
    def audio(f): return f'<td><audio controls preload="none" src="{html.escape(f)}"></audio></td>'
    ths = "".join(f'<th class="cond">{html.escape(v)}</th>' for v in voices)
    t1 = "".join(f'<tr><th class="line">{html.escape(k)}<div class="t">{html.escape(t)}</div></th>' + "".join(audio(f"{k}__{v}_문장별.wav") for v in voices) + "</tr>" for k, t in LINES)
    t2 = "".join(f'<tr><th class="line">{html.escape(lab)}</th>' + "".join(audio(f"all__{v}_{mode}.wav") for v in voices) + "</tr>"
                 for lab, mode in (("문장별 3회 호출을 이어붙임(지금 앱 방식)", "문장별"), ("3문장 통째 1회 호출", "통째")))
    doc = f"""<!doctype html><meta charset="utf-8"><title>Tier-0 A/B</title>
<style>body{{font-family:sans-serif;margin:20px}} th.line{{text-align:left;width:24em;font-weight:600}} .t{{font-weight:normal;color:#555;font-size:.9em}} td,th{{padding:4px 8px;vertical-align:top}} .blind th.cond{{visibility:hidden}}</style>
<h1>Tier-0 A/B · 참조 클립 × 호출 방식</h1>
<p>열 = 참조 클립(voices.json 항목 + --ref). 음색이 아니라 <b>운율·머뭇거림·문장 사이 흐름</b>을 들어 주세요. 생성 파라미터는 운영 워커와 같습니다.</p>
<button onclick="blind()">블라인드(열 이름 숨기고 열 순서 섞기)</button> <button onclick="location.reload()">원래대로</button>
<h2>① 문장별(각 문장 독립 호출)</h2><table class="ab"><tr><th class="line">문장</th>{ths}</tr>{t1}</table>
<h2>② 3문장 연속 — 문장별 이어붙임 vs 통째</h2><table class="ab"><tr><th class="line">호출 방식</th>{ths}</tr>{t2}</table>
<script>
function blind(){{document.body.classList.add('blind');for(const tb of document.querySelectorAll('table.ab')){{const n={html.escape(str(len(voices)))};const ord=[...Array(n).keys()];for(let i=n-1;i>0;i--){{const j=Math.floor(Math.random()*(i+1));[ord[i],ord[j]]=[ord[j],ord[i]];}}for(const r of tb.rows){{const c=[...r.children].slice(1);for(const k of ord)r.appendChild(c[k]);}}}}}}
</script>"""
    (out / "index.html").write_text(doc, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--voices", help="운영 voices.json(항목 전부 조건)"); ap.add_argument("--ref", action="append", default=[], help="이름=wav경로:전사")
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:0"); ap.add_argument("--only", default=None, help="voices.json 에서 이 이름들만(쉼표)")
    a = ap.parse_args(); out = pathlib.Path(os.path.expanduser(a.out)); out.mkdir(parents=True, exist_ok=True)
    voices = {}
    if a.voices:
        for k, v in json.load(open(os.path.expanduser(a.voices), encoding="utf-8")).items():
            if not a.only or k in a.only.split(","): voices[k] = (os.path.expanduser(v["ref_audio"]), v["ref_text"])
    for r in a.ref:
        name, rest = r.split("=", 1); wav, txt = rest.split(":", 1); voices[name] = (os.path.expanduser(wav), txt)
    assert voices, "--voices 또는 --ref 가 필요하다"
    for k, (w, t) in voices.items(): assert os.path.isfile(w), (k, w); print(f"참조 {k}: {w} · {t[:40]}", flush=True)
    import torch; from qwen_tts import Qwen3TTSModel
    m = Qwen3TTSModel.from_pretrained(MODEL_ID, device_map=a.device, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    m.enable_streaming_optimizations()
    log = []
    for v, (ref, rtxt) in voices.items():
        t = time.perf_counter(); synth(m, ref, rtxt, "네, 안녕하세요."); print(f"[{v}] 워밍업·컴파일 {time.perf_counter()-t:.0f}s", flush=True)
        # 문장별: 지금 앱 방식(문장마다 독립 호출, 받은 순서로 이어붙임 — 무음 삽입 없음)
        for key, text in LINES:
            wav, sr, first, el = synth(m, ref, rtxt, text)
            sf.write(out / f"{key}__{v}_문장별.wav", wav, sr); log.append(dict(voice=v, mode="문장별", key=key, first_ms=round(first), s=round(len(wav)/sr, 2), rtf=round(el/(len(wav)/sr), 2)))
            print(f"[{v}/문장별] {key}: 첫 청크 {first:4.0f} ms · {len(wav)/sr:.1f} s · RTF {el/(len(wav)/sr):.2f}", flush=True)
        text_all = " ".join(t for _, t in LINES)
        wav, sr, first, el = synth(m, ref, rtxt, text_all)
        sf.write(out / f"all__{v}_통째.wav", wav, sr); log.append(dict(voice=v, mode="통째", key="all", first_ms=round(first), s=round(len(wav)/sr, 2), rtf=round(el/(len(wav)/sr), 2)))
        print(f"[{v}/통째] 3문장: 첫 청크 {first:4.0f} ms · {len(wav)/sr:.1f} s · RTF {el/(len(wav)/sr):.2f}", flush=True)
        # 문장별 3개를 이어붙인 것도 한 파일로(통째와 나란히 듣기 위해)
        cat = np.concatenate([sf.read(out / f"{key}__{v}_문장별.wav")[0] for key, _ in LINES])
        sf.write(out / f"all__{v}_문장별.wav", cat, sr)
    page(out, list(voices))
    json.dump(log, open(out / "log.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"끝 → {out}/index.html · 조건 {len(voices)}×2 · 파일 {len(list(out.glob('*.wav')))}개", flush=True)


if __name__ == "__main__":
    main()
