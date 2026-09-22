# -*- coding: utf-8 -*-
"""G1 채점 — seed-tts-eval 방식의 WER·SIM + UTMOS(F5-TTS 방식) + 생성 때 잰 TTFA·RTF 를 한 표로. 맥에서 돈다.

조건 하나 = `<문장 id>.wav` 가 든 폴더 하나. 같은 문장 세트(g1_pick.py 의 meta.lst)에 대해 폴더마다 잰다.
  WER/CER  심판 STT 전사 ↔ 합성할 글. 문장부호 제거 → 발화별 (S+D+I)/N → **산술평균**(seed-tts-eval `run_wer.py`·`average_wer.py`).
           한국어는 seed-tts-eval 이 중국어에 하는 처리(글자 단위)를 주 지표(CER)로 삼고, 어절 단위 WER 을 함께 낸다.
  SIM      합성음 ↔ **프롬프트 음성**의 화자 임베딩 코사인 유사도 평균(seed-tts-eval `get_wav_res_ref_text.py`·`cal_sim.sh`).
  UTMOS    `tarepan/SpeechMOS:v1.2.0` `utmos22_strong` 파일별 점수의 평균(F5-TTS `eval_utmos.py`). seed-tts-eval 에는 없다.
  TTFA·RTF 폴더에 g1_generate.py 가 남긴 gen.jsonl 이 있으면 p50/p95 로 합친다. seed-tts-eval 에는 없다.
심판: `gemini-3.5-transcribe-live` — B `app/services/stt.py` 의 GeminiLiveSTTClient 와 같은 설정, 클립 하나 = 세션 하나.
      키는 환경변수 GEMINI_API_KEY 로만 받는다. seed-tts-eval 의 심판(en Whisper-large-v3 · zh Paraformer)이 아니므로 공개 표와 직접 비교 불가.
모든 측정은 `<out>/cache.jsonl` 에 (지표, wav 의 sha1)로 남는다 — 다시 돌려도 과금·점수 흔들림이 없다.

  python g1_score.py --set ~/g1/set --cond human=~/g1/set/human --cond mimi=~/g1/set/mimi --cond a1_p=~/g1/run/a1_prompted --out ~/g1/score
  python g1_score.py --selftest judge --wav x.wav          # 심판이 붙는지 파일 하나로 먼저 (utmos · sim 도 같은 식)
조건 뒤에 `:ref=prompt` 를 붙이면 그 폴더의 음성이 **프롬프트 글**을 말한 것으로 채점한다(프롬프트 원본·프롬프트의 Mimi 재합성).
"""
import argparse, asyncio, hashlib, html, json, math, os, random, re, sys, time, unicodedata, wave
import numpy as np

JUDGE_MODEL, LANGUAGE, EOS_IDLE_S, EOS_MAX_S = "gemini-3.5-transcribe-live", "ko-KR", 2.0, 10.0


# ───────────────────────── 글자 ─────────────────────────
def normalize(text):
    """NFC → `[숨]` 같은 대괄호 태그 제거 → 글자·숫자·공백만 남김(문장부호·기호 제거) → 소문자 → 공백 하나로."""
    t = re.sub(r"\[[^\[\]]*\]", " ", unicodedata.normalize("NFC", text))
    return " ".join("".join(c if c.isalnum() or c.isspace() else " " for c in t).lower().split())


def edits(ref, hyp):
    """Levenshtein 거리 = 치환 + 삭제 + 삽입 (jiwer 의 S+D+I 와 같은 값)."""
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def error_rates(ref_text, hyp_text):
    r, h = normalize(ref_text), normalize(hyp_text)
    rw, hw, rc, hc = r.split(), h.split(), list(r.replace(" ", "")), list(h.replace(" ", ""))
    ew, ec = edits(rw, hw), edits(rc, hc)
    return dict(wer=ew / max(len(rw), 1), cer=ec / max(len(rc), 1), n_word=len(rw), e_word=ew, n_char=len(rc), e_char=ec, ref=r, hyp=h)


def bootstrap_ci(values, seed=0, n_boot=1000):
    """발화별 값의 평균(%)에 대한 95 % 구간. 문장을 복원추출한다."""
    rng, v = random.Random(seed), list(values)
    means = sorted(100 * sum(rng.choices(v, k=len(v))) / len(v) for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def aggregate(rows):
    n = len(rows)
    return dict(n=n, cer_mean=100 * sum(r["cer"] for r in rows) / n, wer_mean=100 * sum(r["wer"] for r in rows) / n,
                cer_corpus=100 * sum(r["e_char"] for r in rows) / max(sum(r["n_char"] for r in rows), 1),
                wer_corpus=100 * sum(r["e_word"] for r in rows) / max(sum(r["n_word"] for r in rows), 1),
                n_over50=sum(r["cer"] > 0.5 for r in rows))


# ───────────────────────── 소리 ─────────────────────────
def read_wav(path):
    """PCM16 wav → (float32 모노, sr). 그 밖의 형식은 soundfile 이 있으면 그걸로."""
    try:
        with wave.open(path, "rb") as w:
            assert w.getsampwidth() == 2
            x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
            return (x.reshape(-1, w.getnchannels()).mean(1) if w.getnchannels() > 1 else x), w.getframerate()
    except (wave.Error, AssertionError):
        import soundfile as sf
        x, sr = sf.read(path, dtype="float32", always_2d=True)
        return x.mean(1), sr


def resample(x, sr_in, sr_out):
    """유리수 비 리샘플(Kaiser-sinc). 24k→16k 는 tok_kspon.py 의 16k→24k 와 같은 48 kHz 격자·같은 필터(385탭, 7.65 kHz, β 8.6)."""
    if sr_in == sr_out:
        return x
    g = math.gcd(sr_in, sr_out); up, down = sr_out // g, sr_in // g
    if up > 4 or down > 8:
        sys.exit(f"{sr_in}→{sr_out} Hz 는 지원하지 않는다 — 16/24/32/48 kHz wav 로 바꿔서 넣어라")
    half = 64 * max(up, down); fc = 0.95625 * min(sr_in, sr_out) / 2 / (sr_in * up)
    n = np.arange(-half, half + 1)
    h = (2 * fc * np.sinc(2 * fc * n) * np.kaiser(2 * half + 1, 8.6) * up).astype(np.float32)
    z = np.zeros(len(x) * up, dtype=np.float32); z[::up] = x
    return np.convolve(z, h)[half:half + len(z)][::down][: -(-len(x) * up // down)]


def sha1(path):
    return hashlib.sha1(open(path, "rb").read()).hexdigest()


class Cache:
    """(지표 이름, 파일 지문) → 값. 한 줄씩 덧붙이는 jsonl."""

    def __init__(self, path):
        self.path, self.d = path, {}
        if os.path.exists(path):
            for line in open(path, encoding="utf-8"):
                r = json.loads(line); self.d[(r["k"], r["h"])] = r["v"]

    def get(self, k, h):
        return self.d.get((k, h))

    def put(self, k, h, v):
        self.d[(k, h)] = v
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(k=k, h=h, v=v), ensure_ascii=False) + "\n")
        return v

    def get_or(self, k, h, fn):
        v = self.get(k, h)
        return v if v is not None else self.put(k, h, fn())


# ───────────────────────── 심판 ─────────────────────────
def pcm16k(path):
    x, sr = read_wav(path)
    return (np.clip(resample(x, sr, 16000), -1, 1) * 32767).astype("<i2").tobytes()


async def gemini_transcribe(client, types, pcm, model=JUDGE_MODEL, language=LANGUAGE, pace=1.0):
    """클립 하나 = 세션 하나. 100 ms 씩 보내고 `audio_stream_end` 로 끝낸 뒤, 마지막 메시지 이후 EOS_IDLE_S 동안 조용하면 닫는다."""
    cfg = types.LiveConnectConfig(
        response_modalities=["TEXT"],
        input_audio_transcription=types.AudioTranscriptionConfig(language_codes=[language]),      # mode 미지정 = VERBATIM(SDK 2.22.0 문서). SMART 는 군말을 지우므로 쓰면 안 된다
        realtime_input_config=types.RealtimeInputConfig(automatic_activity_detection=types.AutomaticActivityDetection(silence_duration_ms=500)))
    texts, last, closing = [], [time.monotonic()], [False]
    async with client.aio.live.connect(model=model, config=cfg) as session:
        async def sender():
            try:
                for i in range(0, len(pcm), 3200):               # 3200 B = 16 kHz PCM16 100 ms
                    await session.send_realtime_input(audio=types.Blob(data=pcm[i:i + 3200], mime_type="audio/pcm;rate=16000"))
                    if pace > 0:
                        await asyncio.sleep(0.1 / pace)
                await session.send_realtime_input(audio_stream_end=True)
                t_end = last[0] = time.monotonic()
                while time.monotonic() - last[0] < EOS_IDLE_S and time.monotonic() - t_end < EOS_MAX_S:
                    await asyncio.sleep(0.1)
            finally:                                             # 보내다 실패해도 세션을 닫아 받는 쪽을 깨운다(안 그러면 영원히 기다린다)
                closing[0] = True
                await session.close()
        task = asyncio.create_task(sender())
        try:
            while not closing[0]:                                # receive() 는 턴이 끝나면 멈춘다(SDK 2.22.0) → 세션이 열려 있는 동안 다시 받는다
                async for msg in session.receive():
                    last[0] = time.monotonic()
                    tr = msg.server_content.input_transcription if msg.server_content is not None else None
                    if tr is not None and tr.text and tr.finished is not False:     # 서버는 finished 를 안 채운다(B DECISIONS F81)
                        texts.append(tr.text)
                await asyncio.sleep(0.01)
        except Exception:
            if not closing[0]:                                   # 우리가 닫아서 생긴 종료만 정상으로 본다
                raise
        finally:
            if not task.done():
                task.cancel()
            res = await asyncio.gather(task, return_exceptions=True)
    if isinstance(res[0], Exception):                            # 보내는 쪽이 실패했으면 전사를 믿지 않는다
        raise res[0]
    return " ".join(t.strip() for t in texts).strip()


def make_judge(a):
    """(경로 목록, 참조 글 목록) → 전사 목록. 캐시는 바깥에서."""
    if a.judge == "fake":                                        # 파이프라인 시험용: 참조의 끝 세 글자를 뗀다(오류율이 0 이 아니게)
        return "judge:fake", lambda paths, refs: [r[:-3] for r in refs]
    from google import genai
    from google.genai import types
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("환경변수 GEMINI_API_KEY 가 없다 — 터미널에서 `read -s GEMINI_API_KEY; export GEMINI_API_KEY` 로 직접 넣어라(채팅에 붙이지 말 것)")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def run(paths, refs):
        async def one(sem, p):
            async with sem:
                for attempt in range(3):
                    try:
                        return await asyncio.wait_for(gemini_transcribe(client, types, pcm16k(p), a.judge_model, a.language, a.pace), 120)
                    except Exception as e:
                        print(f"  심판 재시도 {attempt + 1}/3 ({os.path.basename(p)}): {type(e).__name__}: {e}", flush=True)
                        await asyncio.sleep(2 * (attempt + 1))
                return None                                      # 세 번 다 실패 → 캐시에 남기지 않고 이번 채점에서 뺀다

        async def main():
            sem = asyncio.Semaphore(a.workers)
            return await asyncio.gather(*(one(sem, p) for p in paths))
        return asyncio.run(main())
    return f"judge:{a.judge_model}:{a.language}", run


# ───────────────────────── UTMOS · SIM ─────────────────────────
def make_utmos(a):
    import torch
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()      # torchaudio 필요(내부 리샘플)

    def f(path):
        x, sr = read_wav(path)
        with torch.no_grad():
            return float(predictor(torch.from_numpy(np.ascontiguousarray(x))[None], sr)[0])
    return "utmos:utmos22_strong", f


def make_sim(a):
    """경로 → 단위 길이 임베딩(list). 코사인 = 내적."""
    import torch
    if a.sim == "fake":                                          # 파이프라인 시험용
        def emb(path):
            x, _ = read_wav(path); v = np.array([x.mean(), x.std(), np.abs(x).max(), 1.0]); return (v / np.linalg.norm(v)).tolist()
        return "sim:fake", emb
    if a.sim == "unispeech":                                     # seed-tts-eval 과 같은 모델. thirdparty/UniSpeech 의 코드를 그대로 부른다
        if not (a.sim_ckpt and a.unispeech_dir):
            sys.exit("--sim unispeech 에는 --sim-ckpt wavlm_large_finetune.pth 와 --unispeech-dir <seed-tts-eval>/thirdparty/UniSpeech/downstreams/speaker_verification 이 필요하다")
        sys.path.insert(0, os.path.expanduser(a.unispeech_dir))
        from models.ecapa_tdnn import ECAPA_TDNN_SMALL            # 그 폴더 verification.py 의 init_model('wavlm_large', ckpt) 와 같은 세 줄(fire·librosa import 를 피하려고 직접 부른다)
        model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
        try:
            sd = torch.load(os.path.expanduser(a.sim_ckpt), map_location="cpu")
        except Exception:                                        # 옛 형식 체크포인트: seed-tts-eval README 의 공식 링크에서 받은 파일일 때만
            sd = torch.load(os.path.expanduser(a.sim_ckpt), map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model"], strict=False); model.eval()

        def emb(path):
            x, sr = read_wav(path)
            with torch.no_grad():
                e = model(torch.from_numpy(np.ascontiguousarray(resample(x, sr, 16000)))[None])[0]
            return torch.nn.functional.normalize(e, dim=-1).tolist()
        return "sim:wavlm_large_finetune", emb
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector                                     # 대체: seed-tts-eval 과 **다른** 모델
    fe = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv"); model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").eval()

    def emb(path):
        x, sr = read_wav(path)
        with torch.no_grad():
            e = model(**fe(resample(x, sr, 16000), sampling_rate=16000, return_tensors="pt")).embeddings[0]
        return torch.nn.functional.normalize(e, dim=-1).tolist()
    return "sim:wavlm-base-plus-sv(비교불가)", emb


# ───────────────────────── 세트 · 결과 ─────────────────────────
def load_meta(set_dir):
    items = []
    for line in open(os.path.join(set_dir, "meta.lst"), encoding="utf-8"):
        f = line.rstrip("\n").split("|")
        if len(f) >= 4:
            items.append(dict(id=f[0], prompt_text=f[1], prompt_wav=os.path.join(set_dir, f[2]), infer_text=f[3],
                              gt_wav=os.path.join(set_dir, f[4]) if len(f) > 4 and f[4] else None))
    return items


def pct(v, q):
    v = sorted(v); return v[min(len(v) - 1, int(q * len(v)))] if v else None


def fmt(x, spec=".1f", none="—"):
    return none if x is None else format(x, spec)


def write_html(path, items, conds, per, sort_by, out_dir):
    """문장별로 조건을 나란히 듣는다. sort_by 조건의 CER 이 나쁜 문장부터."""
    key = lambda it: -(per[sort_by].get(it["id"], {}).get("cer") or 0)
    rel = lambda p: html.escape(os.path.relpath(p, out_dir))
    rows = []
    for it in sorted(items, key=key):
        cells = "".join(
            f"<tr><td class=c>{html.escape(c['name'])}</td><td><audio controls preload=none src='{rel(os.path.join(c['dir'], it['id'] + '.wav'))}'></audio></td>"
            f"<td class=n>{fmt(100 * r['cer']) if r.get('cer') is not None else '—'}</td><td class=n>{fmt(r.get('utmos'), '.2f')}</td><td class=n>{fmt(r.get('sim'), '.3f')}</td>"
            f"<td>{html.escape(r.get('hyp') or '')}</td></tr>"
            for c in conds for r in [per[c["name"]].get(it["id"], {})] if r)
        rows.append(f"<section><h3>{html.escape(it['id'])} · {html.escape(it['infer_text'])}</h3>"
                    f"<p class=p>프롬프트: <audio controls preload=none src='{rel(it['prompt_wav'])}'></audio> {html.escape(it['prompt_text'])}</p>"
                    f"<table><tr><th>조건</th><th>소리</th><th>CER %</th><th>UTMOS</th><th>SIM</th><th>심판 전사</th></tr>{cells}</table></section>")
    css = ("body{font:15px/1.5 -apple-system,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222;background:#fff}"
           "h3{margin:28px 0 4px;font-size:16px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:4px 8px;text-align:left;vertical-align:middle}"
           ".n{text-align:right;font-variant-numeric:tabular-nums}.c{white-space:nowrap;font-weight:600}.p{color:#555;margin:0 0 6px}audio{height:32px;vertical-align:middle}"
           "@media(prefers-color-scheme:dark){body{background:#111;color:#ddd}td,th{border-color:#333}.p{color:#aaa}}")
    open(path, "w", encoding="utf-8").write(f"<!doctype html><meta charset=utf-8><title>G1 듣기</title><style>{css}</style>"
                                            f"<h1>G1 듣기 — '{html.escape(sort_by)}' 의 CER 이 나쁜 문장부터</h1>{''.join(rows)}")


def selftest(a):
    if a.selftest == "judge":
        name, run = make_judge(a); print(name, "→", run([a.wav], [""])[0])
    elif a.selftest == "utmos":
        name, f = make_utmos(a); print(name, "→", round(f(a.wav), 3))
    else:
        name, emb = make_sim(a); e1, e2 = emb(a.wav), emb(a.wav2 or a.wav); print(name, "→", round(float(np.dot(e1, e2)), 4))


def main():
    ap = argparse.ArgumentParser(description="G1 채점: WER/CER · SIM · UTMOS · TTFA · RTF")
    ap.add_argument("--set"); ap.add_argument("--cond", action="append", default=[], help="이름=폴더[:ref=prompt]")
    ap.add_argument("--out"); ap.add_argument("--metrics", default="wer,utmos,sim"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--judge", default="gemini", choices=["gemini", "fake"]); ap.add_argument("--judge-model", default=JUDGE_MODEL)
    ap.add_argument("--language", default=LANGUAGE); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pace", type=float, default=1.0, help="실시간의 몇 배로 보낼지. B 가 검증한 것은 1.0(실시간)뿐이다")
    ap.add_argument("--sim", default="unispeech", choices=["unispeech", "xvector", "fake"]); ap.add_argument("--sim-ckpt"); ap.add_argument("--unispeech-dir")
    ap.add_argument("--sort-by", default=None, help="듣기 페이지 정렬 기준 조건(기본: 마지막 --cond)")
    ap.add_argument("--selftest", choices=["judge", "utmos", "sim"]); ap.add_argument("--wav"); ap.add_argument("--wav2")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a)
    if not (a.set and a.cond and a.out):
        ap.error("--set, --cond, --out 이 필요하다")
    set_dir, out = os.path.expanduser(a.set), os.path.expanduser(a.out); os.makedirs(out, exist_ok=True)
    items = load_meta(set_dir); items = items[: a.limit] if a.limit else items
    conds = []
    for c in a.cond:
        name, rest = c.split("=", 1); d, _, opt = rest.partition(":")
        conds.append(dict(name=name, dir=os.path.expanduser(d), ref="prompt_text" if opt == "ref=prompt" else "infer_text"))
    metrics, cache = set(a.metrics.split(",")), Cache(os.path.join(out, "cache.jsonl"))
    jname, judge = make_judge(a) if "wer" in metrics else (None, None)
    uname, utmos = make_utmos(a) if "utmos" in metrics else (None, None)
    sname, emb = make_sim(a) if "sim" in metrics else (None, None)
    print(f"문장 {len(items)}개 · 조건 {len(conds)}개 · 지표 {sorted(metrics)}")

    per, summary = {}, []
    for c in conds:
        have = [(it, os.path.join(c["dir"], it["id"] + ".wav")) for it in items]
        miss = [it["id"] for it, p in have if not os.path.exists(p)]; have = [(it, p) for it, p in have if os.path.exists(p)]
        hs = {p: sha1(p) for _, p in have}; rows = {}
        if judge:
            todo = [(it, p) for it, p in have if cache.get(jname, hs[p]) is None]
            if todo:
                print(f"[{c['name']}] 심판 {len(todo)}개 (캐시 {len(have) - len(todo)}개)", flush=True)
                for (it, p), hyp in zip(todo, judge([p for _, p in todo], [it[c["ref"]] for it, _ in todo])):
                    if hyp is not None:
                        cache.put(jname, hs[p], hyp)
        gen = {}
        if os.path.exists(os.path.join(c["dir"], "gen.jsonl")):
            gen = {r["id"]: r for r in map(json.loads, open(os.path.join(c["dir"], "gen.jsonl"), encoding="utf-8"))}
        for it, p in have:
            r = dict(id=it["id"]); x, sr = read_wav(p); r["audio_s"] = len(x) / sr
            if judge and cache.get(jname, hs[p]) is not None:
                r.update(error_rates(it[c["ref"]], cache.get(jname, hs[p])))
            long_enough = r["audio_s"] >= 0.5                     # 첫 프레임에서 끝난 빈 소리는 UTMOS·SIM 모델에 넣지 않는다(지표에서 빠지고 CER 은 100 % 로 잡힌다)
            if utmos and long_enough:
                r["utmos"] = cache.get_or(uname, hs[p], lambda: utmos(p))
            if emb and long_enough:
                e1 = cache.get_or(sname + ":emb", hs[p], lambda: emb(p)); hp = sha1(it["prompt_wav"])
                e2 = cache.get_or(sname + ":emb", hp, lambda: emb(it["prompt_wav"])); r["sim"] = float(np.dot(e1, e2))
            if it["gt_wav"] and os.path.exists(it["gt_wav"]) and c["ref"] == "infer_text":
                g, gsr = read_wav(it["gt_wav"]); r["len_ratio"] = r["audio_s"] / max(len(g) / gsr, 1e-6)
            r.update({k: gen[it["id"]].get(k) for k in ("ttfa_ms", "rtf", "eos", "frames") if it["id"] in gen})
            rows[it["id"]] = r
        for i in miss if judge else []:                          # 파일이 없으면 = 생성 실패 → 오류율 100 % 로 센다(지표에서 빼 주지 않는다)
            ref = normalize(next(it for it in items if it["id"] == i)[c["ref"]])
            rows[i] = dict(id=i, cer=1.0, wer=1.0, n_char=len(ref.replace(" ", "")), e_char=len(ref.replace(" ", "")), n_word=len(ref.split()), e_word=len(ref.split()), hyp="(파일 없음)")
        per[c["name"]] = rows
        scored = [r for r in rows.values() if "cer" in r]; s = dict(name=c["name"], n=len(rows), missing=len(miss), judge_failed=len(rows) - len(scored) if judge else 0)
        if scored:
            s.update(aggregate(scored)); s["cer_ci"] = bootstrap_ci([r["cer"] for r in scored])
        for k in ("utmos", "sim", "len_ratio"):
            v = [r[k] for r in rows.values() if r.get(k) is not None]; s[k] = sum(v) / len(v) if v else None
        for k in ("ttfa_ms", "rtf"):
            v = [r[k] for r in rows.values() if r.get(k) is not None]; s[k + "_p50"], s[k + "_p95"] = pct(v, 0.5), pct(v, 0.95)
        e = [r["eos"] for r in rows.values() if r.get("eos") is not None]; s["eos_fail"] = 100 * (1 - sum(e) / len(e)) if e else None
        summary.append(s)

    head = "| 조건 | n | CER % [95 % 구간] | WER % | CER>50 % | UTMOS | SIM | TTFA ms p50/p95 | RTF p50/p95 | 끝남 실패 % | 길이 비 |\n|---|---:|---|---:|---:|---:|---:|---|---|---:|---:|"
    lines = [head] + [
        f"| {s['name']} | {s['n']}{'(없음 ' + str(s['missing']) + ')' if s['missing'] else ''}{'(심판 실패 ' + str(s['judge_failed']) + ' — 다시 돌려라)' if s['judge_failed'] else ''} | "
        + (f"**{s['cer_mean']:.2f}** [{s['cer_ci'][0]:.1f}, {s['cer_ci'][1]:.1f}] | {s['wer_mean']:.2f} | {s['n_over50']}" if "cer_mean" in s else "— | — | —")
        + f" | {fmt(s['utmos'], '.2f')} | {fmt(s['sim'], '.3f')} | {fmt(s['ttfa_ms_p50'], '.0f')} / {fmt(s['ttfa_ms_p95'], '.0f')} | {fmt(s['rtf_p50'], '.3f')} / {fmt(s['rtf_p95'], '.3f')}"
        + f" | {fmt(s['eos_fail'])} | {fmt(s['len_ratio'], '.2f')} |" for s in summary]
    note = (f"\n- 집계: 발화별 오류율의 **산술평균**(seed-tts-eval `average_wer.py`). 전체 합산 CER: " + " · ".join(f"{s['name']} {s['cer_corpus']:.2f}" for s in summary if "cer_corpus" in s)
            + f"\n- 심판 `{jname}` · UTMOS `{uname}` · SIM `{sname}` (합성음 ↔ 프롬프트 음성)\n- 정규화: NFC → 대괄호 태그 제거 → 문장부호·기호 제거 → 소문자. CER 은 띄어쓰기를 지운 음절 단위, WER 은 어절 단위.\n")
    md = "\n".join(lines) + "\n" + note
    open(os.path.join(out, "result.md"), "w", encoding="utf-8").write(md)
    json.dump(dict(summary=summary, per=per, judge=jname, utmos=uname, sim=sname), open(os.path.join(out, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    write_html(os.path.join(out, "listen.html"), items, conds, per, a.sort_by or conds[-1]["name"], out)
    print("\n" + md + f"\n[저장] {out}/result.md · result.json · listen.html")


if __name__ == "__main__":
    main()
