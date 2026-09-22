# -*- coding: utf-8 -*-
"""G1 생성 — 시험셋(g1_pick.py)의 문장을 CSM 으로 말하게 하고 TTFA·RTF 를 잰다. 체크포인트가 있는 **학교 서버**에서 돈다.

  --mode prompted   seed-tts-eval 방식: `[0]프롬프트 글 + 프롬프트 음성` 을 문맥으로 주고 `[0]합성할 글` 을 말하게 한다 → WER·SIM·UTMOS·TTFA·RTF
  --mode noprompt   A단계 학습 형식 그대로 `[0]합성할 글` 만 준다                                                    → WER·UTMOS·TTFA·RTF (SIM 은 뜻이 없다)
  --engine static   운영 경로(G0 를 통과한 정적 KV + 컴파일 스텝, g0c_static_loop.py). TTFA·RTF 는 이 경로에서만 의미가 있다. 샘플링은 온도 0.9 · top-k 50.
  --engine hf       HF 기본 generate()(저장소 기본 생성 설정). 점수 대조용 — TTFA 는 못 잰다.
  --check N         탐욕·fp32 로 첫 문장의 앞 N 프레임이 HF generate 와 같은 토큰인지 확인하고 끝낸다(프롬프트 문맥 포함).
재는 법(B `scripts/qwen_tts_experiments/gate6.py` 와 같은 정의, 단 HTTP 구간 없음):
  TTFA = 글을 받은 시점 → 앞 `--first-frames` 프레임(프레임당 80 ms)의 PCM 이 디코드된 시점      RTF = 총 벽시계(마지막 전체 디코드 포함) ÷ 오디오 길이
  끝남 = 앞 31코드북이 전부 0 인 프레임(HF `generation_csm.py` 의 멈춤 조건과 같다). `--max-seconds` 까지 안 나오면 eos=false.
입력 임베딩은 **미리 뽑은 프롬프트 코드**로 직접 만든다(HF `_merge_input_ids_with_input_values` 와 같음은 --check 가 확인).
같은 폴더나 한 단계 위에 g0c_static_loop.py · g0_csm_speed.py · g0b_cards.py 가 있어야 한다. 서버에선 CC/CXX 를 ~/gcc-env 로(labs/g0/명령어.md).

  python g1_generate.py --set ~/CSM/g1/set --weights sesame/csm-1b        --mode prompted --out ~/CSM/g1/run/base_p
  python g1_generate.py --set ~/CSM/g1/set --weights ~/CSM/runs/a1/last   --mode noprompt --out ~/CSM/g1/run/a1_n       # 끊겨도 같은 명령으로 이어서
"""
import argparse, json, os, sys, time, wave
import numpy as np, torch, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
NCB, FRAME_S, SR = 32, 0.08, 24000


def now(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    return time.perf_counter()


def load(repo, weights, dtype, dev):
    from transformers import AutoProcessor, CsmForConditionalGeneration
    proc = AutoProcessor.from_pretrained(repo)                   # 토크나이저는 원본 저장소에서(체크포인트 폴더엔 가중치만 있다)
    model = CsmForConditionalGeneration.from_pretrained(os.path.expanduser(weights), dtype=dtype)
    model.backbone_model.embed_tokens.embed_audio_tokens.weight = model.depth_decoder.model.embed_tokens.weight
    bb = model.backbone_model.embed_tokens.embed_audio_tokens.weight
    assert bb.data_ptr() == model.depth_decoder.model.embed_tokens.weight.data_ptr()
    assert abs(bb.float().std().item() - 0.02) > 0.005, "오디오 임베딩이 무작위다 — 체크포인트를 확인하라"
    return proc, model.to(dev).eval()


def text_ids(tok, text, speaker=0):
    return tok(f"{tok.bos_token}[{speaker}]{text}{tok.eos_token}", add_special_tokens=False).input_ids


def build_ids(tok, cfg, it, codes, mode):
    """→ (ids [L], spans). spans = [(오디오 자리 시작, 코드 [T,32])]. 배치는 처리기의 chat template 과 같다(README 의 검증)."""
    if mode == "noprompt":
        return text_ids(tok, it["infer_text"]), []
    p = text_ids(tok, it["prompt_text"]); T = codes.shape[0]
    return p + [cfg.audio_token_id] * T + [cfg.audio_eos_token_id] + text_ids(tok, it["infer_text"]), [(len(p), codes)]


def embed_inputs(model, ids, spans):
    """글 자리 = embed_text_tokens · 오디오 자리 = 32코드북 임베딩의 합 · audio_eos 자리 = 전부 0 인 프레임의 임베딩."""
    dev = model.device
    x = model.embed_text_tokens(torch.tensor(ids, device=dev)[None])
    for st, c in spans:
        fr = torch.cat([c.to(dev), torch.zeros(1, NCB, dtype=torch.long, device=dev)])
        x[0, st:st + fr.shape[0]] = model.backbone_model.embed_tokens(fr[:, None, :])[:, 0].to(x.dtype)
    return x


@torch.no_grad()
def gen_static(sc, model, ids, spans, max_frames, k_first):
    dev = model.device; t0 = now(dev)
    x = embed_inputs(model, ids, spans); T = x.shape[1]
    sc.bb.prefill(x)
    frames, ttfa, eos = [], None, False
    for f in range(max_frames):
        if f:
            sc.bb_step(sc.P[T + f - 1])
        for p in range(NCB):
            sc.dd_step(sc.P[p])
        fr = sc.s.frame.clone()
        if bool((fr[:, :-1] == 0).all()):
            eos = True; break
        frames.append(fr)
        if len(frames) == k_first:
            model.codec_model.decode(torch.stack(frames, 2)).audio_values.float().cpu(); ttfa = now(dev) - t0
    codes = torch.stack(frames, 2) if frames else torch.zeros(1, NCB, 0, dtype=torch.long, device=dev)     # [1,32,T]
    audio = model.codec_model.decode(codes).audio_values[0, 0].float().cpu() if frames else torch.zeros(int(FRAME_S * SR))
    return codes[0].T, audio, eos, ttfa, now(dev) - t0, T


# ── HF 경로(대조·--check)에는 프롬프트 **오디오**가 필요하다: 16 kHz 원본을 tok_kspon.py 와 같은 필터로 24 kHz 로 올린다 ──
def resample_16k_to_24k(x):
    n = torch.arange(-192, 193, dtype=torch.float64); fc = 7650.0 / 48000
    h = (2 * fc * torch.sinc(2 * fc * n) * torch.kaiser_window(385, periodic=False, beta=8.6, dtype=torch.float64) * 3).float()
    up = torch.zeros(x.numel() * 3); up[::3] = x
    return F.conv1d(up[None, None], h[None, None], padding=192)[0, 0, ::2]


def read_wav16(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2 and w.getnchannels() == 1, path
        return torch.from_numpy(np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0)


def hf_inputs(proc, model, it, set_dir, mode):
    conv = [{"role": "0", "content": [{"type": "text", "text": it["infer_text"]}]}]
    if mode == "prompted":
        au = resample_16k_to_24k(read_wav16(os.path.join(set_dir, "prompt", it["id"] + ".wav"))).numpy()
        conv.insert(0, {"role": "0", "content": [{"type": "text", "text": it["prompt_text"]}, {"type": "audio", "path": au}]})
    inp = proc.apply_chat_template(conv, tokenize=True, return_dict=True).to(model.device)
    if "input_values" in inp:
        inp["input_values"] = inp["input_values"].to(model.dtype)
    return inp


@torch.no_grad()
def gen_hf(model, inp, max_frames):
    dev = model.device; t0 = now(dev)
    out = model.generate(**inp, max_new_tokens=max_frames, output_audio=True, return_dict_in_generate=True)
    total = now(dev) - t0; seq = out.sequences[0]                # [생성 프레임, 32]
    hit = (seq == 0).all(-1).nonzero(); eos = hit.numel() > 0; n = int(hit.min()) if eos else seq.shape[0]
    audio = out.audio[0].float().cpu() if n else torch.zeros(int(FRAME_S * SR))
    return seq[:n], audio, eos, None, total, inp["input_ids"].shape[1]


def write_wav(path, x):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((x.clamp(-1, 1) * 32767).to(torch.int16).numpy().astype("<i2").tobytes())


def main():
    ap = argparse.ArgumentParser(description="G1 생성 + TTFA·RTF")
    ap.add_argument("--set", required=True); ap.add_argument("--out"); ap.add_argument("--repo", default="sesame/csm-1b")
    ap.add_argument("--weights", default="sesame/csm-1b", help="저장소 이름 또는 train_a.py 의 체크포인트 폴더")
    ap.add_argument("--mode", default="prompted", choices=["prompted", "noprompt"]); ap.add_argument("--engine", default="static", choices=["static", "hf"])
    ap.add_argument("--max-seconds", type=float, default=20.0); ap.add_argument("--first-frames", type=int, default=1, help="TTFA 를 재는 첫 청크의 프레임 수")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--dtype", default=None, choices=["bfloat16", "float32"])
    ap.add_argument("--compile", dest="compile_", action=argparse.BooleanOptionalAction, default=None, help="기본: GPU 면 켬"); ap.add_argument("--backend", default="inductor")
    a = ap.parse_args()
    if not a.check and not a.out:
        ap.error("--out 이 필요하다(--check 일 때만 생략)")
    dev = torch.device(a.device); set_dir = os.path.expanduser(a.set)
    dtype = torch.float32 if a.check else getattr(torch, a.dtype or ("bfloat16" if dev.type == "cuda" else "float32"))
    comp = a.compile_ if a.compile_ is not None else dev.type == "cuda"
    items = [json.loads(l) for l in open(os.path.join(set_dir, "set.jsonl"), encoding="utf-8")]; items = items[: a.limit] if a.limit else items
    pc = np.load(os.path.join(set_dir, "prompt_codes.npz"))
    codes_of = lambda it: torch.from_numpy(pc[it["id"]].astype(np.int64)).T                                  # [T,32]
    max_frames = int(a.max_seconds / FRAME_S)
    print(f"G1 생성 · {a.weights} · {a.mode} · {a.engine} · {dev} · {dtype} · 문장 {len(items)} · 최대 {max_frames}프레임")
    proc, model = load(a.repo, a.weights, dtype, dev); tok, cfg = proc.tokenizer, model.config

    if a.check:                                                   # 탐욕·fp32: 내 입력 구성 + 정적 루프 == HF generate ?
        from g0c_static_loop import StaticCsm
        it = items[0]; ids, spans = build_ids(tok, cfg, it, codes_of(it), a.mode); inp = hf_inputs(proc, model, it, set_dir, a.mode)
        print(f"  input_ids 동일: {inp['input_ids'][0].tolist() == ids}  ({len(ids)}위치)")
        if a.mode == "prompted":
            ref = model._merge_input_ids_with_input_values(inp["input_ids"], inp["input_values"], inp["input_values_cutoffs"], None)["inputs_embeds"]
            d_saved = (ref - embed_inputs(model, ids, spans)).abs().max().item()
            # 저장 코드(집 PC 가 뽑음)는 이 기계의 코덱이 뽑는 코드와 세부 코드북이 몇 % 다를 수 있다(2026-09-22 맥 실측: 코드북0 100 %, 1~31 87~95 %).
            # 토큰 비교는 HF 와 **같은 코드**로 해야 뜻이 있으므로 이 기계의 코덱으로 다시 뽑아 쓴다.
            au = resample_16k_to_24k(read_wav16(os.path.join(set_dir, "prompt", it["id"] + ".wav")))
            with torch.no_grad():
                here = model.codec_model.encode(au[None, None].to(dev, model.dtype)).audio_codes[0].T.cpu()
            saved = spans[0][1]; T = min(len(here), len(saved)); eq = (here[:T] == saved[:T])
            print(f"  저장 코드 vs 이 기계 코덱: 코드북0 {eq[:, 0].float().mean() * 100:.0f} % · 1~31 {eq[:, 1:].float().mean() * 100:.1f} % 일치 ({len(saved)}/{len(here)}프레임)")
            if len(here) == len(saved):
                spans = [(spans[0][0], here)]
            print(f"  inputs_embeds 최대 차이: 저장 코드 {d_saved:.2e} · 이 기계 코드 {(ref - embed_inputs(model, ids, spans)).abs().max().item():.2e}  (뒤가 0 이어야 아래 비교가 성립한다)")
        hf = model.generate(**inp, max_new_tokens=a.check, do_sample=False, depth_decoder_do_sample=False, output_audio=False)[0]
        mine = gen_static(StaticCsm(model, True, comp, a.backend), model, ids, spans, a.check, 1)[0]
        n = min(hf.shape[0], mine.shape[0]); same = hf[:n] == mine[:n]
        print(f"  토큰 일치 {int(same.sum())}/{same.numel()} ({n}프레임 × 32)" + ("  → 완전 일치 ✓" if n and bool(same.all()) else "  ✗"))
        return

    out = os.path.expanduser(a.out); os.makedirs(out, exist_ok=True); log = os.path.join(out, "gen.jsonl")
    done = {json.loads(l)["id"] for l in open(log, encoding="utf-8")} if os.path.exists(log) else set()
    if all(it["id"] in done for it in items):
        print("전부 끝나 있다 — 새로 만들 문장이 없다")
    elif a.engine == "static":
        from g0c_static_loop import StaticCsm
        sc = StaticCsm(model, False, comp, a.backend)
        ids, spans = build_ids(tok, cfg, items[0], codes_of(items[0]), a.mode)
        for i in range(3 if comp else 1):                         # 컴파일 + CUDA Graph 캡처 + Mimi 디코드 커널을 데운다(기록 안 함)
            t = time.perf_counter(); gen_static(sc, model, ids, spans, 6, a.first_frames); print(f"  예열 {i + 1}: {time.perf_counter() - t:.1f} s", flush=True)
    rows = []
    for i, it in enumerate(items):
        if it["id"] in done:
            continue
        torch.manual_seed(a.seed * 100003 + i)
        if a.engine == "static":
            ids, spans = build_ids(tok, cfg, it, codes_of(it), a.mode)
            assert len(ids) + max_frames <= cfg.max_position_embeddings, "문맥 + 최대 길이가 2048 위치를 넘는다"
            codes, audio, eos, ttfa, total, T = gen_static(sc, model, ids, spans, max_frames, a.first_frames)
        else:
            codes, audio, eos, ttfa, total, T = gen_hf(model, hf_inputs(proc, model, it, set_dir, a.mode), max_frames)
        write_wav(os.path.join(out, it["id"] + ".wav"), audio); n = int(codes.shape[0]); sec = max(n * FRAME_S, FRAME_S)
        r = dict(id=it["id"], mode=a.mode, engine=a.engine, weights=a.weights, frames=n, eos=bool(eos), audio_s=round(n * FRAME_S, 2),
                 ttfa_ms=None if ttfa is None else round(ttfa * 1e3, 1), total_s=round(total, 3), rtf=round(total / sec, 4), ctx_positions=T, first_frames=a.first_frames, seed=a.seed)
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        rows.append(r)
        print(f"  [{i + 1:>3}/{len(items)}] {it['id']} {n:>3}프레임 {'끝남' if eos else '안끝남'} · TTFA {r['ttfa_ms']} ms · RTF {r['rtf']:.3f} · {it['infer_text'][:30]}", flush=True)
    rows = [json.loads(l) for l in open(log, encoding="utf-8")]
    med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
    t, rt = [r["ttfa_ms"] for r in rows if r["ttfa_ms"] is not None], [r["rtf"] for r in rows if r["frames"]]
    print(f"\n{len(rows)}문장 · TTFA p50 {(format(med(t), '.0f') + ' ms(첫 ' + str(rows[0]['first_frames']) + '프레임)') if t else '—(hf 경로는 못 잰다)'} · RTF p50 {med(rt):.3f} · 끝남 실패 {sum(not r['eos'] for r in rows)}개 → {out}")


if __name__ == "__main__":
    main()
