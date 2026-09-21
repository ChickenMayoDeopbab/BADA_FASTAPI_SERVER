# -*- coding: utf-8 -*-
"""G0 속도 게이트 — CSM-1B 가 3090 에서 실시간·학습 예산 안에 드는지 잰다.

통과 기준(설계 §6 G0):  RTF ≤ 0.6  ·  첫 프레임 ≤ 300 ms  ·  학습 1스텝(seq≈2048)이 VRAM 안
재는 것(설계 §1, A-4):  (a) 프리필  (b) 백본 1스텝  (c) depth decoder 31스텝  (d) Mimi 인코딩/디코딩
재는 대상은 **HF 기본 generate() 경로 그대로**다. CUDA Graph·torch.compile·직접 루프는 여기 없다 —
이 기준선이 "오버헤드 지배냐 대역폭 지배냐"를 알려 준 뒤에 고를 카드다.

사용:
  hf auth login                                        # sesame/csm-1b 는 게이트(자동 승인)
  CUDA_VISIBLE_DEVICES=0 python g0_csm_speed.py        # 기본: sesame/csm-1b, 문맥 10/30/60초, 40프레임
  python g0_csm_speed.py --device cpu --frames 2 --ctx-seconds 2 --skip-train   # 코드 경로만 확인

주의:
- 문맥 오디오는 잡음으로 만든다(속도는 내용과 무관, 프레임 수만 중요). 생성된 소리는 의미 없다.
- generate() 는 EOS(전 코드북 0)에서 일찍 멈출 수 있다. 실제 생성된 프레임 수를 같이 찍는다.
- "첫 프레임"은 기본 경로 기준이라 문맥 전체의 Mimi 인코딩 + 프리필을 매번 다시 한다.
  워커가 세션 캐시를 쓰면 그 둘이 빠진다 → 분해해서 보여 준다.
"""
import argparse, json, math, statistics as st, time
import numpy as np, torch, transformers
from transformers import AutoProcessor, CsmForConditionalGeneration

SR, FRAME_S = 24000, 0.08
GATE_RTF, GATE_TTFF_MS = 0.6, 300.0
# 대역폭 하한용: 프레임마다 통째로 읽히는 가중치 수 (lab03·lab04 에서 센 값)
N_BB = 973_146_112 + 4_200_448                 # 백본 층·norm + 코드북0 헤드(2048×2051)
N_DD = 111_157_248 + 2_097_152 + 1024 * 2051   # depth 층·norm + 2048→1024 투영 + 코드북 헤드 1개
BANDWIDTH_GBPS = {"3090": 936, "4090": 1008}
LINES = ["여보세요, 다음 주 화요일 예약을 바꾸려고 하는데요.", "네, 성함을 말씀해 주시겠어요?",
         "아 네, 김민수요. 화요일 오후 두 시였던 것 같아요.", "잠시만요, 확인해 보겠습니다."]
TARGET = "금요일 오전 열 시와 열한 시 삼십 분이 비어 있습니다. 어느 쪽이 편하세요?"
DEV = torch.device("cpu")


def now():
    if DEV.type == "cuda":
        torch.cuda.synchronize(DEV)
    return time.perf_counter()


def turn(i, seconds, rng, with_audio=True):
    content = [{"type": "text", "text": LINES[i % len(LINES)]}]
    if with_audio:
        content.append({"type": "audio", "path": (0.05 * rng.standard_normal(int(seconds * SR))).astype(np.float32)})
    return {"role": str(i % 2), "content": content}


def load(repo, dtype):
    proc = AutoProcessor.from_pretrained(repo)
    model = CsmForConditionalGeneration.from_pretrained(repo, dtype=dtype)
    model = model.to(device=DEV, dtype=dtype).eval()          # 구버전이 dtype= 을 무시해도 여기서 맞춘다
    # 오디오 임베딩 묶기 — sesame 는 권장, unsloth 는 필수. tie_weights() 로는 안 고쳐진다(설계 부록 A2).
    model.backbone_model.embed_tokens.embed_audio_tokens.weight = model.depth_decoder.model.embed_tokens.weight
    bb = model.backbone_model.embed_tokens.embed_audio_tokens.weight
    assert bb.data_ptr() == model.depth_decoder.model.embed_tokens.weight.data_ptr()
    assert abs(bb.float().std().item() - 0.02) > 0.005, "오디오 임베딩이 무작위다 — 체크포인트를 확인하라"
    return proc, model


def measure_generate(model, inp, frames):
    """generate() 한 번을 돌리며 구간별 시간을 잰다. 기본 루프가 프레임마다 이미 동기화하므로
    (`unfinished_sequences.max() == 0`) 여기서 넣는 동기화는 결과를 거의 바꾸지 않는다."""
    rec = {"enc": [], "bb": [], "dd": [], "end": []}
    codec, dd = model.codec_model, model.depth_decoder
    enc0, gen0 = codec.encode, dd.generate

    def enc(*a, **k):
        t = now(); out = enc0(*a, **k); rec["enc"].append(now() - t); return out

    def gen(*a, **k):
        t = now(); out = gen0(*a, **k); e = now(); rec["dd"].append(e - t); rec["end"].append(e); return out

    codec.encode, dd.generate = enc, gen
    h1 = model.backbone_model.register_forward_pre_hook(lambda m, a: rec.__setitem__("t", now()))
    h2 = model.backbone_model.register_forward_hook(lambda m, a, o: rec["bb"].append(now() - rec["t"]))
    try:
        t0 = now()
        model.generate(**inp, max_new_tokens=frames, output_audio=False)
    finally:
        del codec.encode, dd.generate
        h1.remove(); h2.remove()

    n = len(rec["end"]); ms = lambda s: s * 1000
    frame = st.median(np.diff(rec["end"]).tolist()) if n > 1 else float("nan")
    bb = st.median(rec["bb"][1:]) if n > 1 else float("nan")
    d31 = st.median(rec["dd"][1:] or rec["dd"])
    return dict(frames=n, ttff_ms=ms(rec["end"][0] - t0), enc_ms=ms(sum(rec["enc"])), prefill_ms=ms(rec["bb"][0]),
                first_depth_ms=ms(rec["dd"][0]), frame_ms=ms(frame), backbone_ms=ms(bb), depth31_ms=ms(d31),
                other_ms=ms(frame - bb - d31), rtf=frame / FRAME_S)


def measure_mimi(model, dtype, reps=5):
    codec, rng = model.codec_model, np.random.default_rng(1)
    wav = torch.tensor(0.05 * rng.standard_normal(10 * SR), dtype=dtype, device=DEV)[None, None, :]
    med = lambda f: st.median([(lambda t: (f(), now() - t)[1])(now()) for _ in range(reps)])
    with torch.no_grad():
        codes = codec.encode(wav).audio_codes                       # [1, 32, 125] — 워밍업 겸
        enc_s = med(lambda: codec.encode(wav))
        dec = {n: med(lambda n=n: codec.decode(codes[:, :, :n])) * 1000 for n in (1, 13, 125)}
    return dict(encode_10s_ms=enc_s * 1000, encode_x_realtime=10 / enc_s,
                decode_ms={str(n): v for n, v in dec.items()}, decode_ms_per_frame={str(n): v / n for n, v in dec.items()})


def measure_train(model, proc, positions, dtype):
    rng, n_turns = np.random.default_rng(2), 5
    sec = max(1.0, (positions / n_turns - 40) / 12.5)                # 턴당 글·특수토큰 ≈ 40위치로 잡음
    conv = [turn(i, sec, rng) for i in range(n_turns)]
    batch = proc.apply_chat_template(conv, tokenize=True, return_dict=True,
                                     processor_kwargs={"output_labels": True, "depth_decoder_labels_ratio": 1 / 16}
                                     ).to(DEV, dtype=dtype)
    T = batch["input_ids"].shape[1]
    assert T <= 2048, f"{T} 위치 — 2048 을 넘는다. --train-positions 를 줄여라"
    model.train(); model.codec_model.requires_grad_(False); model.gradient_checkpointing_enable()
    # transformers 5.17 은 위 호출에서 임베딩 출력에 requires_grad_(True) 훅을 건다. 임베딩이 동결돼 있으면(LoRA 등)
    # CSM 의 in-place 대입(`inputs_embeds[:, 0] = ...`)이 RuntimeError 를 낸다. non-reentrant 체크포인팅엔 불필요하므로 끈다.
    model.disable_input_require_grads()
    P = sum(p.numel() for p in {id(p): p for p in model.parameters() if p.requires_grad}.values())
    res = dict(positions=T, trainable_params=P, labels_ratio="1/16", grad_checkpointing=True, batch=1)
    try:
        for i in range(2):                                           # 첫 스텝은 워밍업, 둘째를 기록
            model.zero_grad(set_to_none=True)
            if DEV.type == "cuda":
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(DEV)
            t = now(); out = model(**batch, use_cache=False); out.loss.backward(); step = now() - t
        res.update(step_s=step, loss=out.loss.item(), backbone_loss=out.backbone_loss.item(),
                   depth_loss=out.depth_decoder_loss.item())
        if DEV.type == "cuda":
            res.update(peak_alloc_gb=torch.cuda.max_memory_allocated(DEV) / 1e9,
                       peak_reserved_gb=torch.cuda.max_memory_reserved(DEV) / 1e9)
    except torch.cuda.OutOfMemoryError:
        res["oom"] = True
    model.zero_grad(set_to_none=True); model.eval()
    return res


def main():
    global DEV
    ap = argparse.ArgumentParser(description="CSM-1B G0 속도 게이트")
    ap.add_argument("repo", nargs="?", default="sesame/csm-1b")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--frames", type=int, default=40, help="생성할 프레임 수(80 ms/프레임)")
    ap.add_argument("--ctx-seconds", default="10,30,60", help="문맥 오디오 길이(초), 쉼표로 여러 개")
    ap.add_argument("--train-positions", type=int, default=2000)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--bandwidth-gbps", type=float, default=None, help="GPU 메모리 대역폭(모르는 GPU 일 때)")
    ap.add_argument("--out", default="g0_result.json")
    a = ap.parse_args()
    DEV = torch.device(a.device)
    dtype = getattr(torch, a.dtype or ("bfloat16" if DEV.type == "cuda" else "float32"))
    gpu = torch.cuda.get_device_name(DEV) if DEV.type == "cuda" else "cpu"
    vram = torch.cuda.get_device_properties(DEV).total_memory / 1e9 if DEV.type == "cuda" else None
    R = dict(repo=a.repo, device=gpu, vram_gb=vram, dtype=str(dtype), torch=torch.__version__,
             transformers=transformers.__version__, generate=[])
    bar = "=" * 88
    print(bar); print(f"G0 · {a.repo} · {gpu}" + (f" {vram:.1f} GB" if vram else "") + f" · {dtype} · torch {torch.__version__} · transformers {transformers.__version__}"); print(bar)

    proc, model = load(a.repo, dtype)
    print("[0] 로드 + 오디오 임베딩 묶기 + assert 2개 통과")

    # ── 1. 생성: 문맥 길이별 ──
    print(f"\n[1] generate() — {a.frames}프레임 요청, 샘플링은 저장소 기본값. 시간은 ms")
    print(f"{'문맥':>6}{'위치':>6}{'생성':>5} |{'첫프레임':>9}{'=Mimi인코딩':>11}{'+프리필':>9}{'+depth':>8} |{'프레임':>8}{'=백본':>8}{'+depth31':>9}{'+기타':>8} |{'RTF':>7}")
    torch.manual_seed(0); rng = np.random.default_rng(0)
    for k, ctx in enumerate(float(x) for x in a.ctx_seconds.split(",")):
        n_t = max(1, math.ceil(ctx / 5))
        conv = [turn(i, ctx / n_t, rng) for i in range(n_t)] + [{"role": str(n_t % 2), "content": [{"type": "text", "text": TARGET}]}]
        inp = proc.apply_chat_template(conv, tokenize=True, return_dict=True).to(DEV, dtype=dtype)
        if k == 0:
            model.generate(**inp, max_new_tokens=min(3, a.frames), output_audio=False)      # 워밍업
        m = measure_generate(model, inp, a.frames); m.update(ctx_s=ctx, positions=inp["input_ids"].shape[1])
        R["generate"].append(m)
        print(f"{ctx:>5.0f}s{m['positions']:>6}{m['frames']:>5} |{m['ttff_ms']:>9.1f}{m['enc_ms']:>11.1f}{m['prefill_ms']:>9.1f}{m['first_depth_ms']:>8.1f} |"
              f"{m['frame_ms']:>8.1f}{m['backbone_ms']:>8.1f}{m['depth31_ms']:>9.1f}{m['other_ms']:>8.1f} |{m['rtf']:>7.3f}")
        if m["frames"] < min(8, a.frames):
            print(f"       ⚠ {m['frames']}프레임에서 EOS — 중앙값이 불안정하다. 다시 돌리거나 --frames 를 조정")

    # ── 2. Mimi ──
    R["mimi"] = mm = measure_mimi(model, dtype)
    print(f"\n[2] Mimi (비스트리밍 호출 — 스트리밍 디코더는 아직 없다)")
    print(f"    인코딩 10초: {mm['encode_10s_ms']:.1f} ms = 실시간의 {mm['encode_x_realtime']:.0f}배 → 3,000h 사전 토큰화 ≈ {3000 / mm['encode_x_realtime']:.1f} GPU-시간(배치 1 기준 상한)")
    print("    디코딩: " + " · ".join(f"{n}프레임 {mm['decode_ms'][n]:.1f} ms({mm['decode_ms_per_frame'][n]:.2f}/프레임)" for n in ("1", "13", "125")))

    # ── 3. 학습 1스텝 ──
    if not a.skip_train:
        R["train"] = tr = measure_train(model, proc, a.train_positions, dtype)
        print(f"\n[3] 학습 1스텝 — {tr['positions']}위치 · 배치 1 · 체크포인팅 · ratio 1/16 · 학습 파라미터 {tr['trainable_params']/1e9:.3f}B (Mimi 동결)")
        if tr.get("oom"):
            print("    ✗ forward+backward 에서 OOM")
        else:
            print(f"    forward+backward {tr['step_s']:.2f} s · loss {tr['loss']:.3f} (백본 {tr['backbone_loss']:.3f} + depth {tr['depth_loss']:.3f})")
            if "peak_alloc_gb" in tr:
                M, P = tr["peak_alloc_gb"], tr["trainable_params"] / 1e9
                tr["projected_gb"] = proj = {"8bit_adam": M + 2 * P, "8bit_adam_fp32_master": M + 6 * P, "fp32_adam_fp32_master": M + 12 * P}
                print(f"    peak {M:.2f} GB (reserved {tr['peak_reserved_gb']:.2f}) = 가중치+그래디언트+활성값. 옵티마이저 상태를 더하면(계산값):")
                for name, g in proj.items():
                    print(f"      {name:<24}{g:>6.1f} GB  {'✓' if g <= vram - 1 else '✗'} (VRAM {vram:.0f} GB)")
                print("      백본 LoRA + depth 전체는 8bit_adam 보다 항상 작다")

    # ── 4. 판정 ──
    ref = min(R["generate"], key=lambda m: abs(m["ctx_s"] - 30))
    print(f"\n{bar}\n판정 (문맥 {ref['ctx_s']:.0f}초 기준)\n{bar}")
    ok = lambda b: "통과 ✓" if b else "실패 ✗"
    cached = ref["ttff_ms"] - ref["enc_ms"]
    print(f"  RTF            {ref['rtf']:.3f}  (≤ {GATE_RTF})   {ok(ref['rtf'] <= GATE_RTF)}")
    print(f"  첫 프레임       {ref['ttff_ms']:.0f} ms 기본 경로 / {cached:.0f} ms 문맥 코드 캐시 시 (≤ {GATE_TTFF_MS:.0f})   {ok(cached <= GATE_TTFF_MS)}"
          + ("   ※ 기본 경로로는 초과 — 워커가 문맥을 캐시해야 한다" if ref["ttff_ms"] > GATE_TTFF_MS >= cached else ""))
    if "train" in R and "projected_gb" in R["train"]:
        g = R["train"]["projected_gb"]["8bit_adam_fp32_master"]
        print(f"  학습 메모리      {g:.1f} GB (8-bit Adam + fp32 마스터, 전체 FT)   {ok(g <= vram - 1)}")
    bw = a.bandwidth_gbps or next((v for k, v in BANDWIDTH_GBPS.items() if k in gpu), None)
    if bw and DEV.type == "cuda":
        floor = (N_BB + 31 * N_DD) * 2 / 1e9 / bw / FRAME_S
        R["bandwidth_floor_rtf"], R["rtf_over_floor"] = floor, ref["rtf"] / floor
        regime = "오버헤드 지배 → ① CUDA Graph ② torch.compile ③ 직접 루프 순서로" if ref["rtf"] >= 3 * floor else "대역폭 지배 → ④ 코드북 32→16 만 남는다"
        print(f"  대역폭 하한 RTF {floor:.3f} ({bw:.0f} GB/s) · 실측/하한 = {ref['rtf'] / floor:.1f}배 → {regime}")
    json.dump(R, open(a.out, "w"), ensure_ascii=False, indent=1)
    print(f"\n[저장] {a.out}")


if __name__ == "__main__":
    main()
