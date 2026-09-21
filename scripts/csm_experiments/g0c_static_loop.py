# -*- coding: utf-8 -*-
"""G0 3차 — csm-streaming(원본 Sesame 코드)이 RTF 0.28 을 낸 구조를 **HF 가중치 위에** 짠다.

1·2차 결과(3090·bf16): HF 기본 1.65 · 직접 루프(eager) 1.46 · HF 정적 캐시 자동 컴파일 0.92 — 전부 실패.
원인: 프레임당 층 통과 140회(백본 16 + depth 4×31) × eager ~0.8 ms, 그리고 프레임마다 generate() 재호출.
여기서 하는 것(csm-streaming `models.py`/`generator.py` 752행과 같은 네 가지):
  ① KV 버퍼를 한 번만 할당(백본 2048위치, depth 33위치)   ② 마스크·RoPE 표를 미리 만들어 위치로 인덱싱
  ③ 동기화 없는 샘플링(지수분포 나눗셈 + argmax)            ④ 백본 1스텝·depth 1스텝을 torch.compile(reduce-overhead, fullgraph)
HF 의 선형층·정규화·임베딩 **모듈을 그대로 호출**하므로 학습과 추론이 한 코드베이스에 남는다.

  --mode check   탐욕 디코딩으로 HF generate 와 토큰이 같은지 확인 (CPU 가능, 컴파일 안 함)
  --mode bench   속도 측정 (GPU 면 컴파일, CPU 면 eager)
같은 폴더에 g0_csm_speed.py, g0b_cards.py 가 있어야 한다. 서버에선 CC/CXX 를 ~/gcc-env 로(명령어.md).
"""
import argparse, json, statistics as st
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import g0_csm_speed as g
from g0b_cards import build_inputs

FRAME_S, GATE_RTF, NCB = 0.08, 0.6, 32


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class StaticStack(nn.Module):
    """HF 디코더 층(가중치 그대로) + 고정 KV 버퍼. 프리필(S=T)과 1스텝(S=1)을 같은 코드로 돈다."""

    def __init__(self, layers, norm, rotary, cfg, max_len, device, dtype):
        super().__init__()
        self.layers, self.norm = layers, norm
        self.nh, self.nkv, self.hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        self.register_buffer("k", torch.zeros(len(layers), 1, self.nkv, max_len, self.hd, device=device, dtype=dtype), persistent=False)
        self.register_buffer("v", torch.zeros_like(self.k), persistent=False)
        cos, sin = rotary(torch.zeros(1, 1, 1, device=device, dtype=dtype), position_ids=torch.arange(max_len, device=device)[None])
        self.register_buffer("cos", cos[0], persistent=False); self.register_buffer("sin", sin[0], persistent=False)
        self.register_buffer("mask", torch.tril(torch.ones(max_len, max_len, dtype=torch.bool, device=device)), persistent=False)

    def forward(self, x, pos):                                    # x [1,S,H] · pos [S]
        S = x.shape[1]
        cos, sin, m = self.cos[pos][None, None], self.sin[pos][None, None], self.mask[pos][None, None]
        for i, layer in enumerate(self.layers):
            a, h = layer.self_attn, layer.input_layernorm(x)
            q = a.q_proj(h).view(1, S, self.nh, self.hd).transpose(1, 2)
            k = a.k_proj(h).view(1, S, self.nkv, self.hd).transpose(1, 2)
            v = a.v_proj(h).view(1, S, self.nkv, self.hd).transpose(1, 2)
            q, k = q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
            self.k[i].index_copy_(2, pos, k); self.v[i].index_copy_(2, pos, v)
            o = F.scaled_dot_product_attention(q, self.k[i], self.v[i], attn_mask=m, enable_gqa=True)
            x = x + a.o_proj(o.transpose(1, 2).reshape(1, S, -1))
            x = x + layer.mlp(layer.post_attention_layernorm(x))
        return self.norm(x)


class State(nn.Module):
    """두 스텝이 주고받는 고정 버퍼. 그래프 출력 대신 버퍼에 쓰므로 CUDA Graph 재생이 값을 덮어쓸 걱정이 없다."""

    def __init__(self, hidden, device, dtype):
        super().__init__()
        self.register_buffer("frame", torch.zeros(1, NCB, dtype=torch.long, device=device), persistent=False)   # 이번 프레임의 32코드
        self.register_buffer("emb", torch.zeros(1, 1, hidden, device=device, dtype=dtype), persistent=False)    # depth 의 다음 입력
        self.register_buffer("zero", torch.zeros(1, dtype=torch.long, device=device), persistent=False)


def make_sampler(greedy, temperature=0.9, top_k=50):
    if greedy:
        return lambda z: z.argmax(-1, keepdim=True)
    def f(z):                                                     # csm `sample_topk` 과 같은 방식: 동기화 없음
        z = z.float() / temperature
        p = torch.softmax(z.masked_fill(z < torch.topk(z, top_k)[0][..., -1, None], float("-inf")), -1)
        return torch.argmax(p / torch.empty_like(p).exponential_(1), -1, keepdim=True)
    return f


class BackboneStep(nn.Module):
    def __init__(self, model, stack, state, sample):
        super().__init__()
        self.embed, self.lm_head, self.audio_emb = model.backbone_model.embed_tokens, model.lm_head, model.depth_decoder.model.embed_tokens
        self.stack, self.s, self.sample = stack, state, sample

    def head(self, h):                                            # h [1,2048] → 코드북0 을 뽑아 버퍼에 쓴다
        self.s.frame.index_copy_(1, self.s.zero, self.sample(self.lm_head(h)))
        self.s.emb.copy_(h[:, None, :])                           # depth 위치 0 의 입력 = 백본 은닉값

    def forward(self, pos):                                       # 직전 프레임 32코드 → 한 위치
        self.head(self.stack(self.embed(self.s.frame[:, None, :]), pos)[:, -1])

    def prefill(self, x):                                         # eager 로 한 번. 같은 KV 버퍼에 쓴다
        self.head(self.stack(x, torch.arange(x.shape[1], device=x.device))[:, -1])


class DepthStep(nn.Module):
    def __init__(self, model, stack, state, sample):
        super().__init__()
        dd = model.depth_decoder
        self.proj, self.table, self.W, self.V = dd.model.inputs_embeds_projector, dd.model.embed_tokens, dd.codebooks_head.weight, dd.model.vocab_size
        self.stack, self.s, self.sample = stack, state, sample

    def forward(self, pos):                                       # pos [1] = 0..31. 위치 p 의 출력이 코드북 p 다(p≥1)
        x = self.stack(self.proj(self.s.emb), pos)                # [1,1,1024]
        tok = self.sample(torch.bmm(x, self.W[pos - 1])[:, 0])    # 위치 0 의 출력은 버린다(머리 인덱스 -1 은 유효)
        self.s.frame.index_copy_(1, pos.clamp(min=1), tok)        # 위치 0 의 쓰레기는 위치 1 이 덮어쓴다
        src = torch.where(pos == 0, self.s.frame[:, :1], tok)     # 위치 1 의 입력은 코드북0
        self.s.emb.copy_(self.table(src + pos * self.V))


class StaticCsm:
    def __init__(self, model, greedy, compile_, backend):
        dev, dt, dd = model.device, model.dtype, model.depth_decoder
        bbm, s = model.backbone_model, State(model.config.hidden_size, model.device, model.dtype)
        sample = make_sampler(greedy)
        self.model, self.s = model, s
        self.bb = BackboneStep(model, StaticStack(bbm.layers, bbm.norm, bbm.rotary_emb, model.config, model.config.max_position_embeddings, dev, dt), s, sample)
        self.dd = DepthStep(model, StaticStack(dd.model.layers, dd.model.norm, dd.model.rotary_emb, dd.config, NCB + 1, dev, dt), s, sample)
        self.bb_step, self.dd_step = self.bb, self.dd
        if compile_:
            kw = dict(fullgraph=True, dynamic=False, backend=backend, **({"mode": "reduce-overhead"} if backend == "inductor" else {}))
            self.bb_step, self.dd_step = torch.compile(self.bb, **kw), torch.compile(self.dd, **kw)
        self.P = [torch.tensor([i], device=dev) for i in range(model.config.max_position_embeddings)]

    @torch.no_grad()
    def generate(self, inp, frames):
        rec = {"bb": [], "dd": [], "end": []}
        t0 = g.now()
        x = self.model._merge_input_ids_with_input_values(inp["input_ids"], inp["input_values"], inp["input_values_cutoffs"], None)["inputs_embeds"]
        rec["enc"] = g.now() - t0; t = g.now()
        self.bb.prefill(x); T = x.shape[1]
        rec["prefill"] = g.now() - t
        codes = []
        for f in range(frames):
            ta = g.now()
            if f:
                self.bb_step(self.P[T + f - 1])
            tb = g.now()
            for p in range(NCB):
                self.dd_step(self.P[p])
            codes.append(self.s.frame.clone())
            tc = g.now(); rec["bb"].append(tb - ta); rec["dd"].append(tc - tb); rec["end"].append(tc)
        rec["t0"] = t0
        return torch.stack(codes, 1), rec


def main():
    ap = argparse.ArgumentParser(description="CSM-1B G0 3차: 정적 KV + 컴파일된 스텝 (HF 가중치)")
    ap.add_argument("repo", nargs="?", default="sesame/csm-1b")
    ap.add_argument("--mode", required=True, choices=["check", "bench"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default=None, choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--ctx-seconds", type=float, default=30)
    ap.add_argument("--compile", dest="compile_", action=argparse.BooleanOptionalAction, default=None, help="기본: GPU 면 켬")
    ap.add_argument("--backend", default="inductor", help="추적 가능성만 볼 때는 aot_eager")
    a = ap.parse_args()
    g.DEV = torch.device(a.device)
    dtype = getattr(torch, a.dtype or ("bfloat16" if g.DEV.type == "cuda" else "float32"))
    comp = a.compile_ if a.compile_ is not None else g.DEV.type == "cuda"
    gpu = torch.cuda.get_device_name(g.DEV) if g.DEV.type == "cuda" else "cpu"
    print("=" * 88); print(f"G0-3 · mode={a.mode} · {gpu} · {dtype} · torch {torch.__version__} · compile={comp}({a.backend})"); print("=" * 88)
    proc, model = g.load(a.repo, dtype)
    inp = build_inputs(proc, a.ctx_seconds, dtype); T = inp["input_ids"].shape[1]
    print(f"문맥 {a.ctx_seconds:.0f}초 = {T}위치 · {a.frames}프레임")
    torch.manual_seed(0)

    if a.mode == "check":
        ref = model.generate(**inp, max_new_tokens=a.frames, do_sample=False, depth_decoder_do_sample=False, output_audio=False)
        mine, _ = StaticCsm(model, True, comp, a.backend).generate(inp, a.frames)
        n = min(ref.shape[1], mine.shape[1]); same = ref[:, :n] == mine[:, :n]
        print(f"HF {tuple(ref.shape)} · static {tuple(mine.shape)} · 비교 {n}프레임 × 32코드북")
        print(f"일치 {int(same.sum())}/{same.numel()}" + ("  → 완전 일치 ✓" if bool(same.all()) else
              f"  → 첫 불일치: 프레임 {int((~same).nonzero()[0][1])}, 코드북 {int((~same).nonzero()[0][2])}"))
        return

    sc = StaticCsm(model, False, comp, a.backend)
    for i in range(3):                                            # 1·2회차 = 컴파일 + CUDA Graph 캡처, 3회차를 기록
        t = g.now(); _, rec = sc.generate(inp, a.frames)
        fr = st.median(np.diff(rec["end"]).tolist()) * 1e3           # float64 로 — perf_counter 값은 float32 에 안 담긴다
        print(f"  {i+1}회차 {g.now()-t:6.1f} s · 프레임 {fr:.1f} ms · RTF {fr/80:.3f}" + ("   ← 컴파일 포함" if comp and i < 2 else ""))
    bb, d = st.median(rec["bb"][1:]) * 1e3, st.median(rec["dd"][1:]) * 1e3
    m = dict(mode="static_loop", compile=comp, device=gpu, positions=T, frames=len(rec["end"]), ttff_ms=(rec["end"][0] - rec["t0"]) * 1e3,
             enc_ms=rec["enc"] * 1e3, prefill_ms=rec["prefill"] * 1e3, frame_ms=fr, backbone_ms=bb, depth_ms=d, rtf=fr / 80)
    print(f"\n  첫 프레임 {m['ttff_ms']:.1f} ms (Mimi 인코딩 {m['enc_ms']:.1f} + 프리필 {m['prefill_ms']:.1f} + 첫 프레임)")
    print(f"  프레임 {fr:.1f} ms = 백본 {bb:.1f} + depth 32스텝 {d:.1f} (스텝당 {d/32:.2f})")
    print(f"  RTF {m['rtf']:.3f}  (≤ {GATE_RTF})   {'통과 ✓' if m['rtf'] <= GATE_RTF else '실패 ✗'}   · HF 기본 1.651 대비 {1.651/m['rtf']:.1f}배 · HF 정적 0.92 대비 {0.92/m['rtf']:.1f}배")
    json.dump(m, open("g0c_static_loop.json", "w"), ensure_ascii=False, indent=1); print("\n[저장] g0c_static_loop.json")


if __name__ == "__main__":
    main()
