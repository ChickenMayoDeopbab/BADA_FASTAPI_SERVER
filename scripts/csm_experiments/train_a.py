# -*- coding: utf-8 -*-
"""A단계 학습 — 미리 뽑은 Mimi 토큰으로 `sesame/csm-1b` 에 한국어 발음·즉흥 운율을 가르친다 (전체 파인튜닝).

  입력      tok_kspon.py 의 출력 폴더(codes/*.npz + manifest/*.jsonl). 오디오·Mimi 는 필요 없다.
  정확성    csm_data 가 만드는 입력·레이블·loss 가 HF 경로와 같다(verify_inputs.py, 2026-09-21).
  정밀도    가중치는 fp32 로 들고 bf16 autocast 로 계산한다(bf16 가중치에 lr 3e-5 를 바로 더하면 갱신이 반올림으로 사라진다).
  메모리    fp32 가중치 6.2 + 그래디언트 6.2 + 8-bit Adam 3.1 + 활성값 ≈ 17~18 GB (3090 24 GB). `bitsandbytes` 필요.
  안전장치  오디오 임베딩 수동 묶기 + assert · gradient checkpointing + `disable_input_require_grads()` · Mimi 동결
  검증셋    KsponSpeech_0621~0623(dev.trn) 은 학습에서 빼고 검증 loss 에 쓴다. eval_* 도 뺀다.

사용 (학교 서버):
  uv pip install bitsandbytes
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python train_a.py --data ~/CSM/data/kspon --out ~/CSM/runs/a1
  ... --resume ~/CSM/runs/a1/last        # 컨테이너가 재시작되면 이어서
"""
import argparse, json, math, os, random, re, shutil, time
import torch
from transformers import AutoProcessor, CsmForConditionalGeneration
import csm_data as D


def load_model(repo, weights, device):
    proc = AutoProcessor.from_pretrained(repo)                  # 토크나이저는 항상 원본 저장소에서(체크포인트 폴더엔 가중치만 있다)
    model = CsmForConditionalGeneration.from_pretrained(weights, dtype=torch.float32)
    # 오디오 임베딩 묶기 — tie_weights() 로는 안 고쳐진다. 학습하면 두 벌이 갈라지므로 필수다.
    model.backbone_model.embed_tokens.embed_audio_tokens.weight = model.depth_decoder.model.embed_tokens.weight
    bb = model.backbone_model.embed_tokens.embed_audio_tokens.weight
    assert bb.data_ptr() == model.depth_decoder.model.embed_tokens.weight.data_ptr()
    assert abs(bb.float().std().item() - 0.02) > 0.005, "오디오 임베딩이 무작위다 — 체크포인트를 확인하라"
    model.codec_model.requires_grad_(False)
    model.to(device); model.codec_model.to("cpu")               # 토큰으로 학습하므로 Mimi 는 GPU 에 둘 필요가 없다
    return proc, model


def save_checkpoint(model, optim, state, out, keep):
    d = os.path.join(out, f"step_{state['update']:06d}"); tmp = d + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    sd = model.state_dict()                                     # 묶인 임베딩은 같은 저장공간이라 safetensors 가 거부한다 → 한쪽을 복사해 원본 체크포인트처럼 두 키로 저장
    k = "backbone_model.embed_tokens.embed_audio_tokens.weight"; sd[k] = sd[k].clone()
    model.save_pretrained(tmp, state_dict=sd, safe_serialization=True)
    torch.save(dict(optim=optim.state_dict(), state=state), os.path.join(tmp, "trainer.pt"))
    shutil.rmtree(d, ignore_errors=True); os.replace(tmp, d)
    link = os.path.join(out, "last")
    if os.path.islink(link): os.unlink(link)
    os.symlink(os.path.basename(d), link)
    for old in sorted(p for p in os.listdir(out) if re.fullmatch(r"step_\d{6}", p))[:-keep]:
        shutil.rmtree(os.path.join(out, old), ignore_errors=True)
    return d


def lr_at(u, a):
    if u < a.warmup: return a.lr * (u + 1) / a.warmup
    p = min(1.0, (u - a.warmup) / max(1, a.max_updates - a.warmup))
    return a.lr * (a.lr_floor + (1 - a.lr_floor) * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def evaluate(model, dev, a, device, amp):
    model.eval(); tot = dict(loss=0.0, bb=0.0, dd=0.0); n = 0
    for batch in dev.batches(a.batch_positions // 2, 1.0, seed=0, shuffle=False):       # ratio 1.0 — depth 를 전 프레임으로 잰다
        batch = {k: v.to(device) for k, v in batch.items()}
        with amp():
            emb, lab = D.build_inputs(model, batch)
            o = model(inputs_embeds=emb, attention_mask=batch["attention_mask"], labels=lab, use_cache=False)
        tot["loss"] += o.loss.item(); tot["bb"] += o.backbone_loss.item(); tot["dd"] += o.depth_decoder_loss.item(); n += 1
        if n >= a.eval_batches: break
    model.train()
    return {k: v / max(1, n) for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser(description="CSM A단계 학습 (미리 뽑은 토큰)")
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default="sesame/csm-1b"); ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--text-field", default="mix", choices=["spell", "pron", "mix"], help="spell=철자(30분) · pron=발음(삼십분) · mix=발화마다 반반")
    ap.add_argument("--drop-flags", default="unknown", help="쉼표 목록: unknown,overlap,noise,…  해당 플래그가 있는 발화를 뺀다")
    ap.add_argument("--ratio", type=float, default=1 / 16, help="depth decoder 를 학습할 프레임 비율")
    ap.add_argument("--batch-positions", type=int, default=2048); ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5); ap.add_argument("--lr-floor", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=300); ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--max-updates", type=int, default=0, help="0 = 첫 에폭에서 센다")
    ap.add_argument("--optim", default=None, choices=["adamw8bit", "adamw"]); ap.add_argument("--freeze", default=None, help="이 정규식에 맞는 파라미터는 동결")
    ap.add_argument("--amp-cache", action="store_true", help="autocast 캐스트 캐시를 켠다(+3 GB, 조금 빠름)")
    ap.add_argument("--save-every", type=int, default=250); ap.add_argument("--keep", type=int, default=2)
    ap.add_argument("--eval-every", type=int, default=250); ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--log-every", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert 0 < a.ratio <= 1.0, f"ratio={a.ratio} — 1.0 을 넘으면 HF 에서는 depth 학습이 0 이 된다"
    device = torch.device(a.device); cuda = device.type == "cuda"
    os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed)
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=a.amp_cache)) if cuda else (lambda: torch.autocast("cpu", enabled=False))

    proc, model = load_model(a.repo, a.resume or a.repo, device)
    tok, cfg = proc.tokenizer, model.config
    if a.freeze:
        for n, p in model.named_parameters():
            if re.search(a.freeze, n): p.requires_grad_(False)
    model.train(); model.gradient_checkpointing_enable(); model.disable_input_require_grads()
    params = list({id(p): p for p in model.parameters() if p.requires_grad}.values())
    decay, no_decay = [p for p in params if p.ndim >= 2], [p for p in params if p.ndim < 2]
    groups = [dict(params=decay, weight_decay=a.weight_decay), dict(params=no_decay, weight_decay=0.0)]
    kind = a.optim or ("adamw8bit" if cuda else "adamw")
    if kind == "adamw8bit":
        import bitsandbytes as bnb
        optim = bnb.optim.AdamW8bit(groups, lr=a.lr, betas=(0.9, 0.95))
    else:
        optim = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95))
    state = dict(update=0, micro=0, epoch=0, micro_in_epoch=0)
    if a.resume:
        ck = torch.load(os.path.join(a.resume, "trainer.pt"), map_location="cpu", weights_only=False)
        optim.load_state_dict(ck["optim"]); state = ck["state"]

    drop = [f for f in a.drop_flags.split(",") if f]
    train = D.TokenShards(a.data, tok, cfg, a.text_field, drop, split="train"); dev = D.TokenShards(a.data, tok, cfg, "spell", drop, split="dev")
    print(f"학습 샤드 {len(train.names)} · 검증 샤드 {len(dev.names)} · 학습 파라미터 {sum(p.numel() for p in params)/1e9:.3f}B · {kind} · text={a.text_field} · ratio={a.ratio:.4f} · 배치 {a.batch_positions}위치 × accum {a.grad_accum}")
    assert train.names, f"{a.data}/manifest 에 학습 샤드가 없다"
    if not a.max_updates:                                       # 스케줄 길이를 알려면 에폭당 배치 수가 필요하다 → 매니페스트의 frames 로 센다(토큰화 없이)
        a.max_updates = max(1, a.epochs * count_batches(train, a.batch_positions) // a.grad_accum)
    print(f"예정 업데이트 {a.max_updates:,} (warmup {a.warmup}) · lr {a.lr:g} → {a.lr * a.lr_floor:g}")
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    json.dump({k: v for k, v in vars(a).items()}, open(os.path.join(a.out, "args.json"), "w"), ensure_ascii=False, indent=1)

    t0 = time.time(); seen = 0; acc = dict(loss=0.0, bb=0.0, dd=0.0)
    try:
        while state["update"] < a.max_updates:
            skip = state["micro_in_epoch"]
            for i, batch in enumerate(train.batches(a.batch_positions, a.ratio, seed=a.seed + state["epoch"])):
                if i < skip: continue                           # 이어받기: 이미 본 배치를 건너뛴다(같은 seed 라 같은 순서)
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with amp():
                    emb, lab = D.build_inputs(model, batch)
                    o = model(inputs_embeds=emb, attention_mask=batch["attention_mask"], labels=lab, use_cache=False)
                (o.loss / a.grad_accum).backward()
                for k, v in (("loss", o.loss), ("bb", o.backbone_loss), ("dd", o.depth_decoder_loss)): acc[k] += v.item() / a.grad_accum
                seen += int(batch["attention_mask"].sum()); state["micro"] += 1; state["micro_in_epoch"] += 1
                if state["micro"] % a.grad_accum: continue
                lr = lr_at(state["update"], a)
                for g in optim.param_groups: g["lr"] = lr
                gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
                optim.step(); optim.zero_grad(set_to_none=True); state["update"] += 1; u = state["update"]
                if u % a.log_every == 0 or u == 1:
                    el = time.time() - t0; mem = torch.cuda.max_memory_allocated(device) / 1e9 if cuda else 0
                    rec = dict(update=u, epoch=state["epoch"], **{k: round(v, 4) for k, v in acc.items()}, grad_norm=round(gn, 3), lr=lr, pos_per_s=round(seen / el), mem_gb=round(mem, 2))
                    print(f"[{u:>6}/{a.max_updates}] loss {acc['loss']:.4f} (백본 {acc['bb']:.4f} + depth {acc['dd']:.4f}) · |g| {gn:.2f} · lr {lr:.2e} · {seen/el:,.0f} 위치/s · {mem:.1f} GB", flush=True)
                    log.write(json.dumps(rec) + "\n"); log.flush()
                acc = dict(loss=0.0, bb=0.0, dd=0.0)
                if u % a.eval_every == 0 and dev.names:
                    ev = evaluate(model, dev, a, device, amp); print(f"        검증 loss {ev['loss']:.4f} (백본 {ev['bb']:.4f} + depth {ev['dd']:.4f})", flush=True)
                    log.write(json.dumps(dict(update=u, **{"val_" + k: round(v, 4) for k, v in ev.items()})) + "\n"); log.flush()
                if u % a.save_every == 0: print("        저장 →", save_checkpoint(model, optim, state, a.out, a.keep), flush=True)
                if u >= a.max_updates: break
            else:
                state["epoch"] += 1; state["micro_in_epoch"] = 0; optim.zero_grad(set_to_none=True); state["micro"] -= state["micro"] % a.grad_accum
    except KeyboardInterrupt:
        print("\n중단 — 저장한다")
    print("저장 →", save_checkpoint(model, optim, state, a.out, a.keep))


def count_batches(shards, batch_positions):
    """load()/batches() 와 같은 규칙으로 배치 수만 센다. 글 길이는 모르므로 발화당 25토큰으로 잡는다(스케줄용 근사)."""
    n = 0
    for name in shards.names:
        rows = [json.loads(l) for l in open(os.path.join(shards.root, "manifest", name + ".jsonl"), encoding="utf-8")]
        lens = sorted(r["frames"] + 26 for r in rows if r["has_text"] and not any(r["flags"].get(f, 0) for f in shards.drop) and shards.min_frames <= r["frames"] <= shards.max_frames)
        cur = 0
        for L in lens:
            if cur and L * (cur + 1) > batch_positions: n += 1; cur = 0
            cur += 1
        n += bool(cur)
    return n


if __name__ == "__main__":
    main()
