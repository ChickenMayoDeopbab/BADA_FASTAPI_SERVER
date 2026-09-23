# -*- coding: utf-8 -*-
"""B단계 학습 — 상담 음성 대화 문맥(직전 턴들 → 목표 턴)으로 A단계 모델을 이어 학습한다. 설계: notes/13-B단계-학습-설계.md.

  입력      tok_ktel.py 출력(--data, 행 = 턴) + 선택으로 tok_kspon.py 출력(--mix-data, 마이크로배치 --mix-every 개마다 1개를 A 배치로)
  예제      csm_data_b.TurnShards: 문맥 오디오 ≤ --ctx-frames, 전체 ≤ --max-positions, 목표 턴은 에폭마다 다른 1/--every
  레이블    문맥 턴 = 코드북 0 만(백본), 목표 턴 = 전부(depth, --ratio) — verify_inputs_b.py 가 HF 다중 턴 경로와 같음을 확인(2026-09-23)
  시작점    --init <체크포인트>: 가중치만(옵티마이저·상태는 새로). --resume <체크포인트>: 가중치+옵티마이저+상태(이어받기)
  검증      dev 샤드(이름에 valid)의 같은 목표 턴을 문맥 있음/없음으로 재서 Δ = 없음 − 있음 을 같이 남긴다(문맥을 쓰기 시작하면 Δ 가 커진다)
  나머지    train_a.py 와 같다(fp32 + bf16 autocast · 8-bit AdamW · 체크포인팅 · 임베딩 묶기 · 250업데이트마다 검증·저장). 에폭 끝마다 epoch_k 링크.

사용 (학교 서버):
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 python train_b.py --data ~/CSM/data/ktel --mix-data ~/CSM/data/kspon --init ~/CSM/runs/a2/last --out ~/CSM/runs/b1 --epochs 4 --keep 4
  ... --resume ~/CSM/runs/b1/last        # 끊기면 이어서
"""
import argparse, json, math, os, random, re, time
import numpy as np, torch
import csm_data as D, csm_data_b as B, train_a as TA


@torch.no_grad()
def evaluate(model, dev, a, device, amp, no_ctx):
    model.eval(); tot = dict(loss=0.0, bb=0.0, dd=0.0); n = ex = 0
    for batch in dev.batches(a.batch_positions // 2, 1.0, seed=0, epoch=0, shuffle=False, no_ctx=no_ctx):     # 같은 목표 턴, 같은 순서
        batch = {k: v.to(device) for k, v in batch.items()}
        with amp():
            emb, lab = D.build_inputs(model, batch)
            o = model(inputs_embeds=emb, attention_mask=batch["attention_mask"], labels=lab, use_cache=False)
        tot["loss"] += o.loss.item(); tot["bb"] += o.backbone_loss.item(); tot["dd"] += o.depth_decoder_loss.item(); n += 1; ex += batch["input_ids"].shape[0]
        if ex >= a.eval_examples: break
    model.train()
    return {k: v / max(1, n) for k, v in tot.items()} | dict(n=ex)


def count_batches(shards, batch_positions, epoch, seed):
    """batches() 와 같은 규칙으로 배치 수를 세고 예제 크기 분포를 돌려준다(코드는 안 읽는다)."""
    n, sizes = 0, []
    for name in shards.names:
        ex = [e for s, ts in shards.load(name, random.Random(seed + 1000 * epoch), with_codes=False) for e in shards.examples(s, ts, epoch, seed)]
        lens = sorted(sum(B.turn_len(t) for t in c) + B.turn_len(g) for c, g in ex); sizes += lens
        cur = 0
        for L in lens:
            if cur and L * (cur + 1) > batch_positions: n += 1; cur = 0
            cur += 1
        n += bool(cur)
    return n, sizes


def main():
    ap = argparse.ArgumentParser(description="CSM B단계 학습 (대화 문맥, 미리 뽑은 토큰)")
    ap.add_argument("--data", required=True, help="tok_ktel.py 출력"); ap.add_argument("--out", required=True)
    ap.add_argument("--mix-data", default=None, help="tok_kspon.py 출력 — 주면 마이크로배치 --mix-every 개마다 1개를 A 배치(ratio 1/16)로")
    ap.add_argument("--mix-every", type=int, default=4)
    ap.add_argument("--repo", default="sesame/csm-1b"); ap.add_argument("--init", default=None, help="가중치만 가져올 체크포인트(예: a2/last)"); ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--text-field", default="mix", choices=["spell", "pron", "mix"]); ap.add_argument("--drop-flags", default="unknown", help="A 배치용(B 는 목표 턴의 unknown 만 뺀다)")
    ap.add_argument("--ctx-frames", type=int, default=1500); ap.add_argument("--max-positions", type=int, default=2048); ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--tag-by", default="role", choices=["role", "random"])
    ap.add_argument("--ratio", type=float, default=1.0, help="목표 턴에서 depth decoder 를 학습할 프레임 비율")
    ap.add_argument("--batch-positions", type=int, default=2048); ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1.5e-5); ap.add_argument("--lr-floor", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100); ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--max-updates", type=int, default=0, help="0 = 첫 에폭 배치 수 × epochs 로 센다")
    ap.add_argument("--optim", default=None, choices=["adamw8bit", "adamw"]); ap.add_argument("--freeze", default=None)
    ap.add_argument("--amp-cache", action="store_true")
    ap.add_argument("--save-every", type=int, default=250); ap.add_argument("--keep", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=250); ap.add_argument("--eval-examples", type=int, default=64, help="검증에 볼 목표 턴 수(문맥 있음·없음 각각)")
    ap.add_argument("--log-every", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    assert 0 < a.ratio <= 1.0 and a.mix_every != 1, "ratio 는 (0,1], mix-every 는 0 또는 2 이상"
    device = torch.device(a.device); cuda = device.type == "cuda"
    os.makedirs(a.out, exist_ok=True); torch.manual_seed(a.seed)
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=a.amp_cache)) if cuda else (lambda: torch.autocast("cpu", enabled=False))

    proc, model = TA.load_model(a.repo, a.resume or a.init or a.repo, device)
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

    train = B.TurnShards(a.data, tok, cfg, a.text_field, "train", a.ctx_frames, a.max_positions, a.every, tag_by=a.tag_by)
    dev = B.TurnShards(a.data, tok, cfg, "spell", "dev", a.ctx_frames, a.max_positions, a.every, tag_by=a.tag_by)
    drop = [f for f in a.drop_flags.split(",") if f]
    mix = D.TokenShards(a.mix_data, tok, cfg, a.text_field, drop, split="train") if a.mix_data else None
    assert train.names, f"{a.data}/manifest 에 학습 샤드가 없다"
    print(f"학습 샤드 {len(train.names)} · 검증 샤드 {len(dev.names)} · 혼합 샤드 {len(mix.names) if mix else 0}(every {a.mix_every if mix else 0}) · 학습 파라미터 {sum(p.numel() for p in params)/1e9:.3f}B · {kind}"
          f" · text={a.text_field} · ratio={a.ratio:.2f} · 문맥 ≤ {a.ctx_frames}프레임 · ≤ {a.max_positions}위치 · 목표 1/{a.every} · 태그 {a.tag_by} · 배치 {a.batch_positions}위치 × accum {a.grad_accum}")
    nb, sizes = count_batches(train, a.batch_positions, state["epoch"], a.seed)
    if sizes: print(f"에폭 {state['epoch']} 예제 {len(sizes):,}개 · 위치 p50/p90/최대 {np.percentile(sizes, 50):.0f}/{np.percentile(sizes, 90):.0f}/{max(sizes)} · B 배치 {nb:,}" + (f" + A 배치 ≈ {nb // (a.mix_every - 1):,}" if mix else ""))
    if not a.max_updates:
        per_epoch = nb + (nb // (a.mix_every - 1) if mix else 0)
        a.max_updates = max(1, a.epochs * per_epoch // a.grad_accum)
    print(f"예정 업데이트 {a.max_updates:,} (warmup {a.warmup}) · lr {a.lr:g} → {a.lr * a.lr_floor:g}")
    log = open(os.path.join(a.out, "log.jsonl"), "a")
    json.dump({k: v for k, v in vars(a).items()}, open(os.path.join(a.out, "args.json"), "w"), ensure_ascii=False, indent=1)

    def epoch_batches(epoch):
        b_it = train.batches(a.batch_positions, a.ratio, seed=a.seed, epoch=epoch)
        a_it = mix.batches(a.batch_positions, 1 / 16, seed=a.seed + 7 * epoch) if mix else iter(())
        return B.mixed(b_it, a_it, a.mix_every if mix else 0)

    def run_eval(u):
        ev = evaluate(model, dev, a, device, amp, False); ev0 = evaluate(model, dev, a, device, amp, True); delta = ev0["loss"] - ev["loss"]
        print(f"        검증 loss {ev['loss']:.4f} (백본 {ev['bb']:.4f} + depth {ev['dd']:.4f}) · 문맥 없음 {ev0['loss']:.4f} · Δ {delta:+.4f} · 목표 {ev['n']}턴", flush=True)
        log.write(json.dumps(dict(update=u, epoch=state["epoch"], val_loss=round(ev["loss"], 4), val_bb=round(ev["bb"], 4), val_dd=round(ev["dd"], 4), val_noctx=round(ev0["loss"], 4), val_delta=round(delta, 4), val_n=ev["n"])) + "\n"); log.flush()

    t0 = time.time(); seen = 0; acc = dict(loss=0.0, bb=0.0, dd=0.0); nk = dict(A=0, B=0)
    try:
        while state["update"] < a.max_updates:
            skip = state["micro_in_epoch"]
            for i, (k, batch) in enumerate(epoch_batches(state["epoch"])):
                if i < skip: continue                           # 이어받기: 같은 seed 라 같은 순서
                batch = {kk: v.to(device, non_blocking=True) for kk, v in batch.items()}
                with amp():
                    emb, lab = D.build_inputs(model, batch)
                    o = model(inputs_embeds=emb, attention_mask=batch["attention_mask"], labels=lab, use_cache=False)
                (o.loss / a.grad_accum).backward()
                for kk, v in (("loss", o.loss), ("bb", o.backbone_loss), ("dd", o.depth_decoder_loss)): acc[kk] += v.item() / a.grad_accum
                nk[k] += 1; seen += int(batch["attention_mask"].sum()); state["micro"] += 1; state["micro_in_epoch"] += 1
                if state["micro"] % a.grad_accum: continue
                lr = TA.lr_at(state["update"], a)
                for g in optim.param_groups: g["lr"] = lr
                gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
                optim.step(); optim.zero_grad(set_to_none=True); state["update"] += 1; u = state["update"]
                if u % a.log_every == 0 or u == 1:
                    el = time.time() - t0; mem = torch.cuda.max_memory_allocated(device) / 1e9 if cuda else 0
                    rec = dict(update=u, epoch=state["epoch"], **{kk: round(v, 4) for kk, v in acc.items()}, nA=nk["A"], nB=nk["B"], grad_norm=round(gn, 3), lr=lr, pos_per_s=round(seen / el), mem_gb=round(mem, 2))
                    print(f"[{u:>6}/{a.max_updates}] loss {acc['loss']:.4f} (백본 {acc['bb']:.4f} + depth {acc['dd']:.4f}) · B {nk['B']}/A {nk['A']} · |g| {gn:.2f} · lr {lr:.2e} · {seen/el:,.0f} 위치/s · {mem:.1f} GB", flush=True)
                    log.write(json.dumps(rec) + "\n"); log.flush()
                acc = dict(loss=0.0, bb=0.0, dd=0.0); nk = dict(A=0, B=0)
                if u % a.eval_every == 0 and dev.names: run_eval(u)
                if u % a.save_every == 0: print("        저장 →", TA.save_checkpoint(model, optim, state, a.out, a.keep), flush=True)
                if u >= a.max_updates: break
            else:
                state["epoch"] += 1; state["micro_in_epoch"] = 0; optim.zero_grad(set_to_none=True); state["micro"] -= state["micro"] % a.grad_accum
                d = TA.save_checkpoint(model, optim, state, a.out, a.keep); link = os.path.join(a.out, f"epoch_{state['epoch']}")
                if os.path.islink(link): os.unlink(link)
                os.symlink(os.path.basename(d), link); print(f"        에폭 {state['epoch']} 끝 → {d} (epoch_{state['epoch']})", flush=True)
                if dev.names: run_eval(state["update"])
    except KeyboardInterrupt:
        print("\n중단 — 저장한다")
    print("저장 →", TA.save_checkpoint(model, optim, state, a.out, a.keep))


if __name__ == "__main__":
    main()
