# -*- coding: utf-8 -*-
"""G1b 세트 — 검증 D60 대화(tok_ktel 샤드)에서 **상담원 목표 턴** n개와 문맥(120 초·60 초), 같은 화자의 참조 턴을 고르고 코덱 wav 를 낸다. 서버에서 돈다.

  후보  역할 상담원 · has_text · 3~10 초 · 글 8자 이상(숫자/라틴/괄호 없음) · 표기 laugh/overlap/unknown/repeat/unclear/dual/breath 없음(noise·filler 는 허용 — 상담 음성은 거의 전 턴이 n/)
        · 앞 문맥 오디오 ≥ 60 초 · 120 초 창 안에 같은 화자의 이전 턴 ≥ 2 초(SIM 참조)
  문맥  뒤에서부터 오디오 ≤ 1,500프레임, 위치(글+프레임+eos) ≤ 2,048 − 250(생성 20 초) − 목표 글 → 턴 통째로. ctx60 은 그 안에서 ≤ 750프레임 몫.
  출력  set.jsonl(id, target, ref, ctx120, ctx60, infer_text, prompt_text) · codes.npz(shard:idx → int16 [32,T]) · meta.lst(G1 형식) · mimi/<id>.wav(목표 코드 디코드 = 코덱 천장) · prompt/<id>.wav(참조 턴 디코드 = SIM 참조)
        human/<id>.wav 는 집 PC 의 g1b_human.py 가 원본 zip 에서 만든다.

  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=3 python g1b_pick.py --data ~/CSM/data/ktel --out ~/CSM/g1b/set --n 100
"""
import argparse, glob, json, os, random, re, wave
import numpy as np, torch

BAD = ("laugh", "overlap", "unknown", "repeat", "unclear", "dual", "breath")
SR, GEN_FRAMES, MAX_POS = 24000, 250, 2048


def ok_text(t):
    return not re.search(r"[0-9A-Za-z\[\]|]", t) and sum("가" <= c <= "힣" for c in t) >= 8


def text_ids(tok, text, speaker):
    return tok(f"{tok.bos_token}[{speaker}]{text}{tok.eos_token}", add_special_tokens=False).input_ids


def load_dev(root, pattern):
    """검증 샤드(이름에 pattern) → [(session, [turn …])]. turn 에 shard·idx 가 있어 codes.npz 에서 코드를 찾는다."""
    sessions = []
    for p in sorted(glob.glob(os.path.join(root, "manifest", "*.jsonl"))):
        name = os.path.splitext(os.path.basename(p))[0]
        if pattern not in name: continue
        cur, key = [], None
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if r["session"] != key:
                if cur: sessions.append((key, cur))
                key, cur = r["session"], []
            cur.append(dict(shard=name, idx=r["idx"], role=r["role"], spk_id=r["spk_id"], spell=r["spell"].strip(), pron=r["pron"].strip(), frames=int(r["frames"]),
                            flags=r["flags"], has_text=bool(r["has_text"]), utts=r["utts"], dur_s=r["dur_s"], tag=0 if r["role"] == "상담원" else 1))
        if cur: sessions.append((key, cur))
    return sessions


def candidates(session, turns, tok, lo=38, hi=125, ctx_min=750, ref_min=25):
    out = []
    for i, t in enumerate(turns):
        if not (t["role"] == "상담원" and t["has_text"] and lo <= t["frames"] <= hi and ok_text(t["spell"]) and not any(t["flags"].get(k, 0) for k in BAD)): continue
        if sum(u["frames"] for u in turns[:i]) < ctx_min: continue
        budget = MAX_POS - GEN_FRAMES - len(text_ids(tok, t["spell"], t["tag"])); ctx, frames, pos = [], 0, 0
        for u in reversed(turns[:i]):
            L = len(text_ids(tok, u["spell"], u["tag"])) + u["frames"] + 1
            if frames + u["frames"] > 1500 or pos + L > budget: break
            ctx.append(u); frames += u["frames"]; pos += L
        ctx = ctx[::-1]
        refs = [u for u in ctx if u["spk_id"] == t["spk_id"] and u["frames"] >= ref_min]
        if not refs or sum(u["frames"] for u in ctx) < ctx_min: continue
        c60, f = [], 0
        for u in reversed(ctx):
            if f + u["frames"] > 750: break
            c60.append(u); f += u["frames"]
        out.append(dict(session=session, i=i, target=t, ctx120=ctx, ctx60=c60[::-1], ref=refs[-1]))
    return out


def main():
    ap = argparse.ArgumentParser(description="G1b 세트 뽑기")
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--pattern", default="valid_D60"); ap.add_argument("--n", type=int, default=100); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-session", type=int, default=1); ap.add_argument("--repo", default="sesame/csm-1b"); ap.add_argument("--mimi", default="kyutai/mimi")
    ap.add_argument("--no-audio", action="store_true", help="시험용: 코덱 wav 를 안 만든다"); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    from transformers import AutoProcessor
    tok = AutoProcessor.from_pretrained(a.repo).tokenizer
    sessions = load_dev(os.path.expanduser(a.data), a.pattern); cand = [c for s, ts in sessions for c in candidates(s, ts, tok)]
    n_sess = len({c["session"] for c in cand}); print(f"검증 세션 {len(sessions)} · 후보 {len(cand)}개(세션 {n_sess}개)")
    rng = random.Random(a.seed); rng.shuffle(cand); picked, per = [], {}
    for c in cand:
        if per.get(c["session"], 0) >= a.per_session: continue
        picked.append(c); per[c["session"]] = per.get(c["session"], 0) + 1
        if len(picked) >= a.n: break
    if len(picked) < a.n and a.per_session == 1:
        for c in cand:
            if len(picked) >= a.n: break
            if c not in picked and per.get(c["session"], 0) < 2: picked.append(c); per[c["session"]] += 1
    picked.sort(key=lambda c: (c["session"], c["i"])); print(f"고른 목표 {len(picked)}개 · 세션 {len(per)}개" + (" · 부족" if len(picked) < a.n else ""))
    out = os.path.expanduser(a.out); os.makedirs(out, exist_ok=True)
    shards, need = {}, set()
    for c in picked:
        for u in c["ctx120"] + [c["target"], c["ref"]]: need.add((u["shard"], u["idx"]))
    codes = {}
    for shard, idx in need:
        if shard not in shards: shards[shard] = np.load(os.path.join(os.path.expanduser(a.data), "codes", shard + ".npz"))
        z = shards[shard]; codes[f"{shard}:{idx}"] = z["codes"][:, z["offsets"][idx]:z["offsets"][idx + 1]].astype(np.int16)
    np.savez(os.path.join(out, "codes.npz"), **codes)
    slim = lambda u: dict(shard=u["shard"], idx=u["idx"], tag=u["tag"], spell=u["spell"], frames=u["frames"])
    with open(os.path.join(out, "set.jsonl"), "w", encoding="utf-8") as f, open(os.path.join(out, "meta.lst"), "w", encoding="utf-8") as m:
        for c in picked:
            t = c["target"]; sid = f"{c['session'].split('/')[-1]}_t{c['i']:03d}"
            it = dict(id=sid, session=c["session"], turn=c["i"], tag=t["tag"], infer_text=t["spell"], prompt_text=c["ref"]["spell"],
                      target=dict(**slim(t), pron=t["pron"], utts=t["utts"], dur_s=t["dur_s"], spk_id=t["spk_id"]), ref=slim(c["ref"]),
                      ctx120=[slim(u) for u in c["ctx120"]], ctx60=[slim(u) for u in c["ctx60"]],
                      ctx120_s=round(sum(u["frames"] for u in c["ctx120"]) / 12.5, 1), ctx60_s=round(sum(u["frames"] for u in c["ctx60"]) / 12.5, 1))
            f.write(json.dumps(it, ensure_ascii=False) + "\n"); m.write(f"{sid}|{c['ref']['spell']}|prompt/{sid}.wav|{t['spell']}|human/{sid}.wav\n")
    print(f"→ {out}/set.jsonl · meta.lst · codes.npz({len(codes)}턴) · 문맥 120 초 창 평균 {np.mean([sum(u['frames'] for u in c['ctx120']) for c in picked]) / 12.5:.0f} s · 턴 수 평균 {np.mean([len(c['ctx120']) for c in picked]):.1f}")
    if a.no_audio: return
    dev = torch.device(a.device)
    try:
        from transformers import MimiModel
        mimi = MimiModel.from_pretrained(a.mimi).to(dev).eval()
    except OSError:                                                  # 서버엔 kyutai/mimi 캐시가 없다 → sesame/csm-1b 안의 코덱(비트 단위로 같음, 2026-09-21 확인)
        from transformers import CsmForConditionalGeneration
        print(f"{a.mimi} 캐시 없음 → {a.repo} 의 codec_model 사용"); mimi = CsmForConditionalGeneration.from_pretrained(a.repo, dtype=torch.float32).codec_model.to(dev).eval()
    for sub in ("mimi", "prompt"): os.makedirs(os.path.join(out, sub), exist_ok=True)
    with torch.no_grad():
        for c in picked:
            sid = f"{c['session'].split('/')[-1]}_t{c['i']:03d}"
            for sub, u in (("mimi", c["target"]), ("prompt", c["ref"])):
                x = mimi.decode(torch.from_numpy(codes[f"{u['shard']}:{u['idx']}"].astype(np.int64))[None].to(dev)).audio_values[0, 0].float().cpu()
                with wave.open(os.path.join(out, sub, sid + ".wav"), "wb") as w:
                    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((x.clamp(-1, 1) * 32767).to(torch.int16).numpy().astype("<i2").tobytes())
    print(f"코덱 wav {len(picked)}×2개 → {out}/mimi · {out}/prompt")


if __name__ == "__main__":
    main()
