# -*- coding: utf-8 -*-
"""tok_ktel 샤드(행 = 턴) → CSM B단계 배치: 직전 턴들(글+오디오) → 목표 턴. 설계: notes/13-B단계-학습-설계.md §3~5.

  예제   목표 턴 i(≥1, has_text·unknown 없음·6~400프레임) + 직전 턴들(뒤에서부터, 오디오 합 ≤ ctx_frames · 위치 합 ≤ max_positions, 턴 통째로)
  추출   crc32(f"{session}/{i}/{seed}") % every == epoch % every  → 에폭마다 다른 1/every, every 에폭이면 전부
  시퀀스 턴마다 `<bos>[태그]글<eos>` + <AUDIO>×T + <audio_eos>, 턴 사이 구분 토큰 없음(verify_inputs_b.py 가 HF 다중 턴 템플릿과 대조)
  레이블 [B,L,32]: 글 −100 · 문맥 오디오 프레임 코드북 0 만(1~31 은 −100 = HF 의 −101) · 문맥 audio_eos 전부 0 · 목표 프레임 전부(ratio) · 목표 audio_eos 전부 0
  태그   role 로 상담원 0 · 고객 1 (tag_by="random" 이면 세션마다 0/1 뒤집기)
배치 키 input_ids attention_mask codes audio_mask labels 는 csm_data.build_inputs 가 그대로 쓴다. target_mask 는 목표 턴의 오디오+eos 위치.
"""
import glob, json, os, random, zlib
import numpy as np, torch
from csm_data import NCB


def turn_ids(tok, tag, text):
    return tok(f"{tok.bos_token}[{tag}]{text}{tok.eos_token}", add_special_tokens=False).input_ids


def turn_len(t):
    return len(t["ids"]) + t["frames"] + 1


def select_targets(turns, session, epoch, seed, every):
    """목표 턴 인덱스(1 이상, target_ok). every ≥ 2 면 에폭마다 다른 1/every."""
    return [i for i in range(1, len(turns)) if turns[i]["target_ok"] and (every <= 1 or zlib.crc32(f"{session}/{i}/{seed}".encode()) % every == epoch % every)]


def window(turns, i, ctx_frames, max_positions):
    """턴 i 의 문맥: 직전 턴들을 뒤에서부터 예산 안까지(오래된 것부터 돌려준다). 턴은 통째로만."""
    out, frames, pos = [], 0, turn_len(turns[i])
    for t in reversed(turns[:i]):
        if frames + t["frames"] > ctx_frames or pos + turn_len(t) > max_positions:
            break
        out.append(t); frames += t["frames"]; pos += turn_len(t)
    return out[::-1]


def collate_b(examples, cfg, pad_id, ratio, rng):
    """examples = [(문맥 턴들, 목표 턴)] → 오른쪽 패딩 배치. ratio 는 목표 턴 프레임에만 적용(문맥은 항상 코드북 0 만)."""
    L = max(sum(turn_len(t) for t in c) + turn_len(g) for c, g in examples); Bn = len(examples)
    ids = torch.full((Bn, L), pad_id, dtype=torch.long); att = torch.zeros(Bn, L, dtype=torch.long)
    codes = torch.zeros(Bn, L, NCB, dtype=torch.long); amask = torch.zeros(Bn, L, dtype=torch.bool); tmask = torch.zeros(Bn, L, dtype=torch.bool)
    labels = torch.full((Bn, L, NCB), -100, dtype=torch.long); frames = []
    for b, (ctx, tgt) in enumerate(examples):
        p = 0
        for t in ctx + [tgt]:
            n, T, is_t = len(t["ids"]), t["frames"], t is tgt
            ids[b, p:p + n] = torch.tensor(t["ids"]); ids[b, p + n:p + n + T] = cfg.audio_token_id; ids[b, p + n + T] = cfg.audio_eos_token_id
            codes[b, p + n:p + n + T] = t["codes"]; amask[b, p + n:p + n + T + 1] = True          # audio_eos 위치는 0 프레임(codebook_eos) 그대로
            labels[b, p + n:p + n + T] = t["codes"]; labels[b, p + n + T] = cfg.codebook_eos_token_id
            if is_t:
                tmask[b, p + n:p + n + T + 1] = True; frames += [(b, p + n + k) for k in range(T)]
            else:
                labels[b, p + n:p + n + T, 1:] = -100                                                 # 문맥: 백본(코드북 0)만. eos 프레임은 그대로(HF 처방과 같다)
            p += n + T + 1
        att[b, :p] = 1
    if ratio < 1.0:
        skip = rng.sample(frames, int(len(frames) * (1 - ratio)))
        if skip:
            bi, ti = zip(*skip); labels[list(bi), list(ti), 1:] = -100
    return dict(input_ids=ids, attention_mask=att, codes=codes, audio_mask=amask, labels=labels, target_mask=tmask)


def mixed(b_iter, a_iter, every):
    """("B", 배치) 를 every−1 개 낼 때마다 ("A", 배치) 하나. every 는 0(안 섞음) 또는 2 이상. A 가 바닥나면 B 만 낸다."""
    n = 0
    for b in b_iter:
        yield "B", b; n += 1
        if every > 1 and n % (every - 1) == 0:
            a = next(a_iter, None)
            if a is not None:
                yield "A", a


class TurnShards:
    """manifest/*.jsonl + codes/*.npz(행 = 턴) 를 세션 단위로 읽어 문맥 창 예제 배치를 만든다. dev = 이름에 'valid' 가 든 샤드."""

    def __init__(self, root, tok, cfg, text_field="mix", split="train", ctx_frames=1500, max_positions=2048, every=4, min_frames=6, max_frames=400, tag_by="role"):
        names = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(root, "manifest", "*.jsonl")))
        self.names = [n for n in names if ("valid" in n) == (split == "dev")]
        self.root, self.tok, self.cfg, self.text_field, self.tag_by = root, tok, cfg, text_field, tag_by
        self.ctx_frames, self.max_positions, self.every, self.min_frames, self.max_frames = ctx_frames, max_positions, every, min_frames, max_frames

    def load(self, name, rng, with_codes=True):
        """→ [(세션 키, [턴 dict …])]. 턴 dict: ids codes(Tensor[T,32] | None) frames tag role has_text target_ok"""
        z = np.load(os.path.join(self.root, "codes", name + ".npz")) if with_codes else None
        sessions, cur, key, flip = [], [], None, 0
        for line in open(os.path.join(self.root, "manifest", name + ".jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if r["session"] != key:
                if cur: sessions.append((key, cur))
                key, cur, flip = r["session"], [], int(rng.random() < 0.5) if self.tag_by == "random" else 0
            field = self.text_field if self.text_field != "mix" else rng.choice(("spell", "pron")); text = r[field].strip()
            tag = (0 if r["role"] == "상담원" else 1) ^ flip; has_text = bool(r["has_text"]) and bool(text)
            c = torch.from_numpy(z["codes"][:, z["offsets"][r["idx"]]:z["offsets"][r["idx"] + 1]].T.astype(np.int64)) if with_codes else None
            cur.append(dict(ids=turn_ids(self.tok, tag, text), codes=c, frames=int(r["frames"]), tag=tag, role=r["role"], has_text=has_text,
                            target_ok=has_text and not r["flags"].get("unknown", 0) and self.min_frames <= r["frames"] <= self.max_frames))
        if cur: sessions.append((key, cur))
        return sessions

    def examples(self, session, turns, epoch, seed, no_ctx=False):
        out = []
        for i in select_targets(turns, session, epoch, seed, self.every):
            ctx = [] if no_ctx else [t for t in window(turns, i, self.ctx_frames, self.max_positions) if t["has_text"]]
            if ctx or no_ctx:
                out.append((ctx, turns[i]))
        return out

    def batches(self, batch_positions, ratio, seed, epoch=0, shuffle=True, no_ctx=False):
        """한 에폭. 샤드 순서·배치 순서는 seed+epoch 로 섞는다. 길이순으로 담아 패딩 낭비를 줄인다."""
        rng = random.Random(seed + 1000 * epoch); names = list(self.names)
        if shuffle: rng.shuffle(names)
        for name in names:
            ex = [e for s, ts in self.load(name, rng) for e in self.examples(s, ts, epoch, seed, no_ctx)]
            if shuffle: ex.sort(key=lambda e: sum(turn_len(t) for t in e[0]) + turn_len(e[1]))       # 학습: 길이순(패딩 절약). 검증(shuffle=False): 세션·턴 순서 그대로 → 문맥 있음/없음이 같은 목표 턴을 같은 순서로 본다
            groups, cur = [], []
            for e in ex:
                L = sum(turn_len(t) for t in e[0]) + turn_len(e[1])
                if cur and L * (len(cur) + 1) > batch_positions:
                    groups.append(cur); cur = []
                cur.append(e)
            if cur: groups.append(cur)
            if shuffle: rng.shuffle(groups)
            for g in groups:
                yield collate_b(g, self.cfg, self.tok.pad_token_id, ratio, rng)
