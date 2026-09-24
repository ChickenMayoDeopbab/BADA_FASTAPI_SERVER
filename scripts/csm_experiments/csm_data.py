# -*- coding: utf-8 -*-
"""미리 뽑은 Mimi 토큰(tok_kspon.py 출력) → CSM 학습 배치.

HF CSM 의 학습 경로(`CsmProcessor` + `_merge_input_ids_with_input_values`)가 **오디오에서** 만드는 것을 **토큰에서** 똑같이 만든다:
  글    `<|begin_of_text|>[화자]글<|end_of_text|>` 를 add_special_tokens=False 로 토큰화
  오디오 `<|AUDIO|>` × 프레임 수 + `<|audio_eos|>` 1개
  임베딩 글 위치 = embed_text_tokens · 오디오 위치 = 32코드북 임베딩의 합 · audio_eos 위치 = 전부 0 인 프레임의 임베딩
  레이블 [B, L, 32]: 글 -100 · 오디오 = 진짜 코드 · audio_eos = 0 프레임
        `depth_decoder_labels_ratio` = 오디오 프레임 중 int(n·(1-ratio)) 개를 무작위로 골라 코드북 1~31 을 -100 (코드북 0 은 남는다).
        audio_eos 프레임은 고르는 대상이 아니다(항상 depth 까지 학습) — processing_csm.py 와 같다.
같음은 verify_inputs.py 가 확인한다.
"""
import glob, json, os, random
import numpy as np, torch

NCB = 32
DEV_SHARDS = {"KsponSpeech_0621", "KsponSpeech_0622", "KsponSpeech_0623"}      # dev.trn 의 620001~622545


def encode_sample(tok, text, codes, speaker=0):
    """글 + 코드 [T,32] → (input_ids 리스트, 오디오 위치 시작, 프레임 수)"""
    ids = tok(f"{tok.bos_token}[{speaker}]{text}{tok.eos_token}", add_special_tokens=False).input_ids
    return ids, len(ids), int(codes.shape[0])


def collate(samples, cfg, pad_id, ratio, rng):
    """samples = [(text_ids, codes[T,32] int64 tensor)] → 오른쪽 패딩 배치."""
    L = max(len(t) + c.shape[0] + 1 for t, c in samples); B = len(samples)
    ids = torch.full((B, L), pad_id, dtype=torch.long); att = torch.zeros(B, L, dtype=torch.long)
    codes = torch.zeros(B, L, NCB, dtype=torch.long); amask = torch.zeros(B, L, dtype=torch.bool)
    labels = torch.full((B, L, NCB), -100, dtype=torch.long); frames = []
    for b, (t, c) in enumerate(samples):
        n, T = len(t), c.shape[0]
        ids[b, :n] = torch.tensor(t); ids[b, n:n + T] = cfg.audio_token_id; ids[b, n + T] = cfg.audio_eos_token_id
        att[b, :n + T + 1] = 1
        codes[b, n:n + T] = c                                   # audio_eos 위치는 0 프레임(codebook_eos_token_id) 그대로
        amask[b, n:n + T + 1] = True
        labels[b, n:n + T] = c; labels[b, n + T] = cfg.codebook_eos_token_id
        frames += [(b, n + k) for k in range(T)]
    if ratio < 1.0:                                             # 배치 전체의 오디오 프레임에서 고른다(HF 와 같은 단위)
        skip = rng.sample(frames, int(len(frames) * (1 - ratio)))
        if skip:
            bi, ti = zip(*skip); labels[list(bi), list(ti), 1:] = -100
    return dict(input_ids=ids, attention_mask=att, codes=codes, audio_mask=amask, labels=labels)


def build_inputs(model, batch):
    """배치 → (inputs_embeds, labels). in-place 대입을 쓰지 않는다(체크포인팅×동결 임베딩 함정을 피한다)."""
    emb = model.embed_text_tokens(batch["input_ids"])
    aud = model.backbone_model.embed_tokens(batch["codes"][batch["audio_mask"]][:, None, :])[:, 0]      # [N, H]
    emb = emb.masked_scatter(batch["audio_mask"][..., None].expand_as(emb), aud.to(emb.dtype))
    return emb, batch["labels"]


class TokenShards:
    """manifest/*.jsonl + codes/*.npz 를 샤드 단위로 읽어 위치 예산(batch_positions) 안에서 배치를 만든다."""

    def __init__(self, root, tok, cfg, text_field="mix", drop_flags=("unknown",), min_frames=6, max_frames=400, split="train"):
        names = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(root, "manifest", "*.jsonl")))
        is_dev = lambda n: n in DEV_SHARDS
        self.names = [n for n in names if (is_dev(n) if split == "dev" else not is_dev(n) and not n.startswith("eval"))]
        self.root, self.tok, self.cfg, self.text_field, self.drop = root, tok, cfg, text_field, set(drop_flags)
        self.min_frames, self.max_frames = min_frames, max_frames

    def load(self, name, rng):
        z = np.load(os.path.join(self.root, "codes", name + ".npz")); codes, off = z["codes"], z["offsets"]
        out = []
        for line in open(os.path.join(self.root, "manifest", name + ".jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if not r["has_text"] or any(r["flags"].get(f, 0) for f in self.drop) or not self.min_frames <= r["frames"] <= self.max_frames:
                continue
            field = self.text_field if self.text_field != "mix" else rng.choice(("spell", "pron"))
            text = r[field].strip()
            if not text:
                continue
            c = torch.from_numpy(codes[:, off[r["idx"]]:off[r["idx"] + 1]].T.astype(np.int64))           # [T, 32]
            out.append((self.tok(f"{self.tok.bos_token}[0]{text}{self.tok.eos_token}", add_special_tokens=False).input_ids, c))
        return out

    def batches(self, batch_positions, ratio, seed, shuffle=True):
        """한 에폭. 샤드 순서·샤드 안 배치 순서를 seed 로 섞는다. (배치, 그 배치의 실제 위치 수) 를 낸다."""
        rng = random.Random(seed); names = list(self.names)
        if shuffle: rng.shuffle(names)
        pad = self.tok.pad_token_id
        for name in names:
            samples = sorted(self.load(name, rng), key=lambda s: len(s[0]) + s[1].shape[0])
            groups, cur = [], []
            for s in samples:                                   # 길이순으로 담아 패딩 낭비를 줄인다
                L = len(s[0]) + s[1].shape[0] + 1
                if cur and L * (len(cur) + 1) > batch_positions:
                    groups.append(cur); cur = []
                cur.append(s)
            if cur: groups.append(cur)
            if shuffle: rng.shuffle(groups)
            for g in groups:
                yield collate(g, self.cfg, pad, ratio, rng)
