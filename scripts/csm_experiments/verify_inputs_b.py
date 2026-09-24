# -*- coding: utf-8 -*-
"""csm_data_b 가 만드는 다중 턴 입력·레이블이 HF 경로(apply_chat_template + 설계 §5 의 −101 처방 + _merge)와 같은지. CPU fp32, 합성 잡음 3턴.
  HF_HUB_OFFLINE=1 python verify_inputs_b.py
"""
import random, sys
import numpy as np, torch
from transformers import AutoProcessor, CsmForConditionalGeneration, MimiModel
import csm_data as D, csm_data_b as B

torch.manual_seed(0); nr = np.random.default_rng(0)
clips = [(nr.standard_normal(int(s * 24000)) * 0.05).astype(np.float32) for s in (2.0, 3.5, 1.5)]       # 합성 잡음 3턴 — 길이만 다르면 된다
texts = ["안녕하세요, 무엇을 도와드릴까요?", "어 제가 어제 주문한 게 아직 안 왔어요.", "네, 확인해 드리겠습니다."]; roles = ["0", "1", "0"]
proc = AutoProcessor.from_pretrained("sesame/csm-1b"); tok = proc.tokenizer
model = CsmForConditionalGeneration.from_pretrained("sesame/csm-1b", dtype=torch.float32).eval()
model.backbone_model.embed_tokens.embed_audio_tokens.weight = model.depth_decoder.model.embed_tokens.weight
mimi = MimiModel.from_pretrained("kyutai/mimi").eval(); cfg = model.config
AUD, AEOS = cfg.audio_token_id, cfg.audio_eos_token_id
with torch.no_grad():
    codes = [mimi.encode(torch.from_numpy(c)[None, None]).audio_codes[0].T.long() for c in clips]
ok = True
def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(f"  {'✓' if cond else '✗'} {name} {extra}")

conv = [{"role": r, "content": [{"type": "text", "text": t}, {"type": "audio", "path": c}]} for r, t, c in zip(roles, texts, clips)]
hf = proc.apply_chat_template(conv, tokenize=True, return_dict=True, processor_kwargs={"output_labels": True, "depth_decoder_labels_ratio": 1.0})
lab2d = hf["labels"].clone(); eos = (hf["input_ids"][0] == AEOS).nonzero().flatten(); cut = eos[-2].item()
m = torch.zeros_like(lab2d, dtype=torch.bool); m[0, :cut + 1] = True; lab2d[m & (lab2d == AUD)] = -101              # 설계 §5 처방: 앞 턴의 오디오 프레임만 −101
turns = [dict(ids=B.turn_ids(tok, int(r), t), codes=c, frames=c.shape[0], tag=int(r), role="", has_text=True, target_ok=True) for r, t, c in zip(roles, texts, codes)]
mine = B.collate_b([(turns[:2], turns[2])], cfg, tok.pad_token_id, 1.0, random.Random(0))
print(f"[3턴 대화] HF {hf['input_ids'].shape[1]}위치 · 내 경로 {mine['input_ids'].shape[1]}위치 · 프레임 {[c.shape[0] for c in codes]}")
check("input_ids 동일(턴 사이 구분 토큰 없음)", torch.equal(hf["input_ids"], mine["input_ids"]))
if not torch.equal(hf["input_ids"], mine["input_ids"]):
    print("   HF :", hf["input_ids"][0].tolist()[:40]); print("   내 :", mine["input_ids"][0].tolist()[:40])
with torch.no_grad():
    mg = model._merge_input_ids_with_input_values(hf["input_ids"], hf["input_values"], hf["input_values_cutoffs"], lab2d)
    emb, lab = D.build_inputs(model, mine)
    check("inputs_embeds 동일", torch.equal(mg["inputs_embeds"], emb), f"(최대 차이 {(mg['inputs_embeds'] - emb).abs().max():.2e})")
    check("labels [L,32] 동일(문맥 = 코드북 0 만, 문맥 eos 전부 0)", torch.equal(mg["labels"], lab))
    a = model(inputs_embeds=mg["inputs_embeds"], attention_mask=hf["attention_mask"], labels=mg["labels"]); b = model(inputs_embeds=emb, attention_mask=mine["attention_mask"], labels=lab)
    check("loss 동일", torch.allclose(a.loss, b.loss, atol=1e-5), f"HF {a.loss:.5f} = 백본 {a.backbone_loss:.5f} + depth {a.depth_decoder_loss:.5f} · 내 경로 {b.loss:.5f}")
depth_rows = (~(lab[0, :, 1:] == -100).all(-1)).sum().item(); bb_rows = (lab[0, :, 0] != -100).sum().item(); nf = sum(c.shape[0] for c in codes)
check("백본 레이블 = 전 프레임 + eos 3", bb_rows == nf + 3, f"({bb_rows})")
check("depth 레이블 = 목표 프레임 + eos 3", depth_rows == codes[2].shape[0] + 3, f"({depth_rows})")
single = proc.apply_chat_template(conv[2:], tokenize=True, return_dict=True, processor_kwargs={"output_labels": True, "depth_decoder_labels_ratio": 1.0})
mine1 = B.collate_b([([], turns[2])], cfg, tok.pad_token_id, 1.0, random.Random(0))
check("문맥 없음 예제 = HF 단일 턴 input_ids", torch.equal(single["input_ids"], mine1["input_ids"]))
print("\n전부 통과 ✓" if ok else "\n실패한 항목이 있다 ✗"); sys.exit(0 if ok else 1)
