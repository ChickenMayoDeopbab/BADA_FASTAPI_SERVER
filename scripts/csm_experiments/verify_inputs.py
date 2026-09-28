# -*- coding: utf-8 -*-
"""csm_data 가 만드는 입력이 HF 경로(오디오 → CsmProcessor → _merge)와 **같은지** 확인한다. CPU fp32 로 돈다.
  python verify_inputs.py <24kHz wav> [<24kHz wav 2>]
"""
import random, sys
import soundfile as sf, torch
from transformers import AutoProcessor, CsmForConditionalGeneration, MimiModel
import csm_data as D

wavs = [sf.read(p, dtype="float32")[0] for p in sys.argv[1:3]]
clips = [wavs[0][:6 * 24000], wavs[-1][24000:int(5.5 * 24000)]]                # 길이가 다른 두 발화
texts = ["여보세요, [숨] 저기 2시에 예약을 했는데요.", "어 김민수요."]
proc = AutoProcessor.from_pretrained("sesame/csm-1b"); tok = proc.tokenizer
model = CsmForConditionalGeneration.from_pretrained("sesame/csm-1b", dtype=torch.float32).eval()
model.backbone_model.embed_tokens.embed_audio_tokens.weight = model.depth_decoder.model.embed_tokens.weight
mimi = MimiModel.from_pretrained("kyutai/mimi").eval(); cfg = model.config
with torch.no_grad():
    codes = [mimi.encode(torch.from_numpy(c)[None, None]).audio_codes[0].T.long() for c in clips]      # [T,32] — tok_kspon 과 같은 방식

ok = True
def check(name, cond, extra=""):
    global ok; ok &= bool(cond); print(f"  {'✓' if cond else '✗'} {name} {extra}")

for i in range(2):                                                             # ── 발화 하나씩: HF 와 1:1 ──
    conv = [{"role": "0", "content": [{"type": "text", "text": texts[i]}, {"type": "audio", "path": clips[i]}]}]
    hf = proc.apply_chat_template(conv, tokenize=True, return_dict=True, processor_kwargs={"output_labels": True, "depth_decoder_labels_ratio": 1.0})
    ids, _, _ = D.encode_sample(tok, texts[i], codes[i])
    mine = D.collate([(ids, codes[i])], cfg, tok.pad_token_id, 1.0, random.Random(0))
    print(f"[발화 {i+1}] {hf['input_ids'].shape[1]}위치 (오디오 {codes[i].shape[0]}프레임)")
    check("input_ids 동일", torch.equal(hf["input_ids"], mine["input_ids"]))
    with torch.no_grad():
        m = model._merge_input_ids_with_input_values(hf["input_ids"], hf["input_values"], hf["input_values_cutoffs"], hf["labels"])
        emb, lab = D.build_inputs(model, mine)
        check("inputs_embeds 동일", torch.equal(m["inputs_embeds"], emb), f"(최대 차이 {(m['inputs_embeds']-emb).abs().max():.2e})")
        check("labels [L,32] 동일", torch.equal(m["labels"], lab))
        a = model(**hf); b = model(inputs_embeds=emb, attention_mask=mine["attention_mask"], labels=lab)
        check("loss 동일", torch.allclose(a.loss, b.loss, atol=1e-5), f"HF {a.loss:.5f} = 백본 {a.backbone_loss:.5f} + depth {a.depth_decoder_loss:.5f} · 내 경로 {b.loss:.5f}")

print("[배치 2개] HF 는 왼쪽 패딩, 내 배치는 오른쪽 패딩 — loss 가 같아야 한다")
convs = [[{"role": "0", "content": [{"type": "text", "text": texts[i]}, {"type": "audio", "path": clips[i]}]}] for i in range(2)]
hf = proc.apply_chat_template(convs, tokenize=True, return_dict=True, processor_kwargs={"output_labels": True, "depth_decoder_labels_ratio": 1.0})
mine = D.collate([(D.encode_sample(tok, texts[i], codes[i])[0], codes[i]) for i in range(2)], cfg, tok.pad_token_id, 1.0, random.Random(0))
with torch.no_grad():
    a = model(**hf); emb, lab = D.build_inputs(model, mine); b = model(inputs_embeds=emb, attention_mask=mine["attention_mask"], labels=lab)
check("배치 loss 동일", torch.allclose(a.loss, b.loss, atol=1e-4), f"HF {a.loss:.5f} · 내 경로 {b.loss:.5f}")

print("[ratio] 1/16 이면 depth 학습 프레임이 줄고 audio_eos 는 항상 남는다")
mine = D.collate([(D.encode_sample(tok, texts[i], codes[i])[0], codes[i]) for i in range(2)], cfg, tok.pad_token_id, 1 / 16, random.Random(0))
lab = mine["labels"]; n_frames = sum(c.shape[0] for c in codes)
depth_rows = (~(lab[:, :, 1:] == -100).all(-1)).sum().item(); bb_rows = (lab[:, :, 0] != -100).sum().item()
check("백본 레이블 = 전 프레임 + eos 2개", bb_rows == n_frames + 2, f"({bb_rows})")
check("depth 레이블 = 남긴 프레임 + eos 2개", depth_rows == n_frames - int(n_frames * (15 / 16)) + 2, f"({depth_rows} / {n_frames + 2})")
print("\n전부 통과 ✓" if ok else "\n실패한 항목이 있다 ✗"); sys.exit(0 if ok else 1)
