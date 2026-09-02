import torch; torch.set_float32_matmul_precision("high")
import time, torch, numpy as np, soundfile as sf, pathlib
from qwen_tts import Qwen3TTSModel

m = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0", dtype=torch.bfloat16,
    attn_implementation="flash_attention_2")
if hasattr(m, "enable_streaming_optimizations"):
    m.enable_streaming_optimizations()
    print("[opt] enabled", flush=True)

REF_DIR  = pathlib.Path.home() / "qwen-test/out2b"
REF_TEXT = "아니 그게 아니라, 제가 몇 번을 말씀드렸잖아요."
NEW_LINE = "죄송하지만 그건 저희 쪽에서 처리해 드릴 수가 없어요."
REFS = {"APOLOGETIC": "APOLOGETIC_2.wav", "ANGRY": "ANGRY_0.wav",
        "NEUTRAL": "NEUTRAL_0.wav"}          # 귀로 고른 베스트로 교체
out = pathlib.Path("out5"); out.mkdir(exist_ok=True)

for emo, fn in REFS.items():
    for emit in (1, 2, 4):
        for i in range(3):
            t0 = time.perf_counter(); first = None; parts = []
            for pcm, sr in m.stream_generate_voice_clone(
                    text=NEW_LINE, language="Korean",
                    ref_audio=str(REF_DIR / fn), ref_text=REF_TEXT,
                    emit_every_frames=emit, temperature=0.7, top_k=20):
                if first is None:
                    first = (time.perf_counter() - t0) * 1000
                parts.append(pcm)
            total = time.perf_counter() - t0
            wav = np.concatenate(parts); dur = len(wav) / sr
            if i == 0:
                sf.write(out / f"{emo}-e{emit}_0.wav", wav, sr)
            print(f"{emo:11s} emit={emit}  첫청크 {first:6.0f}ms  "
                  f"총 {total:5.2f}s  오디오 {dur:5.2f}s  RTF {total/dur:.2f}", flush=True)
