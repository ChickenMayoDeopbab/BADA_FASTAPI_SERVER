import torch, torchaudio, numpy as np, soundfile as sf, pathlib, json
from qwen_tts import Qwen3TTSModel

V = json.load(open("/mnt/c/qwen-out/voices.json", encoding="utf-8"))["user"]
LINE = "안녕하세요, 예약을 하고 싶어서 전화드렸어요."

m = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0",
    dtype=torch.bfloat16, attn_implementation="flash_attention_2")
m.enable_streaming_optimizations()

out = pathlib.Path("tail4"); out.mkdir(exist_ok=True)
for i in range(4):
    parts = []
    for pcm, sr in m.stream_generate_voice_clone(
            text=LINE, language="Korean",
            ref_audio=V["ref_audio"], ref_text=V["ref_text"],
            emit_every_frames=2, temperature=0.7, top_k=20):
        parts.append(pcm)
    w24 = np.concatenate(parts).astype(np.float32)

    # A: 24kHz 원본 그대로
    sf.write(out/f"{i}_A_raw24k.wav", w24, sr)

    # B: 서버와 똑같이 16k 리샘플 + int16  ← 같은 생성물, 리샘플만 다름
    w16 = torchaudio.functional.resample(torch.from_numpy(w24), sr, 16000).numpy()
    pcm16 = np.clip(w16, -1.0, 1.0).__mul__(32767).astype("<i2")
    sf.write(out/f"{i}_B_resampled16k.wav", pcm16, 16000)

    print(f"{i}  24k {len(w24)/sr:.3f}s  →  16k {len(w16)/16000:.3f}s", flush=True)
