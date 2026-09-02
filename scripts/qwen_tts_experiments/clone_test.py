import time, torch, numpy as np, soundfile as sf, pathlib
from qwen_tts import Qwen3TTSModel

REF      = "/mnt/c/qwen-out/ref_raw.wav"
REF_TEXT = "안녕하세요 프레임 워크의 채근영입니다."      # ← 반드시 수정
LINES = [
    "부엉이 바위서 떨어졌다 야 기분 좋다",
]

m = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0", dtype=torch.bfloat16,
    attn_implementation="flash_attention_2")
m.enable_streaming_optimizations()

out = pathlib.Path("clone_out"); out.mkdir(exist_ok=True)
for li, line in enumerate(LINES):
    for i in range(3):
        t0 = time.perf_counter(); first = None; parts = []
        for pcm, sr in m.stream_generate_voice_clone(
                text=line, language="Korean", ref_audio=REF, ref_text=REF_TEXT,
                emit_every_frames=2, temperature=0.7, top_k=20):
            if first is None: first = (time.perf_counter()-t0)*1000
            parts.append(pcm)
        wav = np.concatenate(parts)
        sf.write(out / f"L{li}_{i}.wav", wav, sr)
        print(f"L{li} #{i}  첫청크 {first:5.0f}ms  오디오 {len(wav)/sr:.2f}s", flush=True)
