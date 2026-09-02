import torch, numpy as np, soundfile as sf, pathlib
from qwen_tts import Qwen3TTSModel

REF_TEXT = ("네, 안녕하세요. 무엇을 도와드릴까요? 예약 관련해서 문의 주셨군요. "
            "제가 확인해 보고 바로 안내해 드리겠습니다. 잠시만 기다려 주시겠어요? "
            "네, 확인이 끝났습니다. 말씀하신 시간으로 처리해 드릴게요. 감사합니다.")
LINE = "네, 안녕하세요. 바다레스토랑입니다. 무엇을 도와드릴까요?"

m = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0",
    dtype=torch.bfloat16, attn_implementation="flash_attention_2")
m.enable_streaming_optimizations()

out = pathlib.Path("tail3"); out.mkdir(exist_ok=True)
for tag, ref in [("amputated", "/mnt/c/qwen-out/ref_ai.wav"),
                 ("decay",     "/mnt/c/qwen-out/ref_ai_decay.wav")]:
    for i in range(5):
        parts = []
        for pcm, sr in m.stream_generate_voice_clone(
                text=LINE, language="Korean", ref_audio=ref, ref_text=REF_TEXT,
                emit_every_frames=2, temperature=0.7, top_k=20):
            parts.append(pcm)
        w = np.concatenate(parts)
        sf.write(out/f"{tag}_{i}.wav", w, sr)
        print(f"{tag}_{i}  {len(w)/sr:.2f}s", flush=True)
