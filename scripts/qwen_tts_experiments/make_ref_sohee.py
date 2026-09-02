import torch, numpy as np, soundfile as sf
from qwen_tts import Qwen3TTSModel

TEXT = ("네, 안녕하세요. 무엇을 도와드릴까요? 예약 관련해서 문의 주셨군요. "
        "제가 확인해 보고 바로 안내해 드리겠습니다. 잠시만 기다려 주시겠어요? "
        "네, 확인이 끝났습니다. 말씀하신 시간으로 처리해 드릴게요. 감사합니다.")

m = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", device_map="cuda:0",
    dtype=torch.bfloat16, attn_implementation="flash_attention_2")
wavs, sr = m.generate_custom_voice(text=TEXT, language="Korean", speaker="Sohee",
                                   temperature=0.7, top_k=20, max_new_tokens=2048)
sf.write("/mnt/c/qwen-out/ref_ai_raw.wav", wavs[0], sr)
print(f"길이 {len(wavs[0])/sr:.1f}s")
