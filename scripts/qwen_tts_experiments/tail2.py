import torch, numpy as np, soundfile as sf, pathlib
from qwen_tts import Qwen3TTSModel

REF_TEXT = ("네, 안녕하세요. 오늘 날씨가 참 좋네요. 예약 관련해서 문의 주신 것 같은데요, "
            "제가 확인해 보고 바로 안내해 드리겠습니다. 잠시만 기다려 주시겠어요? "
            "네, 확인이 끝났습니다. 말씀하신 시간으로 처리해 드릴게요.")
LINE = "배준하 님, 토요일 저녁 6시에 4분 예약 완료됐습니다. 토요일에 뵙겠습니다."

m = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0",
    dtype=torch.bfloat16, attn_implementation="flash_attention_2")
m.enable_streaming_optimizations()

out = pathlib.Path("tail2"); out.mkdir(exist_ok=True)
for tag, ref in [("plain", "/mnt/c/qwen-out/ai_female.wav"),
                 ("pad",   "/mnt/c/qwen-out/ai_female_pad.wav")]:
    for i in range(2):
        parts = []
        for pcm, sr in m.stream_generate_voice_clone(
                text=LINE, language="Korean", ref_audio=ref, ref_text=REF_TEXT,
                emit_every_frames=2, temperature=0.7, top_k=20):
            parts.append(pcm)
        w = np.concatenate(parts)
        sf.write(out/f"{tag}_{i}.wav", w, sr)
        print(f"{tag}_{i}  {len(w)/sr:.3f}s", flush=True)
