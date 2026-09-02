import torch, numpy as np, soundfile as sf, pathlib
from qwen_tts import Qwen3TTSModel

REF      = "/mnt/c/qwen-out/ai_female.wav"
REF_TEXT = ("네, 안녕하세요. 오늘 날씨가 참 좋네요. 예약 관련해서 문의 주신 것 같은데요, "
            "제가 확인해 보고 바로 안내해 드리겠습니다. 잠시만 기다려 주시겠어요? "
            "네, 확인이 끝났습니다. 말씀하신 시간으로 처리해 드릴게요.")
LINE = "배준하 님, 토요일 저녁 6시에 4분 예약 완료됐습니다. 토요일에 뵙겠습니다."

m = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0",
    dtype=torch.bfloat16, attn_implementation="flash_attention_2")
m.enable_streaming_optimizations()

out = pathlib.Path("tail"); out.mkdir(exist_ok=True)
kw = dict(text=LINE, language="Korean", ref_audio=REF, ref_text=REF_TEXT,
          temperature=0.7, top_k=20)

# 비스트리밍 = 기준선
wavs, sr = m.generate_voice_clone(**kw)
base = len(wavs[0]) / sr
sf.write(out/"nostream.wav", wavs[0], sr)
print(f"비스트리밍      {base:6.3f}s   (기준)")

for emit in (1, 2, 4, 8):
    parts = []
    for pcm, s in m.stream_generate_voice_clone(**kw, emit_every_frames=emit):
        parts.append(pcm)
    w = np.concatenate(parts); d = len(w) / s
    sf.write(out/f"emit{emit}.wav", w, s)
    print(f"emit={emit:<2}         {d:6.3f}s   차이 {base-d:+.3f}s")
