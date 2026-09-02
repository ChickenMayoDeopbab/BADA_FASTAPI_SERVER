import time, torch, torchaudio, numpy as np, soundfile as sf
from qwen_tts import Qwen3TTSModel

REFS = {
  "ai":   ("/mnt/c/qwen-out/ref_ai.wav",
           "네, 안녕하세요. 무엇을 도와드릴까요? 예약 관련해서 문의 주셨군요. "
           "제가 확인해 보고 바로 안내해 드리겠습니다. 잠시만 기다려 주시겠어요? "
           "네, 확인이 끝났습니다. 말씀하신 시간으로 처리해 드릴게요. 감사합니다."),
  "user": ("/mnt/c/qwen-out/ref_user.wav",
           "네, 안녕하세요. 예약 문의드리려고 전화했는데요. 이번 주 토요일 저녁에 자리가 있을까요? "
           "네 명이서 갈 것 같고요. 시간은 여섯 시쯤 생각하고 있습니다. "
           "혹시 창가 자리도 가능한지 궁금해서요."),
}
DIALOGUE = [
    ("ai",   "네, 안녕하세요. 바다레스토랑입니다. 무엇을 도와드릴까요?"),
    ("user", "안녕하세요, 예약을 하고 싶어서 전화드렸어요."),
    ("ai",   "네, 예약 도와드리겠습니다. 원하시는 날짜와 시간이 어떻게 되세요?"),
    ("user", "이번 주 토요일 저녁 6시에 가능할까요?"),
    ("ai",   "네, 토요일 저녁 6시 가능합니다. 몇 분이 방문하실 예정인가요?"),
    ("user", "어른 4명이요."),
    ("ai",   "네, 확인했습니다. 예약자 성함을 말씀해 주시겠어요?"),
    ("user", "배준하입니다."),
    ("ai",   "배준하 님, 토요일 저녁 6시에 4분 예약 완료됐습니다. 토요일에 뵙겠습니다."),
    ("user", "네, 감사합니다. 안녕히 계세요."),
]
GAP = np.zeros(int(16000 * 0.6), dtype=np.float32)     # 400→600ms (여운 확보)

m = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cuda:0",
    dtype=torch.bfloat16, attn_implementation="flash_attention_2")
m.enable_streaming_optimizations()

t_all = time.perf_counter(); parts = []
for i, (spk, text) in enumerate(DIALOGUE):
    ref, rtxt = REFS[spk]
    t0 = time.perf_counter(); chunks = []
    for pcm, sr in m.stream_generate_voice_clone(
            text=text, language="Korean", ref_audio=ref, ref_text=rtxt,
            emit_every_frames=2, temperature=0.7, top_k=20):
        chunks.append(pcm)
    w24 = np.concatenate(chunks).astype(np.float32)
    w16 = torchaudio.functional.resample(torch.from_numpy(w24), sr, 16000).numpy()
    parts += [w16, GAP]
    print(f"  턴 {i:2d} [{spk:4s}] {time.perf_counter()-t0:6.2f}s  오디오 {len(w16)/16000:5.2f}s", flush=True)

full = np.concatenate(parts)
sf.write("example_qwen.wav", full, 16000)
el = time.perf_counter()-t_all
print(f"\n총 {el:.1f}s / 오디오 {len(full)/16000:.1f}s / RTF {el/(len(full)/16000):.2f}")
