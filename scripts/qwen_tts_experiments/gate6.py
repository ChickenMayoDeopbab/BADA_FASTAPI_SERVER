import time, threading, torch, numpy as np, pathlib
from concurrent.futures import ThreadPoolExecutor
from qwen_tts import Qwen3TTSModel

m = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0", dtype=torch.bfloat16,
    attn_implementation="flash_attention_2")
m.enable_streaming_optimizations()

REF  = str(pathlib.Path.home()/"qwen-test/out2b/APOLOGETIC_2.wav")
RTXT = "아니 그게 아니라, 제가 몇 번을 말씀드렸잖아요."
LINE = "죄송하지만 그건 저희 쪽에서 처리해 드릴 수가 없어요."

def one(tag):
    t0 = time.perf_counter(); first = None; parts = []
    try:
        for pcm, sr in m.stream_generate_voice_clone(
                text=LINE, language="Korean", ref_audio=REF, ref_text=RTXT,
                emit_every_frames=2, temperature=0.7, top_k=20):
            if first is None: first = (time.perf_counter()-t0)*1000
            parts.append(pcm)
        tot = time.perf_counter()-t0
        wav = np.concatenate(parts); dur = len(wav)/sr
        return f"  #{tag} 첫청크 {first:6.0f}ms  총 {tot:5.2f}s  오디오 {dur:5.2f}s  RTF {tot/dur:.2f}"
    except Exception as e:
        return f"  #{tag} 실패: {type(e).__name__}: {e}"

one("warmup")                                  # 그래프 캡처
for n in (1, 2, 3):
    print(f"\n=== 동시 {n}개 ===", flush=True)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as ex:
        for r in ex.map(one, range(n)): print(r, flush=True)
    print(f"  벽시계 총 {time.perf_counter()-t0:.2f}s", flush=True)
