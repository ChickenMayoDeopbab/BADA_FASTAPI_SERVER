"""E3 — 한 프로세스 안에서 모델 인스턴스를 GPU 별로 따로 두면 동시 생성이 되는지. 계획 0047.

gate6 은 인스턴스 하나를 스레드들이 공유해 죽었다. 여기서는 스레드마다 자기 인스턴스를 준다.
성공하면 단일 서버 프로세스가 GPU N 장을 쓸 수 있어 배포가 단순해진다.

사용법: CUDA_VISIBLE_DEVICES=1,2 python conc_e3.py
"""

import pathlib
import threading
import time

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel

REF = str(pathlib.Path.home() / "qwen-test/out2b/APOLOGETIC_2.wav")
RTXT = "아니 그게 아니라, 제가 몇 번을 말씀드렸잖아요."
LINE = "죄송하지만 그건 저희 쪽에서 처리해 드릴 수가 없어요."


def load(device: str) -> Qwen3TTSModel:
    print(f"[{device}] 모델 로드", flush=True)
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map=device, dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.enable_streaming_optimizations()
    return model


def generate(model: Qwen3TTSModel, device: str, out: dict) -> None:
    t0 = time.perf_counter()
    first = None
    parts = []
    sr = 0
    try:
        for pcm, chunk_sr in model.stream_generate_voice_clone(
            text=LINE, language="Korean", ref_audio=REF, ref_text=RTXT,
            emit_every_frames=2, temperature=0.7, top_k=20,
        ):
            if first is None:
                first = (time.perf_counter() - t0) * 1000
            parts.append(pcm)
            sr = chunk_sr
        total = time.perf_counter() - t0
        dur = len(np.concatenate(parts)) / sr
        out[device] = f"성공  첫청크 {first:6.0f}ms  총 {total:5.2f}s  RTF {total / dur:.2f}"
    except Exception as exc:  # noqa: BLE001 - 실패 모드 자체가 관측 대상
        out[device] = f"실패  {type(exc).__name__}: {exc}"


def main() -> None:
    n = torch.cuda.device_count()
    print(f"보이는 GPU {n}개", flush=True)
    if n < 2:
        print("GPU 2장이 필요하다 — CUDA_VISIBLE_DEVICES=1,2 로 실행할 것", flush=True)
        return

    devices = ["cuda:0", "cuda:1"]
    models = {d: load(d) for d in devices}

    out: dict[str, str] = {}
    print("워밍업 (그래프 캡처, 순차)", flush=True)
    for d in devices:
        generate(models[d], d, out)
        print(f"  {d} {out[d]}", flush=True)

    print("=== 동시 2개 (인스턴스 분리) ===", flush=True)
    out.clear()
    threads = [threading.Thread(target=generate, args=(models[d], d, out)) for d in devices]
    wall = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"벽시계 총 {time.perf_counter() - wall:.2f}s", flush=True)
    for d in devices:
        print(f"  {d} {out.get(d, '결과 없음')}", flush=True)


if __name__ == "__main__":
    main()
