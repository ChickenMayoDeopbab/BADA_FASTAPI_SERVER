"""E1/E2 — 프로세스를 나누면 동시 생성이 되는지. 계획 0047.

프로세스마다 자기 CUDA 컨텍스트와 모델 인스턴스를 갖는다. 각 워커는 모델을 올리고 워밍업한 뒤
동기화 디렉터리에 ready 파일을 쓰고, 모든 피어가 준비될 때까지 기다렸다가 동시에 생성한다.

사용법 (E1: 같은 GPU / E2: 다른 GPU):
  CUDA_VISIBLE_DEVICES=1 python conc_worker.py A A,B /tmp/e1 &
  CUDA_VISIBLE_DEVICES=1 python conc_worker.py B A,B /tmp/e1 &
"""

import pathlib
import sys
import time

import numpy as np
import torch
from qwen_tts import Qwen3TTSModel

TAG = sys.argv[1]
PEERS = sys.argv[2].split(",")
SYNC = pathlib.Path(sys.argv[3])

REF = str(pathlib.Path.home() / "qwen-test/out2b/APOLOGETIC_2.wav")
RTXT = "아니 그게 아니라, 제가 몇 번을 말씀드렸잖아요."
LINE = "죄송하지만 그건 저희 쪽에서 처리해 드릴 수가 없어요."


def log(msg: str) -> None:
    print(f"[{TAG}] {msg}", flush=True)


def generate(model: Qwen3TTSModel) -> str:
    t0 = time.perf_counter()
    first = None
    parts = []
    sr = 0
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
    return f"첫청크 {first:6.0f}ms  총 {total:5.2f}s  오디오 {dur:5.2f}s  RTF {total / dur:.2f}"


def main() -> None:
    SYNC.mkdir(parents=True, exist_ok=True)
    log(f"모델 로드 (보이는 GPU: {torch.cuda.device_count()}개)")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0", dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.enable_streaming_optimizations()
    log("워밍업 (그래프 캡처)")
    generate(model)

    (SYNC / f"{TAG}.ready").write_text("1")
    log("준비 완료 — 피어 대기")
    deadline = time.time() + 900
    while time.time() < deadline:
        if all((SYNC / f"{p}.ready").exists() for p in PEERS):
            break
        time.sleep(0.2)
    else:
        log("피어 대기 시간 초과 — 중단")
        return

    time.sleep(0.5)  # 마지막 피어의 루프 탈출까지 여유
    log("동시 생성 시작")
    try:
        log(f"성공  {generate(model)}")
    except Exception as exc:  # noqa: BLE001 - 실패 모드 자체가 관측 대상
        log(f"실패  {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
