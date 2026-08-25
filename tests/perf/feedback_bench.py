"""피드백 리포트(good_segments) 지연 벤치.

세션 종료(_teardown) 시점부터 각 단계 지연을 측정한다:
  - upload    : 녹음본 S3 업로드 (기본은 페이크, --upload-delay-ms 로 RTT 시뮬레이션)
  - analyze   : TremorAnalyzer.analyze (실제 연산 — 사용자 체감 지연의 지배 요인)
  - end_frame : 종료 트리거 → end 프레임(feedback 동봉) 송신까지 = 사용자 체감 지연
  - callback  : 종료 트리거 → Spring 콜백 완료까지 (praise LLM 포함)

praise(잘한 점 LLM)는 end 프레임 이후에 실행되어야 하며(파이프라인 주석 참조),
이 순서가 깨지면 CHECK 로 표시한다.

실행:
  python -m tests.perf.feedback_bench --durations 30,60,180 --runs 5
"""
import argparse
import asyncio
import time
from types import SimpleNamespace

import numpy as np

from app.schemas.frames import EndReason
from app.services.pipeline import VoicePipeline
from app.services.tremor import TremorAnalyzer
from tests.perf._stats import print_table, summarize

SAMPLE_RATE = 16000


def make_pcm(duration_s: float) -> bytes:
    """음성 비슷한 합성 PCM(16kHz/int16): f0 억양 + 발화/침묵 반복."""
    t = np.arange(int(duration_s * SAMPLE_RATE)) / SAMPLE_RATE
    f0 = 120 + 8 * np.sin(2 * np.pi * 2.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
    y = 0.4 * np.sin(phase) + 0.15 * np.sin(2 * phase) + 0.05 * np.sin(3 * phase)
    y *= (np.sin(2 * np.pi * 0.15 * t) > -0.3).astype(np.float32)  # 발화/침묵
    return (y * 0.5 * 32767).astype(np.int16).tobytes()


class FakeWS:
    def __init__(self) -> None:
        self.end_at: float | None = None
        self.feedback: dict | None = None

    async def send_json(self, payload: dict) -> None:
        if payload.get("type") == "end":
            self.end_at = time.perf_counter()
            self.feedback = payload.get("feedback")

    async def close(self, code: int = 1000) -> None:
        pass


class FakeStorage:
    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.elapsed: float | None = None

    def upload_pcm(self, session_id: str, pcm: bytes) -> str:
        t0 = time.perf_counter()
        time.sleep(self.delay_s)
        self.elapsed = time.perf_counter() - t0
        return "bench/recording.wav"


class TimedTremor:
    def __init__(self) -> None:
        self.inner = TremorAnalyzer()
        self.elapsed: float | None = None

    def analyze(self, pcm: bytes):
        t0 = time.perf_counter()
        result = self.inner.analyze(pcm)
        self.elapsed = time.perf_counter() - t0
        return result


class FakeLLM:
    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self.called_at: float | None = None

    async def segment_feedback(self, items, **kwargs):
        self.called_at = time.perf_counter()
        await asyncio.sleep(self.delay_s)
        return [("용건을 말했어요", "무슨 일로 걸었는지 밝혔어요.")] * len(items)


class FakeSpring:
    def __init__(self) -> None:
        self.notified_at: float | None = None

    async def notify_session_closed(self, *args, **kwargs) -> None:
        self.notified_at = time.perf_counter()


def build_pipeline(ws, storage, tremor, llm, spring, pcm: bytes, duration_s: float) -> VoicePipeline:
    """무거운 __init__ 없이 _teardown 에 필요한 속성만 채운 최소 객체."""
    p = VoicePipeline.__new__(VoicePipeline)
    p._ws = ws
    p._ws_alive = True
    p._session_id = "feedback-bench"
    p._session = {"type": "SCENARIO"}
    p._history = []
    p._silence_total = 3.0
    p._turn_task = None
    p._end_reason = EndReason.USER_END
    p._recording_storage = storage
    p._tremor = tremor
    p._tremor_buf = bytearray(pcm)
    p._llm = llm
    p._spring = spring
    # AVTI 는 end 프레임/콜백 뒤에서 떼어놓고 도는 섀도 작업이라 벤치에서는 꺼둔다.
    p._settings = SimpleNamespace(avti_enabled=False)
    # 사용자 턴 3개 (good_segments 후보와 겹치도록 녹음 전체에 분산)
    p._user_turn_intervals = [
        (1.0, duration_s * 0.3),
        (duration_s * 0.4, duration_s * 0.6),
        (duration_s * 0.7, duration_s - 1.0),
    ]
    p._user_turn_texts = ["첫 번째 발화", "두 번째 발화", "세 번째 발화"]
    p._turn_open_at = None
    p._script_len = 0
    p._ai_pcm_bytes = 0
    p._server_wait_duration_ms = 0
    p._completed_script_steps = 0
    return p


async def run_once(duration_s: float, upload_delay_s: float, praise_delay_s: float) -> dict:
    pcm = make_pcm(duration_s)
    ws, storage = FakeWS(), FakeStorage(upload_delay_s)
    tremor, llm, spring = TimedTremor(), FakeLLM(praise_delay_s), FakeSpring()
    p = build_pipeline(ws, storage, tremor, llm, spring, pcm, duration_s)

    t0 = time.perf_counter()
    await p._teardown()

    order_ok = (
        ws.end_at is not None
        and (llm.called_at is None or ws.end_at <= llm.called_at)
    )
    return {
        "upload": storage.elapsed,
        "analyze": tremor.elapsed,
        "end_frame": (ws.end_at - t0) if ws.end_at else None,
        "callback": (spring.notified_at - t0) if spring.notified_at else None,
        "order_ok": order_ok,
        "feedback": ws.feedback,
    }


async def main() -> None:
    ap = argparse.ArgumentParser(description="피드백 리포트 지연 벤치")
    ap.add_argument("--durations", default="30,60,180", help="녹음 길이(초), 콤마 구분")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--upload-delay-ms", type=float, default=0, help="S3 업로드 RTT 시뮬레이션")
    ap.add_argument("--praise-delay-ms", type=float, default=0, help="praise LLM 지연 시뮬레이션")
    args = ap.parse_args()

    # librosa/numba 콜드스타트가 첫 샘플을 왜곡하지 않도록 워밍업 1회
    await run_once(5.0, 0, 0)

    for dur in [float(d) for d in args.durations.split(",")]:
        samples: dict[str, list] = {"upload": [], "analyze": [], "end_frame": [], "callback": []}
        all_order_ok, last_feedback = True, None
        for _ in range(args.runs):
            r = await run_once(dur, args.upload_delay_ms / 1000, args.praise_delay_ms / 1000)
            for k in samples:
                samples[k].append(r[k] * 1000 if r[k] is not None else None)
            all_order_ok &= r["order_ok"]
            last_feedback = r["feedback"]

        print(f"\n=== 녹음 {dur:.0f}s × {args.runs}회 ===")
        print_table([(k, summarize(v)) for k, v in samples.items()])
        segs = (last_feedback or {}).get("good_segments") or []
        print(f"good_segments {len(segs)}개, end 프레임 → praise 순서: "
              f"{'PASS' if all_order_ok else 'CHECK (end 프레임보다 praise 가 먼저 실행됨!)'}")


if __name__ == "__main__":
    asyncio.run(main())
