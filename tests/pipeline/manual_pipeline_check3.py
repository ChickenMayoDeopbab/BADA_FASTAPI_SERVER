import asyncio
import logging
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from statistics import mean

import numpy as np
import sounddevice as sd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("pipeline_manual3")

from app.core.config import get_settings
from app.services.pipeline import VoicePipeline, _State
from app.services.spring_client import SpringInternalClient
from app.services.stt import STTEventType

SAMPLE_RATE = 16000
BLOCK = 1600

HALF_DUPLEX = True
INPUT_DEVICE_HINT = "MacBook"

_SESSION = {
    "userId": 1,
    "type": "SCENARIO",
    "aiPersonality": "NORMAL",
    "maxDurationSeconds": 180,
    "scenario": {
        "title": "병원 예약 변경",
        "aiRole": "병원 접수 직원",
        "script": [
            {"step": 1, "aiGoal": "전화를 받고 어느 병원인지 밝히며 용건을 묻는다"},
            {"step": 2, "aiGoal": "기존 예약자 본인 확인(이름, 생년월일)을 요청한다"},
            {"step": 3, "aiGoal": "변경을 원하는 날짜와 시간을 물어본다"},
            {"step": 4, "aiGoal": "변경 가능 여부를 안내하고 확정한다"},
            {"step": 5, "aiGoal": "마무리 인사를 한다"},
        ],
    },
}

class ErrorCollector(logging.Handler):
    _EXPECTED = {"app.services.spring_client"}

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.name in self._EXPECTED:
            return
        self.records.append(f"{record.levelname} {record.name}: {record.getMessage()}")

@dataclass
class Metrics:
    turns: list[dict] = field(default_factory=list)
    barge_ins: int = 0
    end_reason: str | None = None
    _cur: dict | None = None
    _t_final: float | None = None
    _t_emotion: float | None = None

    def begin_turn(self, user: str) -> None:
        self._t_final = time.perf_counter()
        self._t_emotion = None
        self._cur = {"user": user, "emotion": "?", "ttft": None, "ttfb": None, "audio": None}

    def on_emotion(self, value: str) -> None:
        if self._cur is None:
            return
        self._t_emotion = time.perf_counter()
        self._cur["emotion"] = value
        if self._t_final is not None:
            self._cur["ttft"] = (self._t_emotion - self._t_final) * 1000

    def on_first_audio(self) -> None:
        if self._cur is None or self._t_emotion is None:
            return
        self._cur["ttfb"] = (time.perf_counter() - self._t_emotion) * 1000

    def on_audio_done(self, secs: float) -> None:
        if self._cur is not None:
            self._cur["audio"] = secs

    def finalize(self, ai: str, flags: dict) -> None:
        cur = self._cur or {"user": "?", "emotion": "?", "ttft": None, "ttfb": None}
        cur["ai"] = ai
        cur["step_done"] = flags.get("step_done", False)
        cur["end_call"] = flags.get("end_call", False)
        cur["error"] = flags.get("error", False)
        self.turns.append(cur)
        self._cur = None

    def on_barge_in(self) -> None:
        self.barge_ins += 1

class FakeWebSocket:
    def __init__(self, loop: asyncio.AbstractEventLoop, metrics: Metrics) -> None:
        self._loop = loop
        self._metrics = metrics
        self._inbound: asyncio.Queue[dict] = asyncio.Queue()
        self._player = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        self._player.start()
        self._first_audio_seen = True
        self._turn_bytes = 0

    def feed_mic(self, pcm: bytes) -> None:
        self._loop.call_soon_threadsafe(self._inbound.put_nowait, {"bytes": pcm})

    def disconnect(self) -> None:
        self._loop.call_soon_threadsafe(
            self._inbound.put_nowait, {"type": "websocket.disconnect"}
        )

    async def receive(self) -> dict:
        return await self._inbound.get()

    async def send_bytes(self, data: bytes) -> None:
        if not self._first_audio_seen:
            self._first_audio_seen = True
            self._metrics.on_first_audio()
        self._turn_bytes += len(data)
        audio = np.frombuffer(data, dtype=np.int16)
        await asyncio.to_thread(self._player.write, audio)

    async def send_json(self, payload: dict) -> None:
        ftype = payload.get("type")
        if ftype == "emotion":
            self._first_audio_seen = False
            self._turn_bytes = 0
            self._metrics.on_emotion(payload.get("value", "?"))
            logger.info("AI emotion=%s", payload.get("value"))
        elif ftype == "speaking_end":
            self._metrics.on_audio_done(self._turn_bytes / 2 / SAMPLE_RATE)
        elif ftype == "interrupt":
            self._metrics.on_barge_in()
            logger.info(">>> barge-in (interrupt) <<<")
        elif ftype == "end":
            self._metrics.end_reason = payload.get("reason")
            logger.info("=== 세션 종료 reason=%s ===", payload.get("reason"))
        elif ftype == "error":
            self._metrics.end_reason = f"ERROR:{payload.get('code')}"
            logger.info("=== 에러 종료 code=%s ===", payload.get("code"))

    async def close(self, code: int = 1000) -> None:
        with suppress(Exception):
            self._player.stop()
            self._player.close()

async def _state_watcher(pipeline: VoicePipeline) -> None:
    prev = None
    while True:
        cur = pipeline._state
        if cur != prev:
            logger.info("[state] %s", cur.value)
            prev = cur
        await asyncio.sleep(0.05)

def _find_input_device(hint: str) -> int | None:
    for idx, dev in enumerate(sd.query_devices()):
        if hint.lower() in dev["name"].lower() and dev["max_input_channels"] > 0:
            return idx
    return None

def _print_summary(metrics: Metrics, errors: list[str]) -> None:
    turns = metrics.turns
    ttfts = [t["ttft"] for t in turns if t.get("ttft") is not None]
    ttfbs = [t["ttfb"] for t in turns if t.get("ttfb") is not None]

    print("\n================ SUMMARY ================")
    print(f"턴 수             : {len(turns)}")
    print(f"종료 사유         : {metrics.end_reason}")
    print(f"barge-in          : {metrics.barge_ins}회")
    if ttfts:
        print(f"TTFT(FINAL->emotion): 평균 {mean(ttfts):.0f}ms  min {min(ttfts):.0f}  max {max(ttfts):.0f}")
    if ttfbs:
        print(f"TTFB(emotion->오디오): 평균 {mean(ttfbs):.0f}ms  min {min(ttfbs):.0f}  max {max(ttfbs):.0f}")
    print(f"예상외 경고/에러 로그: {len(errors)}건")
    for e in errors[:10]:
        print(f"   - {e}")

    print("\n--- 대화록 ---")
    for i, t in enumerate(turns, 1):
        tags = [k.upper() for k in ("step_done", "end_call", "error") if t.get(k)]
        tagstr = (" [" + ",".join(tags) + "]") if tags else ""
        lat = ""
        if t.get("ttft") is not None and t.get("ttfb") is not None:
            lat = f"  (ttft {t['ttft']:.0f}ms, ttfb {t['ttfb']:.0f}ms)"
        print(f"[{i}] You : {t['user']}")
        print(f"    AI[{t.get('emotion','?')}]: {t.get('ai', '')}{tagstr}{lat}")

    ok = len(turns) >= 2 and len(errors) == 0 and metrics.end_reason is not None
    print("\n판정:", "PASS (멀티턴 + 에러 0 + 정상 종료)" if ok else "CHECK (아래 사유 확인)")
    if not ok:
        if len(turns) < 2:
            print("  - 턴이 2개 미만. STT가 다발화를 못 잡았을 수 있음.")
        if errors:
            print("  - 예상외 경고/에러 로그 있음(위 목록).")
        if metrics.end_reason is None:
            print("  - 종료 프레임을 못 받음(강제 종료).")
    print("========================================")


async def main() -> None:
    settings = get_settings()
    loop = asyncio.get_running_loop()
    metrics = Metrics()

    if INPUT_DEVICE_HINT:
        idx = _find_input_device(INPUT_DEVICE_HINT)
        if idx is not None:
            sd.default.device = (idx, None)
            logger.info("입력 장치 고정: [%d] %s", idx, sd.query_devices(idx)["name"])
        else:
            logger.warning("'%s' 입력 장치 못 찾음, 기본 사용", INPUT_DEVICE_HINT)

    errors = ErrorCollector()
    logging.getLogger().addHandler(errors)

    ws = FakeWebSocket(loop, metrics)
    pipeline = VoicePipeline(
        ws, "manual-session", _SESSION, settings=settings, spring=SpringInternalClient(settings)
    )

    _orig_handle = pipeline._handle_stt_event

    async def _logged_handle(event):
        logger.info("[stt] %-12s %r", event.type.value, (event.text or "")[:40])
        if (
            event.type == STTEventType.FINAL
            and event.text.strip()
            and pipeline._state == _State.LISTENING
            and not pipeline._time_up
        ):
            metrics.begin_turn(event.text.strip())
        await _orig_handle(event)

    pipeline._handle_stt_event = _logged_handle

    _orig_finalize = pipeline._finalize_turn

    async def _logged_finalize(user_utterance, ai_parts, flags):
        metrics.finalize("".join(ai_parts).strip(), flags)
        await _orig_finalize(user_utterance, ai_parts, flags)

    pipeline._finalize_turn = _logged_finalize

    mic_peak = {"v": 0}

    def on_mic(indata, frames, time_info, status_) -> None:
        if status_:
            logger.debug("mic status: %s", status_)
        buf = bytes(indata)
        samples = np.frombuffer(buf, dtype=np.int16)
        if samples.size:
            mic_peak["v"] = max(mic_peak["v"], int(np.abs(samples).max()))
        if HALF_DUPLEX and pipeline._state is not _State.LISTENING:
            buf = b"\x00" * len(buf)
        ws.feed_mic(buf)

    async def _meter() -> None:
        while True:
            await asyncio.sleep(1.0)
            logger.info("[mic] peak=%d  q=%d", mic_peak["v"], pipeline._audio_queue.qsize())
            mic_peak["v"] = 0

    mic = sd.RawInputStream(
        samplerate=SAMPLE_RATE, blocksize=BLOCK, channels=1, dtype="int16", callback=on_mic
    )

    def wait_enter() -> None:
        with suppress(EOFError):
            input()
        ws.disconnect()

    threading.Thread(target=wait_enter, daemon=True).start()
    watcher = asyncio.create_task(_state_watcher(pipeline))
    meter = asyncio.create_task(_meter())

    mode = "하프듀플렉스(스피커 OK)" if HALF_DUPLEX else "풀듀플렉스(유선 이어폰 권장)"
    print(f"=== 총합 하니스 [{mode}] — 시나리오대로 대화. Enter/Ctrl+C 종료 ===")
    try:
        with mic:
            await pipeline.run()
    finally:
        for task in (watcher, meter):
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        logging.getLogger().removeHandler(errors)
        _print_summary(metrics, errors.records)


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n(Ctrl+C) 종료 중...")
