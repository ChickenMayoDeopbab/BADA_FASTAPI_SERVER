import asyncio
import contextlib
import logging
import os
import sys
import wave

os.environ["QWEN_TTS_URL"] = "http://127.0.0.1:8199"
os.environ["QWEN_TTS_REALTIME_ENABLED"] = "true"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from app.core.config import get_settings  # noqa: E402
from app.core.metrics import now_ms  # noqa: E402
from app.services import pipeline as pipeline_mod  # noqa: E402
from app.services import qwen_tts as qwen_mod  # noqa: E402
from app.services.llm import LLMClient  # noqa: E402
from app.services.pipeline import VoicePipeline, _State, _TurnTimings  # noqa: E402
from app.services.qwen_tts import try_acquire_realtime_tts  # noqa: E402
from app.services.tts import ElevenLabsTTSClient  # noqa: E402

UPSTREAM = ("100.82.37.97", 8100)
PROXY_ADDR = ("127.0.0.1", 8199)
OUT_DIR = os.environ.get("E2E_OUT_DIR", ".")

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


class Proxy:

    def __init__(self) -> None:
        self.server = None
        self.writers: list[asyncio.StreamWriter] = []

    async def start(self) -> None:
        self.server = await asyncio.start_server(self._handle, *PROXY_ADDR)

    async def _handle(self, cr, cw) -> None:
        try:
            ur, uw = await asyncio.open_connection(*UPSTREAM)
        except Exception:
            cw.close()
            return
        self.writers += [cw, uw]

        async def pump(r, w):
            try:
                while True:
                    data = await r.read(65536)
                    if not data:
                        break
                    w.write(data)
                    await w.drain()
            except Exception:
                pass
            finally:
                with contextlib.suppress(Exception):
                    w.close()

        await asyncio.gather(pump(cr, uw), pump(ur, cw))

    async def kill(self) -> None:
        self.server.close()
        await self.server.wait_closed()
        for w in self.writers:
            with contextlib.suppress(Exception):
                w.close()


class FakeWS:
    def __init__(self) -> None:
        self.frames: list[dict] = []
        self.pcm = bytearray()

    async def send_json(self, payload: dict) -> None:
        self.frames.append(payload)

    async def send_bytes(self, data: bytes) -> None:
        self.pcm.extend(data)

    def take_pcm(self) -> bytes:
        out = bytes(self.pcm)
        self.pcm.clear()
        return out


METRICS: list[tuple[str, dict]] = []
_orig_log_metric = pipeline_mod.log_metric


def _spy_metric(name: str, **kw) -> None:
    METRICS.append((name, kw))
    _orig_log_metric(name, **kw)


pipeline_mod.log_metric = _spy_metric


def make_pipeline(settings) -> VoicePipeline:
    p = VoicePipeline.__new__(VoicePipeline)
    p._ws = FakeWS()
    p._session_id = "e2e-qwen-realtime"
    p._session = _SESSION
    p._settings = settings
    p._spring = None
    p._llm = LLMClient()
    p._difficulty = None
    p._tts = ElevenLabsTTSClient(settings, None)
    p._qwen_tts = None
    p._state = _State.LISTENING
    p._history = []
    p._current_step = 1
    p._muted = False
    p._ws_alive = True
    p._time_up = False
    p._closing = asyncio.Event()
    p._turn_task = None
    p._listening_since = None
    p._tremor_buf = bytearray()
    p._ai_pcm_bytes = 0
    p._server_wait_duration_ms = 0
    p._completed_script_steps = 0
    p._user_turn_intervals = []
    p._user_turn_texts = []
    p._turn_open_at = None
    p._script_len = len(_SESSION["scenario"]["script"])
    p._voice_id_override = None
    p._max_duration = None
    return p


def save_wav(name: str, pcm: bytes) -> None:
    with wave.open(os.path.join(OUT_DIR, name), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm)


def turn_metric(i: int) -> dict:
    return [kw for name, kw in METRICS if name == "voice_turn"][i]


async def run_turn(p: VoicePipeline, text: str) -> tuple[bytes, dict]:
    n_before = len([1 for name, _ in METRICS if name == "voice_turn"])
    await asyncio.wait_for(p._run_turn(text, _TurnTimings(final_at=now_ms())), timeout=60)
    pcm = p._ws.take_pcm()
    return pcm, turn_metric(n_before)


async def main() -> int:
    settings = get_settings()
    assert settings.qwen_tts_realtime_enabled and settings.qwen_tts_url.endswith(":8199")
    proxy = Proxy()
    await proxy.start()
    failures: list[str] = []

    def check(cond: bool, msg: str) -> None:
        print(("PASS  " if cond else "FAIL  ") + msg)
        if not cond:
            failures.append(msg)

    p = make_pipeline(settings)
    await p._init_qwen_tts()
    check(p._qwen_tts is not None, "통화1: Qwen 슬롯 획득 (engine=qwen)")

    pcm1, m1 = await run_turn(p, "여보세요?")
    check(m1["tts_engine"] == "qwen" and not m1["tts_failed"], f"turn1 qwen 정상 (response_ms={m1['response_ms']})")
    check(len(pcm1) > 16000, f"turn1 오디오 {len(pcm1)/32000:.2f}s")
    save_wav("e2e_turn1_qwen.wav", pcm1)

    pcm2, m2 = await run_turn(p, "안녕하세요, 예약을 다음 주 수요일 오후로 변경하고 싶어서 전화드렸어요.")
    check(m2["tts_engine"] == "qwen" and not m2["tts_failed"], f"turn2 qwen 정상 (response_ms={m2['response_ms']})")
    save_wav("e2e_turn2_qwen.wav", pcm2)

    second, reason = await asyncio.wait_for(try_acquire_realtime_tts(settings), timeout=5)
    check(second is None and reason == "busy", f"통화2 동시 시도 → {reason} (EL 폴백 대상)")

    await proxy.kill()
    pcm3, m3 = await run_turn(p, "네, 김준하이고요. 생년월일은 3월 2일입니다.")
    check(m3["tts_failed"] is True and m3["tts_engine"] == "qwen", "turn3 장애 감지 (tts_failed)")
    check(not p._closing.is_set(), "turn3 통화 유지 (닫히지 않음)")
    check(p._qwen_tts is None, "turn3 편도 전환 + 슬롯 반납")
    check(any(n == "realtime_tts_switch" for n, _ in METRICS), "realtime_tts_switch 메트릭 기록")
    check(len(pcm3) > 8000, f"turn3 폴백 멘트가 EL 로 재생됨 {len(pcm3)/32000:.2f}s")
    check(not qwen_mod._semaphore().locked(), "슬롯 세마포어 해제 → 프리베이크 재사용 가능")
    save_wav("e2e_turn3_fallback_ment.wav", pcm3)

    pcm4, m4 = await run_turn(p, "다음 주 수요일 오후 세 시로 부탁드립니다.")
    check(
        m4["tts_engine"] == "eleven" and not m4["tts_failed"],
        f"turn4 EL 로 통화 계속 (response_ms={m4['response_ms']})",
    )
    check(len(pcm4) > 16000, f"turn4 오디오 {len(pcm4)/32000:.2f}s")
    save_wav("e2e_turn4_eleven.wav", pcm4)

    p._release_qwen_slot()
    engine_metrics = [kw for n, kw in METRICS if n == "realtime_tts_engine"]
    check(engine_metrics and engine_metrics[0]["engine"] == "qwen", "realtime_tts_engine 메트릭")

    print()
    print("턴 지표: " + " | ".join(
        f"t{i+1} {m['tts_engine']} resp={m['response_ms']}ms ttfb={m['tts_ttfb_ms']}ms"
        for i, m in enumerate([m1, m2, m3, m4])))
    print("대화 기록:")
    for h in p._history:
        print(f"  {h['role']}: {h['text'][:80]}")
    print()
    print("E2E PASS — 전 항목 통과" if not failures else f"E2E FAIL — {len(failures)}건: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
