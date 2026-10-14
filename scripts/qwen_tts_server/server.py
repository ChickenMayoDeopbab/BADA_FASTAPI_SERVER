"""Qwen3-TTS 워커 서버 (레포의 이 파일이 원본).

배포 위치: 학교 GPU `~/bada-qwen3-tts/fork/server.py`(launch_workers.sh 가 워커별로 띄움),
집 PC /home/uhihi/Qwen3-TTS-streaming/server.py (WSL2, systemd qwen-tts.service).
계약:
  GET  /health          → {ready, voices, busy}
  POST /v1/tts          {voice, text} → raw PCM 16kHz mono 일괄 (예시 대화 배치용)
  POST /v1/tts/stream   {voice, text} → chunked raw PCM 16kHz mono (실시간 통화용, 계획 0043)
"""

import asyncio
import contextlib
import json
import logging
import math
import os
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import torch
import torchaudio
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from qwen_tts import Qwen3TTSModel

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
VOICES_F = os.environ.get("VOICES_FILE", "/mnt/c/qwen-out/voices.json")
DEVICE = os.environ.get("TTS_DEVICE", "cuda:0")  # 워커를 GPU 별로 띄울 때 지정
OUT_SR = 16000  # 파이프라인이 pcm_16000
EMIT = 2  # 런타임 변경 금지 — 바꾸면 재컴파일
MAX_CHARS = 300  # 긴 텍스트는 호출측이 문장 단위로 쪼개서 보낼 것
STREAM_LOCK_TIMEOUT = 8.0  # 호출측 read 타임아웃(10s)보다 짧게 — busy 를 서버가 먼저 끊는다
# 생성 상한 — 포크 스트리밍 경로는 max_new_tokens 를 버리고 EOS 아니면 max_frames(기본 10000≈800s)에서만 멈춘다.
# 정상 발화 관측(로컬 92개): s/자 p50 0.13·최대 0.27, 앞 무음 최대 4.64s. 안전망이라 넉넉히 —
# FastAPI 문장 가드(qwen_tts.runaway_cap_ms)가 항상 먼저 걸려야 한다(계획 0064 불변식, 테스트로 고정)
FRAME_S = 0.08  # 12Hz 코덱 프레임 = 24kHz 1920 샘플
CAP_BASE_S = 5.0
CAP_PER_CHAR_S = 0.4
# 소비자가 떠난 뒤: drain = 출력 없이 EOS·상한까지(예전 동작) / close = 다음 yield 경계에서 gen.close().
# close 는 GPU 점검(계획 0064 5절) 통과 뒤 env 로 켠다
STOP_MODE = os.environ.get("QWEN_STOP_MODE", "drain")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("tts")
S = {"model": None, "voices": {}, "ready": False, "lock": threading.Lock()}


class CapHitError(RuntimeError):
    """EOS 없이 상한 도달 — 옹알이로 본다"""


def max_frames_for(text: str) -> int:
    cap = math.ceil((CAP_BASE_S + CAP_PER_CHAR_S * len(text)) / FRAME_S)
    # emit 경계에서 끝나야 상한 판정이 마지막 flush 길이(포크 decode 가 자름)에 안 흔들린다
    return cap + (-cap) % EMIT


def _hit_cap(n_samples: int, sr: int | None, cap: int) -> bool:
    # 상한이면 정확히 cap 프레임, EOS 면 많아야 cap-1 프레임 — 반 프레임 여유로 가른다
    return sr is not None and n_samples > (cap - 0.5) * FRAME_S * sr


def _pcm16(w: np.ndarray) -> bytes:
    return (np.clip(w, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def _generate(voice: str, text: str, max_frames: int):
    ref, rtxt = S["voices"][voice]
    return S["model"].stream_generate_voice_clone(
        text=text, language="Korean", ref_audio=ref, ref_text=rtxt,
        emit_every_frames=EMIT, temperature=0.7, top_k=20, max_frames=max_frames)


def _synth(voice: str, text: str) -> bytes:
    cap = max_frames_for(text)
    chunks = []
    sr = OUT_SR
    n = 0
    for pcm, chunk_sr in _generate(voice, text, cap):
        chunks.append(pcm)
        sr = chunk_sr
        n += len(pcm)
    if _hit_cap(n, sr, cap):
        raise CapHitError(f"EOS 없이 {cap}프레임 도달 chars={len(text)}")
    w = np.concatenate(chunks).astype(np.float32)
    w = torchaudio.functional.resample(torch.from_numpy(w), sr, OUT_SR).numpy()
    return _pcm16(w)


class _StreamResampler:
    """청크 단위 리샘플(24k→16k) — 청크 경계 아티팩트를 막는다.

    방출 구간 좌우에 항상 ctx 샘플 이상의 실제 신호를 두고 리샘플하므로, 결과는
    전체를 한 번에 리샘플한 것과 (필터 폭 ≪ ctx 인 한) 동일하다. 방출은 입력
    스텝(24k→16k 이면 3)의 배수 단위로만 해서 출력 인덱스가 정확히 정렬된다.
    대가는 ctx 만큼의 지연(20ms) 하나뿐이다.
    """

    CTX = 480  # 20ms@24k — 리샘플 필터 폭 대비 충분히 넓게

    def __init__(self, in_sr: int) -> None:
        g = math.gcd(in_sr, OUT_SR)
        self.in_sr = in_sr
        self.in_step = in_sr // g
        self.out_step = OUT_SR // g
        self.ctx = self.CTX - self.CTX % self.in_step
        self.left = np.zeros(0, dtype=np.float32)
        self.pending = np.zeros(0, dtype=np.float32)

    def _resample(self, x: np.ndarray) -> np.ndarray:
        return torchaudio.functional.resample(
            torch.from_numpy(x), self.in_sr, OUT_SR).numpy()

    def push(self, chunk: np.ndarray) -> np.ndarray:
        self.pending = np.concatenate([self.pending, chunk.astype(np.float32)])
        emit = len(self.pending) - self.ctx
        emit -= emit % self.in_step
        if emit <= 0:
            return np.zeros(0, dtype=np.float32)
        x = np.concatenate([self.left, self.pending])
        y = self._resample(x)
        start = len(self.left) // self.in_step * self.out_step
        end = (len(self.left) + emit) // self.in_step * self.out_step
        boundary = len(self.left) + emit
        self.left = x[max(0, boundary - self.ctx):boundary]
        self.pending = self.pending[emit:]
        return y[start:end]

    def flush(self) -> np.ndarray:
        if not len(self.pending):
            return np.zeros(0, dtype=np.float32)
        x = np.concatenate([self.left, self.pending])
        y = self._resample(x)
        start = len(self.left) // self.in_step * self.out_step
        self.left = np.zeros(0, dtype=np.float32)
        self.pending = np.zeros(0, dtype=np.float32)
        return y[start:]


_EOF = object()


class _StreamJob:
    """문장 하나의 생성. 락은 생산 스레드가 잡고 푼다 — HTTP 소비자가 어떻게 사라지든 무관하다.

    현행(동기 제너레이터를 StreamingResponse 에 직접)의 문제: Starlette 는 끊긴 응답의
    body_iterator 를 닫지 않는다. 그래서 GeneratorExit 이 순환 GC 가 돌 때에야, 그것도 GC 를
    부른 아무 스레드에서 온다 — 그때까지 락이 잡혀 있다(2026-10-13 46분 busy).
    여기서는 (1) 생성 전체가 한 스레드에서 돌고 (2) 상한 max_frames 가 있어 소비자가 영영
    안 와도 상한 안에 끝나며 (3) 소비자 쪽 finally 가 stop 을 세우면 다음 yield 경계에서 멈춘다.
    """

    def __init__(self, voice: str, text: str, loop: asyncio.AbstractEventLoop, q: asyncio.Queue) -> None:
        self.voice, self.text = voice, text
        self.cap = max_frames_for(text)
        self.stop = threading.Event()
        self._lock = S["lock"]  # 잡은 락을 그대로 푼다
        self._loop, self._q = loop, q

    def _put(self, item) -> None:
        with contextlib.suppress(RuntimeError):  # 루프가 이미 닫힘(종료 중) — 받을 쪽이 없다
            self._loop.call_soon_threadsafe(self._q.put_nowait, item)

    def run(self) -> None:
        if not self._lock.acquire(timeout=STREAM_LOCK_TIMEOUT):
            self._put(RuntimeError("busy"))  # 스트림 중단 → 호출측이 불완전 응답으로 감지하고 폴백
            return
        t = time.perf_counter()
        gen = None
        sr = None
        n = sent = 0
        end: object = _EOF
        outcome = "eos"
        try:
            if self.stop.is_set():  # 락을 기다리는 사이 소비자가 떠남 — 생성을 시작하지 않는다
                outcome = "gone"
                return
            gen = _generate(self.voice, self.text, self.cap)
            rs = None
            for pcm, sr in gen:
                n += len(pcm)
                if self.stop.is_set():
                    if STOP_MODE != "drain":
                        break  # finally 의 gen.close() — 생성한 그 스레드에서, yield 경계에서
                    continue
                if rs is None:
                    rs = _StreamResampler(sr)
                    log.info("stream 첫 청크 voice=%s chars=%d %.3fs",
                             self.voice, len(self.text), time.perf_counter() - t)
                out = rs.push(np.asarray(pcm, dtype=np.float32))
                if len(out):
                    sent += len(out)
                    self._put(_pcm16(out))
            if self.stop.is_set():
                outcome = "stopped" if STOP_MODE != "drain" else "drained"
            elif _hit_cap(n, sr, self.cap):
                outcome = "cap"
                end = CapHitError(f"EOS 없이 {self.cap}프레임 도달 chars={len(self.text)}")
            elif rs is not None:
                tail = rs.flush()
                if len(tail):
                    sent += len(tail)
                    self._put(_pcm16(tail))
        except BaseException as e:  # 무엇이든 소비자에게 넘겨 스트림을 끊게 한다
            outcome = f"error:{type(e).__name__}"
            end = e
        finally:
            try:
                if gen is not None:
                    gen.close()  # 다 돈 제너레이터면 no-op
            except Exception:
                log.exception("stream 제너레이터 close 실패")
            finally:
                self._lock.release()  # 종료 신호보다 먼저 — 호출측이 응답 끝을 보면 이미 비어 있다
                self._put(end)
            frames = n / (FRAME_S * sr) if sr else 0.0
            (log.warning if outcome == "cap" else log.info)(
                "stream 완료 outcome=%s voice=%s chars=%d %.2fs → %.2fs audio frames=%.0f/%d",
                outcome, self.voice, len(self.text), time.perf_counter() - t, sent / OUT_SR, frames, self.cap)


async def _stream_pcm(voice: str, text: str):
    """문장 하나를 16kHz PCM 청크로 흘린다. 비동기 제너레이터여야 한다 — 그래야 Starlette 가
    응답 태스크를 취소할 때(ASGI spec < 2.4 의 끊김 감지) CancelledError 가 이 프레임에 들어와
    finally 가 바로 돈다. 동기 제너레이터는 iterate_in_threadpool 안쪽이라 취소를 못 본다.

    생산 스레드는 첫 __anext__ 에서야 띄운다 — 응답 시작 전에 끊기면 생성도 락도 없다.
    """
    q: asyncio.Queue = asyncio.Queue()  # 무제한이지만 실제 상한은 cap 프레임분(≤ 수 MB)
    job = _StreamJob(voice, text, asyncio.get_running_loop(), q)
    threading.Thread(target=job.run, name="tts-stream", daemon=True).start()
    try:
        while True:
            item = await q.get()
            if item is _EOF:
                return
            if isinstance(item, BaseException):
                raise item  # 응답 시작 뒤 예외 → uvicorn 이 종료 청크 없이 연결을 닫는다 → 호출측 폴백
            yield item
    finally:
        job.stop.set()  # 취소·aclose·정상 종료 어느 길로 와도. GC 로만 닫히는 경우도 상한이 받친다


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(VOICES_F, encoding="utf-8") as f:
        S["voices"] = {k: (v["ref_audio"], v["ref_text"]) for k, v in json.load(f).items()}
    log.info("모델 로드 중...")
    S["model"] = Qwen3TTSModel.from_pretrained(
        MODEL_ID, device_map=DEVICE, dtype=torch.bfloat16,
        attn_implementation="flash_attention_2")
    S["model"].enable_streaming_optimizations()
    for v in S["voices"]:  # 보이스마다 컴파일 1회(~90초)
        t = time.perf_counter()
        try:
            _synth(v, "네, 안녕하세요.")
        except CapHitError as e:  # 컴파일은 이미 됐다 — 워밍업 문장이 옹알이여도 기동은 한다
            log.warning("워밍업 %s %s", v, e)
        log.info("워밍업 %s %.1fs", v, time.perf_counter() - t)
    S["ready"] = True
    log.info("준비 완료 — voices=%s", list(S["voices"]))
    yield


app = FastAPI(lifespan=lifespan)


class Req(BaseModel):
    voice: str
    text: str = Field(min_length=1, max_length=MAX_CHARS)


def _validate(r: Req) -> None:
    if not S["ready"]:
        raise HTTPException(503, "warming up")
    if r.voice not in S["voices"]:
        raise HTTPException(404, f"unknown voice: {r.voice}")


@app.get("/health")
def health():
    return {"ready": S["ready"], "voices": list(S["voices"]), "busy": S["lock"].locked()}


@app.post("/v1/tts")
def tts(r: Req):
    _validate(r)
    if not S["lock"].acquire(timeout=120):  # 동시성 1 — 2 이상이면 CUDA 그래프가 깨진다
        raise HTTPException(503, "busy")
    try:
        t = time.perf_counter()
        try:
            pcm = _synth(r.voice, r.text)
        except CapHitError as e:  # 5xx → 호출측 _attempt 가 재시도(샘플링이라 다시 하면 대개 정상) 후 EL 폴백
            log.warning("tts cap voice=%s %s %.2fs", r.voice, e, time.perf_counter() - t)
            raise HTTPException(502, str(e)) from e
        log.info("tts voice=%s chars=%d %.2fs → %.2fs audio",
                 r.voice, len(r.text), time.perf_counter() - t, len(pcm) / (OUT_SR * 2))
        return Response(pcm, media_type="audio/L16",
                        headers={"X-Sample-Rate": str(OUT_SR)})
    finally:
        S["lock"].release()


@app.post("/v1/tts/stream")
def tts_stream(r: Req):
    """실시간 통화용 — 문장 하나를 chunked 로 흘림"""
    _validate(r)
    return StreamingResponse(_stream_pcm(r.voice, r.text), media_type="audio/L16",
                             headers={"X-Sample-Rate": str(OUT_SR)})
