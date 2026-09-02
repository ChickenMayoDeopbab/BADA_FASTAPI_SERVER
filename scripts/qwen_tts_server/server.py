# 집 PC에 배포된 서버 사본 변경 금지
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
OUT_SR = 16000
EMIT = 2
MAX_CHARS = 300
STREAM_LOCK_TIMEOUT = 8.0

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("tts")
S = {"model": None, "voices": {}, "ready": False, "lock": threading.Lock()}


def _pcm16(w: np.ndarray) -> bytes:
    return (np.clip(w, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def _generate(voice: str, text: str):
    ref, rtxt = S["voices"][voice]
    return S["model"].stream_generate_voice_clone(
        text=text, language="Korean", ref_audio=ref, ref_text=rtxt,
        emit_every_frames=EMIT, temperature=0.7, top_k=20)


def _synth(voice: str, text: str) -> bytes:
    chunks = []
    sr = OUT_SR
    for pcm, chunk_sr in _generate(voice, text):
        chunks.append(pcm)
        sr = chunk_sr
    w = np.concatenate(chunks).astype(np.float32)
    w = torchaudio.functional.resample(torch.from_numpy(w), sr, OUT_SR).numpy()
    return _pcm16(w)


class _StreamResampler:
    """청크 리샘플"""

    CTX = 480

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


def _stream_pcm(voice: str, text: str):
    if not S["lock"].acquire(timeout=STREAM_LOCK_TIMEOUT):
        raise RuntimeError("busy")
    gen = None
    try:
        t = time.perf_counter()
        gen = _generate(voice, text)
        rs = None
        sent = 0
        for pcm, sr in gen:
            if rs is None:
                rs = _StreamResampler(sr)
                log.info("stream 첫 청크 voice=%s chars=%d %.3fs",
                         voice, len(text), time.perf_counter() - t)
            out = rs.push(np.asarray(pcm, dtype=np.float32))
            if len(out):
                sent += len(out)
                yield _pcm16(out)
        if rs is not None:
            tail = rs.flush()
            if len(tail):
                sent += len(tail)
                yield _pcm16(tail)
        log.info("stream 완료 voice=%s chars=%d %.2fs → %.2fs audio",
                 voice, len(text), time.perf_counter() - t, sent / OUT_SR)
    except GeneratorExit:
        if gen is not None:
            for _ in gen:
                pass
        raise
    finally:
        S["lock"].release()


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(VOICES_F, encoding="utf-8") as f:
        S["voices"] = {k: (v["ref_audio"], v["ref_text"]) for k, v in json.load(f).items()}
    log.info("모델 로드 중...")
    S["model"] = Qwen3TTSModel.from_pretrained(
        MODEL_ID, device_map="cuda:0", dtype=torch.bfloat16,
        attn_implementation="flash_attention_2")
    S["model"].enable_streaming_optimizations()
    for v in S["voices"]:
        t = time.perf_counter()
        _synth(v, "네, 안녕하세요.")
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
    if not S["lock"].acquire(timeout=120):
        raise HTTPException(503, "busy")
    try:
        t = time.perf_counter()
        pcm = _synth(r.voice, r.text)
        log.info("tts voice=%s chars=%d %.2fs → %.2fs audio",
                 r.voice, len(r.text), time.perf_counter() - t, len(pcm) / (OUT_SR * 2))
        return Response(pcm, media_type="audio/L16",
                        headers={"X-Sample-Rate": str(OUT_SR)})
    finally:
        S["lock"].release()


@app.post("/v1/tts/stream")
def tts_stream(r: Req):
    _validate(r)
    return StreamingResponse(_stream_pcm(r.voice, r.text), media_type="audio/L16",
                             headers={"X-Sample-Rate": str(OUT_SR)})
