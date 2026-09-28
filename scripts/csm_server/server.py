# -*- coding: utf-8 -*-
"""CSM 워커 HTTP 층 — Qwen 워커(server.py)와 같은 꼴에 세션을 더한 것. 설계 16 §2.1. 프로세스당 세션 1개, 동시성 1.

  GET  /health                       {ready, busy, session, voices}
  POST /v1/session/open              {session_id, voice}                      → {ok, prefill_ms}     참조 음성(voices.json: {voice: {ref_audio, ref_text}}) 프리필
  POST /v1/session/user              {session_id, text, pcm_b64(16 k int16 LE)}   → {ok, frames, prefill_ms, level_db, gain_db}   사용자 턴(정규화 → 인코딩 → [1]글 + 오디오)
  POST /v1/session/context           {session_id, turns:[{tag, text, pcm_b64}]}  → {ok, positions}   평가·재생용: 사람 턴 PCM 을 순서대로 넣는다
  POST /v1/session/context_codes     {session_id, turns:[{tag, text, codes_b64, frames}]} → {ok, positions}   평가·재생용: 토큰화된 코드를 그대로(W2, E-B 재현)
  POST /v1/session/speak             {session_id, text}                       → chunked raw PCM 16 kHz(audio/L16, X-Sample-Rate: 16000, X-TTFA-Ms 는 트레일러 대신 로그)
  POST /v1/session/cancel            {session_id}                             → {ok}
  POST /v1/session/close             {session_id}                             → {ok}

  환경변수: CSM_WEIGHTS(체크포인트 폴더, 기본 ~/CSM/runs/b1/epoch_1) · CSM_REPO(sesame/csm-1b) · VOICES_FILE · CSM_DEVICE(cuda) · CSM_COMPILE(1) · CSM_TARGET_DB(-26) · CSM_GAIN_DB(3)
  기동: HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 VOICES_FILE=~/voices.json uvicorn server:app --host 127.0.0.1 --port 8020
"""
import base64, functools, io, json, logging, os, sys, threading, time, wave
import numpy as np, torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import csm_worker as W

log = logging.getLogger("csm"); logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
MAX_CHARS, SEG_MAX_S = 300, 20.0
S = dict(ready=False, tok=None, model=None, sc=None, codec=None, session=None, session_id=None, voices={}, lock=threading.Lock(), cancel=None, gen_gain=float(os.environ.get("CSM_GAIN_DB", "3")))
app = FastAPI()


def read_wav_any(path):
    """참조 음성 wav → (int16 pcm 16 kHz mono). 16 k 가 아니면 g1_score.resample 로 맞춘다(24/32/48 k)."""
    with wave.open(path, "rb") as w:
        assert w.getsampwidth() == 2; x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0
        x, sr = (x.reshape(-1, w.getnchannels()).mean(1) if w.getnchannels() > 1 else x), w.getframerate()
    if sr != 16000:
        from g1_score import resample; x = resample(x, sr, 16000)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


@app.on_event("startup")
def startup():
    weights = os.path.expanduser(os.environ.get("CSM_WEIGHTS", "~/CSM/runs/b1/epoch_1")); dev = os.environ.get("CSM_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    comp = os.environ.get("CSM_COMPILE", "1" if dev.startswith("cuda") else "0") == "1"
    t0 = time.time(); tok, model, sc = W.load_model(weights, os.environ.get("CSM_REPO", "sesame/csm-1b"), dev, greedy=False, compile_=comp)
    S.update(tok=tok, model=model, sc=sc, codec=W.Codec(model, float(os.environ.get("CSM_TARGET_DB", "-26"))))
    vf = os.environ.get("VOICES_FILE")
    if vf:
        for k, v in json.load(open(os.path.expanduser(vf), encoding="utf-8")).items():
            pcm = read_wav_any(os.path.expanduser(v["ref_audio"])); codes, lv = S["codec"].encode_pcm16k(pcm); S["voices"][k] = dict(text=v["ref_text"], codes=codes, level=lv)
            log.info("voice %s: %d frames, level %.1f dBFS, gain %+.1f", k, codes.shape[0], lv["level_db"], lv["gain_db"])
    # 워밍업: 참조(있으면) + 글 + 6프레임 생성(컴파일·CUDA 그래프)
    s = W.Session(model, sc, tok); v = next(iter(S["voices"].values()), None)
    if v is not None: s.append_turn(0, v["text"], v["codes"])
    s.append_text(0, "네, 안녕하세요."); list(W.Generator(model, sc, s, S["codec"], max_frames=6).run())
    if comp:
        s.reset(); s.append_text(0, "네, 안녕하세요."); list(W.Generator(model, sc, s, S["codec"], max_frames=6).run())
    S["ready"] = True; log.info("ready in %.0f s (%s, compile=%s, voices=%d)", time.time() - t0, dev, comp, len(S["voices"]))


def _session(sid):
    if not S["ready"]: raise HTTPException(503, "warming up")
    if S["session"] is None or S["session_id"] != sid: raise HTTPException(404, f"unknown session: {sid}")
    return S["session"]


def _busy():
    if S["lock"].locked(): raise HTTPException(409, "busy: generating")


def _locked(fn):
    """세션 상태(s.pos·s.turns·S["session"])를 바꾸는 핸들러는 speak 와 같은 락을 쥔 채 돈다 — 동기 핸들러라 스레드풀에서 겹칠 수 있다(리뷰 지적). 생성 중이면 409."""
    @functools.wraps(fn)
    def wrapper(*a, **k):
        if not S["lock"].acquire(timeout=0.5): raise HTTPException(409, "busy: generating")
        try: return fn(*a, **k)
        finally: S["lock"].release()
    return wrapper


class OpenReq(BaseModel): session_id: str; voice: str
class TextReq(BaseModel): session_id: str; text: str = Field(min_length=1, max_length=MAX_CHARS)
class SidReq(BaseModel): session_id: str
class CtxTurn(BaseModel): tag: int; text: str; pcm_b64: str
class CtxReq(BaseModel): session_id: str; turns: list[CtxTurn]


@app.get("/health")
def health():
    return dict(ready=S["ready"], busy=S["lock"].locked(), session=S["session_id"], voices=list(S["voices"]))


@app.post("/v1/session/open")
@_locked
def open_session(r: OpenReq):
    if not S["ready"]: raise HTTPException(503, "warming up")
    if r.voice not in S["voices"] and r.voice != "none": raise HTTPException(404, f"unknown voice: {r.voice}")
    s = W.Session(S["model"], S["sc"], S["tok"]); ms = 0.0
    if r.voice != "none": v = S["voices"][r.voice]; ms = s.append_turn(0, v["text"], v["codes"])
    S["session"], S["session_id"] = s, r.session_id; log.info("open %s voice=%s prefill %.0f ms", r.session_id, r.voice, ms)
    return dict(ok=True, prefill_ms=round(ms, 1), positions=s.pos)


class UserReq(BaseModel): session_id: str; text: str = Field(min_length=1, max_length=MAX_CHARS); pcm_b64: str


@app.post("/v1/session/user")
@_locked
def user_turn(r: UserReq):
    s = _session(r.session_id); body = base64.b64decode(r.pcm_b64)
    if len(body) < 2: raise HTTPException(400, "empty pcm")
    pcm = np.frombuffer(body[: len(body) - len(body) % 2], "<i2"); codes, lv = S["codec"].encode_pcm16k(pcm); ms = s.append_turn(1, r.text, codes)
    log.info("user %s: %.1f s → %d frames, level %.1f dBFS gain %+.1f, prefill %.0f ms, pos %d", r.session_id, len(pcm) / 16000, codes.shape[0], lv["level_db"], lv["gain_db"], ms, s.pos)
    return dict(ok=True, frames=int(codes.shape[0]), prefill_ms=round(ms, 1), level_db=lv["level_db"], gain_db=lv["gain_db"], positions=s.pos)


@app.post("/v1/session/context")
@_locked
def context(r: CtxReq):
    s = _session(r.session_id)
    for t in r.turns:
        pcm = np.frombuffer(base64.b64decode(t.pcm_b64), "<i2"); codes, _ = S["codec"].encode_pcm16k(pcm); s.append_turn(t.tag, t.text, codes)
    return dict(ok=True, positions=s.pos, turns=len(s.turns))


class CodesTurn(BaseModel): tag: int; text: str; codes_b64: str; frames: int
class CodesReq(BaseModel): session_id: str; turns: list[CodesTurn]


@app.post("/v1/session/context_codes")
@_locked
def context_codes(r: CodesReq):
    """평가·재생용(W2): 이미 토큰화된 코드(int16 [T,32] little-endian, base64)를 그대로 문맥으로 넣는다 — E-B 와 같은 코드로 재현하기 위해."""
    s = _session(r.session_id)
    for t in r.turns:
        codes = torch.from_numpy(np.frombuffer(base64.b64decode(t.codes_b64), "<i2").astype(np.int64).reshape(t.frames, W.NCB)); s.append_turn(t.tag, t.text, codes)
    return dict(ok=True, positions=s.pos, turns=len(s.turns))


@app.post("/v1/session/speak")
def speak(r: TextReq):
    s = _session(r.session_id); _busy()
    if not S["lock"].acquire(timeout=0.5): raise HTTPException(409, "busy")
    cancel = threading.Event(); S["cancel"] = cancel

    def gen():
        t0 = time.perf_counter(); g = None; it = None
        try:
            s.append_text(0, r.text); g = W.Generator(S["model"], S["sc"], s, S["codec"], max_frames=int(SEG_MAX_S / W.FRAME_S), gain_db=S["gen_gain"])
            it = g.run(cancel)                                      # 같은 생성기를 잡아 둔다 — 끊김 뒤에도 이 생성기를 이어서 비워야 낸 프레임이 확정된다
            for chunk in it: yield chunk
        except GeneratorExit:                                       # 클라이언트가 끊음 → 멈춰 있던 그 생성기를 닫는다(run 의 finally 가 낸 프레임까지 재인코딩·확정)
            cancel.set()
            if it is not None: it.close()                           # 새 생성기를 만들면 frames 가 비어 0프레임이 확정된다(리뷰 지적)
            raise
        finally:
            info = g.info if g is not None else {}
            if info.get("raw_level_db") is not None: S["gen_gain"] = float(np.clip(S["gen_gain"] + (float(os.environ.get("CSM_TARGET_DB", "-26")) - info["raw_level_db"] - S["gen_gain"]) * 0.5, -6, 12))   # 다음 세그먼트 이득 절반씩 보정
            log.info("speak %s: %s → %s frames eos=%s cancelled=%s ttfa %s ms rtf %s level %s gain→%.1f pos %d (%.2f s)", r.session_id, r.text[:20], info.get("frames"), info.get("eos"), info.get("cancelled"),
                     None if info.get("ttfa_ms") is None else round(info["ttfa_ms"]), None if info.get("rtf") is None else round(info["rtf"], 3), None if info.get("raw_level_db") is None else round(info["raw_level_db"], 1), S["gen_gain"], s.pos, time.perf_counter() - t0)
            S["cancel"] = None; S["lock"].release()
    return StreamingResponse(gen(), media_type="audio/L16", headers={"X-Sample-Rate": "16000"})


@app.post("/v1/session/cancel")
def cancel(r: SidReq):
    _session(r.session_id)
    if S["cancel"] is not None: S["cancel"].set()
    return dict(ok=True)


@app.post("/v1/session/close")
@_locked
def close(r: SidReq):
    if S["session_id"] == r.session_id: S["session"], S["session_id"] = None, None
    return dict(ok=True)
