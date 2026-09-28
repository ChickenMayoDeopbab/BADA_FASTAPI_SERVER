# -*- coding: utf-8 -*-
"""CSM 워커 내부 — 전송(HTTP)과 분리된 세 클래스. 설계: notes/16-CSM-워커-설계.md §2.2, 계획 16a.

  Session    KV 캐시에 턴을 **이어 붙여** 프리필한다(정적 루프 StaticStack 이 위치 배열로 K/V 를 쓴다). 예산을 넘으면 참조 턴 + 최근 턴만 남기고 재프리필.
  Codec      Mimi 인코드(레벨 정규화 뒤) · 세그먼트 프리픽스 재디코드로 새 프레임만 파형으로(창 디코드는 부정확 — 실측).
  Generator  프레임 루프(백본 1 + depth 32) · EOS · 취소 · 청크 · 세그먼트 끝에 이득 준 파형을 재인코딩해 캐시에 확정.
"""
import os, sys, time, threading
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
for d in (HERE, os.path.join(HERE, "..", "csm_experiments")):  # B 레포 배치: g1_generate·g0c_static_loop·level 은 scripts/csm_experiments/
    if d not in sys.path: sys.path.insert(0, d)
import g1_generate as G                                   # load · text_ids · embed_inputs · NCB · FRAME_S · SR
from g0c_static_loop import StaticCsm, make_sampler
from level import analyze, apply_gain, speech_rms_db

NCB, FRAME_S, SR24, SR16 = G.NCB, G.FRAME_S, 24000, 16000


def now(dev):
    if dev.type == "cuda": torch.cuda.synchronize(dev)
    return time.perf_counter()


class Session:
    """캐시 상태. turns[k] = dict(tag, ids, codes[T,32] | None, start) — start = 그 턴의 첫 위치. pos = 캐시에 든 위치 수."""

    def __init__(self, model, sc, tok, budget=2048 - 250 - 32):
        self.model, self.sc, self.tok, self.cfg, self.budget = model, sc, tok, model.config, budget
        self.reset()

    def reset(self):
        self.turns, self.pos = [], 0

    # ── 임베딩 ──
    def _ids_spans(self, tag, text, codes):
        ids = G.text_ids(self.tok, text, tag) if text is not None else []
        if codes is None: return ids, []
        return ids + [self.cfg.audio_token_id] * codes.shape[0] + [self.cfg.audio_eos_token_id], [(len(ids), codes)]

    def _prefill(self, x, start):
        """x [1,S,H] 를 위치 start.. 에 쓴다. 마지막 위치의 백본 출력을 s.frame(코드북0 샘플)·s.emb 에 남긴다."""
        pos = torch.arange(start, start + x.shape[1], device=x.device)
        self.sc.bb.head(self.sc.bb.stack(x, pos)[:, -1]); self.pos = start + x.shape[1]

    def _turn_len(self, t):
        return len(t["ids"]) + (0 if t["codes"] is None else t["codes"].shape[0] + 1)

    # ── 공개 ──
    @torch.no_grad()
    def append_turn(self, tag, text, codes):
        """글 + 오디오(+eos) 턴을 이어 붙인다. 예산을 넘으면 먼저 재프리필. → 프리필 ms"""
        t0 = now(self.model.device); ids, spans = self._ids_spans(tag, text, codes)
        turn = dict(tag=tag, ids=G.text_ids(self.tok, text, tag), codes=codes, start=self.pos)
        if self.pos + len(ids) > self.budget:
            self.turns.append(turn); self.rebase(); return (now(self.model.device) - t0) * 1e3
        self._prefill(G.embed_inputs(self.model, ids, spans), self.pos); self.turns.append(turn)
        return (now(self.model.device) - t0) * 1e3

    @torch.no_grad()
    def append_text(self, tag, text):
        """목표 턴의 글만 붙인다(생성 직전). 예산 검사는 글 + 최대 생성 프레임까지 본다."""
        t0 = now(self.model.device); ids = G.text_ids(self.tok, text, tag)
        turn = dict(tag=tag, ids=ids, codes=None, start=self.pos)
        if self.pos + len(ids) > self.budget:
            self.turns.append(turn); self.rebase(); return (now(self.model.device) - t0) * 1e3
        self._prefill(G.embed_inputs(self.model, ids, []), self.pos); self.turns.append(turn)
        return (now(self.model.device) - t0) * 1e3

    @torch.no_grad()
    def commit_audio(self, codes):
        """생성이 끝난 뒤: 마지막 글 턴의 오디오 자리(start + len(ids) 부터)에 (재인코딩한) 코드 + eos 를 써서 확정한다."""
        t = self.turns[-1]; assert t["codes"] is None, "마지막 턴이 글만인 상태여야 한다"
        a0 = t["start"] + len(t["ids"]); ids = [self.cfg.audio_token_id] * codes.shape[0] + [self.cfg.audio_eos_token_id]
        self._prefill(G.embed_inputs(self.model, ids, [(0, codes)]), a0); t["codes"] = codes

    @torch.no_grad()
    def rebase(self):
        """참조 턴(0번) + 뒤에서부터 예산 안에 드는 턴만 남기고 처음부터 다시 프리필한다."""
        keep = [self.turns[0]] if self.turns else []; used = self._turn_len(keep[0]) if keep else 0
        rest = []
        for t in reversed(self.turns[1:]):
            L = self._turn_len(t)
            if used + L > self.budget: break
            rest.append(t); used += L
        self.turns = keep + rest[::-1]; self.pos = 0
        ids, spans = [], []
        for t in self.turns:
            i, s = self._ids_spans(t["tag"], None, t["codes"]) if False else (None, None)
            tid = t["ids"]; t["start"] = len(ids)
            if t["codes"] is None: ids += tid
            else: spans.append((len(ids) + len(tid), t["codes"])); ids += tid + [self.cfg.audio_token_id] * t["codes"].shape[0] + [self.cfg.audio_eos_token_id]
        if ids: self._prefill(G.embed_inputs(self.model, ids, spans), 0)

    @torch.no_grad()
    def greedy_first_frame(self):
        """시험용: 현재 캐시 끝에서 depth 32스텝으로 첫 프레임을 뽑는다(샘플러가 탐욕이어야 결정적)."""
        for p in range(NCB): self.sc.dd_step(self.sc.P[p])
        return self.sc.s.frame.clone()[0]


class Codec:
    """Mimi 인코드(레벨 정규화 → 16 k→24 k → encode) · 프리픽스 재디코드(세그먼트 처음부터 디코드해 뒤 new 프레임만)."""

    def __init__(self, model, target_db=-26.0):
        self.mimi, self.dev, self.target = model.codec_model, model.codec_model.device, target_db
        self.k16 = G.resample_16k_to_24k                          # torch Kaiser-sinc(tok_kspon 과 같은 필터)

    @torch.no_grad()
    def encode_pcm16k(self, pcm, normalize=True):
        """pcm: int16 ndarray(16 kHz mono) → (codes[T,32] long, level dict)"""
        x = pcm.astype(np.float32) / 32768.0; lv = analyze(x, SR16, self.target)
        if normalize: x = apply_gain(x, lv["gain_db"])
        x24 = self.k16(torch.from_numpy(np.ascontiguousarray(x))).to(self.dev, self.mimi.dtype)
        return self.mimi.encode(x24[None, None]).audio_codes[0].T.long().cpu(), lv

    @torch.no_grad()
    def encode_pcm24k(self, x24):
        """이미 24 kHz float 파형(재인코딩용) → codes[T,32]"""
        return self.mimi.encode(torch.as_tensor(x24, dtype=torch.float32)[None, None].to(self.dev, self.mimi.dtype)).audio_codes[0].T.long().cpu()

    @torch.no_grad()
    def decode_tail(self, codes, new):
        """세그먼트의 코드 codes[T,32] 를 **처음부터** 디코드해 뒤 new 프레임(24 kHz float32)만 돌려준다.
        Mimi 디코더는 인과라 프리픽스 디코드가 전체 디코드의 앞부분과 같다(맥 실측 4e-8). 창(마지막 W 프레임)만 디코드하면 슬라이딩 창 250 때문에 0.3~1 % 어긋나므로 쓰지 않는다.
        세그먼트 ≤ 250프레임이라 호출당 비용이 유계(CPU 150프레임 132 ms, GPU 는 실측 예정)."""
        y = self.mimi.decode(codes.T[None].to(self.dev)).audio_values[0, 0].float().cpu().numpy()
        n = new * 1920                                            # 1,920 샘플 = 1 프레임
        return y[-n:] if n <= len(y) else y

class Resampler24to16:
    """24 k→16 k(2/3) Kaiser-sinc, g1_score.resample 과 같은 설계(48 kHz 격자 385탭). 프리픽스 전체를 torch conv1d 로 돌리고 필터 꼬리(holdback)만 보류한다."""

    def __init__(self, device):
        up, down = 2, 3; half = 64 * max(up, down); fc = 0.95625 * 16000 / 2 / (24000 * up)
        n = np.arange(-half, half + 1); h = (2 * fc * np.sinc(2 * fc * n) * np.kaiser(2 * half + 1, 8.6) * up).astype(np.float32)
        self.h, self.half, self.up, self.down, self.dev = torch.from_numpy(h).to(device), half, up, down, device
        self.holdback = -(-half // down)                          # 16 k 샘플 수(64): 아직 필터 미래 입력이 없는 꼬리

    @torch.no_grad()
    def __call__(self, x24):
        x = torch.as_tensor(np.ascontiguousarray(x24), dtype=torch.float32, device=self.dev)
        z = torch.zeros(x.numel() * self.up, device=self.dev); z[:: self.up] = x
        y = torch.nn.functional.conv1d(z[None, None], self.h[None, None], padding=self.half)[0, 0, :: self.down]
        return y[: -(-x.numel() * self.up // self.down)].cpu().numpy()


class Generator:
    """session.append_text 가 끝난 캐시 끝에서 프레임을 만든다. 청크(16 kHz int16 bytes)를 yield 하고, 끝에 이득 준 파형을 재인코딩해 session.commit_audio."""

    def __init__(self, model, sc, session, codec, max_frames=250, chunk_frames=2, gain_db=3.0, target_db=-26.0):
        self.model, self.sc, self.session, self.codec = model, sc, session, codec
        self.max_frames, self.chunk, self.gain_db, self.target = max_frames, chunk_frames, gain_db, target_db
        self.rs = Resampler24to16(model.device); self.info = {}

    @torch.no_grad()
    def run(self, cancel=None):
        dev, sc, s = self.model.device, self.sc, self.session; t0 = now(dev); T0 = s.pos
        limit = min(self.max_frames, self.model.config.max_position_embeddings - 2 - T0, 2048 - 2 - T0)
        frames, eos, ttfa, sent16, done = [], False, None, 0, False; y24 = np.zeros(0, np.float32); raw_level = None
        try:
            for f in range(limit):
                if cancel is not None and cancel.is_set(): break
                if f: sc.bb_step(sc.P[T0 + f - 1])
                for p in range(NCB): sc.dd_step(sc.P[p])
                fr = sc.s.frame.clone()
                if bool((fr[:, 1:] == 0).all()): eos = True; break
                frames.append(fr[0].cpu())
                if len(frames) % self.chunk == 0:
                    y24 = apply_gain(self.codec.decode_tail(torch.stack(frames), len(frames)), self.gain_db)   # 세그먼트 전체(프리픽스) 디코드
                    y16 = self.rs(y24); settled = len(y16) - self.rs.holdback
                    if settled > sent16:
                        if ttfa is None: ttfa = (now(dev) - t0) * 1e3
                        yield (np.clip(y16[sent16:settled], -1, 1) * 32767).astype("<i2").tobytes(); sent16 = settled
            # 정상 끝(EOS·상한·cancel): 남은 꼬리까지 내보낸다
            if frames:
                y24 = apply_gain(self.codec.decode_tail(torch.stack(frames), len(frames)), self.gain_db); y16 = self.rs(y24)
                if len(y16) > sent16:
                    if ttfa is None: ttfa = (now(dev) - t0) * 1e3
                    yield (np.clip(y16[sent16:], -1, 1) * 32767).astype("<i2").tobytes(); sent16 = len(y16)
            done = True
        finally:
            # 어떻게 끝났든(EOS·상한·cancel·소비자가 끊어 GeneratorExit·예외) 낸 프레임을 이득 준 파형으로 재인코딩해 캐시에 확정한다(여기서는 yield 없음).
            if frames:
                if not done: y24 = apply_gain(self.codec.decode_tail(torch.stack(frames), len(frames)), self.gain_db)   # 끊긴 경우: 마지막 청크 뒤 프레임까지 다시 디코드
                raw_level = speech_rms_db(y24, SR24) - self.gain_db; lv = analyze(y24, SR24, self.target)     # 재인코딩은 정확히 −26 으로
                codes_re = self.codec.encode_pcm24k(apply_gain(y24, lv["gain_db"]))
                s.commit_audio(codes_re[: len(frames)] if codes_re.shape[0] >= len(frames) else codes_re)
            else:
                s.commit_audio(torch.zeros(0, NCB, dtype=torch.long))
            total = (now(dev) - t0) * 1e3
            self.info = dict(frames=len(frames), eos=eos, cancelled=(not done) or bool(cancel is not None and cancel.is_set()), ttfa_ms=ttfa, total_ms=total,
                             rtf=(total / 1e3) / max(len(frames) * FRAME_S, FRAME_S), raw_level_db=raw_level, gain_db=self.gain_db, samples16=sent16)


def load_model(weights, repo="sesame/csm-1b", device="cpu", dtype=None, greedy=False, compile_=False, backend="inductor"):
    dev = torch.device(device); dt = dtype or (torch.bfloat16 if dev.type == "cuda" else torch.float32)
    proc, model = G.load(repo, weights, dt, dev); sc = StaticCsm(model, greedy, compile_, backend)
    return proc.tokenizer, model, sc
