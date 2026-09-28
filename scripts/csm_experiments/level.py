# -*- coding: utf-8 -*-
"""턴 레벨 정규화 — 토큰화(tok_ktel.py)·채점기(g1_score.py)·워커가 **같은 함수**를 쓴다. numpy 만. 설계: notes/15-b2-레벨-정규화-설계.md §2.

  speech_rms_db(x, sr)  말소리 RMS(dBFS): 20 ms 비중첩 프레임 전력(dB) → 게이트 = max(p90 − 20, p10 + 10) → 게이트 위 프레임 전력 평균. 게이트 위 프레임이 없으면 p90.
  analyze(x, sr, target=-26, peak_cap=-2, gain_max=30) → dict(level_db, gain_db, peak_db, floor_db, speech_s, capped, weak)
      gain = target − level, 상한 gain_max 와 (peak_cap − peak) 중 작은 것. 선형 이득만(리미터·압축·게이트 없음).
      weak = 말소리 프레임 < 0.3 s 또는 (말소리 − 바닥) < 10 dB — 1차 라운드는 플래그만 기록하고 이득은 그대로 준다.
  apply_gain(x, gain_db) → x × 10^(gain/20), float32, [-1, 1] 로 clip(걸리면 안전망).
실측 근거(2026-09-28 맥): 상담 원본 −38.5 dBFS · KsponSpeech −31.3 · 원본 CSM 출력 −23.1 · 파고율 최대 24 dB → −26 이면 피크 > −2 dBFS 1/100.
"""
import numpy as np

FRAME_MS, GATE_BELOW_P90, GATE_ABOVE_P10 = 20, 20.0, 10.0
WEAK_SPEECH_S, WEAK_SNR_DB = 0.3, 10.0


def _frames_db(x, sr):
    n = max(1, int(sr * FRAME_MS / 1000)); m = len(x) // n
    if m == 0:
        return np.array([10 * np.log10(np.mean(np.square(x, dtype=np.float64)) + 1e-12)]), np.array([np.mean(np.square(x, dtype=np.float64)) + 1e-12]), n
    fr = np.asarray(x[: m * n], dtype=np.float64).reshape(m, n); pw = (fr ** 2).mean(1) + 1e-12
    return 10 * np.log10(pw), pw, n


def speech_rms_db(x, sr):
    db, pw, _ = _frames_db(x, sr)
    gate = max(np.percentile(db, 90) - GATE_BELOW_P90, np.percentile(db, 10) + GATE_ABOVE_P10); act = pw[db > gate]
    return float(10 * np.log10(act.mean())) if act.size else float(np.percentile(db, 90))


def peak_db(x):
    return float(20 * np.log10(np.max(np.abs(x)) + 1e-9))


def analyze(x, sr, target=-26.0, peak_cap=-2.0, gain_max=30.0):
    db, pw, n = _frames_db(x, sr)
    gate = max(np.percentile(db, 90) - GATE_BELOW_P90, np.percentile(db, 10) + GATE_ABOVE_P10); act = db > gate
    level = float(10 * np.log10(pw[act].mean())) if act.any() else float(np.percentile(db, 90))
    floor = float(np.percentile(db, 10)); peak = peak_db(x); speech_s = float(act.sum() * n / sr)
    want = target - level; gain = min(want, gain_max, peak_cap - peak)
    return dict(level_db=round(level, 2), gain_db=round(float(gain), 2), peak_db=round(peak, 2), floor_db=round(floor, 2), speech_s=round(speech_s, 3),
                capped=bool(gain < want - 1e-9), weak=bool(speech_s < WEAK_SPEECH_S or (level - floor) < WEAK_SNR_DB))


def apply_gain(x, gain_db):
    return np.clip(np.asarray(x, dtype=np.float32) * np.float32(10 ** (gain_db / 20)), -1.0, 1.0)
