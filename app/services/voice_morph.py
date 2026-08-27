from __future__ import annotations

import librosa
import numpy as np
from scipy.signal import butter, filtfilt

DEFAULT_SEMITONES = 5.5

_N_FFT = 2048
_HOP = 512
_LIFTER = 34


def _f0_curve(y: np.ndarray, sr: int, hop: int = 256) -> np.ndarray:
    """결측 구간은 주변값으로 처리"""
    f0, _, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=hop)
    good = ~np.isnan(f0)
    if good.sum() < 2:
        return np.full(len(y), 120.0)

    idx = np.arange(len(f0))
    filled = np.interp(idx, idx[good], f0[good])
    return np.interp(np.arange(len(y)), idx * hop, filled)


def _pitch_marks(y: np.ndarray, sr: int, f0_s: np.ndarray) -> np.ndarray:
    """피치 마킹"""
    b, a = butter(4, 900 / (sr / 2), btype="low")
    lp = filtfilt(b, a, y)

    marks: list[int] = []
    pos = int(sr / f0_s[0])
    while pos < len(y) - 1:
        period = sr / f0_s[min(pos, len(f0_s) - 1)]
        lo, hi = int(pos + 0.8 * period), int(pos + 1.2 * period)
        if hi >= len(y):
            break
        marks.append(lo + int(np.argmax(lp[lo:hi])))
        pos = marks[-1]
    return np.asarray(marks, dtype=int)


def pitch_shift(y: np.ndarray, sr: int, alpha: float) -> np.ndarray:
    """길이는 유지한 채 피치만 alpha 배. 포먼트는 움직이지 않는다."""
    if abs(alpha - 1.0) < 1e-4 or len(y) < sr // 10:
        return y.copy()

    marks = _pitch_marks(y, sr, _f0_curve(y, sr))
    if len(marks) < 3:
        return y.copy()

    periods = np.append(np.diff(marks), np.diff(marks)[-1])

    out = np.zeros(len(y) + sr // 2)
    wsum = np.zeros_like(out)

    synth = float(marks[0])
    while synth < len(y):
        j = int(np.argmin(np.abs(marks - synth)))
        center = int(marks[j])
        half = int(np.clip(periods[j], 8, sr // 40))

        start_out = int(synth) - half
        start_in, end_in = center - half, center + half
        if start_in >= 0 and end_in <= len(y) and start_out >= 0:
            window = np.hanning(2 * half)
            out[start_out : start_out + 2 * half] += y[start_in:end_in] * window
            wsum[start_out : start_out + 2 * half] += window

        synth += half / alpha

    return (out / np.maximum(wsum, 0.35))[: len(y)]


def _envelope(log_mag: np.ndarray) -> np.ndarray:
    cep = np.fft.irfft(log_mag, axis=0)
    cep[_LIFTER:-_LIFTER] = 0.0
    return np.fft.rfft(cep, axis=0).real


def formant_shift(y: np.ndarray, beta: float) -> np.ndarray:
    """스펙트럼만 주파수축으로 beta 배"""
    if abs(beta - 1.0) < 1e-3 or len(y) < _N_FFT:
        return y

    spec = librosa.stft(y, n_fft=_N_FFT, hop_length=_HOP)
    log_mag = np.log(np.abs(spec) + 1e-9)
    env = _envelope(log_mag)

    bins = np.arange(env.shape[0])
    source_bins = bins / beta
    warped = np.empty_like(env)
    for frame in range(env.shape[1]):
        warped[:, frame] = np.interp(source_bins, bins, env[:, frame])

    new_spec = np.exp(log_mag - env + warped) * np.exp(1j * np.angle(spec))
    return librosa.istft(new_spec, hop_length=_HOP, length=len(y))


def morph(y: np.ndarray, sr: int, semitones: float = DEFAULT_SEMITONES) -> np.ndarray:
    """피치를 semitones 반음 옮김"""
    alpha = 2.0 ** (semitones / 12.0)
    return formant_shift(pitch_shift(y, sr, alpha), alpha ** (1 / 3))


def morph_pcm(pcm: bytes, sr: int, semitones: float = DEFAULT_SEMITONES) -> bytes:
    """16-bit PCM 을 받아 변조해서 같은 형식으로 반환"""
    if len(pcm) % 2:
        pcm = pcm[:-1]
    if not pcm:
        return pcm

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    out = morph(samples, sr, semitones)
    clipped = np.clip(out, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()
