from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

SR = 16_000
MAX_ATTEMPTS = 8
ABRUPT_MS = 10.0
TURN_GAP_MS = 400
TURN_GAP_PCM = b"\x00" * (SR * TURN_GAP_MS // 1000 * 2)
_BLOCK_S, _LOUD_DB, _QUIET_DB, _MIN_SIL_S = 0.002, -25.0, -45.0, 0.15  # scripts/tts_bench/post.py 끊김 판정과 같은 값
_FRAME_S, _SOUND_DBFS, _ONSET_FRAMES = 0.02, -45.0, 3
_LEAD_KEEP_S, _TAIL_KEEP_S = 0.10, 0.12
_LEN_FLOOR_S, _LEN_PER_SYLLABLE_S = 1.5, 0.6
_PEAK_CAP = 10 ** (-1 / 20)
LEVEL_DB = {"ai": -16.94749568714168, "user": -22.544892951720513}  # 프리셋 79턴 청취본 역할별 중앙값
_SYLLABLE = re.compile(r"[가-힣0-9A-Za-z]")


@dataclass(frozen=True)
class Render:
    pcm: bytes
    decay_ms: float | None
    span_s: float | None
    cut: bool
    long: bool


def _samples(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0


def _block_db(w: np.ndarray) -> np.ndarray:
    """2ms 블록 최대진폭, 파일 피크 대비 dB"""
    b = int(SR * _BLOCK_S)
    n = len(w) // b
    env = np.abs(w[: n * b]).reshape(n, b).max(1)
    return 20 * np.log10(env / (env.max(initial=0.0) + 1e-12) + 1e-12)


def _ends(w: np.ndarray) -> list[tuple[int, int, float]]:
    """말소리 끝마다 (끝 샘플, 다음 소리 샘플, 감쇠 ms)"""
    b = int(SR * _BLOCK_S)
    db = _block_db(w)
    n = len(db)
    if n == 0:
        return []
    min_sil = int(_MIN_SIL_S / _BLOCK_S)
    res: list[tuple[int, int, float]] = []
    i = 0
    while i < n:
        if db[i] < _QUIET_DB:
            j = i
            while j < n and db[j] < _QUIET_DB:
                j += 1
            if (j - i >= min_sil or j == n) and i > 0:
                k = i - 1
                while k >= 0 and db[k] < _LOUD_DB:
                    k -= 1
                if k >= 0:
                    res.append((i * b, j * b if j < n else len(w), (i - 1 - k) * _BLOCK_S * 1000))
            i = j
        else:
            i += 1
    if db[-1] >= _QUIET_DB:
        res.append((len(w), len(w), 0.0))
    return res


def _onset(w: np.ndarray) -> int | None:
    """20ms RMS 가 -45dBFS 이상으로 60ms 이어지는 첫 샘플"""
    n = int(SR * _FRAME_S)
    k = len(w) // n
    if k == 0:
        return None
    x = w[: k * n].reshape(k, n).astype(np.float64)
    loud = 10 * np.log10(np.mean(x * x, axis=1) + 1e-20) >= _SOUND_DBFS
    run = 0
    for i, v in enumerate(loud):
        run = run + 1 if v else 0
        if run == _ONSET_FRAMES:
            return (i - _ONSET_FRAMES + 1) * n
    return None


def _rms_end(w: np.ndarray) -> int | None:
    """20ms RMS 가 -45dBFS 이상인 마지막 프레임 끝"""
    n = int(SR * _FRAME_S)
    k = len(w) // n
    if k == 0:
        return None
    x = w[: k * n].reshape(k, n).astype(np.float64)
    loud = np.nonzero(10 * np.log10(np.mean(x * x, axis=1) + 1e-20) >= _SOUND_DBFS)[0]
    return int(loud[-1] + 1) * n if len(loud) else None


def end_decay_ms(pcm: bytes) -> float | None:
    """마지막 말소리가 -25dB 에서 -45dB 로 떨어지는 ms, 말소리 없으면 None"""
    ends = _ends(_samples(pcm))
    return ends[-1][2] if ends else None


def is_cut(pcm: bytes) -> bool:
    """끝이 '뚝' 잘렸는지"""
    decay = end_decay_ms(pcm)
    return decay is not None and decay <= ABRUPT_MS


def syllables(text: str) -> int:
    return len(_SYLLABLE.findall(text))


def speech_span(pcm: bytes) -> float | None:
    """첫 소리부터 마지막 말소리 끝까지 초"""
    w = _samples(pcm)
    onset, ends = _onset(w), _ends(w)
    if onset is None or not ends:
        return None
    return max(0, ends[-1][0] - onset) / SR


def too_long(pcm: bytes, text: str) -> bool:
    """글에 비해 말이 너무 긴지"""
    span = speech_span(pcm)
    return span is not None and span > max(_LEN_FLOOR_S, _LEN_PER_SYLLABLE_S * syllables(text))


def tail_after_rms_s(pcm: bytes) -> float:
    """피크 기준 말소리 끝이 절대 RMS 끝보다 얼마나 뒤인지"""
    w = _samples(pcm)
    ends, rms_end = _ends(w), _rms_end(w)
    if not ends or rms_end is None:
        return 0.0
    return max(0, ends[-1][0] - rms_end) / SR


def evaluate(pcm: bytes, text: str) -> Render:
    decay = end_decay_ms(pcm)
    return Render(
        pcm=pcm,
        decay_ms=decay,
        span_s=speech_span(pcm),
        cut=decay is not None and decay <= ABRUPT_MS,
        long=too_long(pcm, text),
    )


def acceptable(render: Render) -> bool:
    return not render.cut and not render.long


def pick(renders: list[Render]) -> Render:
    """첫 통과 렌더, 다 실패면 길이 통과 중 감쇠 최장, 그것도 없으면 가장 짧은 것"""
    for render in renders:
        if acceptable(render):
            return render
    fits = [r for r in renders if not r.long]
    if fits:
        best = fits[0]
        for r in fits[1:]:
            if (r.decay_ms or 0) > (best.decay_ms or 0):
                best = r
        return best
    return min(renders, key=lambda r: r.span_s if r.span_s is not None else float("inf"))


def trim(pcm: bytes) -> bytes:
    """앞 무음은 100ms, 뒤는 마지막 말소리 끝 + 120ms"""
    w = _samples(pcm)
    ends = _ends(w)
    onset = _onset(w)
    if onset is None or not ends:
        return pcm[: len(pcm) // 2 * 2]
    start = max(0, onset - int(SR * _LEAD_KEEP_S))
    stop = min(len(w), ends[-1][0] + int(SR * _TAIL_KEEP_S))
    return pcm[start * 2 : stop * 2]


def speech_db(pcm: bytes) -> float | None:
    """20ms RMS 가 -45dBFS 넘는 프레임의 에너지 평균"""
    x = _samples(pcm)
    n = SR // 50
    k = len(x) // n
    db = 10 * np.log10((x[: k * n].reshape(k, n).astype(np.float64) ** 2).mean(1) + 1e-12)
    loud = db[db > -45]
    return float(10 * np.log10(np.mean(10 ** (loud / 10)))) if len(loud) else None


def match_level(pcm: bytes, target_db: float) -> tuple[bytes, float]:
    """말소리 크기를 목표에 맞추고 피크는 -1dBFS 로 막는다 → (소리, 이득 dB)"""
    x = _samples(pcm)
    level = speech_db(pcm)
    gain = 10 ** ((target_db - level) / 20) if level is not None else 1.0
    y = x * gain
    peak = float(np.abs(y).max()) if len(y) else 0
    if peak > _PEAK_CAP:
        y *= _PEAK_CAP / peak
    return (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes(), float(20 * np.log10(gain))


def join_turns(parts: list[bytes]) -> bytes:
    return TURN_GAP_PCM.join(parts)


def finish(renders: list[Render], target_db: float) -> tuple[bytes, float, Render]:
    """고른 렌더를 다듬고 크기를 맞춘다"""
    chosen = pick(renders)
    pcm, gain = match_level(trim(chosen.pcm), target_db)
    return pcm, gain, chosen
