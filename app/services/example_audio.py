from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import numpy as np

SR = 16_000
TAIL_PHRASE = " ... 네."
MAX_ATTEMPTS = 8
ABRUPT_MS = 10.0
_MAX_CHARS = 300  # 워커 요청 글자 상한
_BLOCK_S, _LOUD_DB, _QUIET_DB, _MIN_SIL_S = 0.002, -25.0, -45.0, 0.15 # scripts/tts_bench/post.py의 끊김 판정과 같은 값
_FRAME_S, _SOUND_DBFS, _ONSET_FRAMES = 0.02, -45.0, 3
_TAIL_KEEP_S, _LEAD_KEEP_S = 0.12, 0.10
_TAIL_MIN_S, _TAIL_MAX_S, _TAIL_MAX_SYLLABLES = 0.05, 0.8, 4.0
_DUMMY_MAX_S = 0.55  # 네 렌더 길이 상한
_WORD_GAP_S = 0.06  # 꼬리 안에 이만큼 쉬면 두 낱말
_CLICK_MAX_S = 0.012  # 쉼 속 이보다 짧은 소리는 딸깍
_MAX_INNER_PAUSE_S = 1.8  # 남길 말 안에 이보다 긴 쉼은 '...' 망설임(실측 한 요청 최대 1.36초)
_MERGED_GAP_S, _MERGED_WORD_S = 0.1, (0.1, 0.55)  # 남길 말 끝이 이 틈 뒤 '네' 길이 조각이면 붙은 꼬리말
_MERGED_PHRASE_MIN, _MERGED_SYLLABLES = 4, 2.5  # 대사 마지막 토막이 이만큼 길고 조각이 이 음절 길이 이하일 때만
_PHRASE_BREAK = re.compile(r"[,.?!…]+")
_SYLLABLE = re.compile(r"[가-힣0-9]")


@dataclass(frozen=True)
class CleanTurn:
    pcm: bytes
    attempts: int
    cut: bool


def _samples(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0


def _block_db(w: np.ndarray) -> np.ndarray:
    """2ms 블록 최대진폭, 파일 피크 대비 dB"""
    b = int(SR * _BLOCK_S)
    n = len(w) // b
    env = np.abs(w[: n * b]).reshape(n, b).max(1)
    return 20 * np.log10(env / (env.max(initial=0.0) + 1e-12) + 1e-12)


def _ends(w: np.ndarray) -> list[tuple[int, int, float]]:
    """말소리 끝마다"""
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


def _longest_quiet_s(w: np.ndarray, start: int, end: int) -> float:
    """start~end 안에서 가장 긴 -45dB 아래 구간 길이"""
    b = int(SR * _BLOCK_S)
    quiet = _block_db(w)[start // b : end // b] < _QUIET_DB
    run = best = 0
    for q in quiet:
        run = run + 1 if q else 0
        best = max(best, run)
    return best * _BLOCK_S


def _last_word_s(w: np.ndarray, start: int, end: int, gap_s: float) -> float | None:
    """start~end 마지막 큰 소리 앞 gap_s 이상 틈부터 그 큰 소리까지 길이, 숨소리는 안 셈"""
    b = int(SR * _BLOCK_S)
    db = _block_db(w)[start // b : end // b]
    loud = np.nonzero(db >= _LOUD_DB)[0]
    if not len(loud):
        return None
    last_loud = int(loud[-1])
    last, run = None, 0
    for i, q in enumerate(db[:last_loud] < _QUIET_DB):
        run = run + 1 if q else 0
        if run * _BLOCK_S >= gap_s:
            last = i + 1
    return None if last is None else (last_loud - last + 1) * _BLOCK_S


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


def end_decay_ms(pcm: bytes) -> float | None:
    """마지막 말소리가 -25dB에서-45dB 로 떨어지는 ms, 말소리 없으면 None"""
    ends = _ends(_samples(pcm))
    return ends[-1][2] if ends else None


def is_cut(pcm: bytes) -> bool:
    """끝이 '뚝' 잘렸는지"""
    decay = end_decay_ms(pcm)
    return decay is not None and decay <= ABRUPT_MS


def syllables(text: str) -> int:
    return len(_SYLLABLE.findall(text))


def _final_phrase_syllables(text: str) -> int:
    """마지막 문장부호 뒤 토막의 음절 수"""
    parts = [p for p in _PHRASE_BREAK.split(text) if p.strip()]
    return syllables(parts[-1]) if parts else 0


def trim_lead(pcm: bytes) -> bytes:
    """앞 무음을 100ms 로 맞춘다"""
    onset = _onset(_samples(pcm))
    start = 0 if onset is None else max(0, onset - int(SR * _LEAD_KEEP_S))
    return pcm[start * 2 : len(pcm) // 2 * 2]


def cut_tail_phrase(pcm: bytes, text: str) -> tuple[bytes, float] | None:
    """꼬리말 앞 마지막 쉼에서 잘라 (턴 오디오, 끝 감쇠 ms), 못 가르면 None"""
    w = _samples(pcm)
    ends = _ends(w)
    if len(ends) < 2:
        return None
    i = len(ends) - 2
    while i > 0 and ends[i][0] - ends[i - 1][1] <= SR * _CLICK_MAX_S:
        i -= 1
    pause_start, decay = ends[i][0], ends[i][2]
    pause_end, tail_end = ends[-2][1], ends[-1][0]
    tail_s = (tail_end - pause_end) / SR
    onset = _onset(w) or 0
    spoken_s = (pause_start - onset) / SR
    if spoken_s <= 0 or _longest_quiet_s(w, onset, pause_start) > _MAX_INNER_PAUSE_S:
        return None
    per_syllable_s = spoken_s / max(syllables(text), 1)
    last_word = _last_word_s(w, onset, pause_start, _MERGED_GAP_S)
    if (
        last_word is not None
        and _MERGED_WORD_S[0] <= last_word <= min(_MERGED_WORD_S[1], _MERGED_SYLLABLES * per_syllable_s)
        and _final_phrase_syllables(text) >= _MERGED_PHRASE_MIN
    ):
        return None
    if not _TAIL_MIN_S <= tail_s <= min(_TAIL_MAX_S, _TAIL_MAX_SYLLABLES * per_syllable_s):
        return None
    if _longest_quiet_s(w, pause_end, tail_end) >= _WORD_GAP_S:
        return None
    if i > 0 and _TAIL_MIN_S <= (pause_start - ends[i - 1][1]) / SR <= _DUMMY_MAX_S:
        return None
    keep = min(pause_start + int(SR * _TAIL_KEEP_S), ends[i][1])
    return pcm[: keep * 2], decay


async def clean_turn(
    synth: Callable[[str], Awaitable[bytes]], text: str, attempts: int = MAX_ATTEMPTS
) -> CleanTurn:
    """꼬리말 붙여 합성해 잘라내고, 끝이 잘렸거나 못 가르면 다시 합성"""
    tried = 0
    if len(text) + len(TAIL_PHRASE) <= _MAX_CHARS:
        best: tuple[bytes, float] | None = None
        for _ in range(attempts):
            tried += 1
            got = cut_tail_phrase(await synth(text + TAIL_PHRASE), text)
            if got is None:
                continue
            if got[1] > ABRUPT_MS:
                return CleanTurn(trim_lead(got[0]), tried, False)
            if best is None or got[1] > best[1]:
                best = got
        if best is not None:
            return CleanTurn(trim_lead(best[0]), tried, True)
    pcm = await synth(text)
    return CleanTurn(trim_lead(pcm), tried + 1, is_cut(pcm))
