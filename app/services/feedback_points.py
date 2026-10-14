from __future__ import annotations

import math
import re
import statistics
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from enum import StrEnum

import numpy as np

__all__ = [
    "Band", "voice_band", "long_pauses", "turn_levels", "quiet_drops", "answer_delay", "is_long_silence",
    "count_syllables", "speaking_seconds", "speech_rate", "fast_turns",
    "is_safe", "polish", "split_title",
]


class Band(StrEnum):
    GOOD = "GOOD"
    MIDDLE = "MIDDLE"
    NEEDS_WORK = "NEEDS_WORK"


VOICE_GOOD_MAX = 3.0
VOICE_WORK_MIN = 5.0


def voice_band(avti: float | None) -> Band | None:
    """목소리 떨림 밴드. 못 쟀으면 None — '측정 실패'를 사용자에게 알리지 않는다."""
    if avti is None:
        return None
    if avti <= VOICE_GOOD_MAX:
        return Band.GOOD
    if avti >= VOICE_WORK_MIN:
        return Band.NEEDS_WORK
    return Band.MIDDLE


# 말 중간에 소리가 이만큼 넘게 끊기면 '멈춤'으로 본다.
# 점수 계산(training_analysis)의 긴 멈춤 기준과 같은 값이라, 글과 점수가 같은 곳을 가리킨다.
LONG_PAUSE_SEC = 1.5


def long_pauses(
    turn: tuple[float, float], voiced_spans: Sequence[tuple[float, float]]
) -> list[float]:
    """턴 안에서 소리와 소리 사이가 길게 빈 곳들의 길이(초).

    말을 꺼내기 전과 마친 뒤의 빈 시간은 세지 않는다 — 말 중간의 멈춤만 본다.
    """
    start, end = turn
    inside = sorted(
        (max(s, start), min(e, end)) for s, e in voiced_spans if e > start and s < end
    )
    gaps = (later[0] - earlier[1] for earlier, later in zip(inside, inside[1:], strict=False))
    return [round(gap, 2) for gap in gaps if gap >= LONG_PAUSE_SEC]


# 같은 통화의 다른 턴(턴 크기의 중앙값)보다 이만큼 넘게 작으면 '작아진 턴'으로 본다. 6 dB = 소리 폭 절반.
# ★ 실음성 캘리브레이션 대상 — 현재 잠정값. 통화용 마이크의 자동 음량 보정이 차이를 줄일 수 있다.
QUIET_DROP_DB = 6.0
# 짧은 대답("네")은 크기가 들쭉날쭉해서 말소리가 이만큼은 있어야 잰다.
MIN_LEVEL_VOICED_SEC = 1.0
# 견줄 기준을 잡으려면 잰 턴이 이만큼은 있어야 한다.
MIN_LEVEL_TURNS = 3

_SAMPLE_RATE = 16000
_FULL_SCALE = 32767.0


def turn_levels(
    pcm: bytes,
    turns: Sequence[tuple[float, float]],
    voiced_spans: Sequence[tuple[float, float]],
) -> dict[int, float]:
    """턴 번호(1부터) → 그 턴에서 말소리가 난 부분만의 평균 크기(dBFS). 못 잰 턴은 키가 없다.

    멈춘 시간은 빼고 재므로 말 중간의 멈춤이 크기를 깎지 않는다.
    """
    samples = np.frombuffer(pcm, dtype=np.int16, count=len(pcm) // 2)
    spans = sorted(voiced_spans)
    span_ends = [end for _, end in spans]
    levels: dict[int, float] = {}
    for index, (start, end) in enumerate(turns, start=1):
        power, count = 0.0, 0
        for span_start, span_end in spans[bisect_right(span_ends, start):]:
            if span_start >= end:
                break
            lo, hi = max(start, span_start), min(end, span_end)
            part = samples[int(lo * _SAMPLE_RATE):int(hi * _SAMPLE_RATE)].astype(np.float32)
            power += float(np.dot(part, part))
            count += part.size
        if count < MIN_LEVEL_VOICED_SEC * _SAMPLE_RATE or power <= 0:
            continue
        levels[index] = 10 * math.log10(power / count / _FULL_SCALE**2)
    return levels


def quiet_drops(levels: Mapping[int, float]) -> dict[int, float]:
    """턴 번호 → 같은 통화의 다른 턴보다 얼마나 작았는지(dB). 기준을 넘게 작아진 턴만 담는다.

    전화기마다 녹음 크기가 달라 절대값으로는 못 견준다. 그 통화에서 잰 턴 크기의 중앙값과 견준다.
    """
    if len(levels) < MIN_LEVEL_TURNS:
        return {}
    usual = statistics.median(levels.values())
    return {
        index: round(usual - level, 1)
        for index, level in levels.items()
        if usual - level >= QUIET_DROP_DB
    }


# 상대 말이 끝난 뒤 이만큼 넘게 지나서 말을 꺼내면 '대답이 늦은 턴'으로 본다.
# Roberts 외(2006)가 0 / 0.6 / 1.2초를 견줬는데, 1초 아래 침묵은 그것만으로는 듣는 사람의 판단이 갈렸다.
# ★ 실음성 캘리브레이션 대상 — 현재 잠정값. 재생 완료 신호와 마이크 소리 사이의 시간 차가 오차로 섞인다.
LATE_ANSWER_SEC = 1.2

# 말 중간의 멈춤이든 대답 전의 멈춤이든, 이만큼 넘게 조용했을 때만 글에서 '길게' 멈췄다고 쓴다.
# 그 아래는 짚기는 하되 그냥 멈췄다고 쓴다 — 1~2초는 알아챌 만하지만 길다고 하기엔 짧다.
# ★ 실음성 캘리브레이션 대상 — 현재 잠정값.
LONG_SILENCE_SEC = 3.0


def is_long_silence(seconds: float) -> bool:
    """글에서 '길게' 멈췄다고 써도 되는 길이인지."""
    return seconds >= LONG_SILENCE_SEC


# 말소리 사이가 이만큼 넘게 비면 쉼으로 보고 말한 시간에서 뺀다. 그보다 짧은 틈은 말하는 중으로 친다
# (ㅂ·ㄷ·ㄱ 같은 소리를 낼 때 입이 잠깐 막히는 순간, 낱말 사이의 짧은 틈). 말 속도 연구에서 흔히 쓰는 값이다.
ARTICULATION_GAP_SEC = 0.25
# 짧은 대답은 속도가 들쭉날쭉해서 음절과 말한 시간이 이만큼은 있어야 잰다.
MIN_RATE_SYLLABLES = 8
MIN_RATE_SPEAKING_SEC = 1.5


def count_syllables(text: str) -> int:
    """받아적은 글의 음절 수. 한글은 글자 하나가 한 음절이다.

    받아쓰기가 숫자와 영문자로 적은 말('17일', 'A동')은 글자 하나를 한 음절로 어림한다 —
    '17일'(십칠일)처럼 대개 맞거나 한 음절쯤 어긋난다. 문장부호와 빈칸은 세지 않는다.
    """
    return sum(
        1 for ch in text if "가" <= ch <= "힣" or ch.isdigit() or (ch.isascii() and ch.isalpha())
    )


def speaking_seconds(
    turn: tuple[float, float], voiced_spans: Sequence[tuple[float, float]]
) -> float:
    """턴 안에서 말하고 있던 시간(초). 쉼(ARTICULATION_GAP_SEC 이상)은 빼고, 그보다 짧은 틈은 잇는다."""
    start, end = turn
    inside = sorted(
        (max(s, start), min(e, end)) for s, e in voiced_spans if e > start and s < end
    )
    total, previous_end = 0.0, None
    for span_start, span_end in inside:
        if previous_end is not None and span_start - previous_end < ARTICULATION_GAP_SEC:
            total += span_start - previous_end
        total += span_end - span_start
        previous_end = span_end
    return total


def speech_rate(
    text: str, turn: tuple[float, float], voiced_spans: Sequence[tuple[float, float]]
) -> float | None:
    """말 속도 — 말한 시간 1초에 몇 음절을 말했는지. 잴 만큼 말이 없으면 None.

    쉼을 뺀 시간으로 나누므로 중간에 멈춘 턴이 느린 턴으로 나오지 않는다. 멈춤은 따로 짚는다.
    받아쓰기가 말을 빠뜨리거나 잡음이 말소리로 잡히면 값이 실제보다 작게 나온다 — 빠른 쪽으로 틀리지는 않는다.
    """
    syllables = count_syllables(text)
    seconds = speaking_seconds(turn, voiced_spans)
    if syllables < MIN_RATE_SYLLABLES or seconds < MIN_RATE_SPEAKING_SEC:
        return None
    return round(syllables / seconds, 2)


# 같은 통화의 다른 턴(말 속도 중앙값)보다 이만큼 넘게 빠르면 '빨라진 턴'으로 본다. 1.25 = 25% 빠름.
# ★ 실음성 캘리브레이션 대상 — 현재 잠정값. 통화 내내 빠르게 말하는 사람은 이 기준으로 잡히지 않는다.
FAST_RATE_RATIO = 1.25
# 견줄 기준을 잡으려면 잰 턴이 이만큼은 있어야 한다.
MIN_RATE_TURNS = 3


def fast_turns(rates: Mapping[int, float]) -> dict[int, float]:
    """턴 번호 → 같은 통화의 다른 턴보다 몇 배 빠르게 말했는지. 기준을 넘게 빨라진 턴만 담는다.

    사람마다 평소 빠르기가 달라 절대값으로는 견주지 않는다. 그 통화에서 잰 말 속도의 중앙값과 견준다.
    """
    if len(rates) < MIN_RATE_TURNS:
        return {}
    usual = statistics.median(rates.values())
    if usual <= 0:
        return {}
    return {
        index: round(rate / usual, 2)
        for index, rate in rates.items()
        if rate / usual >= FAST_RATE_RATIO
    }


def answer_delay(
    turn: tuple[float, float],
    heard_at: float,
    voiced_spans: Sequence[tuple[float, float]],
) -> float | None:
    """상대 말이 다 들린 때(heard_at)부터 말을 꺼내기까지 걸린 시간(초). 그 뒤로 말소리가 없으면 None.

    상대 말이 끝나기 전부터 말하고 있었으면 0 이다 — 겹쳐 말한 것이지 뜸을 들인 게 아니다.
    """
    _, end = turn
    starts = [start for start, stop in voiced_spans if stop > heard_at and start < end]
    if not starts:
        return None
    return round(max(0.0, min(starts) - heard_at), 2)


# --- 문구 검증 -----------------------------------------------------------

_HARD_BANNED = re.compile(
    r"AVTI|FTrC?IP|ATrI|FCo(?:M|HNR)"   # 지표 이름
    r"|지수|점수|\d+\s*점"               # 점수 표현
    r"|\d+\.\d"                          # 소수 (측정값이 그대로 샌 경우)
    r"|파킨슨|떨림증|진단"
)
# '목소리/음성' 과 붙어 있을 때만 문제가 되는 말들.
_VOICE_PATHOLOGY = re.compile(
    r"(목소리|음성|성대)[^.]{0,12}(장애|질환|질병|이상|치료|병원|검사)"
    r"|(장애|질환|질병)[^.]{0,12}(목소리|음성|성대)"
)
# 제목 자리에 오면 안 되는 말. 모델이 종종 분류 라벨을 제목으로 써버린다.
_LABEL_TITLES = {
    "칭찬", "아쉬움", "잘한 점", "아쉬운 점", "좋은 점", "개선점", "피드백",
    "잘했어요", "아쉬웠어요", "good", "improve",
}

_MIND_STATE = re.compile(
    r"자신감[^.!?]{0,6}(?:없|부족|떨어)"
    r"|자신\s*없"
    r"|(?:긴장|불안|초조|위축|주눅|소심|겁먹)"
    r"(?!\s*하지\s*않|\s*되지\s*않|\s*없이|\s*하지\s*마|\s*안\s*하)"
)
# 상대가 '궁금했을 것' 이라는 말은 무엇을 느꼈는지가 흐릿해서 쓰지 않기로 했다. 지시문으로 막아도 가끔 새어 나온다.
# '궁금한 점을 먼저 물어봤어요' 처럼 내가 궁금해한 얘기는 막지 않는다.
_VAGUE_LISTENER = re.compile(r"궁금(?:했을|해할|해했을|해하실|하셨을|할)")
_HEDGED = re.compile(r"들렸|들릴|들리|느꼈|느낄|느껴|것\s*같|수\s*있|거예요|거에요")

_MAX_TITLE_LEN = 30
_MAX_CONTENT_LEN = 150


def _asserts_mind_state(text: str) -> bool:
    """마음 상태를 추측도 없이 단정한 문장이 섞여 있는지."""
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if _MIND_STATE.search(sentence) and not _HEDGED.search(sentence):
            return True
    return False


def is_safe(title: str, content: str) -> bool:
    """사용자에게 내보내도 되는 항목인지. 제목·내용 둘 다 통과해야 한다."""
    title, content = title.strip(), content.strip()
    if not title or not content:
        return False
    if title.rstrip("!.").strip().lower() in _LABEL_TITLES:
        return False  # 무엇에 대한 얘긴지 알 수 없는 제목
    if len(title) > _MAX_TITLE_LEN or len(content) > _MAX_CONTENT_LEN:
        return False
    if _asserts_mind_state(title) or _asserts_mind_state(content):
        return False
    joined = f"{title} {content}"
    if _VAGUE_LISTENER.search(joined):
        return False
    return not _HARD_BANNED.search(joined) and not _VOICE_PATHOLOGY.search(joined)


_REDUPLICATIONS = ("또박또박", "차근차근", "조곤조곤")


def _redup_pattern(word: str) -> re.Pattern[str]:
    """'또박또박' → 또/박 만 3~5글자로 이어진 덩어리. 정상형은 그대로 다시 쓰인다."""
    half = word[: len(word) // 2]
    return re.compile(rf"(?<![가-힣])[{half}]{{3,5}}(?![가-힣])")


_REDUP_FIXES = [(_redup_pattern(word), word) for word in _REDUPLICATIONS]

_PLAIN_WORDING = [
    (re.compile(r"(?:잘\s*)?(?:닿지|전달되지|전해지지)\s*(?:못했|않았)"), "잘 들리지 않았"),
    (re.compile(r"(?:잘\s*)?(?:닿지|전달되지|전해지지)\s*(?:못하|않)"), "잘 들리지 않"),
]


def polish(text: str) -> str:
    out = text.strip()
    if not out:
        return out
    for pattern, word in _REDUP_FIXES:
        out = pattern.sub(word, out)
    for pattern, repl in _PLAIN_WORDING:
        out = pattern.sub(repl, out)
    return re.sub(r"\s{2,}", " ", out).strip()


def split_title(text: str) -> tuple[str, str]:
    body = text.strip()
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", body) if p.strip()]
    if len(parts) >= 2:
        return parts[0].rstrip(".!?"), " ".join(parts[1:])
    if len(body) <= _MAX_TITLE_LEN:
        return body.rstrip(".!?"), body
    head = body[:_MAX_TITLE_LEN]
    cut = head.rfind(" ")
    return (head[:cut] if cut > 8 else head).rstrip(".!?"), body
