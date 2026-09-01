from __future__ import annotations

import re
from enum import StrEnum

__all__ = ["Band", "voice_band", "is_safe", "polish", "split_title"]


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
