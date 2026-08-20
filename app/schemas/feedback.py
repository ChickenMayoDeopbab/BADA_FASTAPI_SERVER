
from pydantic import BaseModel


class FeedbackSegment(BaseModel):
    """구간 하나. 통화의 특정 구간을 짚어서 칭찬하거나 조언한다."""

    start: float
    end: float
    type: str       # GOOD / IMPROVE
    title: str      # 한눈에 읽는 한 줄
    content: str    # 왜 그런지 + 다음에 어떻게 할지
    good_point: str # content 미러 (기존 Spring 매핑 호환)


class FeedbackResponse(BaseModel):
    shake_count: int
    silence_duration: int
    good_segments: list[FeedbackSegment] = []
