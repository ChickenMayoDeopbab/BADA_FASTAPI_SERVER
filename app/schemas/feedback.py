from typing import List

from pydantic import BaseModel

class FeedbackResponse(BaseModel):
    shake_count: int
    silence_duration: int
    highlights: List[str]