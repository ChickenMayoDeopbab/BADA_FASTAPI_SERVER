"""통화 연결 직후 AI 가 먼저 할 첫마디"""

import asyncio
import logging

from app.core.preset_scenarios import PRESET_MAP
from app.db.base import AsyncSessionLocal
from app.db.models import ScenarioORM, is_deleted

logger = logging.getLogger(__name__)

GREETING_FALLBACK = "네, 여보세요?"
GREETING_MAX_CHARS = 60
_LOOKUP_TIMEOUT_SECONDS = 1.0


def greeting_enabled(settings: object, user_id: int | None) -> bool:
    """스위치가 켜졌고 허용 목록이 비었거나 목록 안 사용자"""
    if not getattr(settings, "ai_greeting", False):
        return False
    raw = getattr(settings, "ai_greeting_user_ids", "") or ""
    allowed = {part.strip() for part in str(raw).split(",") if part.strip()}
    return not allowed or (user_id is not None and str(user_id) in allowed)


def first_ai_line(dialogue: object) -> str | None:
    """예시 대화 첫 줄이 쓸 만한 AI 말이면 그 글"""
    if not isinstance(dialogue, list) or not dialogue or not isinstance(dialogue[0], dict):
        return None
    first = dialogue[0]
    if str(first.get("speaker", "")).strip().lower() != "ai":
        return None
    text = str(first.get("text", "")).strip()
    return text if 0 < len(text) <= GREETING_MAX_CHARS else None


async def _custom_dialogue(scenario_id: int) -> object:
    async with AsyncSessionLocal() as db:
        row = await db.get(ScenarioORM, scenario_id)
        return None if row is None or is_deleted(row) else row.example_dialogue


async def pick_greeting(scenario_id: int | None) -> tuple[str, str]:
    """(첫마디, 출처) — 출처는 preset·custom·fallback"""
    if scenario_id is None:
        return GREETING_FALLBACK, "fallback"
    preset = PRESET_MAP.get(scenario_id)
    if preset is not None:
        line = first_ai_line(preset.get("example_dialogue"))
        return (line, "preset") if line else (GREETING_FALLBACK, "fallback")
    try:
        dialogue = await asyncio.wait_for(_custom_dialogue(scenario_id), _LOOKUP_TIMEOUT_SECONDS)
    except Exception:
        logger.warning("첫마디용 예시 대화 조회 실패 scenario_id=%s", scenario_id, exc_info=True)
        return GREETING_FALLBACK, "fallback"
    line = first_ai_line(dialogue)
    return (line, "custom") if line else (GREETING_FALLBACK, "fallback")
