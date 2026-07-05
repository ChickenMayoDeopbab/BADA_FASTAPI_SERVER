from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.preset_scenarios import PRESET_SCENARIOS
from app.db.models import FeedbackORM, ScenarioORM

_PRESET_IDS = {int(s["scenario_id"]) for s in PRESET_SCENARIOS}
_MAX_PRESET_ID = max(_PRESET_IDS, default=0)


def _preset_values(preset: dict[str, Any], created_at: datetime) -> dict[str, Any]:
    return {
        "scenario_id": int(preset["scenario_id"]),
        "title": preset["title"],
        "content": preset["content"],
        "scenario_image": preset["scenario_image"],
        "tts_voice_id": preset["tts_voice_id"],
        "ai_prompt": preset["ai_prompt"],
        "user_id": None,
        "is_custom": False,
        "is_warmup": False,
        "call_target": preset["call_target"],
        "call_purpose": preset["call_purpose"],
        "script": deepcopy(preset["script"]),
        "created_at": created_at,
    }


def _apply_preset(row: ScenarioORM, preset: dict[str, Any]) -> None:
    values = _preset_values(preset, row.created_at)
    for key, value in values.items():
        if key in {"scenario_id", "created_at"}:
            continue
        setattr(row, key, value)


def _copy_custom(row: ScenarioORM, new_id: int) -> ScenarioORM:
    return ScenarioORM(
        scenario_id=new_id,
        title=row.title,
        content=row.content,
        scenario_image=row.scenario_image,
        tts_voice_id=row.tts_voice_id,
        ai_prompt=row.ai_prompt,
        user_id=row.user_id,
        is_custom=True,
        is_warmup=row.is_warmup,
        call_target=row.call_target,
        call_purpose=row.call_purpose,
        script=deepcopy(row.script),
        created_at=row.created_at,
    )


async def _next_custom_id(db: AsyncSession) -> int:
    result = await db.execute(select(func.coalesce(func.max(ScenarioORM.scenario_id), 0)))
    max_id = int(result.scalar_one())
    return max(max_id + 1, _MAX_PRESET_ID + 1)


async def _sync_postgres_sequence(db: AsyncSession) -> None:
    try:
        bind = db.get_bind()
    except Exception:
        return

    if getattr(getattr(bind, "dialect", None), "name", None) != "postgresql":
        return

    await db.execute(
        text(
            """
            SELECT setval(
                pg_get_serial_sequence('scenario', 'scenario_id'),
                GREATEST(COALESCE((SELECT MAX(scenario_id) FROM scenario), 1), 1),
                true
            )
            """
        )
    )


async def seed_preset_scenarios(db: AsyncSession) -> None:
    """Keep reserved preset scenario rows present and move conflicting custom rows."""
    now = datetime.now(UTC).replace(tzinfo=None)
    next_custom_id = await _next_custom_id(db)

    for preset in PRESET_SCENARIOS:
        preset_id = int(preset["scenario_id"])
        row = await db.get(ScenarioORM, preset_id)

        if row is not None and getattr(row, "is_custom", False):
            moved_row = _copy_custom(row, next_custom_id)
            db.add(moved_row)
            await db.flush()
            await db.execute(
                update(FeedbackORM)
                .where(FeedbackORM.scenario_id == preset_id)
                .values(scenario_id=next_custom_id)
            )
            await db.delete(row)
            await db.flush()
            next_custom_id += 1
            row = None

        if row is None:
            db.add(ScenarioORM(**_preset_values(preset, now)))
        else:
            _apply_preset(row, preset)

    await db.flush()
    await _sync_postgres_sequence(db)
    await db.commit()
