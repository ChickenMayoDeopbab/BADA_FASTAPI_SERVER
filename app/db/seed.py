from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

from sqlalchemy import column, func, inspect, or_, select, table, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import AttachmentKind
from app.core.preset_scenarios import NEW_PRESET_SCENARIOS, PRESET_SCENARIOS
from app.core.timeutil import now_utc
from app.db.models import CallBiosignalORM, FeedbackORM, PostAttachmentORM, ScenarioORM, UsageEventORM

_LEGACY_PRESET_MAX_ID = 8


def _preset_values(preset: dict[str, Any], created_at: datetime) -> dict[str, Any]:
    return {
        "scenario_id": int(preset["scenario_id"]),
        "title": preset["title"],
        "content": preset["content"],
        "category": preset["category"].value,
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


def _apply_preset(row: ScenarioORM, preset: dict[str, Any], *, preserve_image: bool = False) -> None:
    values = _preset_values(preset, row.created_at)
    for key, value in values.items():
        if key in {"scenario_id", "created_at"}:
            continue
        if key == "scenario_image" and preserve_image and value is None:
            continue
        setattr(row, key, value)


def _copy_custom(row: ScenarioORM, new_id: int) -> ScenarioORM:
    return ScenarioORM(
        scenario_id=new_id,
        title=row.title,
        content=row.content,
        category=row.category,
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
        deleted_at=row.deleted_at,
    )


async def _postgres_sequence_state(db: AsyncSession) -> tuple[str, int] | None:
    if db.get_bind().dialect.name != "postgresql":
        return None
    # regclass가 현재 search_path에 맞는 인용된 식별자를 반환한다.
    sequence = await db.scalar(text("SELECT pg_get_serial_sequence('scenario', 'scenario_id')::regclass::text"))
    if sequence is None:
        raise RuntimeError("scenario ID sequence is missing")
    last_value, is_called = (await db.execute(text(f"SELECT last_value, is_called FROM {sequence}"))).one()
    return sequence, int(last_value) if is_called else int(last_value) - 1


async def _max_scenario_id(db: AsyncSession) -> int:
    result = await db.execute(select(func.coalesce(func.max(ScenarioORM.scenario_id), 0)))
    return int(result.scalar_one())


async def _next_preset_start_id(db: AsyncSession, sequence_floor: int) -> int:
    # Spring의 훈련 기록은 공유 DB에 있지만 FastAPI 단독 환경에서는 없을 수 있다.
    connection = await db.connection()
    has_training_records = await connection.run_sync(lambda conn: inspect(conn).has_table("training_records"))
    columns = [
        ScenarioORM.scenario_id,
        ScenarioORM.origin_scenario_id,
        FeedbackORM.scenario_id,
        CallBiosignalORM.scenario_id,
        UsageEventORM.scenario_id,
    ]
    if has_training_records:
        columns.append(table("training_records", column("scenario_id")).c.scenario_id)
    attachment_max = (
        select(func.max(PostAttachmentORM.ref_id))
        .where(PostAttachmentORM.kind == AttachmentKind.SCENARIO)
        .scalar_subquery()
    )
    reference_maxima = (await db.execute(select(
        *(select(func.max(ref)).scalar_subquery() for ref in columns), attachment_max,
    ))).one()
    return max(_LEGACY_PRESET_MAX_ID, sequence_floor, *(value or 0 for value in reference_maxima)) + 1


async def next_preset_start_id(db: AsyncSession) -> int:
    sequence_state = await _postgres_sequence_state(db)
    return await _next_preset_start_id(db, sequence_state[1] if sequence_state else 0)


async def _validate_new_presets(db: AsyncSession, sequence_floor: int) -> None:
    expected = {preset["scenario_id"]: preset["title"] for preset in NEW_PRESET_SCENARIOS}
    if not expected:
        return

    existing = (await db.scalars(select(ScenarioORM).where(or_(
        ScenarioORM.scenario_id.in_(expected),
        ScenarioORM.is_custom.is_(False) & ScenarioORM.title.in_(expected.values()),
    )))).all()
    if existing:
        registered = {
            row.scenario_id: row.title for row in existing
            if not row.is_custom and not row.is_warmup and row.user_id is None and row.deleted_at is None
        }
        if len(existing) != len(expected) or registered != expected:
            raise RuntimeError("New preset scenario IDs conflict with existing or partially seeded scenarios")
        return

    next_id = await _next_preset_start_id(db, sequence_floor)
    if min(expected) != next_id:
        raise RuntimeError(f"DEFAULT_SCENARIO_START_ID must be {next_id} for the first seed")


async def seed_preset_scenarios(db: AsyncSession) -> None:
    """프리셋을 등록하고, 신규 프리셋 ID가 충돌하면 등록을 중단한다."""
    if db.get_bind().dialect.name == "postgresql":
        await db.execute(text("SET LOCAL lock_timeout = '5s'"))
        await db.execute(text("LOCK TABLE scenario IN SHARE ROW EXCLUSIVE MODE"))
    sequence_state = await _postgres_sequence_state(db)
    sequence_floor = sequence_state[1] if sequence_state else 0
    await _validate_new_presets(db, sequence_floor)

    now = now_utc()
    max_preset_id = max(preset["scenario_id"] for preset in PRESET_SCENARIOS)
    next_custom_id = max(await _max_scenario_id(db), max_preset_id, sequence_floor) + 1

    for preset in PRESET_SCENARIOS:
        preset_id = int(preset["scenario_id"])
        row = await db.get(ScenarioORM, preset_id)

        if NEW_PRESET_SCENARIOS and preset_id <= _LEGACY_PRESET_MAX_ID and row is not None:
            if row.is_custom:
                raise RuntimeError(f"Legacy preset ID {preset_id} is occupied by a custom scenario")
            continue

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
            _apply_preset(row, preset, preserve_image=preset_id > _LEGACY_PRESET_MAX_ID)

    await db.flush()
    if sequence_state is not None:
        max_id = await _max_scenario_id(db)
        if max_id > sequence_floor:
            # RESTART는 INSERT와 같은 트랜잭션에서 커밋 또는 롤백된다.
            await db.execute(text(f"ALTER SEQUENCE {sequence_state[0]} RESTART WITH {max_id + 1}"))
    await db.commit()
