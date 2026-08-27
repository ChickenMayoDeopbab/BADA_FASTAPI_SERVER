from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import AttachmentKind
from app.core.timeutil import now_utc
from app.db.models import PostAttachmentORM, PostORM, ScenarioORM

# 카피된 시나리오가 받는 거
_CARRIED = (
    "title",
    "content",
    "category",
    "scenario_image",
    "tts_voice_id",
    "ai_prompt",
    "call_target",
    "call_purpose",
    "script",
    "example_dialogue",
)


class PostMissingError(Exception):
    """없거나 삭제된 게시글"""


class NothingToCopyError(Exception):
    """게시글에 시나리오 첨부가 없거나 이미 내거임"""


class OriginGoneError(Exception):
    """원본 소유자가 시나리오를 지움"""


async def _attached_scenario_id(db: AsyncSession, post_id: int) -> int:
    row = await db.get(PostORM, post_id)
    if row is None or row.deleted_at is not None:
        raise PostMissingError

    stmt = select(PostAttachmentORM.ref_id).where(
        PostAttachmentORM.post_id == post_id,
        PostAttachmentORM.kind == AttachmentKind.SCENARIO.value,
    )
    ref_id = (await db.execute(stmt)).scalar_one_or_none()
    if ref_id is None:
        raise NothingToCopyError("이 글에는 시나리오 첨부가 없습니다.")
    return ref_id


async def _already_mine(
    db: AsyncSession, root_id: int, user_id: int
) -> ScenarioORM | None:
    """이미 가지고 있으면 그걸 준다"""
    root = await db.get(ScenarioORM, root_id)
    if root is not None and root.user_id == user_id and root.deleted_at is None:
        return root

    stmt = select(ScenarioORM).where(
        ScenarioORM.origin_scenario_id == root_id,
        ScenarioORM.user_id == user_id,
        ScenarioORM.deleted_at.is_(None),
    )
    return (await db.execute(stmt)).scalars().first()


async def copy_attached_scenario(
    db: AsyncSession, post_id: int, user_id: int
) -> tuple[ScenarioORM, bool]:
    """(시나리오, 이미_가지고_있었나)를 반환"""
    source = await db.get(ScenarioORM, await _attached_scenario_id(db, post_id))
    if source is None:
        raise OriginGoneError
    if source.user_id == user_id:
        raise NothingToCopyError("내가 만든 시나리오입니다.")
    if source.deleted_at is not None:
        raise OriginGoneError

    root_id = source.origin_scenario_id or source.scenario_id

    existing = await _already_mine(db, root_id, user_id)
    if existing is not None:
        return existing, True

    copy = ScenarioORM(
        **{field: getattr(source, field) for field in _CARRIED},
        user_id=user_id,
        is_custom=True,
        is_warmup=False,
        origin_scenario_id=root_id,
        created_at=now_utc(),
    )
    db.add(copy)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        existing = await _already_mine(db, root_id, user_id)
        if existing is None:
            raise
        return existing, True
    await db.refresh(copy)
    return copy, False
