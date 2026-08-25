from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import ReactionKind
from app.core.timeutil import utc_naive_now
from app.db.models import PostReactionORM
from app.schemas.community import ReactionCounts, ReactionStateResponse
from app.services.community_post import alive_post, reaction_summary


async def _state(db: AsyncSession, post_id: int, user_id: int) -> ReactionStateResponse:
    counts, mine = await reaction_summary(db, [post_id], user_id)
    return ReactionStateResponse(
        post_id=post_id,
        reactions=counts.get(post_id, ReactionCounts()),
        my_reaction=mine.get(post_id),
    )


async def _my_reaction_row(
    db: AsyncSession, post_id: int, user_id: int
) -> PostReactionORM | None:
    stmt = select(PostReactionORM).where(
        PostReactionORM.post_id == post_id,
        PostReactionORM.user_id == user_id,
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def set_reaction(
    db: AsyncSession, post_id: int, *, user_id: int, kind: ReactionKind
) -> ReactionStateResponse:
    """좋아요, 공감돼요, 힘내요"""
    await alive_post(db, post_id)

    existing = await _my_reaction_row(db, post_id, user_id)

    if existing is None:
        db.add(
            PostReactionORM(
                post_id=post_id, user_id=user_id, kind=kind.value, created_at=utc_naive_now()
            )
        )
        try:
            await db.commit()
        except IntegrityError:
            await db.rollback()
            existing = await _my_reaction_row(db, post_id, user_id)

    if existing is not None and existing.kind != kind.value:
        existing.kind = kind.value
        await db.commit()

    return await _state(db, post_id, user_id)


async def clear_reaction(db: AsyncSession, post_id: int, *, user_id: int) -> None:
    """공감 취소"""
    await alive_post(db, post_id)

    existing = await _my_reaction_row(db, post_id, user_id)
    if existing is None:
        return

    await db.delete(existing)
    await db.commit()
