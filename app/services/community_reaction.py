from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import CommunityNotificationType, ReactionKind
from app.core.timeutil import now_utc
from app.db.models import PostReactionORM
from app.schemas.community import ReactionCounts, ReactionStateResponse
from app.services.community_post import alive_post, reaction_summary


@dataclass(frozen=True)
class CommunityReactionNotificationEvent:
    notification_type: CommunityNotificationType
    recipient_user_id: int
    actor_user_id: int
    post_id: int
    reaction_id: int
    reaction_kind: ReactionKind


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


async def _set_reaction(
    db: AsyncSession, post_id: int, *, user_id: int, kind: ReactionKind
) -> tuple[ReactionStateResponse, CommunityReactionNotificationEvent | None]:
    """좋아요, 공감돼요, 힘내요"""
    post = await alive_post(db, post_id)

    existing = await _my_reaction_row(db, post_id, user_id)
    notification_event = None

    if existing is None:
        new_reaction = PostReactionORM(
            post_id=post_id, user_id=user_id, kind=kind.value, created_at=now_utc()
        )
        db.add(new_reaction)
        try:
            await db.flush()
            reaction_id = new_reaction.reaction_id
            await db.commit()
            if post.user_id != user_id:
                notification_event = CommunityReactionNotificationEvent(
                    notification_type=CommunityNotificationType.REACTION,
                    recipient_user_id=post.user_id,
                    actor_user_id=user_id,
                    post_id=post_id,
                    reaction_id=reaction_id,
                    reaction_kind=kind,
                )
        except IntegrityError:
            await db.rollback()
            existing = await _my_reaction_row(db, post_id, user_id)

    if existing is not None and existing.kind != kind.value:
        existing.kind = kind.value
        await db.commit()

    return await _state(db, post_id, user_id), notification_event


async def set_reaction(
    db: AsyncSession, post_id: int, *, user_id: int, kind: ReactionKind
) -> ReactionStateResponse:
    state, _ = await _set_reaction(db, post_id, user_id=user_id, kind=kind)
    return state


async def set_reaction_with_notification(
    db: AsyncSession, post_id: int, *, user_id: int, kind: ReactionKind
) -> tuple[ReactionStateResponse, CommunityReactionNotificationEvent | None]:
    return await _set_reaction(db, post_id, user_id=user_id, kind=kind)


async def clear_reaction(db: AsyncSession, post_id: int, *, user_id: int) -> None:
    """공감 취소"""
    await alive_post(db, post_id)

    existing = await _my_reaction_row(db, post_id, user_id)
    if existing is None:
        return

    await db.delete(existing)
    await db.commit()
