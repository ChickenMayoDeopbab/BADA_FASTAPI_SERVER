from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement
from sqlalchemy.sql.selectable import Exists

from app.core.timeutil import now_utc
from app.db.external import users_table
from app.db.models import CommunityUserBlockORM


class BlockedUserNotFoundError(Exception):
    """존재하지 않는 차단 대상 사용자."""


class SelfBlockError(Exception):
    """자기 자신 차단 시도."""


def blocked_user_exists(blocker_user_id: int, blocked_user_id: int | ColumnElement[int]) -> Exists:
    """차단자와 콘텐츠 작성자를 연결하는 EXISTS 조건."""
    return (
        select(CommunityUserBlockORM.block_id)
        .where(
            CommunityUserBlockORM.blocker_user_id == blocker_user_id,
            CommunityUserBlockORM.blocked_user_id == blocked_user_id,
        )
        .exists()
    )


async def is_user_blocked(db: AsyncSession, *, blocker_user_id: int, blocked_user_id: int) -> bool:
    return bool((await db.execute(select(blocked_user_exists(blocker_user_id, blocked_user_id)))).scalar_one())


async def _ensure_target_user(db: AsyncSession, blocked_user_id: int) -> None:
    stmt = select(users_table.c.user_id).where(users_table.c.user_id == blocked_user_id)
    if (await db.execute(stmt)).scalar_one_or_none() is None:
        raise BlockedUserNotFoundError


async def _block_row(db: AsyncSession, blocker_user_id: int, blocked_user_id: int) -> CommunityUserBlockORM | None:
    stmt = select(CommunityUserBlockORM).where(
        CommunityUserBlockORM.blocker_user_id == blocker_user_id,
        CommunityUserBlockORM.blocked_user_id == blocked_user_id,
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def block_user(db: AsyncSession, *, blocker_user_id: int, blocked_user_id: int) -> None:
    if blocker_user_id == blocked_user_id:
        raise SelfBlockError
    await _ensure_target_user(db, blocked_user_id)

    if await _block_row(db, blocker_user_id, blocked_user_id) is not None:
        return

    created_at = now_utc()
    row = CommunityUserBlockORM(blocker_user_id=blocker_user_id, blocked_user_id=blocked_user_id, created_at=created_at)
    db.add(row)
    try:
        await db.commit()
    except IntegrityError:
        await db.rollback()
        if await _block_row(db, blocker_user_id, blocked_user_id) is not None:
            return
        raise


async def unblock_user(db: AsyncSession, *, blocker_user_id: int, blocked_user_id: int) -> None:
    if blocker_user_id == blocked_user_id:
        raise SelfBlockError
    await _ensure_target_user(db, blocked_user_id)

    stmt = delete(CommunityUserBlockORM).where(
        CommunityUserBlockORM.blocker_user_id == blocker_user_id,
        CommunityUserBlockORM.blocked_user_id == blocked_user_id,
    )
    await db.execute(stmt)
    await db.commit()
