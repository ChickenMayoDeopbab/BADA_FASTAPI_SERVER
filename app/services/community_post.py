from __future__ import annotations

import logging

from redis.asyncio import Redis
from sqlalchemy import case, delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.core.enums import ReactionKind
from app.core.security import is_admin
from app.core.timeutil import utc_naive_now
from app.db.external import users_table
from app.db.models import PostAttachmentORM, PostCommentORM, PostORM, PostReactionORM
from app.schemas.community import (
    AuthorInfo,
    PostCreateRequest,
    PostDetailResponse,
    PostListResponse,
    PostSummary,
    PostUpdateRequest,
    ReactionCounts,
    preview_of,
    to_nfc,
)
from app.services.post_attachment import (
    AttachmentInvalidError,
    build_rows,
    commit_translating_conflicts,
    detail_for_post,
    kinds_by_post,
    schedule_morphs,
)

logger = logging.getLogger(__name__)


class PostNotFoundError(Exception):
    """없거나 이미 삭제된 게시글."""


class PostForbiddenError(Exception):
    """작성자도 어드민도 아닌 사용자의 수정·삭제 시도."""

_VIEW_DEDUP_TTL_SEC = 86_400

_COUNT_FIELD_BY_KIND = {
    ReactionKind.CHEER: "cheer",
    ReactionKind.RELATE: "relate",
    ReactionKind.LIKE: "like",
}



async def load_author(db: AsyncSession, user_id: int) -> AuthorInfo:
    """작성자 정보"""
    stmt = select(users_table.c.name, users_table.c.profile_image).where(
        users_table.c.user_id == user_id
    )
    row = (await db.execute(stmt)).first()
    if row is None:
        return AuthorInfo(user_id=user_id)
    return AuthorInfo(user_id=user_id, name=row.name, profile_image_url=row.profile_image)


def _to_detail(row: PostORM, author: AuthorInfo) -> PostDetailResponse:
    return PostDetailResponse(
        post_id=row.post_id,
        title=row.title,
        content=row.content,
        author=author,
        view_count=row.view_count,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


async def _detail_with_aggregates(
    db: AsyncSession, row: PostORM, viewer_id: int
) -> PostDetailResponse:
    """상세도 목록과 같은 통계 싣기"""
    comment_counts = await _comment_counts(db, [row.post_id])
    counts, mine = await reaction_summary(db, [row.post_id], viewer_id)

    detail = _to_detail(row, await load_author(db, row.user_id))
    detail.comment_count = comment_counts.get(row.post_id, 0)
    detail.reactions = counts.get(row.post_id, ReactionCounts())
    detail.my_reaction = mine.get(row.post_id)
    detail.attachments = await detail_for_post(db, row.post_id, viewer_id)
    return detail


async def is_admin_user(db: AsyncSession, user_id: int) -> bool:
    """어드민 여부 확인"""
    stmt = select(users_table.c.role).where(users_table.c.user_id == user_id)
    row = (await db.execute(stmt)).first()
    return row is not None and is_admin(row.role)


async def create_post(
    db: AsyncSession, body: PostCreateRequest, user_id: int
) -> PostDetailResponse:
    now = utc_naive_now()
    row = PostORM(
        user_id=user_id,
        title=body.title,
        content=body.content,
        view_count=0,
        created_at=now,
        updated_at=now,
    )
    db.add(row)

    await db.flush()
    try:
        attachments = await build_rows(db, row.post_id, body.attachments, user_id)
    except AttachmentInvalidError:
        await db.rollback()
        raise
    db.add_all(attachments)
    await commit_translating_conflicts(db)
    await schedule_morphs(db, attachments)

    return await _detail_with_aggregates(db, row, user_id)


async def _should_count_view(redis: Redis, post_id: int, viewer_id: int) -> bool:
    """조회수, 24시간 쿨타임"""
    try:
        stored = await redis.set(
            f"viewed:{post_id}:{viewer_id}", "1", ex=_VIEW_DEDUP_TTL_SEC, nx=True
        )
    except Exception as e:
        logger.warning(
            "조회수 중복 방지 키 실패, 그대로 집계: %s: %s",
            type(e).__name__,
            e,
            extra={"post_id": post_id},
        )
        return True
    return stored is True


async def get_post(
    db: AsyncSession, post_id: int, *, viewer_id: int, redis: Redis
) -> PostDetailResponse | None:
    """삭제되지 않은 게시글 1건 + 조회수 증가"""
    row = await db.get(PostORM, post_id)
    if row is None or row.deleted_at is not None:
        return None

    if await _should_count_view(redis, post_id, viewer_id):
        await db.execute(
            update(PostORM)
            .where(PostORM.post_id == post_id, PostORM.deleted_at.is_(None))
            .values(view_count=PostORM.view_count + 1)
        )
        await db.commit()
        await db.refresh(row)

    return await _detail_with_aggregates(db, row, viewer_id)


def _like_pattern(q: str) -> str:
    normalized = str(to_nfc(q))
    escaped = normalized.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


async def _comment_counts(db: AsyncSession, post_ids: list[int]) -> dict[int, int]:
    """게시글별 댓글+답글 수"""
    if not post_ids:
        return {}

    parent = aliased(PostCommentORM)
    alive_parent = (
        select(parent.comment_id)
        .where(
            parent.comment_id == PostCommentORM.parent_comment_id,
            parent.deleted_at.is_(None),
        )
        .exists()
    )
    stmt = (
        select(PostCommentORM.post_id, func.count())
        .where(
            PostCommentORM.post_id.in_(post_ids),
            PostCommentORM.deleted_at.is_(None),
            or_(PostCommentORM.parent_comment_id.is_(None), alive_parent),
        )
        .group_by(PostCommentORM.post_id)
    )
    return dict((await db.execute(stmt)).all())


async def reaction_summary(
    db: AsyncSession, post_ids: list[int], viewer_id: int
) -> tuple[dict[int, ReactionCounts], dict[int, ReactionKind]]:
    """게시글별 공감 3종 카운트와 내가 누른 종류"""
    if not post_ids:
        return {}, {}

    stmt = (
        select(
            PostReactionORM.post_id,
            PostReactionORM.kind,
            func.count(),
            func.max(case((PostReactionORM.user_id == viewer_id, 1), else_=0)),
        )
        .where(PostReactionORM.post_id.in_(post_ids))
        .group_by(PostReactionORM.post_id, PostReactionORM.kind)
    )

    counts: dict[int, ReactionCounts] = {}
    mine: dict[int, ReactionKind] = {}
    for post_id, raw_kind, count, is_mine in (await db.execute(stmt)).all():
        try:
            kind = ReactionKind(raw_kind)
        except ValueError:
            logger.warning(
                "알 수 없는 공감 종류라 집계에서 제외: %r",
                raw_kind,
                extra={"post_id": post_id, "kind": raw_kind},
            )
            continue
        bucket = counts.setdefault(post_id, ReactionCounts())
        setattr(bucket, _COUNT_FIELD_BY_KIND[kind], count)
        bucket.total += count
        if is_mine:
            mine[post_id] = kind
    return counts, mine


async def _count_posts(db: AsyncSession, conditions: list) -> int:
    """행이 하나도 안 돌아왔을 때만 쓰는 total 보정 쿼리"""
    stmt = select(func.count()).select_from(PostORM).where(*conditions)
    return (await db.execute(stmt)).scalar_one()


async def list_posts(
    db: AsyncSession,
    *,
    viewer_id: int,
    page: int = 1,
    size: int = 20,
    q: str | None = None,
    author_id: int | None = None,
) -> PostListResponse:
    """삭제되지 않은 게시글을 최신순으로. size+1 개를 읽어 has_next 를 판단한다."""
    conditions = [PostORM.deleted_at.is_(None)]
    if author_id is not None:
        conditions.append(PostORM.user_id == author_id)
    if q:
        keyword = _like_pattern(q)
        conditions.append(
            or_(
                PostORM.title.ilike(keyword, escape="\\"),
                PostORM.content.ilike(keyword, escape="\\"),
            )
        )

    stmt = (
        select(
            PostORM,
            users_table.c.name,
            users_table.c.profile_image,
            func.count().over(),
        )
        .join(users_table, users_table.c.user_id == PostORM.user_id, isouter=True)
        .where(*conditions)
        .order_by(PostORM.created_at.desc(), PostORM.post_id.desc())
        .offset((page - 1) * size)
        .limit(size + 1)
    )
    rows = (await db.execute(stmt)).all()

    total = rows[0][3] if rows else await _count_posts(db, conditions)
    has_next = len(rows) > size
    rows = rows[:size]

    post_ids = [row.post_id for row, _, _, _ in rows]
    comment_counts = await _comment_counts(db, post_ids)
    reaction_counts, my_reactions = await reaction_summary(db, post_ids, viewer_id)
    attachment_kinds = await kinds_by_post(db, post_ids)

    posts = [
        PostSummary(
            post_id=row.post_id,
            title=row.title,
            content_preview=preview_of(row.content),
            author=AuthorInfo(user_id=row.user_id, name=name, profile_image_url=image),
            view_count=row.view_count,
            comment_count=comment_counts.get(row.post_id, 0),
            reactions=reaction_counts.get(row.post_id, ReactionCounts()),
            my_reaction=my_reactions.get(row.post_id),
            attachment_kinds=attachment_kinds.get(row.post_id, []),
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
        for row, name, image, _ in rows
    ]
    return PostListResponse(
        posts=posts, page=page, size=size, total=total, has_next=has_next
    )


async def alive_post(db: AsyncSession, post_id: int) -> PostORM:
    row = await db.get(PostORM, post_id)
    if row is None or row.deleted_at is not None:
        raise PostNotFoundError
    return row


async def update_post(
    db: AsyncSession, post_id: int, body: PostUpdateRequest, user_id: int
) -> PostDetailResponse:
    """작성자 본인만 수정 가능"""
    row = await alive_post(db, post_id)
    if row.user_id != user_id:
        raise PostForbiddenError

    changed = False
    pending_morphs: list = []
    if body.title is not None and body.title != row.title:
        row.title = body.title
        changed = True
    if body.content is not None and body.content != row.content:
        row.content = body.content
        changed = True

    if "attachments" in body.model_fields_set:
        try:
            new_rows = await build_rows(db, post_id, body.attachments or [], user_id)
        except AttachmentInvalidError:
            await db.rollback()
            raise
        await db.execute(
            delete(PostAttachmentORM).where(PostAttachmentORM.post_id == post_id)
        )
        db.add_all(new_rows)
        pending_morphs = new_rows
        changed = True

    if changed:
        row.updated_at = utc_naive_now()
        await commit_translating_conflicts(db)
        await schedule_morphs(db, pending_morphs)

    return await _detail_with_aggregates(db, row, user_id)


async def delete_post(db: AsyncSession, post_id: int, *, user_id: int) -> None:
    """작성자 본인 또는 어드민만 삭제 가능"""
    row = await alive_post(db, post_id)
    if row.user_id != user_id and not await is_admin_user(db, user_id):
        raise PostForbiddenError

    row.deleted_at = utc_naive_now()
    await db.commit()
