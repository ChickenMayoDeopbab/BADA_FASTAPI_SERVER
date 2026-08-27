from __future__ import annotations

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.timeutil import now_utc
from app.db.external import users_table
from app.db.models import PostCommentORM
from app.schemas.community import (
    AuthorInfo,
    CommentCreateRequest,
    CommentListResponse,
    CommentResponse,
    CommentThread,
    CommentUpdateRequest,
)
from app.services.community_post import alive_post, is_admin_user, load_author


class CommentNotFoundError(Exception):
    """없거나 이미 삭제된 댓글"""


class CommentForbiddenError(Exception):
    """작성자도 어드민도 아닌 사용자의 댓글 수정·삭제 시도"""


class InvalidParentCommentError(Exception):
    """답글의 부모로 쓸 수 없는 댓글,없거나, 삭제됐거나, 다른 글이거나, 답글인거"""



async def _check_parent(db: AsyncSession, post_id: int, parent_id: int) -> None:
    """답글 깊이를 1단으로 고정"""
    parent = await db.get(PostCommentORM, parent_id)
    if parent is None or parent.deleted_at is not None:
        raise InvalidParentCommentError
    if parent.post_id != post_id:
        raise InvalidParentCommentError
    if parent.parent_comment_id is not None:
        raise InvalidParentCommentError


async def create_comment(
    db: AsyncSession, post_id: int, body: CommentCreateRequest, user_id: int
) -> CommentResponse:
    """댓글 생성"""
    await alive_post(db, post_id)
    if body.parent_comment_id is not None:
        await _check_parent(db, post_id, body.parent_comment_id)

    now = now_utc()
    row = PostCommentORM(
        post_id=post_id,
        parent_comment_id=body.parent_comment_id,
        user_id=user_id,
        content=body.content,
        created_at=now,
        updated_at=now,
    )
    db.add(row)
    await db.commit()

    return CommentResponse(
        comment_id=row.comment_id,
        parent_comment_id=row.parent_comment_id,
        content=row.content,
        author=await load_author(db, user_id),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


async def list_comments(db: AsyncSession, post_id: int) -> CommentListResponse:
    """살아있는 댓글·답글을 오래된 순 2뎁스 트리로. 작성자는 users 조인 1회로 붙임"""
    await alive_post(db, post_id)

    stmt = (
        select(PostCommentORM, users_table.c.name, users_table.c.profile_image)
        .join(users_table, users_table.c.user_id == PostCommentORM.user_id, isouter=True)
        .where(
            PostCommentORM.post_id == post_id,
            PostCommentORM.deleted_at.is_(None),
        )
        .order_by(PostCommentORM.created_at, PostCommentORM.comment_id)
    )
    rows = (await db.execute(stmt)).all()

    threads: dict[int, CommentThread] = {}
    replies: list[tuple[int, CommentResponse]] = []
    for row, name, image in rows:
        author = AuthorInfo(user_id=row.user_id, name=name, profile_image_url=image)
        if row.parent_comment_id is None:
            threads[row.comment_id] = CommentThread(
                comment_id=row.comment_id,
                content=row.content,
                author=author,
                created_at=row.created_at,
                updated_at=row.updated_at,
                replies=[],
            )
        else:
            replies.append((
                row.parent_comment_id,
                CommentResponse(
                    comment_id=row.comment_id,
                    parent_comment_id=row.parent_comment_id,
                    content=row.content,
                    author=author,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                ),
            ))

    for parent_id, reply in replies:
        thread = threads.get(parent_id)
        if thread is not None:
            thread.replies.append(reply)

    return CommentListResponse(comments=list(threads.values()))


def _to_comment(row: PostCommentORM, author: AuthorInfo) -> CommentResponse:
    return CommentResponse(
        comment_id=row.comment_id,
        parent_comment_id=row.parent_comment_id,
        content=row.content,
        author=author,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


async def _alive_comment(db: AsyncSession, comment_id: int) -> PostCommentORM:
    row = await db.get(PostCommentORM, comment_id)
    if row is None or row.deleted_at is not None:
        raise CommentNotFoundError
    return row


async def update_comment(
    db: AsyncSession, comment_id: int, body: CommentUpdateRequest, user_id: int
) -> CommentResponse:
    """작성자 본인만 수정 가능"""
    row = await _alive_comment(db, comment_id)
    if row.user_id != user_id:
        raise CommentForbiddenError

    if body.content != row.content:
        row.content = body.content
        row.updated_at = now_utc()
        await db.commit()

    return _to_comment(row, await load_author(db, row.user_id))


async def delete_comment(db: AsyncSession, comment_id: int, *, user_id: int) -> None:
    """작성자 본인 또는 어드민만 삭제 가능"""
    row = await _alive_comment(db, comment_id)
    if row.user_id != user_id and not await is_admin_user(db, user_id):
        raise CommentForbiddenError

    now = now_utc()
    row.deleted_at = now

    if row.parent_comment_id is None:
        await db.execute(
            update(PostCommentORM)
            .where(
                PostCommentORM.parent_comment_id == comment_id,
                PostCommentORM.deleted_at.is_(None),
            )
            .values(deleted_at=now)
        )

    await db.commit()
