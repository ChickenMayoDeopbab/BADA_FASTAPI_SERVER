from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Response, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.deps.auth import get_current_user_id
from app.deps.db import get_db
from app.deps.redis import get_redis
from app.deps.spring import get_spring_client
from app.schemas.community import (
    CommentCreateRequest,
    CommentListResponse,
    CommentResponse,
    CommentUpdateRequest,
    PostCreateRequest,
    PostDetailResponse,
    PostListResponse,
    PostUpdateRequest,
    ReactionRequest,
    ReactionStateResponse,
    ScenarioCopyResponse,
)
from app.services.community_comment import (
    CommentForbiddenError,
    CommentNotFoundError,
    InvalidParentCommentError,
)
from app.services.community_comment import create_comment as svc_create_comment
from app.services.community_comment import delete_comment as svc_delete_comment
from app.services.community_comment import list_comments as svc_list_comments
from app.services.community_comment import update_comment as svc_update_comment
from app.services.community_post import PostForbiddenError, PostNotFoundError
from app.services.community_post import create_post as svc_create_post
from app.services.community_post import delete_post as svc_delete_post
from app.services.community_post import get_post as svc_get_post
from app.services.community_post import list_posts as svc_list_posts
from app.services.community_post import update_post as svc_update_post
from app.services.community_reaction import clear_reaction as svc_clear_reaction
from app.services.community_reaction import (
    set_reaction_with_notification as svc_set_reaction,
)
from app.services.post_attachment import AttachmentInvalidError
from app.services.scenario_share import (
    NothingToCopyError,
    OriginGoneError,
    PostMissingError,
)
from app.services.scenario_share import copy_attached_scenario as svc_copy_scenario
from app.services.spring_client import SpringInternalClient

router = APIRouter(prefix="/api/v1/community", tags=["community"])



@router.post(
    "/posts",
    response_model=PostDetailResponse,
    status_code=status.HTTP_201_CREATED,
    summary="게시글 작성",
    description="극복 사례나 팁을 제목·내용으로 등록한다.",
)
async def create_post(
    body: PostCreateRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
) -> PostDetailResponse:
    try:
        return await svc_create_post(db, body, user_id)
    except AttachmentInvalidError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e


@router.get(
    "/posts",
    response_model=PostListResponse,
    summary="게시글 목록 조회",
    description="극복 사례·팁 게시글을 최신순으로 조회한다.",
)
async def list_posts(
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=50),
    q: str | None = Query(None, description="제목 또는 내용 검색어"),
) -> PostListResponse:
    return await svc_list_posts(db, viewer_id=user_id, page=page, size=size, q=q)


@router.get(
    "/posts/{post_id}",
    response_model=PostDetailResponse,
    summary="게시글 단건 조회",
    description="게시글 상세 내용과 작성 정보를 반환한다.",
)
async def get_post(
    post_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
    redis: Redis = Depends(get_redis),
) -> PostDetailResponse:
    detail = await svc_get_post(db, post_id, viewer_id=user_id, redis=redis)
    if detail is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="게시글을 찾을 수 없습니다.",
        )
    return detail


@router.patch(
    "/posts/{post_id}",
    response_model=PostDetailResponse,
    summary="게시글 수정",
    description="작성자 본인만 수정할 수 있다.",
)
async def update_post(
    post_id: int,
    body: PostUpdateRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
) -> PostDetailResponse:
    try:
        return await svc_update_post(db, post_id, body, user_id)
    except PostNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="게시글을 찾을 수 없습니다.",
        ) from e
    except PostForbiddenError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="본인이 작성한 게시글만 수정할 수 있습니다.",
        ) from e
    except AttachmentInvalidError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        ) from e


@router.delete(
    "/posts/{post_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="게시글 삭제",
    description="작성자 본인 또는 어드민이 삭제한다. 실제로는 soft delete 다.",
)
async def delete_post(
    post_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
) -> None:
    try:
        await svc_delete_post(db, post_id, user_id=user_id)
    except PostNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="게시글을 찾을 수 없습니다.",
        ) from e
    except PostForbiddenError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="본인이 작성한 게시글만 삭제할 수 있습니다.",
        ) from e


@router.put(
    "/posts/{post_id}/reaction",
    response_model=ReactionStateResponse,
    summary="공감 등록·변경",
    description="힘내요/공감돼요/좋아요 중 하나. 게시글당 하나만 고를 수 있고 다른 종류를 누르면 갈아탄다.",
)
async def set_reaction(
    post_id: int,
    body: ReactionRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
    spring: SpringInternalClient = Depends(get_spring_client),
) -> ReactionStateResponse:
    try:
        reaction, notification_event = await svc_set_reaction(
            db, post_id, user_id=user_id, kind=body.kind
        )
        if notification_event is not None:
            background_tasks.add_task(
                spring.notify_community_notification,
                notification_type=notification_event.notification_type,
                recipient_user_id=notification_event.recipient_user_id,
                actor_user_id=notification_event.actor_user_id,
                post_id=notification_event.post_id,
                reaction_id=notification_event.reaction_id,
                reaction_kind=notification_event.reaction_kind,
            )
        return reaction
    except PostNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="게시글을 찾을 수 없습니다.",
        ) from e


@router.delete(
    "/posts/{post_id}/reaction",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="공감 취소",
    description="눌러둔 공감을 내린다. 공감한 적 없어도 204 로 통과한다.",
)
async def clear_reaction(
    post_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
) -> None:
    try:
        await svc_clear_reaction(db, post_id, user_id=user_id)
    except PostNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="게시글을 찾을 수 없습니다.",
        ) from e


@router.post(
    "/posts/{post_id}/comments",
    response_model=CommentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="댓글·답글 작성",
    description="parent_comment_id 를 주면 그 댓글의 답글이 된다. 답글에 답글은 달 수 없다.",
)
async def create_comment(
    post_id: int,
    body: CommentCreateRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
    spring: SpringInternalClient = Depends(get_spring_client),
) -> CommentResponse:
    try:
        comment, notification_event = await svc_create_comment(
            db, post_id, body, user_id
        )
        if notification_event is not None:
            background_tasks.add_task(
                spring.notify_community_notification,
                notification_type=notification_event.notification_type,
                recipient_user_id=notification_event.recipient_user_id,
                actor_user_id=notification_event.actor_user_id,
                post_id=notification_event.post_id,
                comment_id=notification_event.comment_id,
            )
        return comment
    except PostNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="게시글을 찾을 수 없습니다.",
        ) from e
    except InvalidParentCommentError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="답글을 달 수 없는 댓글입니다. 답글에는 답글을 달 수 없습니다.",
        ) from e


@router.get(
    "/posts/{post_id}/comments",
    response_model=CommentListResponse,
    summary="게시글별 댓글 조회",
    description="댓글과 그 답글을 오래된 순으로 반환한다.",
)
async def list_comments(
    post_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
) -> CommentListResponse:
    try:
        return await svc_list_comments(db, post_id)
    except PostNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="게시글을 찾을 수 없습니다.",
        ) from e


@router.patch(
    "/comments/{comment_id}",
    response_model=CommentResponse,
    summary="댓글 수정",
    description="작성자 본인만 수정할 수 있다.",
)
async def update_comment(
    comment_id: int,
    body: CommentUpdateRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
) -> CommentResponse:
    try:
        return await svc_update_comment(db, comment_id, body, user_id)
    except CommentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="댓글을 찾을 수 없습니다.",
        ) from e
    except CommentForbiddenError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="본인이 작성한 댓글만 수정할 수 있습니다.",
        ) from e


@router.delete(
    "/comments/{comment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="댓글 삭제",
    description="작성자 본인 또는 어드민이 삭제한다. 실제로는 soft delete 다.",
)
async def delete_comment(
    comment_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
) -> None:
    try:
        await svc_delete_comment(db, comment_id, user_id=user_id)
    except CommentNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="댓글을 찾을 수 없습니다.",
        ) from e
    except CommentForbiddenError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="본인이 작성한 댓글만 삭제할 수 있습니다.",
        ) from e


@router.get(
    "/me/posts",
    response_model=PostListResponse,
    summary="내가 작성한 글 조회",
    description="본인이 쓴 게시글만 최신순으로 반환한다. 삭제한 글은 제외된다.",
)
async def list_my_posts(
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=50),
) -> PostListResponse:
    return await svc_list_posts(
        db, viewer_id=user_id, page=page, size=size, author_id=user_id
    )


@router.post(
    "/posts/{post_id}/scenario/copy",
    response_model=ScenarioCopyResponse,
    summary="첨부된 시나리오를 내 목록으로 가져오기",
)
async def copy_attached_scenario(
    post_id: int,
    response: Response,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
) -> ScenarioCopyResponse:
    try:
        scenario, already = await svc_copy_scenario(db, post_id, user_id)
    except PostMissingError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="게시글을 찾을 수 없습니다."
        ) from e
    except NothingToCopyError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except OriginGoneError as e:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="원본 시나리오가 삭제되어 가져올 수 없습니다.",
        ) from e

    response.status_code = status.HTTP_200_OK if already else status.HTTP_201_CREATED
    return ScenarioCopyResponse(
        scenario_id=scenario.scenario_id,
        title=scenario.title,
        category=scenario.category,
        already_copied=already,
    )
