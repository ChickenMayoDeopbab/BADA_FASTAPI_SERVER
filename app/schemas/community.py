import unicodedata
from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, BeforeValidator, Field, StringConstraints

from app.core.enums import AttachmentKind, ReactionKind


def to_nfc(value: object) -> object:
    """NFC로 변경"""
    return unicodedata.normalize("NFC", value) if isinstance(value, str) else value


def _clean_text(value: object) -> object:
    """text 정규화"""
    return to_nfc(value).strip() if isinstance(value, str) else value


TITLE_MAX = 100
POST_BODY_MAX = 5_000
COMMENT_BODY_MAX = 1_000

Title = Annotated[
    str, BeforeValidator(_clean_text), StringConstraints(min_length=1, max_length=TITLE_MAX)
]
PostBody = Annotated[
    str, BeforeValidator(_clean_text), StringConstraints(min_length=1, max_length=POST_BODY_MAX)
]
CommentBody = Annotated[
    str, BeforeValidator(_clean_text), StringConstraints(min_length=1, max_length=COMMENT_BODY_MAX)
]


class AuthorInfo(BaseModel):
    """게시글·댓글 작성자"""

    user_id: int
    name: str | None = None
    profile_image_url: str | None = None


class ReactionCounts(BaseModel):
    cheer: int = 0
    relate: int = 0
    like: int = 0
    total: int = 0


class AttachmentRequest(BaseModel):
    """첨부하는거 종류별 최대 1개"""

    kind: AttachmentKind
    ref_id: int


class AttachedScenario(BaseModel):
    """첨부된 시나리오"""

    title: str
    content: str
    category: str
    is_available: bool = True
    is_mine: bool = False


class AttachedTrainingRecord(BaseModel):
    scenario_name: str | None = None
    session_type: str | None = None
    started_at: datetime | None = None
    duration_seconds: int | None = None
    anxiety_score: int | None = None
    is_available: bool = True


class PostAttachment(BaseModel):
    kind: AttachmentKind
    ref_id: int
    scenario: AttachedScenario | None = None
    training_record: AttachedTrainingRecord | None = None


class ScenarioCopyResponse(BaseModel):
    """시나리오 카피 결과"""

    scenario_id: int
    title: str
    category: str
    already_copied: bool = False


class PostCreateRequest(BaseModel):
    title: Title
    content: PostBody
    attachments: list[AttachmentRequest] = Field(default_factory=list)


class PostUpdateRequest(BaseModel):
    title: Title | None = None
    content: PostBody | None = None
    attachments: list[AttachmentRequest] | None = None


class PostDetailResponse(BaseModel):
    post_id: int
    title: str
    content: str
    author: AuthorInfo
    view_count: int
    comment_count: int = 0
    reactions: ReactionCounts = Field(default_factory=lambda: ReactionCounts())
    my_reaction: ReactionKind | None = None
    attachments: list[PostAttachment] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


_PREVIEW_CHARS = 100
_ZWJ = "\u200d"


class PostSummary(BaseModel):
    """목록용 요약, 본문은 앞 100자만 담음"""

    post_id: int
    title: str
    content_preview: str
    author: AuthorInfo
    view_count: int
    comment_count: int
    reactions: ReactionCounts
    my_reaction: ReactionKind | None = None
    attachment_kinds: list[AttachmentKind] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class PostListResponse(BaseModel):
    posts: list[PostSummary]
    page: int
    size: int
    total: int
    has_next: bool


def preview_of(content: str) -> str:
    """프리뷰"""
    cut = content[:_PREVIEW_CHARS]
    while cut and (cut[-1] == _ZWJ or unicodedata.combining(cut[-1])):
        cut = cut[:-1]
    return cut


class ReactionRequest(BaseModel):
    kind: ReactionKind


class ReactionStateResponse(BaseModel):
    """공감 등록·변경 직후 상태"""

    post_id: int
    reactions: ReactionCounts
    my_reaction: ReactionKind | None = None


class CommentCreateRequest(BaseModel):
    content: CommentBody
    parent_comment_id: int | None = None


class CommentUpdateRequest(BaseModel):
    content: CommentBody


class CommentResponse(BaseModel):
    comment_id: int
    parent_comment_id: int | None = None
    content: str
    author: AuthorInfo
    created_at: datetime
    updated_at: datetime


class CommentThread(CommentResponse):
    """최상위 댓글 + 그 답글"""

    replies: list[CommentResponse] = []


class CommentListResponse(BaseModel):
    comments: list[CommentThread]
