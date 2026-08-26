from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def is_deleted(row: object) -> bool:
    return getattr(row, "deleted_at", None) is not None


_AUTO_PK = BigInteger().with_variant(Integer, "sqlite")


class ScenarioORM(Base):
    __tablename__ = "scenario"
    __table_args__ = (
        Index("ix_scenario_origin_user", "origin_scenario_id", "user_id"),
    )

    # _AUTO_PK 를 쓰는 이유: 맨 BigInteger 는 SQLite 에서 BIGINT 로 렌더돼
    # rowid 자동 부여가 안 된다(INTEGER PRIMARY KEY 에서만 된다).
    # PostgreSQL DDL 은 양쪽 다 BIGSERIAL 로 같아서 마이그레이션은 필요 없다.
    scenario_id: Mapped[int] = mapped_column(_AUTO_PK, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(20), nullable=False)
    scenario_image: Mapped[str | None] = mapped_column(String(255), nullable=True)
    tts_voice_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    ai_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    is_custom: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    is_warmup: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    call_target: Mapped[str] = mapped_column(String(100), nullable=False)
    call_purpose: Mapped[str] = mapped_column(String(200), nullable=False)
    script: Mapped[list | None] = mapped_column(JSON, nullable=True) # [{"step", "ai_goal", "hint"}]
    example_dialogue: Mapped[list | None] = mapped_column(JSON, nullable=True) # 커스텀 전용 [{"speaker", "text"}]
    example_audio_url: Mapped[str | None] = mapped_column(String(255), nullable=True)
    origin_scenario_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

class FeedbackORM(Base):
    __tablename__ = "feedback"

    feedback_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    scenario_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("scenario.scenario_id"), nullable=False)
    shake_count: Mapped[int] = mapped_column(BigInteger, nullable=False)
    silence_duration: Mapped[int] = mapped_column(BigInteger, nullable=False)
    highlights: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

class VoiceTremorMetricORM(Base):
    __tablename__ = "voice_tremor_metric"
    __table_args__ = (UniqueConstraint("session_id", "part_index", name="uq_vtm_session_part"),)

    metric_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    part_index: Mapped[int] = mapped_column(Integer, nullable=False)  # 0 = 세션 전체
    start_sec: Mapped[float] = mapped_column(Float, nullable=False)
    end_sec: Mapped[float] = mapped_column(Float, nullable=False)

    status: Mapped[str] = mapped_column(String(16), nullable=False)
    avti: Mapped[float | None] = mapped_column(Float, nullable=True)
    ftrcip: Mapped[float | None] = mapped_column(Float, nullable=True)
    atri: Mapped[float | None] = mapped_column(Float, nullable=True)
    fcohnr: Mapped[float | None] = mapped_column(Float, nullable=True)
    fcom: Mapped[float | None] = mapped_column(Float, nullable=True)

    script_version: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

class FileORM(Base):
    __tablename__ = "file"

    file_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    file_type: Mapped[str] = mapped_column(String(32), nullable=False)
    s3_key: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)


class PostORM(Base):
    """커뮤니티 게시글"""

    __tablename__ = "post"
    __table_args__ = (
        Index("ix_post_alive_recent", "deleted_at", "created_at"),
        Index("ix_post_user_recent", "user_id", "created_at"),
    )

    post_id: Mapped[int] = mapped_column(_AUTO_PK, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    title: Mapped[str] = mapped_column(String(100), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    view_count: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class PostCommentORM(Base):
    """댓글과 답글들"""

    __tablename__ = "post_comment"
    __table_args__ = (
        Index("ix_post_comment_post_created", "post_id", "created_at"),
        Index("ix_post_comment_parent", "parent_comment_id"),
    )

    comment_id: Mapped[int] = mapped_column(_AUTO_PK, primary_key=True, autoincrement=True)
    post_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("post.post_id"), nullable=False)
    parent_comment_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("post_comment.comment_id"), nullable=True
    )
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class PostReactionORM(Base):
    """게시글 공감, 변경은 UPDATE, 취소는 DELETE."""

    __tablename__ = "post_reaction"
    __table_args__ = (UniqueConstraint("post_id", "user_id", name="uq_post_reaction_post_user"),)

    reaction_id: Mapped[int] = mapped_column(_AUTO_PK, primary_key=True, autoincrement=True)
    post_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("post.post_id"), nullable=False)
    user_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class PostAttachmentORM(Base):
    __tablename__ = "post_attachment"
    __table_args__ = (
        UniqueConstraint("post_id", "kind", name="uq_post_attachment_post_kind"),
    )

    attachment_id: Mapped[int] = mapped_column(_AUTO_PK, primary_key=True, autoincrement=True)
    post_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("post.post_id"), nullable=False)
    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    ref_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
