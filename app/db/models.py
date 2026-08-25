from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def is_deleted(row: object) -> bool:
    return getattr(row, "deleted_at", None) is not None


class ScenarioORM(Base):
    __tablename__ = "scenario"

    scenario_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
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
