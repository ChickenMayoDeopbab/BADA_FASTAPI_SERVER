from datetime import datetime

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ScenarioORM(Base):
    __tablename__ = "scenario"

    scenario_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(50), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    scenario_image: Mapped[str | None] = mapped_column(String(255), nullable=True)
    tts_voice_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    ai_prompt: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    is_custom: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    call_target: Mapped[str] = mapped_column(String(100), nullable=False)
    call_purpose: Mapped[str] = mapped_column(String(200), nullable=False)
    script: Mapped[list | None] = mapped_column(JSON, nullable=True) # [{"step", "ai_goal", "hint"}]
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
