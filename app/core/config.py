from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent.parent.parent / ".env",
        extra="ignore",
    )

    # App
    env: str = "dev"
    cors_origins: list[str] = ["*"]

    # Infra
    redis_url: str
    rabbitmq_url: str

    # Security
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    internal_secret: str

    # STT
    google_stt_credentials_path: str
    google_stt_language: str = "ko-KR"

    # LLM(실시간)
    gemini_api_key: str
    llm_realtime_model: str = "gemini-2.5-flash"

    # LLM(분석)
    anthropic_api_key: str
    llm_analysis_model: str = "claude-sonnet-4-6"

    # TTS
    elevenlabs_api_key: str
    elevenlabs_model: str = "eleven_flash_v2_5"
    elevenlabs_voice_id: str

    # Internal callback
    spring_boot_internal_url: str

    # DB
    database_url: str

@lru_cache
def get_settings() -> Settings:
    return Settings()