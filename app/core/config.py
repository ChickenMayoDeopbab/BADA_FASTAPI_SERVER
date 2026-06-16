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
    google_stt_credentials_path: str | None = None
    google_project_id: str
    google_stt_location: str = "asia-northeast1"
    google_stt_model: str = "chirp_3"
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
    elevenlabs_ws_host: str = "wss://api.elevenlabs.io"
    elevenlabs_output_format: str = "pcm_16000" # PCM 16k로
    elevenlabs_language_code: str = "ko"
    elevenlabs_apply_text_normalization: str = "on"
    elevenlabs_auto_mode: bool = True # 짧은 대사 즉시 생성
    elevenlabs_stability: float = 0.5
    elevenlabs_similarity_boost: float = 0.75
    elevenlabs_style: float = 0.0
    elevenlabs_speaker_boost: bool = False
    elevenlabs_speed: float = 1.0

    # Internal callback
    spring_boot_internal_url: str


@lru_cache
def get_settings() -> Settings:
    return Settings()
