from functools import lru_cache
from pathlib import Path
from typing import Literal

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

    # Security
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    internal_secret: str
    ws_auth_timeout_seconds: float = 10.0

    # STT
    google_stt_credentials_path: str | None = None
    google_project_id: str
    google_stt_location: str = "asia-northeast1"
    google_stt_model: str = "chirp_3"
    google_stt_language: str = "ko-KR"
    google_stt_denoise: bool = False
    google_stt_denoise_snr_threshold: float = 0.0
    stt_engine: Literal["chirp", "gemini_live"] = "chirp"
    gemini_stt_model: str = "gemini-3.5-transcribe-live"
    gemini_stt_silence_ms: int = 500

    # LLM(실시간)
    gemini_api_key: str
    llm_realtime_model: str = "gemini-3.5-flash-lite"
    llm_thinking_budget: int | None = None

    # 커뮤니티 유해 콘텐츠 검사
    community_moderation_model: str = "gemini-3.5-flash-lite"
    community_moderation_timeout_seconds: float = 5.0

    # 커뮤니티 신고 운영 알림
    community_report_alert_webhook_url: str | None = None

    # 문의·피드백 전송
    inquiry_webhook_url: str | None = None

    # LLM(분석)
    anthropic_api_key: str
    llm_analysis_model: str = "claude-sonnet-4-20250514"

    # 시나리오 썸네일 이미지
    gemini_image_model: str = "gemini-2.5-flash-image"

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

    qwen_tts_url: str | None = None
    qwen_tts_urls: str | None = None
    qwen_tts_timeout: float = 30.0
    qwen_tts_health_timeout: float = 1.0
    qwen_tts_realtime_enabled: bool = False
    qwen_tts_speed: float = 1.0 # 실시간 말 속도(0.8~1.25)
    qwen_tts_req_metric: bool = True  # 실시간 요청별 끝·쉼·시간 계측 줄
    qwen_tts_non_streaming: bool = False  # 실시간 N 요청 + 말소리 뒤 쉼 압축
    qwen_tts_non_streaming_voices: str = "ai"  # N 을 쓸 보이스(쉼표 구분)
    qwen_tts_pause_keep_ms: float = 160.0  # 말소리 뒤 남길 쉼(들리는 ms)
    qwen_tts_pause_cap_over_ms: float = 350.0  # 안쪽 쉼이 이보다 길면
    qwen_tts_pause_cap_to_ms: float = 250.0  # 이 길이로 줄임
    qwen_tts_pause_min_buffer_ms: float = 300.0  # 앱 버퍼가 이보다 얇으면 쉼을 덜 줄임
    qwen_tts_ns_gain_db: float = -3.0  # N 적용 요청에 곱할 크기(dB, ai 기준 실측)
    qwen_tts_gap_voices: str = ""  # 요청 사이 쉼을 채울 보이스(쉼표 구분)
    qwen_tts_gap_min_ms: float = 250.0  # 그 보이스의 요청 사이 들리는 쉼 하한
    example_qwen_ns: bool = False  # 프리셋·새 커스텀 예시를 Qwen non_streaming_mode 로 굽기
    example_request_wait_sec: float = 20.0  # 예시 요청이 굽기를 기다리는 상한(앱 30초 타임아웃 안)
    example_turn_prior_sec: float = 6.0  # 남은 굽기 시간 추정용 턴당 사전값
    example_level_ai_db: float = -16.94749568714168  # 예시 AI 말소리 크기 목표(청취본 중앙값)
    example_level_user_db: float = -22.544892951720513  # 예시 user 말소리 크기 목표(청취본 중앙값)
    example_ns_skip_presets: str = ""  # N 굽기에서 뺄 프리셋 id 콤마 목록(그 프리셋은 예전 굽기 그대로)
    qwen_scenario_voices: bool = False  # Qwen에 시나리오, 난이도 칸 보이스를 보낼지
    csm_tts_urls: str | None = None
    csm_tts_realtime_enabled: bool = False
    csm_tts_voice: str = "ai"
    csm_tts_health_timeout: float = 1.0
    tts_coalesce_ms: int = 320

    # 통화 첫마디
    ai_greeting: bool = False  # 연결 직후 AI 가 먼저 말하기
    ai_greeting_user_ids: str = ""  # 먼저 말할 userId(쉼표 구분, 비우면 전체)

    # Internal callback
    spring_boot_internal_url: str

    # S3 recordings
    aws_access_key: str | None = None
    aws_secret_key: str | None = None
    aws_region: str = "ap-northeast-2"
    s3_bucket: str | None = None
    # 녹음 원본을 잠그는 KMS 키(ARN 또는 alias/...). 없으면 녹음을 올리지 않는다.
    recording_kms_key_id: str | None = None

    # DB
    database_url: str

    avti_enabled: bool = True
    praat_bin: str = "praat"
    avti_script_path: str = str(
        Path(__file__).resolve().parent.parent.parent / "vendor" / "tremor3.05" / "tremor.praat"
    )
    avti_window_sec: float = 2.5
    avti_min_sustained_sec: float = 2.5
    avti_timeout_sec: float = 30.0

@lru_cache
def get_settings() -> Settings:
    return Settings()
