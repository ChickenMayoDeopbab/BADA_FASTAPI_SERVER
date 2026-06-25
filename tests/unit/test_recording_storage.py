import wave
from io import BytesIO

from app.core.config import Settings
from app.services.recording_storage import RecordingStorageService


class _S3Client:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def put_object(self, **kwargs) -> None:
        self.calls.append(kwargs)


def _settings() -> Settings:
    return Settings(
        redis_url="redis://localhost:6379/0",
        rabbitmq_url="amqp://guest:guest@localhost:5672/",
        jwt_secret="secret",
        internal_secret="internal",
        google_project_id="project",
        gemini_api_key="gemini",
        anthropic_api_key="anthropic",
        elevenlabs_api_key="eleven",
        elevenlabs_voice_id="voice",
        spring_boot_internal_url="http://spring",
        database_url="postgresql+asyncpg://user:pass@localhost/db",
        s3_bucket="bucket",
        aws_region="ap-northeast-2",
    )


def test_upload_pcm_wraps_wav_and_returns_url() -> None:
    s3 = _S3Client()
    storage = RecordingStorageService(_settings(), client=s3)

    url = storage.upload_pcm("sess-123", b"\x01\x00\x02\x00")

    assert url == "https://bucket.s3.ap-northeast-2.amazonaws.com/recordings/sess-123.wav"
    assert len(s3.calls) == 1
    call = s3.calls[0]
    assert call["Bucket"] == "bucket"
    assert call["Key"] == "recordings/sess-123.wav"
    assert call["ContentType"] == "audio/wav"

    with wave.open(BytesIO(call["Body"]), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16_000
        assert wav.readframes(2) == b"\x01\x00\x02\x00"


def test_upload_pcm_returns_none_without_bucket() -> None:
    settings = _settings()
    settings.s3_bucket = None
    storage = RecordingStorageService(settings, client=_S3Client())

    assert storage.upload_pcm("sess-123", b"\x01\x00") is None
