import wave
from io import BytesIO

from app.core.config import Settings
from app.services.recording_storage import RecordingStorageService


class _S3Client:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.presign_calls: list[tuple] = []

    def put_object(self, **kwargs) -> None:
        self.calls.append(kwargs)

    def generate_presigned_url(self, operation, Params=None, ExpiresIn=None):  # noqa: N803
        self.presign_calls.append((operation, Params, ExpiresIn))
        return f"https://signed.test/{Params['Key']}?X-Amz-Expires={ExpiresIn}"


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


def test_upload_pcm_wraps_wav_and_returns_key() -> None:
    s3 = _S3Client()
    storage = RecordingStorageService(_settings(), client=s3)

    key = storage.upload_pcm("sess-123", b"\x01\x00\x02\x00")

    assert key is not None
    assert key.startswith("recordings/sess-123/")
    assert key.endswith(".wav")
    assert len(s3.calls) == 1
    call = s3.calls[0]
    assert call["Bucket"] == "bucket"
    assert call["Key"] == key
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


def test_presigned_url_signs_get_object() -> None:
    s3 = _S3Client()
    storage = RecordingStorageService(_settings(), client=s3)

    url = storage.presigned_url("examples/1-abcd.wav")

    assert url == "https://signed.test/examples/1-abcd.wav?X-Amz-Expires=600"
    assert s3.presign_calls == [
        ("get_object", {"Bucket": "bucket", "Key": "examples/1-abcd.wav"}, 600)
    ]


def test_presigned_url_passes_custom_expiry() -> None:
    s3 = _S3Client()
    storage = RecordingStorageService(_settings(), client=s3)

    storage.presigned_url("examples/1-abcd.wav", expires_in=60)

    assert s3.presign_calls[0][2] == 60


def test_presigned_url_returns_none_without_bucket() -> None:
    settings = _settings()
    settings.s3_bucket = None
    storage = RecordingStorageService(settings, client=_S3Client())

    assert storage.presigned_url("examples/1-abcd.wav") is None


def test_presigned_url_returns_none_for_empty_key() -> None:
    storage = RecordingStorageService(_settings(), client=_S3Client())

    assert storage.presigned_url("") is None
