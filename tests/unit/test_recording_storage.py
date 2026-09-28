import wave
from io import BytesIO

import pytest

from app.core.config import Settings
from app.services import recording_crypto
from app.services.recording_storage import RecordingStorageService
from tests.unit.fake_kms import FakeKms


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


class _S3Client:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.delete_calls: list[dict] = []
        self.presign_calls: list[tuple] = []
        self.objects: dict[str, bytes] = {}

    def put_object(self, **kwargs) -> None:
        self.calls.append(kwargs)
        self.objects[kwargs["Key"]] = kwargs["Body"]

    def get_object(self, *, Bucket, Key):  # noqa: N803
        return {"Body": _Body(self.objects[Key])}

    def delete_object(self, **kwargs) -> None:
        self.delete_calls.append(kwargs)

    def generate_presigned_url(self, operation, Params=None, ExpiresIn=None):  # noqa: N803
        self.presign_calls.append((operation, Params, ExpiresIn))
        return f"https://signed.test/{Params['Key']}?X-Amz-Expires={ExpiresIn}"


def _settings() -> Settings:
    return Settings(
        redis_url="redis://localhost:6379/0",
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
        recording_kms_key_id="alias/bada-recording",
    )


def test_upload_pcm_encrypts_wav_and_returns_key() -> None:
    s3, kms = _S3Client(), FakeKms()
    storage = RecordingStorageService(_settings(), client=s3, kms_client=kms)

    key = storage.upload_pcm("sess-123", b"\x01\x00\x02\x00")

    assert key is not None
    assert key.startswith("recordings/sess-123/")
    assert key.endswith(".wav")
    assert len(s3.calls) == 1
    call = s3.calls[0]
    assert call["Bucket"] == "bucket"
    assert call["Key"] == key
    assert call["ContentType"] == "application/octet-stream"
    assert recording_crypto.is_encrypted(call["Body"])
    assert kms.generate_calls[0]["KeyId"] == "alias/bada-recording"

    plain = recording_crypto.decrypt(call["Body"], s3_key=key, kms=kms)
    with wave.open(BytesIO(plain), "rb") as wav:
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        assert wav.getframerate() == 16_000
        assert wav.readframes(2) == b"\x01\x00\x02\x00"


def test_upload_pcm_refuses_plaintext_without_kms_key() -> None:
    settings = _settings()
    settings.recording_kms_key_id = None
    s3 = _S3Client()
    storage = RecordingStorageService(settings, client=s3, kms_client=FakeKms())

    with pytest.raises(RuntimeError):
        storage.upload_pcm("sess-123", b"\x01\x00")
    assert s3.calls == []


def test_upload_pcm_returns_none_without_bucket() -> None:
    settings = _settings()
    settings.s3_bucket = None
    storage = RecordingStorageService(settings, client=_S3Client())

    assert storage.upload_pcm("sess-123", b"\x01\x00") is None


def test_download_pcm_opens_encrypted_recording() -> None:
    s3, kms = _S3Client(), FakeKms()
    storage = RecordingStorageService(_settings(), client=s3, kms_client=kms)
    key = storage.upload_pcm("sess-123", b"\x01\x00\x02\x00")

    assert storage.download_pcm(key) == b"\x01\x00\x02\x00"


def test_download_pcm_still_reads_legacy_plain_recording() -> None:
    s3 = _S3Client()
    storage = RecordingStorageService(_settings(), client=s3, kms_client=FakeKms())
    s3.objects["recordings/old.wav"] = RecordingStorageService._to_wav(b"\x03\x00")

    assert storage.download_pcm("recordings/old.wav") == b"\x03\x00"


def test_download_pcm_returns_none_when_key_cannot_be_opened() -> None:
    s3, kms = _S3Client(), FakeKms()
    storage = RecordingStorageService(_settings(), client=s3, kms_client=kms)
    key = storage.upload_pcm("sess-123", b"\x01\x00")
    s3.objects["recordings/moved.wav"] = s3.objects[key]

    assert storage.download_pcm("recordings/moved.wav") is None


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


def test_delete_removes_s3_object() -> None:
    s3 = _S3Client()
    storage = RecordingStorageService(_settings(), client=s3)

    assert storage.delete("community/morphed/test.wav") is True
    assert s3.delete_calls == [
        {"Bucket": "bucket", "Key": "community/morphed/test.wav"}
    ]


def test_delete_is_noop_without_bucket() -> None:
    settings = _settings()
    settings.s3_bucket = None
    s3 = _S3Client()
    storage = RecordingStorageService(settings, client=s3)

    assert storage.delete("community/morphed/test.wav") is False
    assert s3.delete_calls == []
