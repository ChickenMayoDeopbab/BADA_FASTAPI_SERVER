import io
import logging
import uuid
import wave
from typing import Any

from app.core.config import Settings
from app.services import recording_crypto

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000
_CHANNELS = 1
_SAMPLE_WIDTH_BYTES = 2


def _boto_client(service: str, settings: Settings) -> Any:
    import boto3

    if settings.aws_access_key and settings.aws_secret_key:
        return boto3.client(
            service,
            aws_access_key_id=settings.aws_access_key,
            aws_secret_access_key=settings.aws_secret_key,
            region_name=settings.aws_region,
        )
    return boto3.client(service, region_name=settings.aws_region)


class RecordingStorageService:
    def __init__(
        self, settings: Settings, client: Any | None = None, kms_client: Any | None = None
    ) -> None:
        self._settings = settings
        self._bucket = settings.s3_bucket
        self._client = client
        self._kms = kms_client
        self._kms_key_id = getattr(settings, "recording_kms_key_id", None)
        if self._bucket and self._client is None:
            self._client = _boto_client("s3", settings)

    def upload_pcm(self, session_id: str, pcm: bytes) -> str | None:
        """녹음 원본은 KMS로 잠가서 올린다. 키가 없으면 평문으로 올리지 않고 실패한다."""
        if not self._bucket or not pcm or self._client is None:
            return None
        if not self._kms_key_id:
            raise RuntimeError("recording_kms_key_id 미설정 - 녹음을 평문으로 올리지 않음")

        key = f"recordings/{session_id}/{uuid.uuid4()}.wav"
        body = recording_crypto.encrypt(
            self._to_wav(pcm), s3_key=key, kms=self._kms_client(), key_id=self._kms_key_id
        )
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=body,
            ContentType="application/octet-stream",
        )
        return key

    def upload_wav(self, key: str, pcm: bytes) -> str | None:
        if not self._bucket or not pcm:
            return None

        if self._client is None:
            return None

        wav_bytes = self._to_wav(pcm)
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=wav_bytes,
            ContentType="audio/wav",
        )
        return key

    def download_pcm(self, key: str) -> bytes | None:
        """WAV읽고 raw PCM만 줌. 잠긴 녹음이면 풀어서 준다."""
        if not self._bucket or not key or self._client is None:
            return None
        try:
            body = self._client.get_object(Bucket=self._bucket, Key=key)["Body"].read()
            # 일괄 암호화 전에 올라간 평문 녹음도 그대로 읽는다.
            if recording_crypto.is_encrypted(body):
                body = recording_crypto.decrypt(body, s3_key=key, kms=self._kms_client())
            with wave.open(io.BytesIO(body), "rb") as wav:
                return wav.readframes(wav.getnframes())
        except Exception:
            logger.warning("녹음 읽기 실패", extra={"recording_key": key}, exc_info=True)
            return None

    def exists(self, key: str) -> bool:
        if not self._bucket or not key or self._client is None:
            return False
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
        except Exception:
            return False
        return True

    def delete(self, key: str) -> bool:
        if not self._bucket or not key or self._client is None:
            return False
        self._client.delete_object(Bucket=self._bucket, Key=key)
        return True

    def presigned_url(self, key: str, expires_in: int = 600) -> str | None:
        if not self._bucket or not key or self._client is None:
            return None
        return self._client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self._bucket, "Key": key},
            ExpiresIn=expires_in,
        )

    def _kms_client(self) -> Any:
        if self._kms is None:
            self._kms = _boto_client("kms", self._settings)
        return self._kms

    @staticmethod
    def _to_wav(pcm: bytes) -> bytes:
        if len(pcm) % _SAMPLE_WIDTH_BYTES != 0:
            pcm = pcm[:-1]

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(_CHANNELS)
            wav.setsampwidth(_SAMPLE_WIDTH_BYTES)
            wav.setframerate(_SAMPLE_RATE)
            wav.writeframes(pcm)
        return buffer.getvalue()
