import io
import uuid
import wave
from typing import Any

from app.core.config import Settings

_SAMPLE_RATE = 16_000
_CHANNELS = 1
_SAMPLE_WIDTH_BYTES = 2


class RecordingStorageService:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._bucket = settings.s3_bucket
        self._client = client
        if self._bucket and self._client is None:
            import boto3

            if settings.aws_access_key and settings.aws_secret_key:
                self._client = boto3.client(
                    "s3",
                    aws_access_key_id=settings.aws_access_key,
                    aws_secret_access_key=settings.aws_secret_key,
                    region_name=settings.aws_region,
                )
            else:
                self._client = boto3.client("s3", region_name=settings.aws_region)

    def upload_pcm(self, session_id: str, pcm: bytes) -> str | None:
        if not self._bucket or not pcm:
            return None

        if self._client is None:
            return None

        key = f"recordings/{session_id}/{uuid.uuid4()}.wav"
        wav_bytes = self._to_wav(pcm)
        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=wav_bytes,
            ContentType="audio/wav",
        )
        return key

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
