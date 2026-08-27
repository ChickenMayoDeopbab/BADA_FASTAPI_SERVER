from __future__ import annotations

import hashlib
import logging

from app.core.config import Settings
from app.services.recording_storage import RecordingStorageService
from app.services.voice_morph import DEFAULT_SEMITONES, morph_pcm

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000
_URL_TTL_SEC = 600


def morphed_key(recording_key: str, semitones: float = DEFAULT_SEMITONES) -> str:
    """원본 키와 변조 강도가 결정됨"""
    digest = hashlib.sha1(f"{recording_key}|{semitones}".encode()).hexdigest()[:12]
    return f"community/morphed/{digest}.wav"


def build_storage(settings: Settings) -> RecordingStorageService:
    return RecordingStorageService(settings)


def ensure_morphed(
    storage: RecordingStorageService,
    recording_key: str,
    semitones: float = DEFAULT_SEMITONES,
) -> str | None:
    """변조본이 없으면 만들고 키 반환"""
    key = morphed_key(recording_key, semitones)
    if storage.exists(key):
        return key

    pcm = storage.download_pcm(recording_key)
    if not pcm:
        logger.warning("원본 녹음을 못 읽어 변조 생략", extra={"recording_key": recording_key})
        return None

    return storage.upload_wav(key, morph_pcm(pcm, _SAMPLE_RATE, semitones))


def morphed_url(
    storage: RecordingStorageService,
    recording_key: str,
    semitones: float = DEFAULT_SEMITONES,
) -> str | None:
    """이미 만들어져 있으면 재생 URL, 없으면 None"""
    key = morphed_key(recording_key, semitones)
    if not storage.exists(key):
        return None
    return storage.presigned_url(key, expires_in=_URL_TTL_SEC)
