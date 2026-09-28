"""녹음 원본 봉투 암호화.

파일마다 KMS에서 새 데이터 키를 받아 AES-256-GCM 으로 잠그고, KMS로 잠긴
데이터 키를 파일 앞머리에 붙여 둔다. 여는 쪽(변조 작업, Spring 재생 API)은
앞머리의 데이터 키를 KMS로 풀어 본문을 연다.

    MAGIC(8) | edk_len(2, big-endian) | edk | nonce(12) | ciphertext + tag(16)

앞머리(MAGIC ~ edk)는 GCM 의 AAD 로 묶고, KMS 암호화 컨텍스트에는 S3 키를
넣어 파일을 다른 키 자리로 옮기면 열리지 않게 한다.
형식을 바꾸면 Spring RecordingCipher 도 같이 바꿔야 한다.
"""

from __future__ import annotations

import os
import struct
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"BADAREC1"
_LEN = struct.Struct(">H")
_NONCE_BYTES = 12


def is_encrypted(blob: bytes) -> bool:
    return blob.startswith(MAGIC)


def _context(s3_key: str) -> dict[str, str]:
    return {"purpose": "bada-recording", "s3_key": s3_key}


def encrypt(plain: bytes, *, s3_key: str, kms: Any, key_id: str) -> bytes:
    data_key = kms.generate_data_key(
        KeyId=key_id, KeySpec="AES_256", EncryptionContext=_context(s3_key)
    )
    edk = data_key["CiphertextBlob"]
    header = MAGIC + _LEN.pack(len(edk)) + edk
    nonce = os.urandom(_NONCE_BYTES)
    return header + nonce + AESGCM(data_key["Plaintext"]).encrypt(nonce, plain, header)


def decrypt(blob: bytes, *, s3_key: str, kms: Any) -> bytes:
    if not is_encrypted(blob):
        raise ValueError("암호화된 녹음이 아님")
    edk_start = len(MAGIC) + _LEN.size
    (edk_len,) = _LEN.unpack_from(blob, len(MAGIC))
    header_end = edk_start + edk_len
    body_start = header_end + _NONCE_BYTES
    if len(blob) < body_start:
        raise ValueError("녹음 앞머리가 잘림")

    header = blob[:header_end]
    data_key = kms.decrypt(
        CiphertextBlob=blob[edk_start:header_end], EncryptionContext=_context(s3_key)
    )["Plaintext"]
    return AESGCM(data_key).decrypt(blob[header_end:body_start], blob[body_start:], header)
