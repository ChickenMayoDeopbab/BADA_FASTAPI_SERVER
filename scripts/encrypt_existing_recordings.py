"""S3 에 평문으로 남은 녹음 원본(recordings/)을 KMS 봉투 암호화로 바꾼다.

기본은 건수만 세는 점검 모드다. --apply 를 붙여야 실제로 덮어쓴다.
이미 잠긴 파일은 건너뛰므로 여러 번 돌려도 된다. 덮어쓰기 전에 메모리에서
다시 풀어 원본과 같은지 확인하고, 다를 때는 올리지 않는다.
S3 키는 그대로 두므로 Spring DB 의 recording_key 는 바꿀 필요가 없다.

버킷 버전 관리가 켜져 있으면 평문이 이전 버전으로 남는다. 확인 후 이전 버전을 지울 것.

예)
  .venv/bin/python scripts/encrypt_existing_recordings.py
  .venv/bin/python scripts/encrypt_existing_recordings.py --apply
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import get_settings  # noqa: E402
from app.services import recording_crypto  # noqa: E402
from app.services.recording_storage import _boto_client  # noqa: E402

PREFIX = "recordings/"


@dataclass
class Tally:
    plain: list[str] = field(default_factory=list)
    already_encrypted: int = 0
    not_wav: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def _keys(s3: Any, bucket: str, prefix: str):
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]


def _head(s3: Any, bucket: str, key: str) -> bytes:
    size = len(recording_crypto.MAGIC)
    return s3.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{size - 1}")["Body"].read()


def migrate(
    s3: Any, kms: Any, *, bucket: str, key_id: str, prefix: str = PREFIX, apply: bool = False
) -> Tally:
    tally = Tally()
    for key in _keys(s3, bucket, prefix):
        head = _head(s3, bucket, key)
        if recording_crypto.is_encrypted(head):
            tally.already_encrypted += 1
            continue
        if not head.startswith(b"RIFF"):
            tally.not_wav.append(key)
            continue
        tally.plain.append(key)
        if not apply:
            continue

        try:
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            blob = recording_crypto.encrypt(body, s3_key=key, kms=kms, key_id=key_id)
            if recording_crypto.decrypt(blob, s3_key=key, kms=kms) != body:
                raise RuntimeError("다시 풀어 본 내용이 원본과 다름")
            s3.put_object(
                Bucket=bucket, Key=key, Body=blob, ContentType="application/octet-stream"
            )
        except Exception as e:
            tally.failed.append(f"{key}: {e}")
    return tally


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="실제로 암호화해서 덮어쓴다")
    parser.add_argument("--prefix", default=PREFIX)
    args = parser.parse_args()

    settings = get_settings()
    if not settings.s3_bucket or not settings.recording_kms_key_id:
        print("S3_BUCKET 과 RECORDING_KMS_KEY_ID 가 모두 있어야 한다", file=sys.stderr)
        return 2

    tally = migrate(
        _boto_client("s3", settings),
        _boto_client("kms", settings),
        bucket=settings.s3_bucket,
        key_id=settings.recording_kms_key_id,
        prefix=args.prefix,
        apply=args.apply,
    )

    verb = "암호화함" if args.apply else "암호화 대상"
    print(f"{verb}: {len(tally.plain) - len(tally.failed)}")
    print(f"이미 잠김: {tally.already_encrypted}")
    for key in tally.not_wav:
        print(f"WAV 아님(건너뜀): {key}")
    for line in tally.failed:
        print(f"실패: {line}", file=sys.stderr)
    if not args.apply and tally.plain:
        print("--apply 를 붙여 다시 실행하면 덮어쓴다")
    return 1 if tally.failed else 0


if __name__ == "__main__":
    sys.exit(main())
