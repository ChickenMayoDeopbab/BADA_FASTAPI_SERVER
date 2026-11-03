from __future__ import annotations

import functools
import threading
from typing import Any

from app.core.config import Settings

_lock = threading.Lock()


def shared_boto_client(service: str, settings: Settings) -> Any:
    """설정별 boto3 클라이언트를 프로세스에 하나만 만듦"""
    with _lock:
        return _client(service, settings.aws_region, settings.aws_access_key, settings.aws_secret_key)


@functools.cache
def _client(service: str, region: str, access_key: str | None, secret_key: str | None) -> Any:
    import boto3

    if access_key and secret_key:
        return boto3.client(
            service,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
    return boto3.client(service, region_name=region)
