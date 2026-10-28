from __future__ import annotations

import functools
import ssl

import httpx


@functools.cache
def shared_ssl_context() -> ssl.SSLContext:
    """httpx 기본 검증과 같은 SSL 컨텍스트를 프로세스에 하나만 만듦"""
    return httpx.create_ssl_context()
