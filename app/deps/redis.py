import logging

from fastapi import Request
from redis.asyncio import Redis, from_url

logger = logging.getLogger(__name__)

async def init_redis(url: str) -> Redis:
    """풀 생성 및 연결 확인"""
    client: Redis = from_url(
        url,
        encoding="utf-8",
        decode_responses=True,
        max_connections=20,
        socket_timeout=5,
        socket_connect_timeout=3,
        health_check_interval=30,
    )
    await client.ping()
    logger.info("redis connected",
                extra={"url": _mask_url(url)})
    return client

async def close_redis(client: Redis) -> None:
    """풀 정리"""
    await client.aclose()
    logger.info("redis closed")

def _mask_url(url: str) -> str:
    """비밀번호 마스킹된 URL"""
    if "@" not in url:
        return url
    scheme_userpass, host = url.rsplit("@", 1)
    if ":" in scheme_userpass.split("//", 1)[-1]:
        prefix = scheme_userpass.rsplit(":", 1)[0]
        return f"{prefix}:***@{host}"
    return url

def get_redis(request: Request) -> Redis:
    """레디스 얻는거"""
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        raise RuntimeError("레디스 초기화 안됨")
    return redis
