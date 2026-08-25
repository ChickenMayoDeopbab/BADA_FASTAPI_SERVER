import hmac

import jwt
from fastapi import Header, HTTPException, WebSocket, status

from app.core.config import get_settings

DB_ADMIN_ROLE = "ADMIN"


def is_admin(role: str | None) -> bool:
    """admin인지"""
    return role == DB_ADMIN_ROLE


async def require_internal_secret(
    x_internal_secret: str | None = Header(default=None),
) -> None:
    expected = get_settings().internal_secret
    if not x_internal_secret or not hmac.compare_digest(
        x_internal_secret.encode(), expected.encode()
    ):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)

def decode_access_token(token: str) -> dict:
    """Spring이 발급해준 Access 토큰 검증"""
    settings = get_settings()
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=[settings.jwt_algorithm],
            # PyJWT 는 exp 가 "있으면" 검사하고 "없으면" 통과시킨다.
            # 필수로 요구하지 않으면 만료 없는 토큰이 영구 유효해진다.
            options={"require": ["exp", "sub", "type"]},
        )
    except jwt.ExpiredSignatureError as e:
        raise HTTPException(status_code=401, detail="TOKEN_EXPIRED") from e
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail="TOKEN_INVALID") from e

    if payload.get("type") != "ACCESS":
        raise HTTPException(status_code=401, detail="NOT_ACCESS_TOKEN")

    return payload

async def authenticate_ws(ws: WebSocket, token: str) -> tuple[int, str | None]:
    """Websocket 전용 검증"""
    try:
        payload = decode_access_token(token)
    except HTTPException as e:
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason=str(e.detail))
        raise
    user_id = int(payload["sub"])
    role = payload.get("role")
    return user_id, role
