from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import authenticate_ws as authenticate_ws_token
from app.core.security import decode_access_token
from app.core.timeutil import ensure_utc, now_utc
from app.db.base import AsyncSessionLocal
from app.db.external import users_table
from app.deps.db import get_db

_bearer = HTTPBearer()


async def ensure_user_can_access(db: AsyncSession, user_id: int) -> None:
    stmt = select(users_table.c.status, users_table.c.suspended_until).where(users_table.c.user_id == user_id)
    row = (await db.execute(stmt)).first()
    if row is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="USER_NOT_FOUND")
    if row.status == "ACTIVE":
        return
    if row.status == "BANNED":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="USER_BANNED")
    if row.status == "SUSPENDED":
        if row.suspended_until is not None and ensure_utc(row.suspended_until) <= now_utc():
            return
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="USER_SUSPENDED")
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="USER_INACTIVE")


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer), db: AsyncSession = Depends(get_db)
) -> int:
    payload = decode_access_token(credentials.credentials)
    try:
        user_id = int(payload["sub"])
    except (KeyError, ValueError) as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="유효하지 않은 토큰") from e
    await ensure_user_can_access(db, user_id)
    return user_id


async def authenticate_ws_user(ws, token: str) -> tuple[int, str | None]:
    user_id, role = await authenticate_ws_token(ws, token)
    try:
        async with AsyncSessionLocal() as db:
            await ensure_user_can_access(db, user_id)
    except HTTPException as exc:
        await ws.close(code=status.WS_1008_POLICY_VIOLATION, reason=str(exc.detail))
        raise
    return user_id, role
