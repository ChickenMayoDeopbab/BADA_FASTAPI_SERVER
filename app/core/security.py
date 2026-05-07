from fastapi import Header, HTTPException, status

from app.core.config import get_settings

async def require_internal_secret(
    x_internal_secret: str | None = Header(default=None),
) -> None:
    expected = get_settings().internal_secret
    if not x_internal_secret or x_internal_secret != expected:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN)