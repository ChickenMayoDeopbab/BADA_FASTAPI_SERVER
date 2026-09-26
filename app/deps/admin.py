from fastapi import Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import is_admin
from app.db.external import users_table
from app.deps.auth import get_current_user_id
from app.deps.db import get_db


async def require_admin_user_id(
    user_id: int = Depends(get_current_user_id), db: AsyncSession = Depends(get_db)
) -> int:
    role = await db.scalar(select(users_table.c.role).where(users_table.c.user_id == user_id))
    if not is_admin(role):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="ADMIN_REQUIRED")
    return user_id
