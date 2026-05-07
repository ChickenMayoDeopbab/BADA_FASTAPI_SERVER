from fastapi import APIRouter, Depends

from app.core.security import require_internal_secret

router = APIRouter(dependencies=[Depends(require_internal_secret)])

@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}