from anthropic import APIError
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.enums import ScenarioCategory
from app.deps.auth import get_current_user_id
from app.deps.db import get_db
from app.schemas.scenario import (
    CustomScenarioResponse,
    CustomSessionRequest,
    ScenarioListResponse,
)
from app.services.scenario_service import (
    create_custom_scenario as svc_create_custom_scenario,
)
from app.services.scenario_service import (
    get_scenarios as svc_get_scenarios,
)

router = APIRouter(prefix="/api/v1/scenario", tags=["scenario"])

@router.get(
    "/scenarios",
    response_model=ScenarioListResponse,
    summary="훈련 시나리오 목록 조회",
    description="카테고리, 난이도 선택 가능",
)
async def list_scenarios(
    db: AsyncSession = Depends(get_db),
    category:   ScenarioCategory | None = Query(None, description="카테고리 선택"),
) -> ScenarioListResponse:
    return await svc_get_scenarios(db, category)

@router.post(
    "/custom",
    response_model=CustomScenarioResponse,
    status_code=status.HTTP_201_CREATED,
    summary="커스텀 시나리오 생성",
    description=(
        "전화 목적·상대·성격·난이도를 입력하면 AI가 맞춤 시나리오를 생성해 저장합니다. "
        "실제 훈련 세션 생성은 Spring(POST /api/v1/session)이 담당합니다."
    ),
)
async def create_custom_scenario(
        body: CustomSessionRequest,
        db: AsyncSession = Depends(get_db),
        user_id: int = Depends(get_current_user_id),
) -> CustomScenarioResponse:
    try:
        response = await svc_create_custom_scenario(db, body, user_id)
    except APIError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI 시나리오 생성 실패: {e.message}",
        ) from e
    except (ValueError, KeyError) as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"응답 파싱 오류: {str(e)}",
        ) from e
    return response
