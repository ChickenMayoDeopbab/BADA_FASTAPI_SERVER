from anthropic import APIError
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from websockets.exceptions import WebSocketException

from app.core.enums import ScenarioCategory
from app.deps.auth import get_current_user_id
from app.deps.db import get_db
from app.schemas.scenario import (
    CustomScenarioResponse,
    CustomSessionRequest,
    ExampleConversationResponse,
    ScenarioListResponse,
)
from app.services.example_service import (
    ScenarioNotFoundError,
)
from app.services.example_service import (
    get_example_conversation as svc_get_example_conversation,
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
    description="카테고리, 난이도 선택 가능. 커스텀 시나리오는 본인이 만든 것만 노출된다.",
)
async def list_scenarios(
    db: AsyncSession = Depends(get_db),
    category:   ScenarioCategory | None = Query(None, description="카테고리 선택"),
    user_id: int = Depends(get_current_user_id),
) -> ScenarioListResponse:
    return await svc_get_scenarios(db, category, user_id)

@router.get(
    "/{scenario_id}/example",
    response_model=ExampleConversationResponse,
    summary="시나리오 예시 대화 들어보기",
    description=(
        "시나리오별 모범 예시 대화(대본 및 두 화자 TTS 오디오 URL)를 반환합니다. "
        "오디오는 최초 요청 시 생성돼 캐시되며, 커스텀 시나리오는 본인 것만 조회할 수 있습니다."
    ),
)
async def get_example_conversation(
        scenario_id: int,
        db: AsyncSession = Depends(get_db),
        user_id: int = Depends(get_current_user_id),
) -> ExampleConversationResponse:
    try:
        return await svc_get_example_conversation(db, scenario_id, user_id)
    except ScenarioNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="시나리오를 찾을 수 없습니다.",
        ) from e
    except APIError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI 예시 대화 생성 실패: {e.message}",
        ) from e
    except WebSocketException as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"TTS 오디오 생성 실패: {e}",
        ) from e
    except (ValueError, KeyError) as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"응답 파싱 오류: {str(e)}",
        ) from e


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
