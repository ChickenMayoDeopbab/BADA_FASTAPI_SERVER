
import asyncio

from anthropic import APIError
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
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
    ScenarioRecommendationResponse,
)
from app.services.example_service import (
    ScenarioNotFoundError,
    bake_example_audio,
)
from app.services.example_service import (
    get_example_conversation as svc_get_example_conversation,
)
from app.services.scenario_image_service import generate_scenario_thumbnail
from app.services.scenario_service import (
    ScenarioConfigError,
    ScenarioGenInvalidError,
    ScenarioRefusedError,
)
from app.services.scenario_service import (
    create_custom_scenario as svc_create_custom_scenario,
)
from app.services.scenario_service import (
    delete_custom_scenario as svc_delete_custom_scenario,
)
from app.services.scenario_service import (
    get_recommended_scenario as svc_get_recommended_scenario,
)
from app.services.scenario_service import (
    get_scenarios as svc_get_scenarios,
)

router = APIRouter(prefix="/api/v1/scenario", tags=["scenario"])

# 왜 막혔는지 사용자가 알 수 있어야 한다. 서버 결함이 아니므로 4xx.
_REFUSAL_MESSAGES = {
    "ILLEGAL": "불법 행위를 연습하는 시나리오는 만들 수 없습니다. 통화 목적을 다시 적어주세요.",
    "HARMFUL": "다른 사람에게 해가 될 수 있는 시나리오는 만들 수 없습니다. 통화 목적을 다시 적어주세요.",
    "SEXUAL": "선정적인 내용의 시나리오는 만들 수 없습니다. 통화 목적을 다시 적어주세요.",
    "INJECTION": "시나리오 설정에 넣을 수 없는 지시가 포함돼 있습니다. 통화 목적을 다시 적어주세요.",
}

@router.get(
    "/scenarios",
    response_model=ScenarioListResponse,
    summary="훈련 시나리오 목록 조회",
    description=("업무·일상·학교·기타 카테고리로 조회, 커스텀 시나리오는 본인이 만든 것만 노출"
),)
async def list_scenarios(
    db: AsyncSession = Depends(get_db),
    category: ScenarioCategory | None = Query(
        None,
        description="work, daily, school, other 중 하나",
    ),
    user_id: int = Depends(get_current_user_id),
) -> ScenarioListResponse:
    return await svc_get_scenarios(db, category, user_id)


@router.get(
    "/recommendation",
    response_model=ScenarioRecommendationResponse,
    summary="오늘의 훈련 시나리오 추천",
    description=(
        "본인의 활성 시나리오 중 미연습 커스텀, 미연습 시나리오, "
        "가장 오래전에 연습한 시나리오 순으로 하나를 추천합니다."
    ),
)
async def recommend_scenario(
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
) -> ScenarioRecommendationResponse:
    recommendation = await svc_get_recommended_scenario(db, user_id)
    if recommendation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="추천할 시나리오를 찾을 수 없습니다.",
        )
    return recommendation


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
        "카테고리·전화 목적·상대·성격·난이도를 입력하면 AI가 맞춤 시나리오를 생성해 저장합니다. "
        "실제 훈련 세션 생성은 Spring(POST /api/v1/session)이 담당합니다."
    ),
)
async def create_custom_scenario(
        body: CustomSessionRequest,
        background: BackgroundTasks,
        db: AsyncSession = Depends(get_db),
        user_id: int = Depends(get_current_user_id),
) -> CustomScenarioResponse:
    try:
        response = await svc_create_custom_scenario(db, body, user_id)
    except ScenarioRefusedError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=_REFUSAL_MESSAGES.get(e.code, _REFUSAL_MESSAGES["INJECTION"]),
        ) from e
    except ScenarioConfigError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI 시나리오 생성이 일시적으로 불가합니다. 잠시 후 다시 시도해 주세요.",
        ) from e
    except APIError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"AI 시나리오 생성 실패: {e.message}",
        ) from e
    except ScenarioGenInvalidError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="AI가 올바른 시나리오를 만들지 못했습니다. 다시 시도해 주세요.",
        ) from e
    except (ValueError, KeyError) as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"응답 파싱 오류: {str(e)}",
        ) from e

    background.add_task(_post_create_tasks, response.scenario.scenario_id)
    return response


async def _post_create_tasks(scenario_id: int) -> None:
    """시나리오 생성 후 작업들을 동시에 실행"""
    await asyncio.gather(
        generate_scenario_thumbnail(scenario_id),
        bake_example_audio(scenario_id),
        return_exceptions=True,
    )


@router.delete(
    "/custom/{scenario_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="커스텀 시나리오 삭제",
    description=(
        "본인이 만든 커스텀 시나리오를 삭제합니다. "
        "기본 시나리오·타인의 시나리오·이미 삭제된 시나리오는 404를 반환합니다."
    ),
)
async def delete_custom_scenario(
        scenario_id: int,
        db: AsyncSession = Depends(get_db),
        user_id: int = Depends(get_current_user_id),
) -> None:
    deleted = await svc_delete_custom_scenario(db, scenario_id, user_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="시나리오를 찾을 수 없습니다.",
        )
