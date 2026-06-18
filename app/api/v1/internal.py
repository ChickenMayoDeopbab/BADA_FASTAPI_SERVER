from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.preset_scenarios import PRESET_MAP
from app.core.security import require_internal_secret
from app.db.models import ScenarioORM
from app.deps.db import get_db
from app.schemas.scenario import ScenarioContextResponse, ScriptTurnContext

router = APIRouter(dependencies=[Depends(require_internal_secret)])

@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get(
    "/scenarios/{scenario_id}/context",
    response_model=ScenarioContextResponse,
    response_model_by_alias=True,
    summary="시나리오 컨텍스트 조회 (내부용)",
    description="Spring이 세션 생성 시 Redis에 저장할 시나리오 컨텍스트를 반환한다.",
)
async def get_scenario_context(
    scenario_id: int,
    db: AsyncSession = Depends(get_db),
) -> ScenarioContextResponse:
    """프리셋은 PRESET_MAP에서, 커스텀은 DB(ScenarioORM)에서 컨텍스트를 만든다."""
    preset = PRESET_MAP.get(scenario_id)
    if preset is not None:
        return ScenarioContextResponse(
            title=preset["title"],
            ai_role=preset["ai_role"],
            script=[
                ScriptTurnContext(
                    step=turn["step"],
                    ai_goal=turn["ai_goal"],
                    hint=turn.get("hint", ""),
                )
                for turn in preset["script"]
            ],
        )

    row = await db.get(ScenarioORM, scenario_id)
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="SCENARIO_NOT_FOUND",
        )

    # 커스텀 시나리오는 단계별 script 데이터가 없어 자유 대화로 진행된다.
    return ScenarioContextResponse(
        title=row.title,
        ai_role=row.call_target,
        script=[],
    )
