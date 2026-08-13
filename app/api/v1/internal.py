from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.preset_scenarios import PRESET_MAP
from app.core.security import require_internal_secret
from app.db.models import ScenarioORM, is_deleted
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
    row = await db.get(ScenarioORM, scenario_id)
    if row is not None and is_deleted(row):
        row = None
    if row is not None and getattr(row, "is_custom", False):
        return _custom_context(row)

    preset = PRESET_MAP.get(scenario_id)
    if preset is not None:
        return ScenarioContextResponse(
            title=preset["title"],
            ai_role=preset["ai_role"],
            ai_prompt=preset["ai_prompt"],
            script=[
                ScriptTurnContext(
                    step=turn["step"],
                    ai_goal=turn["ai_goal"],
                    hint=turn.get("hint", ""),
                )
                for turn in preset["script"]
            ],
            tts_voice_id=preset["tts_voice_id"],
        )

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="SCENARIO_NOT_FOUND",
        )

    return _custom_context(row)


def _custom_context(row: ScenarioORM) -> ScenarioContextResponse:
    # 커스텀 시나리오 생성은 AI가 만든 script 사용(없으면 자유 대화)
    raw_script = row.script if isinstance(row.script, list) else []
    script = [
        ScriptTurnContext(
            step=int(turn["step"]),
            ai_goal=str(turn["ai_goal"]),
            hint=str(turn.get("hint", "")),
        )
        for turn in raw_script
        if isinstance(turn, dict) and "step" in turn and "ai_goal" in turn
    ]
    return ScenarioContextResponse(
        title=row.title,
        ai_role=row.call_target,
        ai_prompt=getattr(row, "ai_prompt", ""),
        script=script,
        tts_voice_id=getattr(row, "tts_voice_id", None),
    )
