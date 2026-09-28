from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.preset_scenarios import PRESET_MAP
from app.core.security import require_internal_secret
from app.db.models import ScenarioORM, is_deleted
from app.deps.db import get_db
from app.schemas.scenario import ScenarioContextResponse, ScriptTurnContext
from app.services.feedback_service import delete_feedback as svc_delete_feedback
from app.services.training_data_cleanup import delete_training_data as svc_delete_training_data
from app.services.user_data_cleanup import delete_user_data as svc_delete_user_data

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


@router.delete(
    "/feedback/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="세션 피드백 삭제",
    description=(
        "Spring이 훈련 기록을 삭제할 때 해당 세션의 feedback 행을 삭제. "
    ),
)
async def delete_feedback(
    session_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    await svc_delete_feedback(db, session_id)


@router.delete(
    "/training-data/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="세션 훈련 데이터 삭제",
    description=(
        "Spring이 훈련 기록을 삭제하기 전에 해당 세션의 피드백, "
        "음성 떨림 지표, 커뮤니티 첨부 및 음성 변조본을 삭제한다."
    ),
)
async def delete_training_data(
    session_id: str,
    db: AsyncSession = Depends(get_db),
) -> None:
    await svc_delete_training_data(db, session_id)


@router.delete(
    "/users/{user_id}/data",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="회원 탈퇴 데이터 삭제",
    description=(
        "Spring의 회원 행 삭제 전에 FastAPI 소유의 회원 데이터와 파일을 삭제한다. "
        "신고·차단 기록과 사용량·세션 기록은 보유정책에 따라 유지한다."
    ),
)
async def delete_user_data(
    user_id: int,
    db: AsyncSession = Depends(get_db),
) -> None:
    await svc_delete_user_data(db, user_id)


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
