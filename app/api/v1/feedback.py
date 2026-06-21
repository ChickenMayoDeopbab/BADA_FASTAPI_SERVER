from fastapi import APIRouter

router = APIRouter(prefix="/api/v1/feedback", tags=["feedback"])

# @router.get(
#     "/feedback",
#             response_model=FeedbackResponse,
#             status_code=200,
#             summary="피드백 조회 api",
#             description="떨림 횟수 + 침묵구간 + 잘한점을 조회합니다.",
# )
