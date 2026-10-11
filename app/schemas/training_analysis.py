from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnalysisQualityStatus(StrEnum):
    PASS = "PASS"
    PARTIAL = "PARTIAL"
    FAIL = "FAIL"


class TrainingAnalysisPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stability_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )
    conversation_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )
    fluency_score: float | None = Field(
        default=None,
        ge=0,
        le=100,
    )

    user_speech_duration_ms: int = Field(ge=0)
    ai_speech_duration_ms: int = Field(ge=0)
    server_wait_duration_ms: int = Field(ge=0)

    valid_user_turn_count: int = Field(ge=0)

    user_tremor_duration_ms: int = Field(ge=0)
    user_sustained_speech_duration_ms: int = Field(ge=0)

    completed_script_steps: int = Field(ge=0)
    script_step_count: int = Field(ge=0)

    analysis_quality_status: AnalysisQualityStatus
    analysis_exclusion_reason: str | None = None

    analyzer_version: str
    analysis_policy_version: str
    metric_quality: dict[str, AnalysisQualityStatus] = Field(default_factory=dict)
    unanswered_user_turn_count: int = Field(default=0, ge=0)
    unrecognized_user_turn_count: int = Field(default=0, ge=0)
    leading_silence_duration_ms: int = Field(default=0, ge=0)
    trailing_silence_duration_ms: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_quality_result(self) -> TrainingAnalysisPayload:
        scores = (
            self.stability_score,
            self.conversation_score,
            self.fluency_score,
        )

        if self.analysis_quality_status is AnalysisQualityStatus.PASS:
            if any(score is None for score in scores):
                raise ValueError(
                    "PASS 분석에는 객관 점수 3개가 모두 필요합니다."
                )

            if self.analysis_exclusion_reason is not None:
                raise ValueError(
                    "PASS 분석에는 제외 사유가 없어야 합니다."
                )

        if self.analysis_quality_status is AnalysisQualityStatus.PARTIAL and (
            all(score is None for score in scores) or all(score is not None for score in scores)
        ):
            raise ValueError("PARTIAL 분석에는 유효한 지표와 평가 불가 지표가 모두 필요합니다.")

        for name, score in zip(("stability", "conversation", "fluency"), scores, strict=True):
            quality = self.metric_quality.get(name)
            if quality is AnalysisQualityStatus.PARTIAL:
                raise ValueError("개별 지표의 품질은 PASS 또는 FAIL이어야 합니다.")
            if quality is not None and (quality is AnalysisQualityStatus.PASS) != (score is not None):
                raise ValueError("지표 품질 상태와 점수 유무가 일치해야 합니다.")

        if (
            self.analysis_quality_status is AnalysisQualityStatus.FAIL
            and any(score is not None for score in scores)
        ):
            raise ValueError(
                "FAIL 분석에는 객관 점수가 없어야 합니다."
            )

        if (
            self.analysis_quality_status is AnalysisQualityStatus.FAIL
            and not self.analysis_exclusion_reason
        ):
            raise ValueError(
                "FAIL 분석에는 제외 사유가 필요합니다."
            )

        return self
