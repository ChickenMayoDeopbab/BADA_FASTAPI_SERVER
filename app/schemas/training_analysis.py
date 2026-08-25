from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AnalysisQualityStatus(StrEnum):
    PASS = "PASS"
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
