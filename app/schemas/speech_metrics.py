from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MetricKey = Literal[
    "tremor_turn_rate", "long_pause_turn_rate", "quiet_drop_turn_rate", "fast_rate_turn_rate", "response_delay_ms"
]


class SpeechMeasurement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["MEASURED", "INSUFFICIENT_DATA", "UNAVAILABLE"]
    eligible_turn_count: int = Field(ge=0)
    affected_turn_count: int | None = Field(default=None, ge=0)
    total_delay_ms: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_measurement(self) -> SpeechMeasurement:
        if self.status == "MEASURED":
            if self.eligible_turn_count == 0:
                raise ValueError("측정값에는 유효한 발화가 필요합니다.")
            if (self.affected_turn_count is None) == (self.total_delay_ms is None):
                raise ValueError("비율 분자 또는 대답 대기 시간 합계 중 하나가 필요합니다.")
        elif self.affected_turn_count is not None or self.total_delay_ms is not None:
            raise ValueError("측정 불가 항목에는 결과값을 저장하지 않습니다.")
        if self.affected_turn_count is not None and self.affected_turn_count > self.eligible_turn_count:
            raise ValueError("해당 발화 수는 측정 가능 발화 수를 넘을 수 없습니다.")
        return self


class SpeechMetricsPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    measurement_version: Literal["FEEDBACK_METRICS_V1"] = "FEEDBACK_METRICS_V1"
    policy_version: Literal["FEEDBACK_POLICY_V1"] = "FEEDBACK_POLICY_V1"
    metrics: dict[MetricKey, SpeechMeasurement]
