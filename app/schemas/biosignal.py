from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_BODY_BYTES = 900 * 1024
MAX_ROWS = 7200
MAX_IBI_ROWS = 15_000
MAX_PPG_CHUNKS = 200
MAX_PPG_SAMPLES = 30_000
_DAY_MS = 86_400_000

Row = tuple[float, float, int | None]


class BiosignalDevice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str | None = Field(default=None, max_length=64)
    sdk: str | None = Field(default=None, max_length=32)


class BiosignalClock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phone_ws_open_ms: int | None = None
    phone_call_end_ms: int | None = None
    watch_minus_phone_ms: float | None = Field(default=None, ge=-_DAY_MS, le=_DAY_MS)


class BiosignalMarkers(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ringing_ms: int | None = None
    meditation: tuple[int, int] | None = None
    rest_baseline: tuple[int, int] | None = None
    recovery_end_ms: int | None = None


class PpgChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    t0: float
    hz: float = Field(ge=10, le=200)
    green: list[float] = Field(max_length=MAX_PPG_SAMPLES)


class BiosignalUpload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    v: Literal[1]
    source: Literal["samsung_sensor_sdk", "health_services"]
    device: BiosignalDevice | None = None
    clock: BiosignalClock = Field(default_factory=BiosignalClock)
    markers: BiosignalMarkers = Field(default_factory=BiosignalMarkers)
    hr: list[Row] = Field(default_factory=list, max_length=MAX_ROWS)
    ibi: list[Row] = Field(default_factory=list, max_length=MAX_IBI_ROWS)
    eda: list[Row] = Field(default_factory=list, max_length=MAX_ROWS)
    skin_temp: list[Row] = Field(default_factory=list, max_length=MAX_ROWS)
    acc: list[tuple[float, float, float]] = Field(default_factory=list, max_length=MAX_ROWS)
    ppg: list[PpgChunk] = Field(default_factory=list, max_length=MAX_PPG_CHUNKS)

    @model_validator(mode="after")
    def limit_ppg_samples(self) -> "BiosignalUpload":
        if sum(len(chunk.green) for chunk in self.ppg) > MAX_PPG_SAMPLES:
            raise ValueError(f"PPG 표본은 모두 합쳐 {MAX_PPG_SAMPLES}개까지입니다.")
        return self
