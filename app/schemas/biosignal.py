from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_BODY_BYTES = 512 * 1024
MAX_ROWS = 7200
MAX_IBI_ROWS = 15_000
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
