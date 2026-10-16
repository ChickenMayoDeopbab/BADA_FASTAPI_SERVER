from __future__ import annotations

import math
from collections.abc import Mapping

from app.schemas.speech_metrics import SpeechMeasurement, SpeechMetricsPayload
from app.services.feedback_points import MIN_LEVEL_TURNS, MIN_RATE_TURNS, VOICE_WORK_MIN


def _numbers(metrics: Mapping, key: str, *, nonnegative: bool = False) -> dict:
    values = metrics.get(key) or {}
    return {
        turn: float(value) for turn, value in values.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and (not nonnegative or value >= 0)
    }


def _rate(eligible: int, affected: int, minimum: int = 1) -> SpeechMeasurement:
    if eligible < minimum:
        return SpeechMeasurement(
            status="INSUFFICIENT_DATA" if eligible else "UNAVAILABLE", eligible_turn_count=eligible
        )
    return SpeechMeasurement(status="MEASURED", eligible_turn_count=eligible, affected_turn_count=affected)


def summarize_speech_metrics(turn_metrics: Mapping | None) -> SpeechMetricsPayload:
    """선택된 피드백 구간을 사용하지 않고, 전체 측정값에서 통화별 분자와 분모를 만든다."""
    metrics = turn_metrics or {}
    avti = _numbers(metrics, "avti", nonnegative=True)
    levels = _numbers(metrics, "loudness_db")
    rates = _numbers(metrics, "syllables_per_sec", nonnegative=True)
    delays = _numbers(metrics, "answer_delay_sec", nonnegative=True)
    quiet = _numbers(metrics, "quiet_drop_db", nonnegative=True)
    fast = _numbers(metrics, "fast_rate_ratio", nonnegative=True)
    # 턴 구간만으로 발화 측정 성공 여부를 알 수 없으므로 파이프라인이 기록한 분모만 사용한다.
    pause_turns = set(metrics.get("pause_eligible_turns") or [])
    pauses = metrics.get("pause_sec") or {}
    delay = SpeechMeasurement(status="UNAVAILABLE", eligible_turn_count=0)
    if delays:
        delay = SpeechMeasurement(
            status="MEASURED", eligible_turn_count=len(delays),
            total_delay_ms=round(sum(delays.values()) * 1000),
        )
    return SpeechMetricsPayload(metrics={
        "tremor_turn_rate": _rate(len(avti), sum(value >= VOICE_WORK_MIN for value in avti.values())),
        "long_pause_turn_rate": _rate(len(pause_turns), sum(bool(pauses.get(turn)) for turn in pause_turns)),
        # 음량 원본은 저장 시 반올림되므로 재판정하지 않고 같은 통화에서 계산된 판정을 보존한다.
        "quiet_drop_turn_rate": _rate(len(levels), len(levels.keys() & quiet.keys()), MIN_LEVEL_TURNS),
        "fast_rate_turn_rate": _rate(len(rates), len(rates.keys() & fast.keys()), MIN_RATE_TURNS),
        "response_delay_ms": delay,
    })
