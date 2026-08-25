from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from typing import TypeAlias

from app.core.enums import SessionType
from app.schemas.frames import EndReason
from app.schemas.training_analysis import (
    AnalysisQualityStatus,
    TrainingAnalysisPayload,
)
from app.services.tremor import TremorResult

ANALYZER_VERSION = "SPEECH_ANALYZER_V1"
ANALYSIS_POLICY_VERSION = "ANALYSIS_POLICY_V1"

_SAMPLE_RATE = 16000
_SAMPLE_BYTES = 2
_PCM_BYTES_PER_SECOND = _SAMPLE_RATE * _SAMPLE_BYTES

_MIN_VALID_TURN_SPEECH_SECONDS = 0.5
_MIN_VALID_USER_TURNS = 2
_MIN_TOTAL_USER_SPEECH_SECONDS = 3.0
_MIN_SUSTAINED_SPEECH_SECONDS = 1.2
_LONG_PAUSE_SECONDS = 1.5

_SUPPORTED_SESSION_TYPES = {
    SessionType.SCENARIO.value,
    SessionType.CUSTOM.value,
}

_VALID_END_REASONS = {
    EndReason.SCENARIO_DONE.value,
    EndReason.END_CALL.value,
    EndReason.TIMEOUT.value,
}

_TOKEN_PATTERN = re.compile(r"[0-9A-Za-z가-힣]+")
_FILLER_WORDS = {
    "음",
    "어",
    "저",
    "그",
    "아",
    "음음",
    "어어",
    "뭐지",
    "그러니까",
}

Span: TypeAlias = tuple[float, float]  # noqa: UP040 - Python 3.11 compatibility


def _enum_value(value: object | None) -> str | None:
    if value is None:
        return None

    raw = getattr(value, "value", value)
    return str(raw).strip().upper()


def _normalize_spans(
    spans: Iterable[Sequence[float]],
) -> list[Span]:
    cleaned: list[Span] = []

    for span in spans:
        if len(span) < 2:
            continue

        try:
            start = float(span[0])
            end = float(span[1])
        except (TypeError, ValueError):
            continue

        if not math.isfinite(start) or not math.isfinite(end):
            continue

        start = max(0.0, start)

        if end <= start:
            continue

        cleaned.append((start, end))

    cleaned.sort()

    merged: list[Span] = []

    for start, end in cleaned:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
            continue

        previous_start, previous_end = merged[-1]
        merged[-1] = (
            previous_start,
            max(previous_end, end),
        )

    return merged


def _intersect_spans(
    left: Iterable[Sequence[float]],
    right: Iterable[Sequence[float]],
) -> list[Span]:
    left_spans = _normalize_spans(left)
    right_spans = _normalize_spans(right)

    intersections: list[Span] = []
    left_index = 0
    right_index = 0

    while (
        left_index < len(left_spans)
        and right_index < len(right_spans)
    ):
        left_start, left_end = left_spans[left_index]
        right_start, right_end = right_spans[right_index]

        start = max(left_start, right_start)
        end = min(left_end, right_end)

        if end > start:
            intersections.append((start, end))

        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1

    return _normalize_spans(intersections)


def _duration_seconds(spans: Iterable[Sequence[float]]) -> float:
    return sum(
        end - start
        for start, end in _normalize_spans(spans)
    )


def _to_milliseconds(seconds: float) -> int:
    return int(round(max(0.0, seconds) * 1000))


def _clamp_score(value: float) -> float:
    return round(min(max(value, 0.0), 100.0), 2)


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in _TOKEN_PATTERN.findall(text)
    ]


def _long_pause_duration(
    user_turns: list[Span],
    voiced_spans: list[Span],
) -> float:
    total = 0.0

    for turn_start, turn_end in user_turns:
        turn_voice = _intersect_spans(
            [(turn_start, turn_end)],
            voiced_spans,
        )

        cursor = turn_start

        for voice_start, voice_end in turn_voice:
            gap = voice_start - cursor

            if gap >= _LONG_PAUSE_SECONDS:
                total += gap

            cursor = max(cursor, voice_end)

        tail_gap = turn_end - cursor

        if tail_gap >= _LONG_PAUSE_SECONDS:
            total += tail_gap

    return total


class TrainingPerformanceAnalyzer:
    def analyze(
        self,
        *,
        session_type: SessionType | str | None,
        reason: EndReason | str,
        user_turn_intervals: list[tuple[float, float]],
        user_turn_texts: list[str],
        tremor_result: TremorResult | None,
        completed_script_steps: int,
        script_step_count: int,
        ai_pcm_bytes: int,
        server_wait_duration_ms: int,
        analyzer_failed: bool = False,
    ) -> TrainingAnalysisPayload:
        voice_activity_spans = _normalize_spans(
            tremor_result.voiced_spans
            if tremor_result is not None
            else []
        )

        sustained_spans = _normalize_spans(
            tremor_result.sustained_spans
            if tremor_result is not None
            else []
        )

        tremor_spans = _normalize_spans(
            [
                episode[:2]
                for episode in (
                    tremor_result.episodes
                    if tremor_result is not None
                    else []
                )
                if len(episode) >= 2
            ]
        )

        turn_data_mismatch = (
            len(user_turn_intervals)
            != len(user_turn_texts)
        )

        valid_turns: list[Span] = []
        valid_texts: list[str] = []

        for interval, text in zip(
            user_turn_intervals,
            user_turn_texts,
            strict=False,
        ):
            normalized = _normalize_spans([interval])

            if not normalized:
                continue

            turn = normalized[0]
            spoken_duration = _duration_seconds(
                _intersect_spans([turn], voice_activity_spans)
            )

            if spoken_duration < _MIN_VALID_TURN_SPEECH_SECONDS:
                continue

            if not _tokens(text):
                continue

            valid_turns.append(turn)
            valid_texts.append(text.strip())

        valid_voice_spans = _intersect_spans(
            valid_turns,
            voice_activity_spans,
        )

        valid_sustained_spans = _intersect_spans(
            valid_turns,
            sustained_spans,
        )

        valid_tremor_spans = _intersect_spans(
            valid_turns,
            tremor_spans,
        )

        valid_tremor_spans = _intersect_spans(
            valid_tremor_spans,
            valid_sustained_spans,
        )

        user_speech_seconds = _duration_seconds(
            valid_voice_spans
        )

        sustained_speech_seconds = _duration_seconds(
            valid_sustained_spans
        )

        tremor_seconds = _duration_seconds(
            valid_tremor_spans
        )

        user_speech_duration_ms = _to_milliseconds(
            user_speech_seconds
        )

        sustained_speech_duration_ms = _to_milliseconds(
            sustained_speech_seconds
        )

        tremor_duration_ms = _to_milliseconds(
            tremor_seconds
        )

        ai_speech_duration_ms = _to_milliseconds(
            max(0, ai_pcm_bytes) / _PCM_BYTES_PER_SECOND
        )

        server_wait_duration_ms = max(
            0,
            int(round(server_wait_duration_ms)),
        )

        script_step_count = max(
            0,
            int(script_step_count),
        )

        completed_script_steps = max(
            0,
            int(completed_script_steps),
        )

        if script_step_count > 0:
            completed_script_steps = min(
                completed_script_steps,
                script_step_count,
            )

        base_fields = {
            "user_speech_duration_ms":
                user_speech_duration_ms,
            "ai_speech_duration_ms":
                ai_speech_duration_ms,
            "server_wait_duration_ms":
                server_wait_duration_ms,
            "valid_user_turn_count":
                len(valid_turns),
            "user_tremor_duration_ms":
                tremor_duration_ms,
            "user_sustained_speech_duration_ms":
                sustained_speech_duration_ms,
            "completed_script_steps":
                completed_script_steps,
            "script_step_count":
                script_step_count,
            "analyzer_version":
                ANALYZER_VERSION,
            "analysis_policy_version":
                ANALYSIS_POLICY_VERSION,
        }

        def failed(reason_code: str) -> TrainingAnalysisPayload:
            return TrainingAnalysisPayload(
                stability_score=None,
                conversation_score=None,
                fluency_score=None,
                analysis_quality_status=
                    AnalysisQualityStatus.FAIL,
                analysis_exclusion_reason=reason_code,
                **base_fields,
            )

        normalized_session_type = _enum_value(session_type)
        normalized_end_reason = _enum_value(reason)

        if normalized_session_type not in _SUPPORTED_SESSION_TYPES:
            return failed("UNSUPPORTED_SESSION_TYPE")

        if normalized_end_reason not in _VALID_END_REASONS:
            return failed("INCOMPLETE_TRAINING")

        if turn_data_mismatch:
            return failed("TURN_DATA_MISMATCH")

        if analyzer_failed:
            return failed("ANALYZER_ERROR")

        if tremor_result is None:
            return failed("MISSING_OR_UNREADABLE_AUDIO")

        if script_step_count <= 0:
            return failed("MISSING_SCRIPT")

        if len(valid_turns) < _MIN_VALID_USER_TURNS:
            return failed("INSUFFICIENT_VALID_TURNS")

        if (
            user_speech_seconds
            < _MIN_TOTAL_USER_SPEECH_SECONDS
        ):
            return failed("INSUFFICIENT_USER_SPEECH")

        if (
            sustained_speech_seconds
            < _MIN_SUSTAINED_SPEECH_SECONDS
        ):
            return failed("INSUFFICIENT_SUSTAINED_SPEECH")

        tremor_ratio = min(
            tremor_seconds / sustained_speech_seconds,
            1.0,
        )

        stability_score = _clamp_score(
            100.0 * (1.0 - tremor_ratio)
        )

        conversation_score = _clamp_score(
            100.0
            * completed_script_steps
            / script_step_count
        )

        user_turn_duration = _duration_seconds(valid_turns)
        long_pause_duration = _long_pause_duration(
            valid_turns,
            valid_voice_spans,
        )

        pause_ratio = (
            min(long_pause_duration / user_turn_duration, 1.0)
            if user_turn_duration > 0
            else 1.0
        )

        pause_score = 100.0 * (1.0 - pause_ratio)

        all_tokens: list[str] = []

        for text in valid_texts:
            all_tokens.extend(_tokens(text))

        filler_count = sum(
            token in _FILLER_WORDS
            for token in all_tokens
        )

        repeated_count = sum(
            previous == current
            for previous, current in zip(
                all_tokens,
                all_tokens[1:],
                strict=False,
            )
        )

        lexical_disfluency_ratio = (
            min(
                2.0
                * (filler_count + repeated_count)
                / len(all_tokens),
                1.0,
            )
            if all_tokens
            else 1.0
        )

        lexical_score = (
            100.0
            * (1.0 - lexical_disfluency_ratio)
        )

        fluency_score = _clamp_score(
            pause_score * 0.7
            + lexical_score * 0.3
        )

        return TrainingAnalysisPayload(
            stability_score=stability_score,
            conversation_score=conversation_score,
            fluency_score=fluency_score,
            analysis_quality_status=
                AnalysisQualityStatus.PASS,
            analysis_exclusion_reason=None,
            **base_fields,
        )
