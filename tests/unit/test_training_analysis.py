import pytest
from pydantic import ValidationError

from app.core.enums import SessionType
from app.schemas.frames import EndReason
from app.schemas.training_analysis import (
    AnalysisQualityStatus,
    TrainingAnalysisPayload,
)
from app.services.training_analysis import (
    TrainingPerformanceAnalyzer,
)
from app.services.tremor import TremorResult


def test_rejects_failed_analysis_with_objective_scores() -> None:
    with pytest.raises(
        ValidationError,
        match="FAIL 분석에는 객관 점수가 없어야 합니다.",
    ):
        TrainingAnalysisPayload(
            stability_score=75.0,
            conversation_score=75.0,
            fluency_score=75.0,
            user_speech_duration_ms=1000,
            ai_speech_duration_ms=1000,
            server_wait_duration_ms=500,
            valid_user_turn_count=1,
            user_tremor_duration_ms=0,
            user_sustained_speech_duration_ms=0,
            completed_script_steps=0,
            script_step_count=4,
            analysis_quality_status=AnalysisQualityStatus.FAIL,
            analysis_exclusion_reason="INSUFFICIENT_USER_SPEECH",
            analyzer_version="SPEECH_ANALYZER_V1",
            analysis_policy_version="ANALYSIS_POLICY_V1",
        )


def _valid_tremor_result() -> TremorResult:
    return TremorResult(
        shake_count=2,
        episodes=[
            (1.0, 1.5, 5.0, 12.0),
            (6.0, 6.5, 5.2, 11.0),
        ],
        voiced_sec=6.0,
        sustained_sec=4.0,
        voiced_spans=[
            (0.5, 3.5),
            (5.5, 8.5),
        ],
        sustained_spans=[
            (1.0, 3.0),
            (6.0, 8.0),
        ],
    )


def _sentence_ratio_tremor_result() -> TremorResult:
    return TremorResult(
        shake_count=1,
        episodes=[
            (1.0, 1.5, 5.0, 12.0),
        ],
        voiced_sec=11.0,
        sustained_sec=7.5,
        voiced_spans=[
            (0.5, 3.5),
            (7.0, 9.0),
            (10.5, 13.5),
            (15.5, 18.5),
        ],
        sustained_spans=[
            (1.0, 3.0),
            (7.0, 8.5),
            (11.0, 13.0),
            (16.0, 18.0),
        ],
    )


def _clean_tremor_result() -> TremorResult:
    return TremorResult(
        shake_count=0,
        episodes=[],
        voiced_sec=6.0,
        sustained_sec=4.0,
        voiced_spans=[
            (0.5, 3.5),
            (5.5, 8.5),
        ],
        sustained_spans=[
            (1.0, 3.0),
            (6.0, 8.0),
        ],
    )


def test_calculates_scores_from_clean_utterance_ratios() -> None:
    analyzer = TrainingPerformanceAnalyzer()

    result = analyzer.analyze(
        session_type=SessionType.SCENARIO,
        reason=EndReason.SCENARIO_DONE,
        user_turn_intervals=[
            (0.0, 4.0),
            (5.0, 9.0),
            (10.0, 14.0),
            (15.0, 19.0),
        ],
        user_turn_texts=[
            "예약하고 싶어요",
            "예약 가능한가요",
            "음 날짜를 확인할게요",
            "네 감사합니다",
        ],
        tremor_result=_sentence_ratio_tremor_result(),
        completed_script_steps=0,
        script_step_count=0,
        ai_pcm_bytes=32000,
        server_wait_duration_ms=1200,
    )

    assert (
        result.analysis_quality_status
        is AnalysisQualityStatus.PASS
    )

    # 4개 발화 중 떨림 발화 1개
    assert result.stability_score == 75.0

    # 4개 발화 중 1.5초 이상 침묵 발화 1개
    assert result.conversation_score == 75.0

    # 4개 발화 중 필러가 있는 발화 1개
    assert result.fluency_score == 75.0

    assert result.user_speech_duration_ms == 11000
    assert result.user_tremor_duration_ms == 500

    assert (
        result.user_sustained_speech_duration_ms
        == 7500
    )

    assert result.valid_user_turn_count == 4
    assert result.analyzer_version == "SPEECH_ANALYZER_V2"
    assert (
        result.analysis_policy_version
        == "ANALYSIS_POLICY_V2"
    )


def test_does_not_count_repetition_across_turn_boundaries() -> None:
    analyzer = TrainingPerformanceAnalyzer()

    result = analyzer.analyze(
        session_type=SessionType.SCENARIO,
        reason=EndReason.SCENARIO_DONE,
        user_turn_intervals=[
            (0.0, 4.0),
            (5.0, 9.0),
        ],
        user_turn_texts=[
            "네 감사합니다",
            "감사합니다 다음에 연락드릴게요",
        ],
        tremor_result=_valid_tremor_result(),
        completed_script_steps=2,
        script_step_count=4,
        ai_pcm_bytes=32000,
        server_wait_duration_ms=1200,
    )

    assert (
        result.analysis_quality_status
        is AnalysisQualityStatus.PASS
    )
    assert result.fluency_score == 100.0


def test_rejects_insufficient_valid_turns() -> None:
    analyzer = TrainingPerformanceAnalyzer()

    result = analyzer.analyze(
        session_type=SessionType.SCENARIO,
        reason=EndReason.SCENARIO_DONE,
        user_turn_intervals=[
            (0.0, 4.0),
        ],
        user_turn_texts=[
            "예약하고 싶어요",
        ],
        tremor_result=_valid_tremor_result(),
        completed_script_steps=1,
        script_step_count=4,
        ai_pcm_bytes=32000,
        server_wait_duration_ms=1000,
    )

    assert (
        result.analysis_quality_status
        is AnalysisQualityStatus.FAIL
    )

    assert (
        result.analysis_exclusion_reason
        == "INSUFFICIENT_VALID_TURNS"
    )

    assert result.stability_score is None
    assert result.conversation_score is None
    assert result.fluency_score is None


def test_rejects_user_aborted_training() -> None:
    analyzer = TrainingPerformanceAnalyzer()

    result = analyzer.analyze(
        session_type=SessionType.SCENARIO,
        reason=EndReason.USER_END,
        user_turn_intervals=[
            (0.0, 4.0),
            (5.0, 9.0),
        ],
        user_turn_texts=[
            "예약하고 싶어요",
            "네 감사합니다",
        ],
        tremor_result=_valid_tremor_result(),
        completed_script_steps=2,
        script_step_count=4,
        ai_pcm_bytes=32000,
        server_wait_duration_ms=1000,
    )

    assert (
        result.analysis_quality_status
        is AnalysisQualityStatus.FAIL
    )

    assert (
        result.analysis_exclusion_reason
        == "INCOMPLETE_TRAINING"
    )


def test_rejects_missing_sustained_speech() -> None:
    analyzer = TrainingPerformanceAnalyzer()

    tremor_result = TremorResult(
        shake_count=0,
        episodes=[],
        voiced_sec=6.0,
        voiced_spans=[
            (0.5, 3.5),
            (5.5, 8.5),
        ],
        sustained_spans=[],
    )

    result = analyzer.analyze(
        session_type=SessionType.SCENARIO,
        reason=EndReason.SCENARIO_DONE,
        user_turn_intervals=[
            (0.0, 4.0),
            (5.0, 9.0),
        ],
        user_turn_texts=[
            "예약하고 싶어요",
            "네 감사합니다",
        ],
        tremor_result=tremor_result,
        completed_script_steps=2,
        script_step_count=4,
        ai_pcm_bytes=32000,
        server_wait_duration_ms=1000,
    )

    assert (
        result.analysis_quality_status
        is AnalysisQualityStatus.FAIL
    )

    assert (
        result.analysis_exclusion_reason
        == "INSUFFICIENT_SUSTAINED_SPEECH"
    )


def test_counts_repetition_inside_a_turn_as_stutter() -> None:
    analyzer = TrainingPerformanceAnalyzer()

    result = analyzer.analyze(
        session_type=SessionType.SCENARIO,
        reason=EndReason.SCENARIO_DONE,
        user_turn_intervals=[
            (0.0, 4.0),
            (5.0, 9.0),
        ],
        user_turn_texts=[
            "예약 예약 하고 싶어요",
            "네 감사합니다",
        ],
        tremor_result=_clean_tremor_result(),
        completed_script_steps=0,
        script_step_count=0,
        ai_pcm_bytes=0,
        server_wait_duration_ms=0,
    )

    assert (
        result.analysis_quality_status
        is AnalysisQualityStatus.PASS
    )
    assert result.fluency_score == 50.0


def test_does_not_require_a_script_for_v2_scores() -> None:
    analyzer = TrainingPerformanceAnalyzer()

    result = analyzer.analyze(
        session_type=SessionType.CUSTOM,
        reason=EndReason.END_CALL,
        user_turn_intervals=[
            (0.0, 4.0),
            (5.0, 9.0),
        ],
        user_turn_texts=[
            "예약하고 싶어요",
            "네 감사합니다",
        ],
        tremor_result=_clean_tremor_result(),
        completed_script_steps=0,
        script_step_count=0,
        ai_pcm_bytes=0,
        server_wait_duration_ms=0,
    )

    assert (
        result.analysis_quality_status
        is AnalysisQualityStatus.PASS
    )
    assert result.stability_score == 100.0
    assert result.conversation_score == 100.0
    assert result.fluency_score == 100.0


@pytest.mark.parametrize(
    ("first_voice_start", "expected_score"),
    [
        (1.49, 100.0),
        (1.5, 50.0),
    ],
)
def test_long_pause_boundary(
    first_voice_start: float,
    expected_score: float,
) -> None:
    analyzer = TrainingPerformanceAnalyzer()

    tremor_result = TremorResult(
        shake_count=0,
        episodes=[],
        voiced_sec=5.5,
        sustained_sec=3.5,
        voiced_spans=[
            (first_voice_start, 4.0),
            (5.5, 8.5),
        ],
        sustained_spans=[
            (first_voice_start, 3.0),
            (6.0, 8.0),
        ],
    )

    result = analyzer.analyze(
        session_type=SessionType.SCENARIO,
        reason=EndReason.SCENARIO_DONE,
        user_turn_intervals=[
            (0.0, 4.0),
            (5.0, 9.0),
        ],
        user_turn_texts=[
            "예약하고 싶어요",
            "네 감사합니다",
        ],
        tremor_result=tremor_result,
        completed_script_steps=0,
        script_step_count=0,
        ai_pcm_bytes=0,
        server_wait_duration_ms=0,
    )

    assert (
        result.analysis_quality_status
        is AnalysisQualityStatus.PASS
    )
    assert result.conversation_score == expected_score


def test_missing_audio_result_is_null_safe() -> None:
    analyzer = TrainingPerformanceAnalyzer()

    result = analyzer.analyze(
        session_type=SessionType.SCENARIO,
        reason=EndReason.SCENARIO_DONE,
        user_turn_intervals=[
            (0.0, 4.0),
            (5.0, 9.0),
        ],
        user_turn_texts=[
            "예약하고 싶어요",
            "네 감사합니다",
        ],
        tremor_result=None,
        completed_script_steps=0,
        script_step_count=0,
        ai_pcm_bytes=0,
        server_wait_duration_ms=0,
    )

    assert (
        result.analysis_quality_status
        is AnalysisQualityStatus.FAIL
    )
    assert (
        result.analysis_exclusion_reason
        == "MISSING_OR_UNREADABLE_AUDIO"
    )
    assert result.stability_score is None
    assert result.conversation_score is None
    assert result.fluency_score is None


def test_analyzer_error_is_null_safe() -> None:
    analyzer = TrainingPerformanceAnalyzer()

    result = analyzer.analyze(
        session_type=SessionType.SCENARIO,
        reason=EndReason.SCENARIO_DONE,
        user_turn_intervals=[],
        user_turn_texts=[],
        tremor_result=None,
        completed_script_steps=0,
        script_step_count=0,
        ai_pcm_bytes=0,
        server_wait_duration_ms=0,
        analyzer_failed=True,
    )

    assert (
        result.analysis_quality_status
        is AnalysisQualityStatus.FAIL
    )
    assert result.analysis_exclusion_reason == "ANALYZER_ERROR"
