from app.core.enums import SessionType
from app.schemas.frames import EndReason
from app.schemas.training_analysis import (
    AnalysisQualityStatus,
)
from app.services.training_analysis import (
    TrainingPerformanceAnalyzer,
)
from app.services.tremor import TremorResult


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


def test_calculates_three_objective_scores() -> None:
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

    assert result.stability_score == 75.0
    assert result.conversation_score == 50.0
    assert result.fluency_score == 100.0

    assert result.user_speech_duration_ms == 6000
    assert result.user_tremor_duration_ms == 1000

    assert (
        result.user_sustained_speech_duration_ms
        == 4000
    )

    assert result.ai_speech_duration_ms == 1000
    assert result.valid_user_turn_count == 2
    assert result.analysis_exclusion_reason is None


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
