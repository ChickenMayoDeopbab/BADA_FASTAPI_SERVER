import json
import math
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from app.schemas.frames import EndReason
from app.schemas.speech_metrics import SpeechMeasurement
from app.services.pipeline import VoicePipeline
from app.services.speech_metrics import summarize_speech_metrics
from app.services.spring_client import SpringInternalClient


def test_all_measured_turns_are_used_independently_of_selected_feedback():
    result = summarize_speech_metrics({
        "avti": {1: 2, 2: 5, 3: 3, 4: 7, 5: 1},
        "picked": [{"turn": 2}, {"turn": 4}],
        "pause_eligible_turns": [1, 2, 3, 4, 5],
        "pause_sec": {3: [1.5, 2.0], 99: [3.0]},
        "loudness_db": {1: -20, 2: -20, 3: -26},
        "quiet_drop_db": {3: 6.01},
        "syllables_per_sec": {1: 4, 2: 4, 3: 6},
        "fast_rate_ratio": {3: 1.5},
        "answer_delay_sec": {2: 0.5, 3: 1.5},
    })
    assert result.metrics["tremor_turn_rate"].eligible_turn_count == 5
    assert result.metrics["tremor_turn_rate"].affected_turn_count == 2
    assert result.metrics["long_pause_turn_rate"].affected_turn_count == 1
    assert result.metrics["quiet_drop_turn_rate"].affected_turn_count == 1
    assert result.metrics["fast_rate_turn_rate"].affected_turn_count == 1
    assert result.metrics["response_delay_ms"].total_delay_ms == 2000
    assert result.metrics["response_delay_ms"].eligible_turn_count == 2


def test_missing_measurements_are_not_zero_and_empty_selected_feedback_is_irrelevant():
    missing = summarize_speech_metrics({"turn_sec": {1: [0, 3]}, "picked": []})
    assert all(metric.status == "UNAVAILABLE" for metric in missing.metrics.values())
    assert all(metric.affected_turn_count is None for metric in missing.metrics.values())
    measured = summarize_speech_metrics({"avti": {1: 2}, "pause_eligible_turns": [1], "pause_sec": {}, "picked": []})
    assert measured.metrics["tremor_turn_rate"].affected_turn_count == 0
    assert measured.metrics["long_pause_turn_rate"].status == "MEASURED"
    assert measured.metrics["long_pause_turn_rate"].affected_turn_count == 0


def test_relative_metrics_require_three_eligible_turns():
    result = summarize_speech_metrics({"loudness_db": {1: -20, 2: -30}, "syllables_per_sec": {1: 4, 2: 7}})
    for key in ("quiet_drop_turn_rate", "fast_rate_turn_rate"):
        assert result.metrics[key].status == "INSUFFICIENT_DATA"
        assert result.metrics[key].eligible_turn_count == 2
        assert result.metrics[key].affected_turn_count is None


def test_rounding_stored_loudness_does_not_change_the_original_classification():
    result = summarize_speech_metrics({"loudness_db": {1: -20, 2: -20, 3: -26}, "quiet_drop_db": {}})
    assert result.metrics["quiet_drop_turn_rate"].affected_turn_count == 0


def test_nonfinite_and_invalid_measurements_are_excluded():
    result = summarize_speech_metrics({
        "avti": {1: math.nan, 2: math.inf, 3: -1, 4: True, 5: 0},
        "answer_delay_sec": {1: -2, 2: math.inf, 3: 0},
    })
    assert result.metrics["tremor_turn_rate"].eligible_turn_count == 1
    assert result.metrics["response_delay_ms"].eligible_turn_count == 1
    assert result.metrics["response_delay_ms"].total_delay_ms == 0


@pytest.mark.parametrize("fields", [
    {"status": "MEASURED", "eligible_turn_count": 0, "affected_turn_count": 0},
    {"status": "MEASURED", "eligible_turn_count": 1, "affected_turn_count": 2},
    {"status": "MEASURED", "eligible_turn_count": 1},
    {"status": "MEASURED", "eligible_turn_count": 1, "affected_turn_count": 0, "total_delay_ms": 0},
    {"status": "UNAVAILABLE", "eligible_turn_count": 0, "affected_turn_count": 0},
])
def test_summary_contract_rejects_inconsistent_counts(fields):
    with pytest.raises(ValidationError):
        SpeechMeasurement(**fields)


def test_pipeline_records_pause_denominator_only_for_measurable_speech():
    pipeline = VoicePipeline.__new__(VoicePipeline)
    pipeline._user_turn_intervals = [(0, 3), (3, 6), (6, 8)]
    pipeline._long_pauses_by_turn(SimpleNamespace(voiced_spans=[(0, .5), (2, 2.5), (3, 3.1)]))
    result = summarize_speech_metrics(pipeline._turn_metrics_for_storage())
    assert result.metrics["long_pause_turn_rate"].eligible_turn_count == 1
    assert result.metrics["long_pause_turn_rate"].affected_turn_count == 1


@pytest.mark.asyncio
async def test_callback_retries_the_same_versioned_summary(monkeypatch):
    monkeypatch.setattr("app.services.spring_client._RETRY_BASE_DELAY_SECONDS", 0)
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(500 if len(calls) == 1 else 200)

    client = SpringInternalClient(SimpleNamespace(spring_boot_internal_url="http://spring", internal_secret="test"),
                                  transport=httpx.MockTransport(handler))
    summary = summarize_speech_metrics({"avti": {1: 2}})
    assert await client.notify_session_closed("session", reason=EndReason.END_CALL, transcript=[],
                                             silence_total=0, speech_metrics=summary)
    assert calls[0] == calls[1]
    assert calls[0]["speech_metrics"]["schema_version"] == 1
    assert calls[0]["speech_metrics"]["metrics"]["tremor_turn_rate"]["affected_turn_count"] == 0
