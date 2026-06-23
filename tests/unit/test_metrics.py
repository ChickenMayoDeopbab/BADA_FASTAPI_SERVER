import logging

import app.core.metrics as metrics
from app.core.metrics import Stopwatch, log_metric, now_ms
from app.services.pipeline import _TurnTimings


def test_now_ms_is_monotonic_nondecreasing() -> None:
    a = now_ms()
    b = now_ms()
    assert b >= a


def test_stopwatch_elapsed_nonnegative() -> None:
    sw = Stopwatch()
    assert sw.elapsed_ms >= 0


def test_stopwatch_between_marks() -> None:
    sw = Stopwatch()
    sw.mark("start")
    sw.mark("end")
    d = sw.between("start", "end")
    assert d is not None and d >= 0


def test_stopwatch_missing_mark_returns_none() -> None:
    sw = Stopwatch()
    assert sw.since("nope") is None
    assert sw.between("a", "b") is None
    assert sw.at("nope") is None


def test_log_metric_emits_record_with_fields(caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    log_metric("voice_turn", session_id="s1", response_ms=123.4)
    records = [r for r in caplog.records if r.name == "app.metrics"]
    assert len(records) == 1
    rec = records[0]
    assert rec.metric == "voice_turn"
    assert rec.session_id == "s1"
    assert rec.response_ms == 123.4
    msg = rec.getMessage()
    assert "metric=voice_turn" in msg
    assert "response_ms=123.4" in msg


def test_log_metric_swallows_logger_exceptions(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise RuntimeError("logging broke")

    monkeypatch.setattr(metrics._metric_logger, "info", boom)
    log_metric("http_request", path="/x")


def _full_timings() -> _TurnTimings:
    return _TurnTimings(
        final_at=100.0,
        last_audio_at=90.0,
        ctx_sent_at=101.0,
        first_token_at=150.0,
        last_token_at=300.0,
        first_pcm_at=200.0,
        last_pcm_at=500.0,
    )


def test_turn_timings_full_metrics() -> None:
    m = _full_timings().as_metrics()
    assert m["stt_ms"] == 10.0          # final - last_audio
    assert m["llm_ttft_ms"] == 49.0     # first_token - ctx_sent
    assert m["llm_total_ms"] == 199.0   # last_token - ctx_sent
    assert m["tts_ttfb_ms"] == 50.0     # first_pcm - first_token
    assert m["tts_total_ms"] == 350.0   # last_pcm - first_token
    assert m["response_ms"] == 100.0    # first_pcm - final
    assert m["turn_total_ms"] == 400.0  # last_pcm - final


def test_turn_timings_partial_turn_yields_none() -> None:
    m = _TurnTimings(
        final_at=100.0,
        last_audio_at=90.0,
        ctx_sent_at=101.0,
        first_token_at=150.0,
        last_token_at=160.0,
    ).as_metrics()
    assert m["llm_ttft_ms"] == 49.0
    assert m["tts_ttfb_ms"] is None
    assert m["response_ms"] is None
    assert m["turn_total_ms"] is None
