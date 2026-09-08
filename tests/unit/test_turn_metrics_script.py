
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "turn_metrics.py"


@pytest.fixture(scope="module")
def tm():
    spec = importlib.util.spec_from_file_location("turn_metrics_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_parses_text_and_json_lines(tm, tmp_path) -> None:
    text = ("2026-09-07 19:15:24.526 [INFO] app.metrics: metric=voice_turn session_id=s1 step=1 "
            "error=False watchdog=False fallback=True tts_engine=qwen tts_failed=True tts_ttfb_ms=None "
            "pcm_chunks=0 gap0_count=0 arrival_rtf=None\n")
    prod = json.dumps({
        "message": "metric=voice_turn session_id=s2 ...", "metric": "voice_turn", "session_id": "s2",
        "step": 2, "error": False, "watchdog": False, "fallback": False, "tts_engine": "eleven",
        "tts_failed": False, "tts_ttfb_ms": 257.3, "pcm_chunks": 3, "gap0_count": 0, "arrival_rtf": 0.02,
    }) + "\n"
    noise = "2026-09-07 19:15:15.092 [INFO] app.services.pipeline: STT FINAL 수신 state=LISTENING\n"
    log = tmp_path / "x.log"
    log.write_text(text + prod + noise, encoding="utf-8")

    rows = tm.parse([str(log)])
    turns = [r for r in rows if r["metric"] == "voice_turn"]
    assert [t["session_id"] for t in turns] == ["s1", "s2"]
    assert turns[0]["tts_failed"] is True and turns[0]["fallback"] is True
    assert turns[0]["tts_ttfb_ms"] is None and turns[0]["pcm_chunks"] == 0
    assert turns[1]["tts_ttfb_ms"] == 257.3 and turns[1]["pcm_chunks"] == 3
    assert turns[1]["arrival_rtf"] == 0.02 and turns[1]["tts_failed"] is False


def test_wilson_interval(tm) -> None:
    p, lo, hi = tm.wilson(2, 19)
    assert (round(p, 3), round(lo, 3), round(hi, 3)) == (0.105, 0.029, 0.314)
    assert tm.wilson(0, 0) == (0.0, 0.0, 0.0)
    p, lo, hi = tm.wilson(0, 5)
    assert lo == 0.0 and 0.43 < hi < 0.44


def test_percentile_interpolates(tm) -> None:
    assert tm.pct([1, 2, 3, 4], 0.5) == 2.5
    assert tm.pct([None, 10], 0.9) == 10
    assert tm.pct([], 0.5) is None
