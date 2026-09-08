
import importlib.util
import sys
from pathlib import Path

import pytest

import app.core.audio_stats as audio_stats

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ws_listen.py"


@pytest.fixture(scope="module")
def ws_listen():
    spec = importlib.util.spec_from_file_location("ws_listen_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_recorder_uses_shared_starvation(ws_listen) -> None:
    assert ws_listen.starvation is audio_stats.starvation


def test_recorder_as_heard_inserts_silence_per_shared_algorithm(ws_listen) -> None:
    rec = ws_listen.Recorder(jitter_ms=0.0, gap_threshold_ms=80.0)
    chunk = b"\x01\x02" * 1600
    rec.add(chunk, at=0.0)
    rec.add(chunk, at=0.25)
    rec.add(chunk, at=0.30)

    heard, gaps = rec.as_heard()

    silence = int(0.150 * 16000) * 2
    assert len(heard) == 3 * len(chunk) + silence
    assert heard[len(chunk): len(chunk) + silence] == b"\x00" * silence
    assert [(round(at, 3), round(dur, 3)) for at, dur in gaps] == [(0.25, 0.15)]


def test_recorder_add_without_at_uses_wall_clock(ws_listen) -> None:
    rec = ws_listen.Recorder(jitter_ms=0.0, gap_threshold_ms=80.0)
    rec.add(b"\x00\x00")
    rec.add(b"\x00\x00")
    assert not rec.empty
    assert len(rec.raw()) == 4


def test_default_gap_threshold_matches_operational_definition(ws_listen) -> None:
    parser = ws_listen.build_parser()
    args = parser.parse_args(["ws", "--audio", "x.wav"])
    assert args.gap_ms == audio_stats.GAP_THRESHOLD_MS
