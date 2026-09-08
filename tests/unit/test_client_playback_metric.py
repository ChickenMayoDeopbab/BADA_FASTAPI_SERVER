
import asyncio
import json
import logging

import pytest

from app.services.pipeline import VoicePipeline, _State


def _make_pipeline() -> VoicePipeline:
    p = VoicePipeline.__new__(VoicePipeline)
    p._session_id = "sess-play"
    p._state = _State.LISTENING
    p._muted = False
    p._closing = asyncio.Event()
    p._ws_alive = True
    return p


def _client_playback(caplog):
    return [
        r for r in caplog.records
        if r.name == "app.metrics" and getattr(r, "metric", None) == "client_playback"
    ]


@pytest.mark.asyncio
async def test_playback_stats_frame_becomes_client_playback_metric(caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    p = _make_pipeline()
    frame = {
        "type": "playback_stats",
        "turn": 3,
        "chunks": 41,
        "gaps80": 2,
        "max_gap_ms": 213.5,
        "total_gap_ms": 380,
        "played_ms": 4120,
        "dropped_chunks": 0,
        "route": "speaker",
        "platform": "android",
        "os": "14",
    }

    await p._handle_client_text(json.dumps(frame))

    recs = _client_playback(caplog)
    assert len(recs) == 1
    rec = recs[0]
    assert rec.session_id == "sess-play"
    assert rec.turn == 3
    assert rec.chunks == 41
    assert rec.gaps80 == 2
    assert rec.max_gap_ms == 213.5
    assert rec.total_gap_ms == 380
    assert rec.played_ms == 4120
    assert rec.dropped_chunks == 0
    assert rec.route == "speaker"
    assert rec.platform == "android"
    assert rec.os == "14"


@pytest.mark.asyncio
async def test_playback_stats_drops_unknown_and_malformed_fields(caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    p = _make_pipeline()
    frame = {
        "type": "playback_stats",
        "chunks": "41",
        "gaps80": True,
        "max_gap_ms": float("nan"),
        "route": "x" * 500,
        "platform": {"nested": 1},
        "evil": "rm -rf",
        "session_id": "spoof",
        "metric": "voice_turn",
    }

    await p._handle_client_text(json.dumps(frame))

    recs = _client_playback(caplog)
    assert len(recs) == 1
    rec = recs[0]
    assert rec.session_id == "sess-play"
    assert rec.metric == "client_playback"
    for f in ("chunks", "gaps80", "max_gap_ms", "route", "platform", "evil"):
        assert not hasattr(rec, f), f


@pytest.mark.asyncio
async def test_playback_stats_does_not_change_pipeline_state(caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    p = _make_pipeline()
    p._state = _State.SPEAKING

    await p._handle_client_text(json.dumps({"type": "playback_stats", "chunks": 1}))

    assert p._state == _State.SPEAKING
    assert p._muted is False
    assert not p._closing.is_set()


@pytest.mark.asyncio
async def test_other_client_frames_still_work(caplog) -> None:
    caplog.set_level(logging.INFO, logger="app.metrics")
    p = _make_pipeline()
    await p._handle_client_text(json.dumps({"type": "mute", "muted": True}))
    assert p._muted is True
    assert _client_playback(caplog) == []
