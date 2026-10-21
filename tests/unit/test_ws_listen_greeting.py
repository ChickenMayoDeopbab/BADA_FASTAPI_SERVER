import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ws_listen.py"
_SILENCE = b"\x00" * 640


@pytest.fixture(scope="module")
def ws_listen():
    spec = importlib.util.spec_from_file_location("ws_listen_greeting_under_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class _ServerThatGreetsOnAudio:
    """첫 오디오를 받아야 인사를 흘려보내는 서버 흉내"""

    def __init__(self, frames: list) -> None:
        self.sent: list = []
        self._frames: asyncio.Queue = asyncio.Queue()
        for frame in frames:
            self._frames.put_nowait(frame)

    async def send(self, data) -> None:
        self.sent.append(data)

    async def recv(self):
        while not self.sent:
            await asyncio.sleep(0.001)
        return await self._frames.get()


def _args() -> SimpleNamespace:
    return SimpleNamespace(chunk_ms=20, turn_timeout=5.0)


async def test_greeting_is_woken_by_silence_and_heard_before_turn_one(ws_listen, monkeypatch) -> None:
    monkeypatch.setattr(ws_listen, "_GREETING_RESUME_GRACE_S", 0.0)
    pcm = b"\x01\x02" * 160
    ws = _ServerThatGreetsOnAudio([
        json.dumps({"type": "emotion", "emotion": "NEUTRAL"}),
        pcm,
        json.dumps({"type": "transcript", "role": "ai", "text": "네, 여보세요?"}),
        json.dumps({"type": "speaking_end", "turn_id": 1}),
    ])
    rec = ws_listen.Recorder(jitter_ms=0.0, gap_threshold_ms=80.0)
    assert await ws_listen._run_ws_greeting(_args(), rec, ws, 640) is True
    assert ws.sent and all(chunk == _SILENCE for chunk in ws.sent)
    assert not rec.empty
    sent = len(ws.sent)
    await asyncio.sleep(0.05)
    assert len(ws.sent) == sent, "인사가 시작되면 앱처럼 마이크를 막는다"


@pytest.mark.parametrize("ftype", ["end", "error"])
async def test_greeting_stops_the_call_on_end_or_error(ws_listen, ftype) -> None:
    ws = _ServerThatGreetsOnAudio([json.dumps({"type": ftype})])
    rec = ws_listen.Recorder(jitter_ms=0.0, gap_threshold_ms=80.0)
    assert await ws_listen._run_ws_greeting(_args(), rec, ws, 640) is False
