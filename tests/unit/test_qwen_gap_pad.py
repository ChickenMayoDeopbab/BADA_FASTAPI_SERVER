import json
import math

import httpx
import numpy as np
import pytest

from app.services import qwen_tts as qwen_mod
from app.services.qwen_tts import NS_HEADER, QwenRealtimeTTSClient, QwenTTSUnavailableError
from tests.unit.test_qwen_realtime_pause import _Clock, _settings, _turn, _Worker, chunked, ms, pcm, speech
from tests.unit.test_qwen_realtime_tts import _real_settings

TWO = ("안녕하세요. ", "반갑습니다.")
BODY = pcm(np.concatenate([speech(400), np.zeros(ms(50))]))
LEAD30 = pcm(np.concatenate([np.zeros(ms(30)), speech(400), np.zeros(ms(50))]))
LONG = pcm(np.concatenate([speech(400), np.zeros(ms(300))]))
TAIL150 = pcm(np.concatenate([speech(400), np.zeros(ms(150))]))
LOUD_END = pcm(speech(400))
LEAD80 = pcm(np.concatenate([np.zeros(ms(80)), speech(400), np.zeros(ms(50))]))
LEAD300 = pcm(np.concatenate([np.zeros(ms(300)), speech(400), np.zeros(ms(50))]))
LONG_SPEECH = pcm(np.concatenate([speech(3000), np.zeros(ms(50))]))
TAIL600 = pcm(np.concatenate([speech(400), np.zeros(ms(600))]))
HUM = np.random.default_rng(0).normal(0.0, 5e-5, ms(80))  # 조용함 문턱보다 훨씬 작은 잡음
NOISY_LEAD80 = pcm(np.concatenate([HUM, speech(400), np.zeros(ms(50))]))


class _SeqWorker:
    """요청마다 다음 body 를 흘리는 가짜 워커, fail_at 번째 요청은 다 흘린 뒤 끊김"""

    def __init__(
        self, *bodies: bytes, header: str | None = None, fail_at: int | None = None, first: int = 4480,
        rest: int = 5120,
    ) -> None:
        self.bodies = list(bodies)
        self.header = header
        self.fail_at = fail_at
        self.first = first
        self.rest = rest
        self.requests: list[dict] = []

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(json.loads(request.read()))
            body = self.bodies[min(len(self.requests), len(self.bodies)) - 1]
            fail = len(self.requests) == self.fail_at

            async def gen():
                for c in chunked(body, first=self.first, rest=self.rest):
                    yield c
                if fail:
                    raise httpx.ReadError("worker gone")

            headers = {} if self.header is None else {NS_HEADER: self.header}
            return httpx.Response(200, content=gen(), headers=headers)

        return httpx.MockTransport(handler)


def _session(worker, *, gap_ms: float = 250.0, speed: float = 1.0, voice: str = "female_young_soft", **kw):
    client = httpx.AsyncClient(transport=worker.transport())
    return qwen_mod.QwenRealtimeTTSSession(
        client, "http://tts.test", speed, voice, gap_ms=gap_ms, clock=kw.pop("clock", None) or _Clock(), **kw
    )


def _inner_quiet_ms(out: bytes) -> list[float]:
    """출력 안쪽 조용한 구간 길이 — 2ms 블록, 전체 최대값 대비 -45dB 아래, 앞뒤 끝 구간은 뺌"""
    x = np.frombuffer(out, dtype="<i2")
    n = len(x) // 32
    blocks = np.abs(x[: n * 32].astype(np.int32)).reshape(n, 32).max(1)
    quiet = blocks < blocks.max() * 10 ** (-45 / 20)
    runs, i = [], 0
    while i < n:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < n and quiet[j]:
            j += 1
        if i > 0 and j < n:
            runs.append((j - i) * 2.0)
        i = j
    return runs


async def test_short_pause_between_requests_is_filled_to_the_floor() -> None:
    session = _session(_SeqWorker(BODY, BODY))
    out = await _turn(session, TWO)
    assert _inner_quiet_ms(out) == [pytest.approx(250, abs=2)]
    assert len(out) == 2 * len(BODY) + ms(200) * 2


async def test_lead_silence_of_the_next_request_counts() -> None:
    session = _session(_SeqWorker(BODY, LEAD30))
    out = await _turn(session, TWO)
    assert _inner_quiet_ms(out) == [pytest.approx(250, abs=2)]
    assert session.request_log[1]["gap_pad_ms"] == pytest.approx(170, abs=2)


async def test_quieter_next_request_still_measures_its_lead() -> None:
    quieter = pcm(np.concatenate([np.zeros(ms(30)), speech(400, db=-30.0), np.zeros(ms(50))]))
    session = _session(_SeqWorker(BODY, quieter))
    out = await _turn(session, TWO)
    assert session.request_log[1]["gap_pad_ms"] == pytest.approx(170, abs=2)
    assert _inner_quiet_ms(out) == [pytest.approx(250, abs=2)]


async def test_onset_after_the_first_chunk_still_counts_its_whole_lead() -> None:
    session = _session(_SeqWorker(BODY, LEAD80, first=ms(20) * 2))
    out = await _turn(session, TWO)
    assert session.request_log[1]["gap_pad_ms"] == pytest.approx(120, abs=2)
    assert _inner_quiet_ms(out) == [pytest.approx(250, abs=2)]


@pytest.mark.parametrize(("first", "rest"), [(650, 5010), (641, 5001), (30, 46), (31, 47)])
async def test_unaligned_odd_or_tiny_chunks_keep_the_floor(first, rest) -> None:
    session = _session(_SeqWorker(BODY, NOISY_LEAD80, first=first, rest=rest))
    out = await _turn(session, TWO)
    assert session.request_log[1]["gap_pad_ms"] == pytest.approx(120, abs=2)
    assert _inner_quiet_ms(out) == [pytest.approx(250, abs=2)]
    onset = ms(80) * 2  # 첫 소리 블록 시작(2ms 블록 경계)
    assert out == BODY + NOISY_LEAD80[:onset] + bytes(ms(120) * 2) + NOISY_LEAD80[onset:]


@pytest.mark.parametrize(("first", "rest"), [(4480, 5120), (31, 47)])
async def test_request_without_sound_gives_back_held_bytes(first, rest) -> None:
    quiet = pcm(np.zeros(ms(100) + 13))  # 2ms 블록으로 안 나눠떨어지는 끝
    session = _session(_SeqWorker(BODY, quiet, first=first, rest=rest))
    out = await _turn(session, TWO)
    assert out == BODY + quiet
    assert "gap_pad_ms" not in session.request_log[1]


async def test_lead_over_several_chunks_that_passes_the_floor_needs_no_pad() -> None:
    session = _session(_SeqWorker(BODY, LEAD300, first=ms(20) * 2))
    out = await _turn(session, TWO)
    assert session.request_log[1]["gap_pad_ms"] == 0
    assert out == BODY + LEAD300


async def test_tail_is_tracked_in_a_short_window_only() -> None:
    session = _session(_SeqWorker(LONG_SPEECH, BODY))
    out = await _turn(session, TWO)
    assert session.request_log[1]["gap_pad_ms"] == pytest.approx(200, abs=2)
    assert _inner_quiet_ms(out) == [pytest.approx(250, abs=2)]
    assert session._gap._tail.maxlen == math.ceil(250 / 2) + 1


async def test_trailing_silence_longer_than_the_window_needs_no_pad() -> None:
    session = _session(_SeqWorker(TAIL600, BODY))
    out = await _turn(session, TWO)
    assert session.request_log[1]["gap_pad_ms"] == 0
    assert out == TAIL600 + BODY


async def test_failed_request_still_sets_the_tail_for_the_next_one() -> None:
    session = _session(_SeqWorker(BODY, LOUD_END, BODY, fail_at=2))
    await session.begin()

    async def source(*texts):
        for text in texts:
            yield text

    try:
        with pytest.raises(QwenTTSUnavailableError):
            async for _ in session.stream(source(*TWO)):
                pass
        out = b"".join([c async for c in session.stream(source("다시 말할게요."))])
    finally:
        await session.aclose()
    assert session.request_log[1]["outcome"] == "error"
    assert session.request_log[2]["gap_pad_ms"] == pytest.approx(250, abs=2)
    assert out[: ms(250) * 2] == bytes(ms(250) * 2)


async def test_already_long_pause_is_left_alone() -> None:
    session = _session(_SeqWorker(LONG, LONG))
    out = await _turn(session, TWO)
    assert out == LONG + LONG
    assert session.request_log[1]["gap_pad_ms"] == 0


async def test_first_request_of_a_turn_gets_no_pad() -> None:
    session = _session(_SeqWorker(BODY))
    out = await _turn(session, ("안녕하세요.",))
    assert out == BODY
    assert "gap_pad_ms" not in session.request_log[0]


async def test_floor_is_in_heard_time_after_speed() -> None:
    session = _session(_SeqWorker(TAIL150, TAIL150), speed=1.1)
    out = await _turn(session, TWO)
    assert _inner_quiet_ms(out) == [pytest.approx(250, abs=3)]


async def test_turn_gap_metric_counts_the_pad() -> None:
    session = _session(_SeqWorker(BODY, BODY))
    await _turn(session, TWO)
    assert session.request_log[1]["gap_pad_ms"] == pytest.approx(200, abs=2)
    assert session.turn_metrics()["tts_gap_max_ms"] == pytest.approx(250, abs=4)


async def test_floor_applies_after_e_compression() -> None:
    worker = _Worker(clock=_Clock())
    session = _session(
        worker, voice="ai", ns="want", call=qwen_mod._CallState(), pause=qwen_mod._PauseConfig(), clock=worker.clock
    )
    out = await _turn(session, TWO)
    assert [r["e"] for r in session.request_log] == ["on", "on"]
    assert _inner_quiet_ms(out) == [pytest.approx(250, abs=4)]
    assert session.request_log[1]["gap_pad_ms"] == pytest.approx(70, abs=4)


async def test_listed_voice_through_client_gets_the_floor() -> None:
    worker = _SeqWorker(BODY, BODY)
    settings = _settings(False, 1.0, qwen_tts_gap_voices="female_young_soft, female_young_rough")
    client = QwenRealtimeTTSClient(settings, transport=worker.transport())
    out = await _turn(await client.open("female_young_rough"), TWO)
    assert _inner_quiet_ms(out) == [pytest.approx(250, abs=2)]


@pytest.mark.parametrize(
    ("voices", "voice", "floor"),
    [
        ("female_young_soft", "ai", 250.0),
        ("", "female_young_soft", 250.0),
        ("female_young_soft", "female_young_soft", 0.0),
    ],
)
async def test_unlisted_voice_or_zero_floor_passes_bytes_through(voices, voice, floor) -> None:
    worker = _SeqWorker(BODY, BODY)
    settings = _settings(False, 1.0, qwen_tts_gap_voices=voices, qwen_tts_gap_min_ms=floor)
    client = QwenRealtimeTTSClient(settings, transport=worker.transport())
    session = await client.open(voice)
    out = await _turn(session, TWO)
    assert out == BODY + BODY
    assert all("gap_pad_ms" not in r for r in session.request_log)


def test_gap_voices_default_empty_and_parse() -> None:
    assert QwenRealtimeTTSClient(_settings())._gap_voices == frozenset()
    client = QwenRealtimeTTSClient(_settings(qwen_tts_gap_voices=" female_young_soft , ,female_young_rough "))
    assert client._gap_voices == frozenset({"female_young_soft", "female_young_rough"})


@pytest.mark.parametrize("bad", ["abc", -5, float("nan"), None])
def test_unusable_floor_falls_back_to_default(bad) -> None:
    assert QwenRealtimeTTSClient(_settings(qwen_tts_gap_min_ms=bad))._gap_ms == 250.0


def test_env_reaches_realtime_client(monkeypatch) -> None:
    for name in ("QWEN_TTS_GAP_VOICES", "QWEN_TTS_GAP_MIN_MS"):
        monkeypatch.delenv(name, raising=False)
    default = QwenRealtimeTTSClient(_real_settings())
    assert default._gap_voices == frozenset() and default._gap_ms == 250.0
    monkeypatch.setenv("QWEN_TTS_GAP_VOICES", "female_young_soft,female_young_rough")
    monkeypatch.setenv("QWEN_TTS_GAP_MIN_MS", "300")
    client = QwenRealtimeTTSClient(_real_settings())
    assert client._gap_voices == frozenset({"female_young_soft", "female_young_rough"})
    assert client._gap_ms == 300.0
