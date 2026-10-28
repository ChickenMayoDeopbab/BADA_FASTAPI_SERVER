import asyncio
import json
import random
from types import SimpleNamespace

import pytest

from app.services import example_service as es
from app.services.qwen_tts import QwenTTSUnavailableError, QwenUnreachableError
from tests.unit.test_example_jobs import (
    _DB,
    _EL,
    _bake,
    _clean,  # noqa: F401
    _Qwen,
    _row,
    _Storage,
    _wire,
)

_TURNS = [
    {"speaker": "ai", "text": "네, 바다집입니다."},
    {"speaker": "user", "text": "월세 날짜 \"조정\" 문의요 {중괄호}."},
    {"speaker": "ai", "text": "네, 말씀하세요 \\ 확인할게요."},
    {"speaker": "user", "text": "다음 달 5일로요."},
]


def _text(turns=_TURNS, *, fence: bool = True, tail: str = "") -> str:
    body = json.dumps(turns, ensure_ascii=False, indent=1)
    return (f"```json\n{body}\n```" if fence else body) + tail


def _chunks(text: str, seed: int = 0) -> list[str]:
    rng = random.Random(seed)
    out, i = [], 0
    while i < len(text):
        n = rng.randint(1, 9)
        out.append(text[i : i + n])
        i += n
    return out


class _FakeStream:
    def __init__(self, owner, chunks, *, stop_reason="end_turn", fail_after=None, stall_after=None, delay=0.0):
        self.owner = owner
        self.chunks = chunks
        self.stop_reason = stop_reason
        self.fail_after = fail_after
        self.stall_after = stall_after
        self.delay = delay
        self.current_message_snapshot = SimpleNamespace(usage=SimpleNamespace(input_tokens=10, output_tokens=3))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a) -> bool:
        return False

    @property
    def text_stream(self):
        async def _gen():
            for i, chunk in enumerate(self.chunks):
                if self.fail_after is not None and i >= self.fail_after:
                    raise ConnectionError("스트림 끊김")
                if self.stall_after is not None and i >= self.stall_after:
                    await asyncio.sleep(10)
                if self.delay:
                    await asyncio.sleep(self.delay)
                yield chunk
            self.owner.stream_ended_at = asyncio.get_running_loop().time()

        return _gen()

    async def get_final_message(self):
        return SimpleNamespace(stop_reason=self.stop_reason, usage=SimpleNamespace(input_tokens=10, output_tokens=40))


class _FakeLLM:
    def __init__(self, text: str, *, retry_turns=None, create_delay: float = 0.0, **stream_kw) -> None:
        self.text = text
        self.retry_turns = retry_turns or _TURNS
        self.create_delay = create_delay
        self.stream_kw = stream_kw
        self.streams = 0
        self.creates: list[dict] = []
        self.create_kwargs: list[dict] = []
        self.stream_ended_at = None
        self.qwen = None

    def __call__(self, api_key: str):
        owner = self

        class _Messages:
            def stream(self, **_kw):
                owner.streams += 1
                return _FakeStream(owner, _chunks(owner.text), **owner.stream_kw)

            async def create(self, **kw):
                owner.creates.append({"active_connects": getattr(owner.qwen, "active", 0)})
                owner.create_kwargs.append(kw)
                if owner.create_delay:
                    await asyncio.sleep(owner.create_delay)
                payload = json.dumps(owner.retry_turns, ensure_ascii=False)
                return SimpleNamespace(content=[SimpleNamespace(text=payload)], usage=None)

        return SimpleNamespace(messages=_Messages())


class _TimedQwen(_Qwen):
    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.first_call_at = None
        self.active = 0

    def connect(self, exclude=()):
        outer = super().connect(exclude)
        qwen = self

        class _Ctx:
            async def __aenter__(self):
                qwen.active += 1
                return await outer.__aenter__()

            async def __aexit__(self, *a):
                qwen.active -= 1
                return await outer.__aexit__(*a)

        return _Ctx()

    async def synth_ns(self, voice, text):
        if self.first_call_at is None:
            self.first_call_at = asyncio.get_running_loop().time()
        return await super().synth_ns(voice, text)


def _llm_usage(monkeypatch) -> list[dict]:
    rows: list[dict] = []
    monkeypatch.setattr(es, "log_llm_usage", lambda *a, **kw: rows.append(kw))
    return rows


def _wire_stream(monkeypatch, llm: _FakeLLM, *, qwen=None, el=None, rows=None):
    rows = rows if rows is not None else {42: _row(42, custom=True)}
    qwen = qwen or _TimedQwen()
    llm.qwen = qwen
    env = _wire(monkeypatch, rows, qwen=qwen, el=el)
    monkeypatch.setattr(es, "AsyncAnthropic", llm)
    return env, rows


async def _until(predicate, timeout: float = 5.0) -> None:
    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(_poll(), timeout)


def test_parser_is_chunking_invariant_and_handles_strings() -> None:
    text = _text(tail="\n")
    expected = json.loads(json.dumps(_TURNS, ensure_ascii=False))
    for seed in range(200):
        parser = es._ArrayStreamParser()
        got = [el for chunk in _chunks(text, seed) for el in parser.feed(chunk)]
        assert got == expected and parser.closed


def test_parser_skips_a_malformed_element() -> None:
    parser = es._ArrayStreamParser()
    got = parser.feed('[{"speaker": "ai", "text": "네"}, {"speaker": }, {"speaker": "user", "text": "예"}]')
    assert got == [{"speaker": "ai", "text": "네"}, {"speaker": "user", "text": "예"}]


async def test_synthesis_starts_before_the_stream_ends(monkeypatch) -> None:
    llm = _FakeLLM(_text(), delay=0.003)
    env, rows = _wire_stream(monkeypatch, llm)
    usage = _llm_usage(monkeypatch)
    await _bake(42, 5)
    assert env.qwen.first_call_at < llm.stream_ended_at
    assert rows[42].example_dialogue == _TURNS and rows[42].example_audio_url.endswith("-qn.wav")
    assert rows[42].example_audio_url.split("-")[1] == es._ns_d8(_TURNS)
    assert llm.creates == [] and [u.get("ok", True) for u in usage] == [True]


@pytest.mark.parametrize(
    ("kw", "text"), [({"stop_reason": "max_tokens"}, _text()), ({}, _text(fence=False, tail=" 끝"))]
)
async def test_failed_end_check_retries_without_streaming(monkeypatch, kw, text) -> None:
    retry = [{"speaker": "ai", "text": "다시 만든 대본."}, {"speaker": "user", "text": "네."}]
    llm = _FakeLLM(text, retry_turns=retry, **kw)
    env, rows = _wire_stream(monkeypatch, llm)
    usage = _llm_usage(monkeypatch)
    await _bake(42, 5)
    assert llm.streams == 1 and len(llm.creates) == 1
    assert rows[42].example_dialogue == retry
    assert rows[42].example_audio_url.split("-")[1] == es._ns_d8(retry)
    assert usage[-1].get("attempt") == 2


async def test_consumer_failure_falls_back_with_the_committed_dialogue(monkeypatch) -> None:
    first = _TURNS[0]["text"]
    qwen = _TimedQwen(script={first: [QwenTTSUnavailableError("HTTP 500")]})
    llm = _FakeLLM(_text(), delay=0.002)
    env, rows = _wire_stream(monkeypatch, llm, qwen=qwen)
    await _bake(42, 5)
    assert llm.streams == 1 and llm.creates == []
    assert rows[42].example_dialogue == _TURNS
    plain = es._plain_keys(env.settings, rows[42], 42, _TURNS)
    assert rows[42].example_audio_url == plain.eleven and env.el.opens == len(_TURNS)
    assert not es._cooling(42)


async def test_unreachable_worker_in_stream_rebakes_on_another_worker(monkeypatch) -> None:
    first = _TURNS[0]["text"]
    qwen = _TimedQwen(script={first: [QwenUnreachableError("x", worker="http://w1.test")]})
    llm = _FakeLLM(_text(), delay=0.002)
    env, rows = _wire_stream(monkeypatch, llm, qwen=qwen)
    await _bake(42, 5)
    assert rows[42].example_dialogue == _TURNS and rows[42].example_audio_url.endswith("-qn.wav")
    assert env.qwen.connects == [set(), {"http://w1.test"}] and env.el.opens == 0


async def test_unexpected_consumer_error_keeps_the_dialogue(monkeypatch) -> None:
    first = _TURNS[0]["text"]
    qwen = _TimedQwen(script={first: [RuntimeError("뜻밖")]})
    llm = _FakeLLM(_text(), delay=0.002)
    env, rows = _wire_stream(monkeypatch, llm, qwen=qwen)
    usage = _llm_usage(monkeypatch)
    await _bake(42, 5)
    assert rows[42].example_dialogue == _TURNS and llm.stream_ended_at is not None
    assert rows[42].example_audio_url == es._plain_keys(env.settings, rows[42], 42, _TURNS).eleven
    assert [u.get("ok", True) for u in usage] == [True]


async def test_request_after_commit_waits_for_the_last_turns(monkeypatch) -> None:
    llm = _FakeLLM(_text())
    qwen = _TimedQwen(delay=0.05)
    env, rows = _wire_stream(monkeypatch, llm, qwen=qwen)
    env.settings.example_request_wait_sec = 5.0
    baker = asyncio.ensure_future(_bake(42, 5))
    await _until(lambda: rows[42].example_dialogue is not None and es._loop_jobs()[42].turns_done >= 2)
    resp = await asyncio.wait_for(es.get_example_conversation(_DB(rows), 42, user_id=7), 3)
    assert resp.audio_url and resp.audio_url.endswith("-qn.wav")
    wait = [kw for name, kw in env.metrics if name == "example_wait"][0]
    assert wait["early_return"] is False and wait["served"] is True
    await baker


_LONG = [{"speaker": "ai" if i % 2 == 0 else "user", "text": f"긴 대본 {i}번째 줄입니다."} for i in range(12)]
_RETRY = [{"speaker": "ai", "text": "다시 만든 대본."}, {"speaker": "user", "text": "네."}]


async def test_failed_end_check_stops_baking_discarded_turns_at_once(monkeypatch) -> None:
    llm = _FakeLLM(_text(_LONG), retry_turns=_RETRY, stop_reason="max_tokens")
    qwen = _TimedQwen(delay=0.05)
    env, rows = _wire_stream(monkeypatch, llm, qwen=qwen)
    await _bake(42, 5)
    old = {t["text"] for t in _LONG}
    assert sum(text in old for _, text in qwen.calls) <= 1
    assert rows[42].example_dialogue == _RETRY and rows[42].example_audio_url.endswith("-qn.wav")


async def test_queued_consumer_never_bakes_discarded_turns(monkeypatch) -> None:
    llm = _FakeLLM(_text(_LONG), retry_turns=_RETRY, stop_reason="max_tokens")
    qwen = _TimedQwen()
    env, rows = _wire_stream(monkeypatch, llm, qwen=qwen)

    async def _hold():
        async with es._prebake_semaphore():
            while not llm.creates:
                await asyncio.sleep(0.005)

    holder = asyncio.ensure_future(_hold())
    await asyncio.sleep(0)
    await _bake(42, 5)
    await holder
    old = {t["text"] for t in _LONG}
    assert not any(text in old for _, text in qwen.calls)
    assert rows[42].example_dialogue == _RETRY and len(qwen.connects) == 1


async def test_retry_call_has_a_time_limit(monkeypatch) -> None:
    llm = _FakeLLM(_text(), stop_reason="max_tokens", create_delay=1.0)
    env, rows = _wire_stream(monkeypatch, llm)
    monkeypatch.setattr(es, "_LLM_TIMEOUT_SEC", 0.1)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    with pytest.raises(TimeoutError):
        await _bake(42, 5)
    assert loop.time() - t0 < 0.8
    assert llm.create_kwargs[0]["timeout"].read == 0.1 and llm.create_kwargs[0]["timeout"].connect == 5.0
    assert rows[42].example_audio_url is None


async def test_broken_stream_releases_the_worker_before_the_retry(monkeypatch) -> None:
    llm = _FakeLLM(_text(), fail_after=40, delay=0.002)
    env, rows = _wire_stream(monkeypatch, llm)
    usage = _llm_usage(monkeypatch)
    await _bake(42, 5)
    assert llm.creates == [{"active_connects": 0}]
    assert rows[42].example_dialogue == _TURNS and rows[42].example_audio_url.endswith("-qn.wav")
    assert [u.get("ok", True) for u in usage] == [False, True] and usage[-1]["attempt"] == 2


async def test_consumer_budget_expiry_still_commits_the_dialogue(monkeypatch) -> None:
    llm = _FakeLLM(_text())
    env, rows = _wire_stream(monkeypatch, llm, qwen=_TimedQwen(delay=0.2))
    monkeypatch.setattr(es, "_CUSTOM_NS_BUDGET_SEC", 0.05)
    await _bake(42, 5)
    assert rows[42].example_dialogue == _TURNS
    assert "-qn" not in rows[42].example_audio_url and env.el.opens == len(_TURNS)


async def test_stalled_stream_hits_the_llm_limit_and_retries(monkeypatch) -> None:
    llm = _FakeLLM(_text(), stall_after=30)
    env, rows = _wire_stream(monkeypatch, llm)
    monkeypatch.setattr(es, "_LLM_TIMEOUT_SEC", 0.1)
    await _bake(42, 5)
    assert len(llm.creates) == 1 and rows[42].example_audio_url.endswith("-qn.wav")


async def test_streamed_result_equals_the_non_streaming_result(monkeypatch) -> None:
    llm = _FakeLLM(_text())
    env_a, rows_a = _wire_stream(monkeypatch, llm)
    await _bake(42, 5)

    rows_b = {42: _row(42, custom=True, dialogue=list(_TURNS))}
    env_b = _wire(monkeypatch, rows_b, qwen=_Qwen())
    await _bake(42, 5)

    key_a, key_b = rows_a[42].example_audio_url, rows_b[42].example_audio_url
    assert key_a == key_b and env_a.storage.objects[key_a] == env_b.storage.objects[key_b]


async def test_request_during_streaming_returns_at_once(monkeypatch) -> None:
    llm = _FakeLLM(_text(), delay=0.01)
    env, rows = _wire_stream(monkeypatch, llm)
    resp = await asyncio.wait_for(es.get_example_conversation(_DB(rows), 42, user_id=7), 2)
    assert resp.audio_url is None
    wait = [kw for name, kw in env.metrics if name == "example_wait"][0]
    assert wait["early_return"] is True and wait["waited_ms"] < 100
    job = es._loop_jobs()[42]
    assert await asyncio.wait_for(asyncio.shield(job.task), 5)


def test_fakes_are_used() -> None:
    assert _EL and _Storage


async def test_streamed_turns_are_normalized_like_the_batch_parser(monkeypatch) -> None:
    raw = [{"speaker": "AI ", "text": "  네, 바다집입니다. "}, {"speaker": "user", "text": "예약이요."}, 7,
           {"speaker": "narrator", "text": "무시"}, {"speaker": "ai", "text": "네."}]
    llm = _FakeLLM(json.dumps(raw, ensure_ascii=False))
    env, rows = _wire_stream(monkeypatch, llm)
    await _bake(42, 5)
    assert llm.creates == []
    assert rows[42].example_dialogue == [
        {"speaker": "ai", "text": "네, 바다집입니다."},
        {"speaker": "user", "text": "예약이요."},
        {"speaker": "ai", "text": "네."},
    ]


async def test_request_while_baking_but_dialogue_still_streaming_returns_at_once(monkeypatch) -> None:
    llm = _FakeLLM(_text(), delay=0.02)
    env, rows = _wire_stream(monkeypatch, llm)
    baker = asyncio.ensure_future(_bake(42, 5))
    await _until(lambda: es._loop_jobs().get(42) is not None and es._loop_jobs()[42].phase == "baking")
    assert not es._loop_jobs()[42].dialogue.done()
    resp = await asyncio.wait_for(es.get_example_conversation(_DB(rows), 42, user_id=7), 2)
    assert resp.audio_url is None
    assert [kw for name, kw in env.metrics if name == "example_wait"][0]["early_return"] is True
    await asyncio.wait_for(baker, 5)


async def test_failure_mid_bake_stops_after_the_turn_in_progress(monkeypatch) -> None:
    llm = _FakeLLM(_text(_LONG), retry_turns=_RETRY, stop_reason="max_tokens", delay=0.001)
    qwen = _TimedQwen(delay=0.05)
    env, rows = _wire_stream(monkeypatch, llm, qwen=qwen)
    await _bake(42, 5)
    old = [text for _, text in qwen.calls if text in {t["text"] for t in _LONG}]
    assert 1 <= len(old) <= 5
    assert rows[42].example_dialogue == _RETRY and rows[42].example_audio_url.endswith("-qn.wav")
