import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from app.core.tts_voices import EXAMPLE_USER_VOICE_ID, VOICE_REGISTRY, pick_example_user_voice
from app.services import example_service
from app.services.example_service import ScenarioNotFoundError, get_example_conversation
from app.services.qwen_tts import QwenTTSUnavailableError


def test_user_voice_default_when_no_collision() -> None:
    assert pick_example_user_voice("some-ai-voice") == EXAMPLE_USER_VOICE_ID


def test_user_voice_avoids_collision_with_ai_voice() -> None:
    picked = pick_example_user_voice(EXAMPLE_USER_VOICE_ID)
    assert picked != EXAMPLE_USER_VOICE_ID
    assert picked in {v.voice_id for v in VOICE_REGISTRY}



class _FakeTTSSession:
    def __init__(self, pcm: bytes) -> None:
        self._pcm = pcm

    async def begin(self, emotion=None) -> None:
        pass

    async def stream(self, text_source):
        async for _ in text_source:
            pass
        yield self._pcm

    async def aclose(self) -> None:
        pass


class _FakeTTSClient:
    def __init__(self, pcm_per_turn: bytes = b"\x01\x02") -> None:
        self.open_calls: list[str | None] = []
        self._pcm = pcm_per_turn

    async def open(self, voice_id: str | None = None) -> _FakeTTSSession:
        self.open_calls.append(voice_id)
        return _FakeTTSSession(self._pcm)


class _FakeStorage:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, bytes]] = []
        self.presigned: list[str] = []

    def upload_wav(self, key: str, pcm: bytes) -> str | None:
        self.uploads.append((key, pcm))
        return key

    def presigned_url(self, key: str, expires_in: int = 600) -> str | None:
        self.presigned.append(key)
        return f"https://signed.test/{key}"


class _FakeDB:
    def __init__(
        self,
        row: object | None = None,
        refresh_sets_url: str | None = None,
        refresh_marks_deleted: bool = False,
    ) -> None:
        self._row = row
        self.commits = 0
        self.refreshes = 0
        self.refresh_sets_url = refresh_sets_url
        self.refresh_marks_deleted = refresh_marks_deleted

    async def get(self, _model: object, _pk: int) -> object | None:
        return self._row

    async def commit(self) -> None:
        self.commits += 1

    async def refresh(self, obj: object, attrs: list[str] | None = None) -> None:
        self.refreshes += 1
        if self.refresh_sets_url is not None:
            obj.example_audio_url = self.refresh_sets_url
        if self.refresh_marks_deleted:
            obj.deleted_at = "2026-09-01T00:00:00Z"


_DIALOGUE = [
    {"speaker": "ai", "text": "네, 바다레스토랑입니다."},
    {"speaker": "user", "text": "예약하려고 전화드렸어요."},
]


def _settings(s3: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        elevenlabs_voice_id="voice-default",
        s3_bucket="bucket" if s3 else None,
        anthropic_api_key="k",
        llm_analysis_model="m",
        qwen_tts_url=None,
        qwen_tts_timeout=30.0,
        qwen_tts_health_timeout=1.0,
    )


def _wire(monkeypatch, *, tts: _FakeTTSClient | None = None, s3: bool = True):
    tts = tts or _FakeTTSClient()
    storage = _FakeStorage()
    monkeypatch.setattr(example_service, "get_settings", lambda: _settings(s3))
    monkeypatch.setattr(example_service, "ElevenLabsTTSClient", lambda _s: tts)
    monkeypatch.setattr(example_service, "RecordingStorageService", lambda _s: storage)
    monkeypatch.setattr(example_service, "PRESET_MAP", {1: {"example_dialogue": _DIALOGUE}})
    return tts, storage


def _preset_row(**kw) -> SimpleNamespace:
    base = dict(
        scenario_id=1, is_custom=False, user_id=None, tts_voice_id=None,
        example_dialogue=None, example_audio_url=None,
        title="음식점 예약", call_target="레스토랑 직원", call_purpose="음식점 예약",
        script=[{"step": 1, "ai_goal": "인사", "hint": "인사하세요"}],
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _custom_row(**kw) -> SimpleNamespace:
    base = dict(
        scenario_id=42, is_custom=True, user_id=7, tts_voice_id="v-ai",
        example_dialogue=None, example_audio_url=None,
        title="집주인 통화", call_target="집주인", call_purpose="월세 납부일 조정",
        script=[{"step": 1, "ai_goal": "인사", "hint": "인사하세요"}],
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _fake_anthropic(payload: object) -> type:
    class _Messages:
        async def create(self, **_kw) -> SimpleNamespace:
            text = json.dumps(payload, ensure_ascii=False)
            return SimpleNamespace(content=[SimpleNamespace(text=text)])

    class _Client:
        def __init__(self, api_key: str) -> None:
            self.messages = _Messages()

    return _Client


async def test_preset_first_call_generates_audio(monkeypatch) -> None:
    tts, storage = _wire(monkeypatch)
    row = _preset_row()
    db = _FakeDB(row)

    resp = await get_example_conversation(db, 1, user_id=7)

    assert [t.speaker for t in resp.dialogue] == ["ai", "user"]
    assert tts.open_calls == ["voice-default", pick_example_user_voice("voice-default")]
    assert len(storage.uploads) == 1
    key, _pcm = storage.uploads[0]
    assert key.startswith("examples/1-") and key.endswith(".wav")
    assert resp.audio_url == f"https://signed.test/{key}"
    assert row.example_audio_url == key
    assert db.commits >= 1


async def test_audio_concatenates_turns_with_silence_gap(monkeypatch) -> None:
    tts = _FakeTTSClient(pcm_per_turn=b"\x01\x02")
    _, storage = _wire(monkeypatch, tts=tts)

    await get_example_conversation(_FakeDB(_preset_row()), 1, user_id=7)

    _key, pcm = storage.uploads[0]
    gap = int(16_000 * 0.4) * 2
    assert len(pcm) == 2 + gap + 2


async def test_second_call_hits_cache(monkeypatch) -> None:
    tts, storage = _wire(monkeypatch)
    row = _preset_row()
    db = _FakeDB(row)

    first = await get_example_conversation(db, 1, user_id=7)
    calls_after_first = list(tts.open_calls)
    second = await get_example_conversation(db, 1, user_id=7)

    assert second.audio_url == first.audio_url
    assert tts.open_calls == calls_after_first
    assert len(storage.uploads) == 1
    assert len(storage.presigned) == 2, "캐시 적중이어도 서명은 요청마다 새로 발급한다"


async def test_dialogue_change_regenerates(monkeypatch) -> None:
    tts, storage = _wire(monkeypatch)
    row = _preset_row()
    db = _FakeDB(row)
    first = await get_example_conversation(db, 1, user_id=7)

    changed = [
        {"speaker": "ai", "text": "네, 새로 바뀐 인사말입니다."},
        {"speaker": "user", "text": "예약이요."},
    ]
    monkeypatch.setattr(example_service, "PRESET_MAP", {1: {"example_dialogue": changed}})
    second = await get_example_conversation(db, 1, user_id=7)

    assert second.audio_url != first.audio_url
    assert len(storage.uploads) == 2


async def test_preset_without_db_row_still_works(monkeypatch) -> None:
    _wire(monkeypatch)

    resp = await get_example_conversation(_FakeDB(None), 1, user_id=7)

    assert resp.audio_url and resp.audio_url.startswith("https://signed.test/examples/1-")
    assert [t.speaker for t in resp.dialogue] == ["ai", "user"]


async def test_cache_hit_with_legacy_static_url_row(monkeypatch) -> None:
    tts, storage = _wire(monkeypatch)
    user_voice = pick_example_user_voice("voice-default")
    key = example_service._audio_key(1, _DIALOGUE, "voice-default", user_voice)
    row = _preset_row(example_audio_url=f"https://bucket.s3.ap-northeast-2.amazonaws.com/{key}")

    resp = await get_example_conversation(_FakeDB(row), 1, user_id=7)

    assert resp.audio_url == f"https://signed.test/{key}"
    assert tts.open_calls == []
    assert storage.uploads == []


async def test_s3_unset_returns_dialogue_without_audio(monkeypatch) -> None:
    tts, storage = _wire(monkeypatch, s3=False)

    resp = await get_example_conversation(_FakeDB(_preset_row()), 1, user_id=7)

    assert resp.audio_url is None
    assert [t.text for t in resp.dialogue]
    assert tts.open_calls == []
    assert storage.uploads == []
    assert storage.presigned == []


# --- 커스텀: LLM 대본 생성/재사용/소유권 ---


async def test_custom_generates_dialogue_via_llm_and_saves(monkeypatch) -> None:
    tts, _storage = _wire(monkeypatch)
    payload = [
        {"speaker": "ai", "text": "여보세요."},
        {"speaker": "user", "text": "안녕하세요, 302호 세입자입니다."},
    ]
    monkeypatch.setattr(example_service, "AsyncAnthropic", _fake_anthropic(payload))
    row = _custom_row()
    db = _FakeDB(row)

    resp = await get_example_conversation(db, 42, user_id=7)

    assert row.example_dialogue == payload
    assert [t.text for t in resp.dialogue] == ["여보세요.", "안녕하세요, 302호 세입자입니다."]
    assert tts.open_calls == ["v-ai", pick_example_user_voice("v-ai")]
    assert db.commits >= 1


async def test_custom_reuses_saved_dialogue(monkeypatch) -> None:
    _wire(monkeypatch)
    saved = [
        {"speaker": "ai", "text": "네."},
        {"speaker": "user", "text": "문의드릴 게 있어요."},
    ]
    row = _custom_row(example_dialogue=saved)

    class _BoomAnthropicClient:
        def __init__(self, api_key: str) -> None:
            raise AssertionError("저장된 대본이 있으면 LLM을 호출하면 안 된다")

    monkeypatch.setattr(example_service, "AsyncAnthropic", _BoomAnthropicClient)

    resp = await get_example_conversation(_FakeDB(row), 42, user_id=7)

    assert [t.speaker for t in resp.dialogue] == ["ai", "user"]


async def test_custom_llm_bad_output_raises(monkeypatch) -> None:
    _wire(monkeypatch)
    monkeypatch.setattr(example_service, "AsyncAnthropic", _fake_anthropic({"not": "a list"}))

    with pytest.raises(ValueError):
        await get_example_conversation(_FakeDB(_custom_row()), 42, user_id=7)


async def test_custom_of_other_user_not_found(monkeypatch) -> None:
    _wire(monkeypatch)
    row = _custom_row(user_id=99)

    with pytest.raises(ScenarioNotFoundError):
        await get_example_conversation(_FakeDB(row), 42, user_id=7)


async def test_unknown_scenario_not_found(monkeypatch) -> None:
    _wire(monkeypatch)

    with pytest.raises(ScenarioNotFoundError):
        await get_example_conversation(_FakeDB(None), 555, user_id=7)


# --- 리소스: 락 정리 / TTS 병렬화 ---


async def test_generation_lock_not_retained_after_request(monkeypatch) -> None:
    """요청이 끝나면 scenario_id별 락이 딕셔너리에 남지 않는다(메모리 누수 방지)."""
    _wire(monkeypatch)

    await get_example_conversation(_FakeDB(_preset_row()), 1, user_id=7)

    assert 1 not in example_service._generation_locks


class _TrackingSession:
    def __init__(self, client: "_ConcurrencyTrackingTTSClient") -> None:
        self._client = client

    async def begin(self, emotion=None) -> None:
        pass

    async def stream(self, text_source):
        async for _ in text_source:
            pass
        self._client.active += 1
        self._client.peak = max(self._client.peak, self._client.active)
        await asyncio.sleep(0.01)
        self._client.active -= 1
        yield b"\x00\x00"

    async def aclose(self) -> None:
        pass


class _ConcurrencyTrackingTTSClient:
    def __init__(self) -> None:
        self.active = 0
        self.peak = 0

    async def open(self, voice_id: str | None = None) -> _TrackingSession:
        return _TrackingSession(self)


def _long_dialogue(n: int = 6) -> list[dict]:
    return [
        {"speaker": "ai" if i % 2 == 0 else "user", "text": f"{i}번째 대사입니다."}
        for i in range(n)
    ]


async def test_turns_synthesized_concurrently_but_bounded(monkeypatch) -> None:
    """턴 합성은 병렬로 하되, ElevenLabs 동시 연결 한도 때문에 3개로 제한한다."""
    tts = _ConcurrencyTrackingTTSClient()
    storage = _FakeStorage()
    monkeypatch.setattr(example_service, "get_settings", lambda: _settings())
    monkeypatch.setattr(example_service, "ElevenLabsTTSClient", lambda _s: tts)
    monkeypatch.setattr(example_service, "RecordingStorageService", lambda _s: storage)
    monkeypatch.setattr(example_service, "PRESET_MAP", {1: {"example_dialogue": _long_dialogue(6)}})

    await get_example_conversation(_FakeDB(_preset_row()), 1, user_id=7)

    assert tts.peak >= 2, "턴 합성이 병렬로 실행되지 않음"
    assert tts.peak <= 3, "동시 TTS 연결이 한도(3)를 초과함"


class _EchoSession:
    def __init__(self, delay: float) -> None:
        self._delay = delay

    async def begin(self, emotion=None) -> None:
        pass

    async def stream(self, text_source):
        text = ""
        async for chunk in text_source:
            text += chunk
        await asyncio.sleep(self._delay)
        yield text.encode()

    async def aclose(self) -> None:
        pass


class _EchoTTSClient:
    """앞 턴일수록 느리게 완성돼, 완료 순서로 이어붙이면 순서가 뒤집힌다."""

    def __init__(self, total_turns: int) -> None:
        self._remaining = total_turns

    async def open(self, voice_id: str | None = None) -> _EchoSession:
        delay = self._remaining * 0.005
        self._remaining -= 1
        return _EchoSession(delay)


async def test_parallel_synthesis_preserves_turn_order(monkeypatch) -> None:
    dialogue = _long_dialogue(4)
    tts = _EchoTTSClient(total_turns=4)
    storage = _FakeStorage()
    monkeypatch.setattr(example_service, "get_settings", lambda: _settings())
    monkeypatch.setattr(example_service, "ElevenLabsTTSClient", lambda _s: tts)
    monkeypatch.setattr(example_service, "RecordingStorageService", lambda _s: storage)
    monkeypatch.setattr(example_service, "PRESET_MAP", {1: {"example_dialogue": dialogue}})

    await get_example_conversation(_FakeDB(_preset_row()), 1, user_id=7)

    _key, pcm = storage.uploads[0]
    gap = b"\x00" * (int(16_000 * 0.4) * 2)
    expected = gap.join(turn["text"].encode() for turn in dialogue)
    assert pcm == expected


class _FakeQwenClient:
    def __init__(
        self,
        *,
        enabled: bool = True,
        ready: bool = True,
        fail_at: int | None = None,
        pcm: bytes = b"\x09\x09",
    ) -> None:
        self.enabled = enabled
        self._ready = ready
        self._fail_at = fail_at
        self._pcm = pcm
        self.calls: list[tuple[str, str]] = []

    @asynccontextmanager
    async def connect(self):
        yield self

    async def healthy(self) -> bool:
        return self.enabled and self._ready

    async def synth(self, voice: str, text: str) -> bytes:
        index = len(self.calls)
        self.calls.append((voice, text))
        if self._fail_at is not None and index == self._fail_at:
            raise QwenTTSUnavailableError("boom")
        return self._pcm


def _wire_qwen(monkeypatch, client: _FakeQwenClient) -> _FakeQwenClient:
    monkeypatch.setattr(example_service, "QwenTTSClient", lambda _s: client)
    return client


def _eleven_key() -> str:
    return example_service._audio_key(
        1, _DIALOGUE, "voice-default", pick_example_user_voice("voice-default")
    )


async def test_qwen_disabled_uses_elevenlabs_with_plain_key(monkeypatch) -> None:
    tts, storage = _wire(monkeypatch)
    _wire_qwen(monkeypatch, _FakeQwenClient(enabled=False, ready=False))

    await get_example_conversation(_FakeDB(_preset_row()), 1, user_id=7)

    assert tts.open_calls == ["voice-default", pick_example_user_voice("voice-default")]
    key, _ = storage.uploads[0]
    assert not key.endswith("-q.wav")


async def test_qwen_healthy_uses_qwen_with_suffixed_key(monkeypatch) -> None:
    tts, storage = _wire(monkeypatch)
    qwen = _wire_qwen(monkeypatch, _FakeQwenClient())

    await get_example_conversation(_FakeDB(_preset_row()), 1, user_id=7)

    assert [voice for voice, _ in qwen.calls] == ["ai", "user"]
    assert tts.open_calls == []
    key, _ = storage.uploads[0]
    assert key.endswith("-q.wav")


async def test_qwen_mid_turn_failure_refalls_back_whole_dialogue(monkeypatch) -> None:
    tts, storage = _wire(monkeypatch)
    _wire_qwen(monkeypatch, _FakeQwenClient(fail_at=1))

    await get_example_conversation(_FakeDB(_preset_row()), 1, user_id=7)

    assert tts.open_calls == ["voice-default", pick_example_user_voice("voice-default")]
    key, pcm = storage.uploads[0]
    assert not key.endswith("-q.wav")
    assert b"\x09" not in pcm


async def test_qwen_unhealthy_skips_synth_entirely(monkeypatch) -> None:
    tts, _ = _wire(monkeypatch)
    qwen = _wire_qwen(monkeypatch, _FakeQwenClient(ready=False))

    await get_example_conversation(_FakeDB(_preset_row()), 1, user_id=7)

    assert qwen.calls == []
    assert tts.open_calls


async def test_realtime_slot_busy_skips_qwen_without_health_check(monkeypatch) -> None:
    tts, storage = _wire(monkeypatch)
    qwen = _wire_qwen(monkeypatch, _FakeQwenClient())
    checked: list[int] = []
    orig = qwen.healthy

    async def _spy():
        checked.append(1)
        return await orig()

    qwen.healthy = _spy
    monkeypatch.setattr(example_service, "realtime_slot_active", lambda: True)

    await get_example_conversation(_FakeDB(_preset_row()), 1, user_id=7)

    assert checked == [], "실시간 통화가 슬롯을 잡고 있으면 헬스체크 없이 즉시 폴백"
    assert qwen.calls == []
    assert tts.open_calls, "ElevenLabs 로 합성돼야 한다"
    key, _ = storage.uploads[0]
    assert not key.endswith("-q.wav")


async def test_cache_hit_skips_health_check(monkeypatch) -> None:
    _wire(monkeypatch)
    qwen = _FakeQwenClient()
    checked = []
    orig = qwen.healthy

    async def _spy():
        checked.append(1)
        return await orig()

    qwen.healthy = _spy
    _wire_qwen(monkeypatch, qwen)

    await get_example_conversation(
        _FakeDB(_preset_row(example_audio_url=_eleven_key())), 1, user_id=7
    )

    assert checked == [], "캐시 히트면 Qwen 헬스체크를 하지 않아야 한다"


async def test_qwen_key_ignores_elevenlabs_voice_ids() -> None:
    a = example_service._audio_key(1, _DIALOGUE, "voice-A", "user-A", qwen=True)
    b = example_service._audio_key(1, _DIALOGUE, "voice-B", "user-B", qwen=True)
    assert a == b, "voice_id 변경이 Qwen 캐시를 무효화하면 안 된다"

    c = example_service._audio_key(1, _DIALOGUE, "voice-A", "user-A")
    d = example_service._audio_key(1, _DIALOGUE, "voice-B", "user-B")
    assert c != d


async def test_existing_eleven_cache_kept_when_qwen_becomes_available(monkeypatch) -> None:
    tts, storage = _wire(monkeypatch)
    qwen = _wire_qwen(monkeypatch, _FakeQwenClient())

    resp = await get_example_conversation(
        _FakeDB(_preset_row(example_audio_url=_eleven_key())), 1, user_id=7
    )

    assert storage.uploads == []
    assert qwen.calls == []
    assert tts.open_calls == []
    assert resp.audio_url


async def test_concurrent_request_rereads_row_inside_lock(monkeypatch) -> None:
    tts, storage = _wire(monkeypatch)
    _wire_qwen(monkeypatch, _FakeQwenClient())
    db = _FakeDB(_preset_row(), refresh_sets_url=_eleven_key())

    resp = await get_example_conversation(db, 1, user_id=7)

    assert db.refreshes == 1
    assert storage.uploads == [], "앞선 요청이 이미 구웠으면 다시 굽지 않는다"
    assert tts.open_calls == []
    assert resp.audio_url



def _wire_prebake(monkeypatch, row, *, storage=None, tts=None, qwen=None):
    storage = storage or _FakeStorage()
    tts = tts or _FakeTTSClient()
    db = _FakeDB(row)

    class _SessionCtx:
        async def __aenter__(self): return db
        async def __aexit__(self, *a): return False

    monkeypatch.setattr(example_service, "get_settings", lambda: _settings(True))
    monkeypatch.setattr(example_service, "AsyncSessionLocal", lambda: _SessionCtx())
    monkeypatch.setattr(example_service, "ElevenLabsTTSClient", lambda _s: tts)
    monkeypatch.setattr(example_service, "RecordingStorageService", lambda _s: storage)
    off = qwen or _FakeQwenClient(enabled=False, ready=False)
    monkeypatch.setattr(example_service, "QwenTTSClient", lambda _s: off)
    monkeypatch.setattr(example_service, "is_deleted", lambda _r: False)
    return db, storage, tts


async def test_prebake_generates_and_stores(monkeypatch) -> None:
    row = _custom_row(example_dialogue=_DIALOGUE)
    db, storage, tts = _wire_prebake(monkeypatch, row)

    await example_service.bake_example_audio(42)

    assert len(storage.uploads) == 1
    assert row.example_audio_url == storage.uploads[0][0]
    assert db.commits >= 1


async def test_prebake_skips_when_already_baked(monkeypatch) -> None:
    row = _custom_row(example_dialogue=_DIALOGUE, example_audio_url="examples/42-abc.wav")
    _, storage, tts = _wire_prebake(monkeypatch, row)

    await example_service.bake_example_audio(42)

    assert storage.uploads == []
    assert tts.open_calls == []


async def test_prebake_swallows_failure(monkeypatch) -> None:
    row = _custom_row(example_dialogue=_DIALOGUE)

    class _BoomTTS:
        async def open(self, voice_id=None):
            raise RuntimeError("TTS 폭발")

    _wire_prebake(monkeypatch, row, tts=_BoomTTS())

    await example_service.bake_example_audio(42)  # 예외가 새어나오면 실패

    assert row.example_audio_url is None


async def test_prebaked_audio_makes_request_a_cache_hit(monkeypatch) -> None:
    row = _preset_row()
    tts, storage = _wire(monkeypatch)
    _wire_qwen(monkeypatch, _FakeQwenClient(enabled=False, ready=False))

    first = await get_example_conversation(_FakeDB(row), 1, user_id=7)  # 여기서 구워짐
    calls_after_bake = list(tts.open_calls)
    second = await get_example_conversation(_FakeDB(row), 1, user_id=7)

    assert second.audio_url == first.audio_url
    assert tts.open_calls == calls_after_bake, "두 번째는 합성하지 않는다"
    assert len(storage.uploads) == 1


async def test_metric_records_trigger(monkeypatch) -> None:
    metrics = []
    monkeypatch.setattr(example_service, "log_metric", lambda n, **kw: metrics.append((n, kw)))
    _wire(monkeypatch)
    _wire_qwen(monkeypatch, _FakeQwenClient(enabled=False, ready=False))

    await get_example_conversation(_FakeDB(_preset_row()), 1, user_id=7)

    assert metrics and metrics[0][0] == "example_tts"
    assert metrics[0][1]["trigger"] == "request"

    metrics.clear()
    row = _custom_row(example_dialogue=_DIALOGUE)
    _wire_prebake(monkeypatch, row)
    monkeypatch.setattr(example_service, "log_metric", lambda n, **kw: metrics.append((n, kw)))
    await example_service.bake_example_audio(42)
    assert metrics[0][1]["trigger"] == "prebake"


async def test_prebake_generates_dialogue_via_llm_when_absent(monkeypatch) -> None:
    payload = [
        {"speaker": "ai", "text": "네, 안녕하세요."},
        {"speaker": "user", "text": "예약하려고요."},
    ]
    row = _custom_row(example_dialogue=None)
    _, storage, _ = _wire_prebake(monkeypatch, row)
    monkeypatch.setattr(example_service, "AsyncAnthropic", _fake_anthropic(payload))

    await example_service.bake_example_audio(42)

    assert row.example_dialogue == payload, "대본이 생성·저장돼야 한다"
    assert len(storage.uploads) == 1
    assert row.example_audio_url == storage.uploads[0][0]


async def test_dialogue_generated_once_when_prebake_and_request_race(monkeypatch) -> None:
    payload = [
        {"speaker": "ai", "text": "네, 안녕하세요."},
        {"speaker": "user", "text": "예약하려고요."},
    ]
    shared = {"example_dialogue": None, "example_audio_url": None}
    calls = []

    class _Messages:
        async def create(self, **_kw):
            calls.append(1)
            await asyncio.sleep(0)
            return SimpleNamespace(
                content=[SimpleNamespace(text=json.dumps(payload, ensure_ascii=False))]
            )

    class _Anthropic:
        def __init__(self, api_key: str) -> None:
            self.messages = _Messages()

    class _SessionDB:

        def __init__(self) -> None:
            self.row = _custom_row(**shared)

        async def get(self, _model, _pk):
            return self.row

        async def refresh(self, obj, attrs=None):
            for k, v in shared.items():
                setattr(obj, k, v)

        async def commit(self):
            for k in shared:
                shared[k] = getattr(self.row, k)

    tts, storage = _wire(monkeypatch)
    _wire_qwen(monkeypatch, _FakeQwenClient(enabled=False, ready=False))
    monkeypatch.setattr(example_service, "AsyncAnthropic", _Anthropic)
    monkeypatch.setattr(example_service, "PRESET_MAP", {})

    await asyncio.gather(
        get_example_conversation(_SessionDB(), 42, user_id=7),
        get_example_conversation(_SessionDB(), 42, user_id=7),
    )

    assert calls == [1], f"LLM 을 {len(calls)}번 불렀다 — 락 밖에서 대본을 만들고 있다"
    assert len(storage.uploads) == 1, "대본이 하나면 오디오도 하나여야 한다"


async def test_deleted_during_lock_wait_skips_synthesis(monkeypatch) -> None:
    row = _custom_row(example_dialogue=_DIALOGUE)
    tts, storage = _wire(monkeypatch)
    _wire_qwen(monkeypatch, _FakeQwenClient(enabled=False, ready=False))
    monkeypatch.setattr(example_service, "PRESET_MAP", {})
    db = _FakeDB(row, refresh_marks_deleted=True)

    resp = await get_example_conversation(db, 42, user_id=7)

    assert tts.open_calls == [], "지워진 시나리오를 합성하고 있다"
    assert storage.uploads == []
    assert resp.audio_url is None


async def test_prebake_queue_wait_is_outside_timeout(monkeypatch) -> None:
    monkeypatch.setattr(example_service, "_PREBAKE_TIMEOUT_SEC", 0.45)
    baked: list[int] = []

    class _SlowTTS:
        async def open(self, voice_id=None):
            await asyncio.sleep(0.3)  # 첫 작업이 세마포어를 오래 점유한다

            class _S:
                async def begin(self, emotion=None) -> None:
                    return None

                async def stream(self, src):
                    async for _ in src:
                        pass
                    yield b"\x01\x02"

                async def aclose(self) -> None:
                    return None

            return _S()

    rows = {i: _custom_row(scenario_id=i, example_dialogue=_DIALOGUE) for i in (1, 2)}

    class _Ctx:
        def __init__(self, sid): self.sid = sid
        async def __aenter__(self):
            db = _FakeDB(rows[self.sid])
            return db
        async def __aexit__(self, *a): return False

    seq = iter([1, 2])
    monkeypatch.setattr(example_service, "get_settings", lambda: _settings(True))
    monkeypatch.setattr(example_service, "AsyncSessionLocal", lambda: _Ctx(next(seq)))
    monkeypatch.setattr(example_service, "ElevenLabsTTSClient", lambda _s: _SlowTTS())
    monkeypatch.setattr(example_service, "RecordingStorageService",
                        lambda _s: _RecordingSpy(baked))
    monkeypatch.setattr(example_service, "QwenTTSClient",
                        lambda _s: _FakeQwenClient(enabled=False, ready=False))
    monkeypatch.setattr(example_service, "is_deleted", lambda _r: False)

    await asyncio.gather(example_service.bake_example_audio(1),
                         example_service.bake_example_audio(2))

    assert len(baked) == 2, f"줄서기 대기가 타임아웃을 먹어 {2 - len(baked)}건이 취소됐다"


class _RecordingSpy:
    def __init__(self, sink: list) -> None:
        self._sink = sink

    def upload_wav(self, key: str, pcm: bytes) -> str | None:
        self._sink.append(key)
        return key

    def presigned_url(self, key: str, expires_in: int = 600) -> str | None:
        return f"https://signed.test/{key}"
