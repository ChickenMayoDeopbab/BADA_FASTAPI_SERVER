import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from app.core.tts_voices import EXAMPLE_USER_VOICE_ID, VOICE_REGISTRY, pick_example_user_voice
from app.services import example_service
from app.services.example_audio import MAX_ATTEMPTS, TAIL_PHRASE
from app.services.example_service import ScenarioNotFoundError, get_example_conversation
from app.services.qwen_tts import QwenTTSUnavailableError
from tests.unit.test_example_audio import pcm as _pcm
from tests.unit.test_example_audio import silence as _silence
from tests.unit.test_example_audio import tone as _tone


def test_user_voice_default_when_no_collision() -> None:
    assert pick_example_user_voice("some-ai-voice") == EXAMPLE_USER_VOICE_ID


def test_user_voice_avoids_collision_with_ai_voice() -> None:
    picked = pick_example_user_voice(EXAMPLE_USER_VOICE_ID)
    assert picked != EXAMPLE_USER_VOICE_ID
    assert picked in {v.el_voice_id for v in VOICE_REGISTRY}



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
        qwen_tts_urls=None,
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
        pcm: bytes | None = None,
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
        return self._out(text)

    def _out(self, text: str) -> bytes:
        return self._pcm if self._pcm is not None else _voice_pcm(text)


def _voice_pcm(text: str, decay_ms: float = 40.0) -> bytes:
    if text.endswith(TAIL_PHRASE):
        return _pcm(_tone(2.0, decay_ms), _silence(0.3), _tone(0.3, 30.0), _silence(0.1))
    return _pcm(_tone(2.0, decay_ms), _silence(0.1))


def _wire_qwen(monkeypatch, client: _FakeQwenClient) -> _FakeQwenClient:
    monkeypatch.setattr(example_service, "QwenTTSClient", lambda _s: client)
    monkeypatch.setattr(example_service, "has_free_worker", lambda _s: True)
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
    assert pcm == example_service._TURN_GAP_PCM.join([b"\x01\x02"] * 2)


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
    monkeypatch.setattr(example_service, "has_free_worker", lambda _s: False)

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


async def test_qwen_key_depends_on_qwen_voice_names_and_keeps_legacy_digest() -> None:
    a = example_service._audio_key(1, _DIALOGUE, "male_old_soft", "user", qwen=True)
    b = example_service._audio_key(1, _DIALOGUE, "female_old_soft", "user", qwen=True)
    assert a != b, "Qwen 보이스가 바뀌면 다시 굽는다"
    payload = json.dumps([_DIALOGUE, "qwen-ai", "qwen-user"], ensure_ascii=False)
    legacy = f"examples/1-{hashlib.sha1(payload.encode()).hexdigest()[:8]}-q.wav"
    assert example_service._audio_key(1, _DIALOGUE, "ai", "user", qwen=True) == legacy, "옛 캐시 키와 같아야 한다"

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


async def test_custom_generation_prompt_writes_people_counter_in_hangul(monkeypatch) -> None:
    _wire(monkeypatch)
    calls: list[dict] = []

    class _Messages:
        async def create(self, **kw) -> SimpleNamespace:
            calls.append(kw)
            turns = [{"speaker": "ai", "text": "여보세요."}, {"speaker": "user", "text": "네."}]
            text = json.dumps(turns, ensure_ascii=False)
            return SimpleNamespace(content=[SimpleNamespace(text=text)])

    class _Client:
        def __init__(self, api_key: str) -> None:
            self.messages = _Messages()

    monkeypatch.setattr(example_service, "AsyncAnthropic", _Client)

    await get_example_conversation(_FakeDB(_custom_row()), 42, user_id=7)

    system = calls[0]["system"]
    assert "In \"ai\" lines" in system
    assert "never digits like '4분'" in system
    assert "own party with 명" in system


class _MissingQwen(_FakeQwenClient):
    def __init__(self, missing: set[str]) -> None:
        super().__init__()
        self._missing = missing

    async def synth(self, voice: str, text: str) -> bytes:
        self.calls.append((voice, text))
        if voice in self._missing:
            from app.services.qwen_tts import QwenVoiceMissingError

            raise QwenVoiceMissingError(voice)
        return self._out(text)


def _flag_on(monkeypatch, on: bool = True) -> None:
    monkeypatch.setattr(example_service, "get_settings", lambda: SimpleNamespace(**vars(_settings()),
                                                                                 qwen_scenario_voices=on))


def _qkey(ai: str) -> str:
    return example_service._audio_key(1, _DIALOGUE, ai, "user", qwen=True, method=example_service._CLEAN_METHOD)


def _old_qkey(scenario_id: int = 1) -> str:
    return example_service._audio_key(scenario_id, _DIALOGUE, "ai", "user", qwen=True)


def _capture(monkeypatch) -> list:
    metrics: list = []
    monkeypatch.setattr(example_service, "log_metric", lambda n, **kw: metrics.append((n, kw)))
    return metrics


async def test_slot_voices_bake_new_example_with_low_slot_and_user(monkeypatch) -> None:
    tts, storage = _wire(monkeypatch)
    _flag_on(monkeypatch)
    qwen = _wire_qwen(monkeypatch, _FakeQwenClient())
    metrics = _capture(monkeypatch)

    await get_example_conversation(_FakeDB(_preset_row(tts_voice_id="male_middle_rough")), 1, user_id=7)

    assert [v for v, _ in qwen.calls] == ["male_middle_soft", "user"], "예시는 난이도 하 칸"
    assert storage.uploads[0][0] == _qkey("male_middle_soft") and tts.open_calls == []
    assert metrics[0][1]["voice"] == "male_middle_soft" and metrics[0][1]["fallback_reason"] is None


async def test_switch_off_keeps_ai_user_on_qwen_and_elevenlabs_gets_slot_id(monkeypatch) -> None:
    tts, storage = _wire(monkeypatch)
    _flag_on(monkeypatch, on=False)
    qwen = _wire_qwen(monkeypatch, _FakeQwenClient())
    await get_example_conversation(_FakeDB(_preset_row(tts_voice_id="male_middle_soft")), 1, user_id=7)
    assert [v for v, _ in qwen.calls] == ["ai", "user"] and storage.uploads[0][0] == _qkey("ai")

    tts, storage = _wire(monkeypatch)
    _wire_qwen(monkeypatch, _FakeQwenClient(enabled=False, ready=False))
    await get_example_conversation(_FakeDB(_preset_row(tts_voice_id="male_middle_soft")), 1, user_id=7)
    el = "Ctl3Pjfp5c2p3ThSA8Hn"
    assert tts.open_calls == [el, pick_example_user_voice(el)], "ElevenLabs 에는 칸 키가 아니라 EL id"


async def test_custom_example_baked_with_old_qwen_key_is_kept_with_slot_voices(monkeypatch) -> None:
    _, storage = _wire(monkeypatch)
    _flag_on(monkeypatch)
    qwen = _wire_qwen(monkeypatch, _FakeQwenClient())
    row = _custom_row(tts_voice_id="female_young_soft", example_dialogue=_DIALOGUE, example_audio_url=_old_qkey(42))
    await get_example_conversation(_FakeDB(row), 42, user_id=7)
    assert qwen.calls == [] and storage.uploads == [], "이미 구운 커스텀 예시는 다시 굽지 않는다"


async def test_preset_request_serves_old_example_while_prebake_holds_the_lock(monkeypatch) -> None:
    _, storage = _wire(monkeypatch)
    _flag_on(monkeypatch)
    qwen = _wire_qwen(monkeypatch, _FakeQwenClient())
    row = _preset_row(tts_voice_id="female_young_soft", example_audio_url=_old_qkey())
    lock = example_service._generation_locks.setdefault(1, asyncio.Lock())
    async with lock:
        resp = await asyncio.wait_for(get_example_conversation(_FakeDB(row), 1, user_id=7), timeout=1.0)
    assert resp.audio_url and qwen.calls == [] and storage.uploads == []


async def test_prebake_replaces_old_preset_example_with_the_slot_voice(monkeypatch) -> None:
    _, storage = _wire(monkeypatch)
    _flag_on(monkeypatch)
    qwen = _wire_qwen(monkeypatch, _FakeQwenClient())
    row = _preset_row(tts_voice_id="female_young_soft", example_audio_url=_old_qkey())
    assert await _bake_qwen_only(row, storage) == _qkey("female_young_soft")
    assert [v for v, _ in qwen.calls] == ["female_young_soft", "user"]


async def test_preset_eleven_example_from_before_casting_is_served_then_prebaked(monkeypatch) -> None:
    _, storage = _wire(monkeypatch)
    qwen = _wire_qwen(monkeypatch, _FakeQwenClient())
    row = _preset_row(tts_voice_id="female_young_soft", example_audio_url=_eleven_key())
    await get_example_conversation(_FakeDB(row), 1, user_id=7)
    assert qwen.calls == [] and storage.uploads == []
    assert await _bake_qwen_only(row, storage) == _qkey("ai")


async def test_custom_legacy_eleven_id_cache_still_hits(monkeypatch) -> None:
    _, storage = _wire(monkeypatch)
    qwen = _wire_qwen(monkeypatch, _FakeQwenClient())
    legacy = "4JJwo477JUAx3HV0T7n7"
    old_key = example_service._audio_key(42, _DIALOGUE, legacy, pick_example_user_voice(legacy))
    row = _custom_row(tts_voice_id=legacy, example_dialogue=_DIALOGUE, example_audio_url=old_key)
    await get_example_conversation(_FakeDB(row), 42, user_id=7)
    assert qwen.calls == [] and storage.uploads == []


async def test_missing_slot_voice_rebakes_with_ai_and_stores_clean_ai_key(monkeypatch) -> None:
    tts, storage = _wire(monkeypatch)
    _flag_on(monkeypatch)
    qwen = _wire_qwen(monkeypatch, _MissingQwen({"male_middle_soft"}))
    metrics = _capture(monkeypatch)
    await get_example_conversation(_FakeDB(_preset_row(tts_voice_id="male_middle_soft")), 1, user_id=7)
    assert [v for v, _ in qwen.calls] == ["male_middle_soft", "ai", "user"]
    assert storage.uploads[0][0] == _qkey("ai") and tts.open_calls == [], "ElevenLabs 로 굳지 않는다"
    assert (metrics[0][1]["engine"], metrics[0][1]["voice"], metrics[0][1]["fallback_reason"]) == (
        "qwen", "ai", "voice_missing")


async def test_missing_voice_and_failed_retry_falls_back_to_elevenlabs(monkeypatch) -> None:
    tts, storage = _wire(monkeypatch)
    _flag_on(monkeypatch)
    _wire_qwen(monkeypatch, _MissingQwen({"male_middle_soft", "user"}))
    metrics = _capture(monkeypatch)
    await get_example_conversation(_FakeDB(_preset_row(tts_voice_id="male_middle_soft")), 1, user_id=7)
    el = "Ctl3Pjfp5c2p3ThSA8Hn"
    assert tts.open_calls == [el, pick_example_user_voice(el)] and not storage.uploads[0][0].endswith("-q.wav")
    assert (metrics[0][1]["engine"], metrics[0][1]["voice"], metrics[0][1]["fallback_reason"]) == (
        "eleven", el, "voice_missing")


class _RetryFailsQwen(_MissingQwen):
    async def synth(self, voice: str, text: str) -> bytes:
        if voice == "ai":
            self.calls.append((voice, text))
            raise QwenTTSUnavailableError("HTTP 503")
        return await super().synth(voice, text)


async def test_retry_failing_for_another_reason_reports_synth_failed(monkeypatch) -> None:
    tts, _ = _wire(monkeypatch)
    _flag_on(monkeypatch)
    _wire_qwen(monkeypatch, _RetryFailsQwen({"male_middle_soft"}))
    metrics = _capture(monkeypatch)
    await get_example_conversation(_FakeDB(_preset_row(tts_voice_id="male_middle_soft")), 1, user_id=7)
    assert tts.open_calls and metrics[0][1]["fallback_reason"] == "synth_failed"


async def test_missing_user_voice_does_not_retry_with_ai(monkeypatch) -> None:
    tts, _ = _wire(monkeypatch)
    _flag_on(monkeypatch)
    qwen = _wire_qwen(monkeypatch, _MissingQwen({"user"}))
    metrics = _capture(monkeypatch)
    await get_example_conversation(_FakeDB(_preset_row(tts_voice_id="male_middle_soft")), 1, user_id=7)
    assert [v for v, _ in qwen.calls] == ["male_middle_soft", "user"], "칸이 아니라 user 가 없으면 다시 굽지 않는다"
    assert tts.open_calls and metrics[0][1]["fallback_reason"] == "voice_missing"


async def test_switch_off_missing_ai_goes_straight_to_elevenlabs(monkeypatch) -> None:
    tts, _ = _wire(monkeypatch)
    _flag_on(monkeypatch, on=False)
    qwen = _wire_qwen(monkeypatch, _MissingQwen({"ai"}))
    metrics = _capture(monkeypatch)
    await get_example_conversation(_FakeDB(_preset_row(tts_voice_id="male_middle_soft")), 1, user_id=7)
    assert [v for v, _ in qwen.calls] == ["ai"] and tts.open_calls
    assert metrics[0][1]["fallback_reason"] == "voice_missing"


async def test_example_baked_under_slot_qwen_key_is_a_cache_hit(monkeypatch) -> None:
    _, storage = _wire(monkeypatch)
    _flag_on(monkeypatch)
    qwen = _wire_qwen(monkeypatch, _FakeQwenClient())
    row = _preset_row(tts_voice_id="male_middle_rough", example_audio_url=_qkey("male_middle_soft"))
    await get_example_conversation(_FakeDB(row), 1, user_id=7)
    assert qwen.calls == [] and storage.uploads == []


async def test_example_baked_under_slot_eleven_key_is_a_cache_hit(monkeypatch) -> None:
    _, storage = _wire(monkeypatch)
    qwen = _wire_qwen(monkeypatch, _FakeQwenClient())
    el = "uyVNoMrnUku1dZyVEXwD"
    row = _preset_row(tts_voice_id="female_young_soft",
                      example_audio_url=example_service._audio_key(1, _DIALOGUE, el, pick_example_user_voice(el)))
    await get_example_conversation(_FakeDB(row), 1, user_id=7)
    assert qwen.calls == [] and storage.uploads == []


async def test_eleven_example_uses_the_low_slot_el_voice(monkeypatch) -> None:
    tts, _ = _wire(monkeypatch)
    _wire_qwen(monkeypatch, _FakeQwenClient(enabled=False, ready=False))
    await get_example_conversation(_FakeDB(_preset_row(tts_voice_id="male_young_rough")), 1, user_id=7)
    assert tts.open_calls[0] == "yUhh6u2RvOjeCsWPEaEg", "거친 칸 저장이어도 예시는 하 칸 EL 보이스"


class _AbruptOnceQwen(_FakeQwenClient):
    def __init__(self, abrupt_voice: str) -> None:
        super().__init__()
        self._abrupt_voice = abrupt_voice

    def _out(self, text: str) -> bytes:
        if [v for v, _ in self.calls].count(self._abrupt_voice) == 1:
            return _voice_pcm(text, decay_ms=0.0)
        return _voice_pcm(text)


async def test_qwen_bake_sends_tail_phrase_and_records_clean_stats(monkeypatch) -> None:
    _, storage = _wire(monkeypatch)
    qwen = _wire_qwen(monkeypatch, _FakeQwenClient())
    metrics = _capture(monkeypatch)
    await get_example_conversation(_FakeDB(_preset_row()), 1, user_id=7)
    assert [t for _, t in qwen.calls] == [turn["text"] + TAIL_PHRASE for turn in _DIALOGUE]
    assert (metrics[0][1]["clean_attempts"], metrics[0][1]["clean_residual_cuts"]) == (2, 0)
    assert len(storage.uploads[0][1]) < 2 * len(_voice_pcm("x" + TAIL_PHRASE)), "꼬리말은 잘려 나간다"


async def test_qwen_bake_resynthesizes_only_the_turn_with_a_cut_end(monkeypatch) -> None:
    _wire(monkeypatch)
    qwen = _wire_qwen(monkeypatch, _AbruptOnceQwen("user"))
    metrics = _capture(monkeypatch)
    await get_example_conversation(_FakeDB(_preset_row()), 1, user_id=7)
    assert [v for v, _ in qwen.calls] == ["ai", "user", "user"]
    assert (metrics[0][1]["clean_attempts"], metrics[0][1]["clean_residual_cuts"]) == (3, 0)


async def test_eleven_bake_has_no_clean_stats(monkeypatch) -> None:
    _wire(monkeypatch)
    _wire_qwen(monkeypatch, _FakeQwenClient(enabled=False, ready=False))
    metrics = _capture(monkeypatch)
    await get_example_conversation(_FakeDB(_preset_row()), 1, user_id=7)
    assert metrics[0][1]["clean_attempts"] is None and metrics[0][1]["clean_residual_cuts"] is None


def test_clean_method_changes_the_qwen_key() -> None:
    assert _qkey("ai") != _old_qkey()
    assert _qkey("ai").endswith("-q.wav")


async def _bake_qwen_only(row, storage) -> str | None:
    return await example_service._bake_locked(
        _FakeDB(row), example_service.get_settings(), row, 1, _DIALOGUE, storage,
        trigger="preset_prebake", qwen_only=True,
    )


async def test_qwen_only_rebakes_preset_that_was_baked_by_elevenlabs(monkeypatch) -> None:
    tts, storage = _wire(monkeypatch)
    _wire_qwen(monkeypatch, _FakeQwenClient())
    row = _preset_row(example_audio_url=_eleven_key())
    assert await _bake_qwen_only(row, storage) == _qkey("ai")
    assert tts.open_calls == [] and row.example_audio_url == _qkey("ai")


async def test_qwen_only_skips_when_already_baked_with_the_current_key(monkeypatch) -> None:
    _, storage = _wire(monkeypatch)
    qwen = _wire_qwen(monkeypatch, _FakeQwenClient())
    assert await _bake_qwen_only(_preset_row(example_audio_url=_qkey("ai")), storage) == _qkey("ai")
    assert qwen.calls == [] and storage.uploads == []


@pytest.mark.parametrize("client", [_FakeQwenClient(ready=False), _FakeQwenClient(fail_at=1)], ids=["down", "fails"])
async def test_qwen_only_never_falls_back_to_elevenlabs(monkeypatch, client) -> None:
    tts, storage = _wire(monkeypatch)
    _wire_qwen(monkeypatch, client)
    metrics = _capture(monkeypatch)
    row = _preset_row(example_audio_url=_eleven_key())
    assert await _bake_qwen_only(row, storage) is None
    assert tts.open_calls == [] and storage.uploads == [] and row.example_audio_url == _eleven_key()
    assert metrics[0][0] == "example_tts_deferred"


async def test_qwen_only_defers_when_realtime_calls_hold_every_worker(monkeypatch) -> None:
    tts, storage = _wire(monkeypatch)
    _wire_qwen(monkeypatch, _FakeQwenClient())
    monkeypatch.setattr(example_service, "has_free_worker", lambda _s: False)
    assert await _bake_qwen_only(_preset_row(), storage) is None and tts.open_calls == []


def _wire_rounds(monkeypatch, healthy: list[bool] | None = None) -> None:
    states = list(healthy or [])

    class _Health:
        def __init__(self, _settings) -> None:
            pass

        async def healthy(self) -> bool:
            return states.pop(0) if states else True

    monkeypatch.setattr(example_service, "get_settings", lambda: _settings(True))
    monkeypatch.setattr(example_service, "worker_urls", lambda _s: ["http://w"])
    monkeypatch.setattr(example_service, "QwenTTSClient", _Health)


async def test_prebake_presets_retries_deferred_presets_then_stops(monkeypatch) -> None:
    _wire_rounds(monkeypatch)
    monkeypatch.setattr(example_service, "PRESET_MAP", {1: {}, 2: {}, 3: {}})
    results = {1: [True], 2: [False, True], 3: [False, False, True]}
    seen: list[int] = []
    sleeps: list[float] = []

    async def _one(_settings, sid):
        seen.append(sid)
        return results[sid].pop(0)

    async def _sleep(sec):
        sleeps.append(sec)

    monkeypatch.setattr(example_service, "_prebake_preset", _one)
    monkeypatch.setattr(example_service.asyncio, "sleep", _sleep)
    await example_service.prebake_presets()
    assert seen == [1, 2, 3, 2, 3, 3] and sleeps == [example_service._PRESET_RETRY_SEC] * 2


async def test_prebake_presets_gives_up_after_the_round_limit(monkeypatch) -> None:
    _wire_rounds(monkeypatch)
    monkeypatch.setattr(example_service, "PRESET_MAP", {1: {}})
    calls: list[int] = []

    async def _never(_settings, sid):
        calls.append(sid)
        return False

    async def _sleep(_sec):
        return None

    monkeypatch.setattr(example_service, "_prebake_preset", _never)
    monkeypatch.setattr(example_service.asyncio, "sleep", _sleep)
    await example_service.prebake_presets()
    assert len(calls) == example_service._PRESET_PREBAKE_ROUNDS


@pytest.mark.parametrize(
    "settings",
    [_settings(False), SimpleNamespace(**vars(_settings(True)), example_preset_prebake=False)],
    ids=["no-s3", "switched-off"],
)
async def test_prebake_presets_does_nothing_without_s3_or_when_off(monkeypatch, settings) -> None:
    monkeypatch.setattr(example_service, "get_settings", lambda: settings)
    monkeypatch.setattr(example_service, "worker_urls", lambda _s: ["http://w"])

    async def _boom(*_a):
        raise AssertionError("굽지 않아야 한다")

    monkeypatch.setattr(example_service, "_prebake_preset", _boom)
    await example_service.prebake_presets()


async def test_prebake_preset_bakes_with_qwen_only_and_reports_done(monkeypatch) -> None:
    row = _preset_row(example_audio_url=_eleven_key())
    _, storage, tts = _wire_prebake(monkeypatch, row, qwen=_FakeQwenClient())
    monkeypatch.setattr(example_service, "has_free_worker", lambda _s: True)
    monkeypatch.setattr(example_service, "PRESET_MAP", {1: {"example_dialogue": _DIALOGUE}})
    assert await example_service._prebake_preset(example_service.get_settings(), 1) is True
    assert row.example_audio_url == _qkey("ai") and tts.open_calls == []


async def test_prebake_preset_reports_deferred_and_swallows_errors(monkeypatch) -> None:
    row = _preset_row()
    _wire_prebake(monkeypatch, row)
    monkeypatch.setattr(example_service, "has_free_worker", lambda _s: True)
    monkeypatch.setattr(example_service, "PRESET_MAP", {1: {"example_dialogue": _DIALOGUE}})
    assert await example_service._prebake_preset(example_service.get_settings(), 1) is False

    async def _boom(*_a, **_k):
        raise RuntimeError("DB 끊김")

    monkeypatch.setattr(example_service, "_resolve_and_bake", _boom)
    assert await example_service._prebake_preset(example_service.get_settings(), 1) is False


class _AlwaysAbruptQwen(_FakeQwenClient):
    def _out(self, text: str) -> bytes:
        voice = self.calls[-1][0]
        return _voice_pcm(text, decay_ms=0.0 if voice == "user" else 40.0)


async def test_turn_that_stays_cut_is_counted_in_the_metric(monkeypatch) -> None:
    _wire(monkeypatch)
    _wire_qwen(monkeypatch, _AlwaysAbruptQwen())
    metrics = _capture(monkeypatch)
    await get_example_conversation(_FakeDB(_preset_row()), 1, user_id=7)
    assert (metrics[0][1]["clean_attempts"], metrics[0][1]["clean_residual_cuts"]) == (1 + MAX_ATTEMPTS, 1)


async def test_preset_baked_with_ai_after_missing_slot_is_a_cache_hit(monkeypatch) -> None:
    _, storage = _wire(monkeypatch)
    _flag_on(monkeypatch)
    qwen = _wire_qwen(monkeypatch, _FakeQwenClient())
    row = _preset_row(tts_voice_id="male_middle_soft", example_audio_url=_qkey("ai"))
    await get_example_conversation(_FakeDB(row), 1, user_id=7)
    assert qwen.calls == [] and storage.uploads == [], "보이스가 없어 ai 로 구운 예시는 요청마다 다시 굽지 않는다"


async def test_qwen_only_ignores_old_keys_even_for_custom_rows(monkeypatch) -> None:
    _, storage = _wire(monkeypatch)
    _wire_qwen(monkeypatch, _FakeQwenClient())
    row = _custom_row(tts_voice_id=None, example_dialogue=_DIALOGUE, example_audio_url=_old_qkey())
    assert await _bake_qwen_only(row, storage) == _qkey("ai")


async def test_prebake_presets_skips_a_round_when_qwen_is_down(monkeypatch) -> None:
    _wire_rounds(monkeypatch, healthy=[False, True])
    monkeypatch.setattr(example_service, "PRESET_MAP", {1: {}, 2: {}})
    seen: list[tuple[int, int]] = []
    sleeps: list[float] = []

    async def _one(_settings, sid):
        seen.append((len(sleeps), sid))
        return True

    async def _sleep(sec):
        sleeps.append(sec)

    monkeypatch.setattr(example_service, "_prebake_preset", _one)
    monkeypatch.setattr(example_service.asyncio, "sleep", _sleep)
    await example_service.prebake_presets()
    assert seen == [(1, 1), (1, 2)], "워커가 죽은 라운드는 프리셋마다 헬스체크하지 않고 통째로 넘긴다"


async def test_prebake_presets_does_nothing_without_qwen_workers(monkeypatch) -> None:
    monkeypatch.setattr(example_service, "get_settings", lambda: _settings(True))

    async def _boom(*_a):
        raise AssertionError("워커가 없으면 미리 굽기를 하지 않는다")

    monkeypatch.setattr(example_service, "_prebake_preset", _boom)
    await example_service.prebake_presets()


class _VoicesQwen(_FakeQwenClient):
    def __init__(self, voices) -> None:
        super().__init__()
        self.voices = None
        self._advertised = voices

    async def health_state(self) -> str:
        self.voices = self._advertised
        return "ready"


async def test_qwen_only_keeps_ai_fallback_while_the_slot_voice_is_still_missing(monkeypatch) -> None:
    _, storage = _wire(monkeypatch)
    _flag_on(monkeypatch)
    qwen = _wire_qwen(monkeypatch, _VoicesQwen(frozenset({"ai", "user"})))
    row = _preset_row(tts_voice_id="male_middle_soft", example_audio_url=_qkey("ai"))
    assert await _bake_qwen_only(row, storage) == _qkey("ai")
    assert qwen.calls == [] and storage.uploads == [], "서버를 켤 때마다 다시 굽지 않는다"


async def test_qwen_only_rebakes_ai_fallback_once_the_slot_voice_appears(monkeypatch) -> None:
    _, storage = _wire(monkeypatch)
    _flag_on(monkeypatch)
    qwen = _wire_qwen(monkeypatch, _VoicesQwen(frozenset({"ai", "user", "male_middle_soft"})))
    row = _preset_row(tts_voice_id="male_middle_soft", example_audio_url=_qkey("ai"))
    assert await _bake_qwen_only(row, storage) == _qkey("male_middle_soft")
    assert [v for v, _ in qwen.calls] == ["male_middle_soft", "user"]


async def test_disabled_qwen_is_reported_as_disabled_not_busy(monkeypatch) -> None:
    _wire(monkeypatch)
    _wire_qwen(monkeypatch, _FakeQwenClient(enabled=False, ready=False))
    monkeypatch.setattr(example_service, "has_free_worker", lambda _s: False)
    metrics = _capture(monkeypatch)
    await get_example_conversation(_FakeDB(_preset_row()), 1, user_id=7)
    assert metrics[0][1]["fallback_reason"] == "disabled"


async def test_qwen_example_joins_clean_turns_with_the_turn_gap(monkeypatch) -> None:
    _, storage = _wire(monkeypatch)
    _wire_qwen(monkeypatch, _FakeQwenClient())
    await get_example_conversation(_FakeDB(_preset_row()), 1, user_id=7)
    audio = storage.uploads[0][1]
    assert example_service._TURN_GAP_PCM in audio
    from app.services.example_audio import cut_tail_phrase, trim_lead

    turns = [trim_lead(cut_tail_phrase(_voice_pcm(t["text"] + TAIL_PHRASE), t["text"])[0]) for t in _DIALOGUE]
    assert audio == example_service._TURN_GAP_PCM.join(turns)


async def test_prebake_preset_shares_the_prebake_queue_and_waits_outside_its_timeout(monkeypatch) -> None:
    row = _preset_row(example_audio_url=_eleven_key())
    _wire_prebake(monkeypatch, row, qwen=_FakeQwenClient())
    monkeypatch.setattr(example_service, "has_free_worker", lambda _s: True)
    monkeypatch.setattr(example_service, "PRESET_MAP", {1: {"example_dialogue": _DIALOGUE}})
    monkeypatch.setattr(example_service, "_PRESET_PREBAKE_TIMEOUT_SEC", 0.2)
    metrics = _capture(monkeypatch)

    async with example_service._prebake_semaphore():
        task = asyncio.create_task(example_service._prebake_preset(example_service.get_settings(), 1))
        await asyncio.sleep(0.4)
        assert not task.done(), "커스텀 미리 굽기가 큐를 잡고 있으면 기다린다"
    assert await task is True, "큐 대기는 타임아웃에 들어가지 않는다"
    assert metrics[0][1]["trigger"] == "preset_prebake"


async def test_prebake_preset_gives_up_a_stuck_bake_at_its_timeout(monkeypatch) -> None:
    class _Stuck(_FakeQwenClient):
        async def synth(self, voice: str, text: str) -> bytes:
            await asyncio.sleep(10)
            return b""

    _wire_prebake(monkeypatch, _preset_row(), qwen=_Stuck())
    monkeypatch.setattr(example_service, "has_free_worker", lambda _s: True)
    monkeypatch.setattr(example_service, "PRESET_MAP", {1: {"example_dialogue": _DIALOGUE}})
    monkeypatch.setattr(example_service, "_PRESET_PREBAKE_TIMEOUT_SEC", 0.2)
    assert await asyncio.wait_for(example_service._prebake_preset(example_service.get_settings(), 1), 2.0) is False


async def test_qwen_only_rebakes_when_the_worker_does_not_report_voices(monkeypatch) -> None:
    _, storage = _wire(monkeypatch)
    _flag_on(monkeypatch)
    qwen = _wire_qwen(monkeypatch, _VoicesQwen(None))
    row = _preset_row(tts_voice_id="male_middle_soft", example_audio_url=_qkey("ai"))
    assert await _bake_qwen_only(row, storage) == _qkey("male_middle_soft") and qwen.calls


class _CountingVoicesQwen(_VoicesQwen):
    def __init__(self, voices, state: str = "ready") -> None:
        super().__init__(voices)
        self.health_calls = 0
        self._state = state

    async def health_state(self) -> str:
        self.health_calls += 1
        await super().health_state()
        return self._state

    async def healthy(self) -> bool:
        self.health_calls += 1
        return True


async def test_qwen_only_checks_worker_health_once_when_reusing_the_slot_check(monkeypatch) -> None:
    _, storage = _wire(monkeypatch)
    _flag_on(monkeypatch)
    qwen = _wire_qwen(monkeypatch, _CountingVoicesQwen(frozenset({"ai", "user", "male_middle_soft"})))
    row = _preset_row(tts_voice_id="male_middle_soft", example_audio_url=_qkey("ai"))
    assert await _bake_qwen_only(row, storage) == _qkey("male_middle_soft")
    assert qwen.health_calls == 1, "/health 는 한 번만"


@pytest.mark.parametrize("state", ["warming", "down", "unhealthy"])
async def test_qwen_only_defers_when_the_reused_health_state_is_not_ready(monkeypatch, state) -> None:
    tts, storage = _wire(monkeypatch)
    _flag_on(monkeypatch)
    qwen = _wire_qwen(monkeypatch, _CountingVoicesQwen(None, state=state))
    row = _preset_row(tts_voice_id="male_middle_soft", example_audio_url=_qkey("ai"))
    assert await _bake_qwen_only(row, storage) is None
    assert qwen.calls == [] and tts.open_calls == [] and qwen.health_calls == 1


async def test_qwen_only_still_bakes_when_the_reused_health_state_is_busy(monkeypatch) -> None:
    _, storage = _wire(monkeypatch)
    _flag_on(monkeypatch)
    qwen = _wire_qwen(monkeypatch, _CountingVoicesQwen(None, state="busy"))
    row = _preset_row(tts_voice_id="male_middle_soft", example_audio_url=_qkey("ai"))
    assert await _bake_qwen_only(row, storage) == _qkey("male_middle_soft") and qwen.health_calls == 1


async def test_prebake_presets_keeps_waiting_for_a_late_worker_without_using_the_budget(monkeypatch) -> None:
    down = example_service._PRESET_PREBAKE_ROUNDS + 5
    _wire_rounds(monkeypatch, healthy=[False] * down)
    monkeypatch.setattr(example_service, "PRESET_MAP", {1: {}})
    seen: list[int] = []
    sleeps: list[float] = []
    results = [False, True]

    async def _one(_settings, sid):
        seen.append(sid)
        return results.pop(0)

    async def _sleep(sec):
        sleeps.append(sec)

    monkeypatch.setattr(example_service, "_prebake_preset", _one)
    monkeypatch.setattr(example_service.asyncio, "sleep", _sleep)
    await example_service.prebake_presets()
    assert seen == [1, 1] and len(sleeps) == down + 1, "워커가 늦게 떠도 재시도 예산이 남아 있다"


async def test_prebake_presets_does_not_sleep_after_the_last_attempt(monkeypatch) -> None:
    _wire_rounds(monkeypatch)
    monkeypatch.setattr(example_service, "PRESET_MAP", {1: {}})
    sleeps: list[float] = []

    async def _never(_settings, _sid):
        return False

    async def _sleep(sec):
        sleeps.append(sec)

    monkeypatch.setattr(example_service, "_prebake_preset", _never)
    monkeypatch.setattr(example_service.asyncio, "sleep", _sleep)
    await example_service.prebake_presets()
    assert len(sleeps) == example_service._PRESET_PREBAKE_ROUNDS - 1


async def test_qwen_only_skips_the_health_check_when_every_worker_is_busy(monkeypatch) -> None:
    tts, storage = _wire(monkeypatch)
    _flag_on(monkeypatch)
    qwen = _wire_qwen(monkeypatch, _CountingVoicesQwen(frozenset({"ai", "user"})))
    monkeypatch.setattr(example_service, "has_free_worker", lambda _s: False)
    row = _preset_row(tts_voice_id="male_middle_soft", example_audio_url=_qkey("ai"))
    assert await _bake_qwen_only(row, storage) is None, "이번 라운드는 미루고 다음에 칸을 다시 확인"
    assert qwen.health_calls == 0 and qwen.calls == [] and tts.open_calls == []
