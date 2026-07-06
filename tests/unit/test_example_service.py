import json
from types import SimpleNamespace

import pytest

from app.core.tts_voices import EXAMPLE_USER_VOICE_ID, VOICE_REGISTRY, pick_example_user_voice
from app.services import example_service
from app.services.example_service import ScenarioNotFoundError, get_example_conversation

# --- user 역할 보이스 선택 ---


def test_user_voice_default_when_no_collision() -> None:
    assert pick_example_user_voice("some-ai-voice") == EXAMPLE_USER_VOICE_ID


def test_user_voice_avoids_collision_with_ai_voice() -> None:
    picked = pick_example_user_voice(EXAMPLE_USER_VOICE_ID)
    assert picked != EXAMPLE_USER_VOICE_ID
    assert picked in {v.voice_id for v in VOICE_REGISTRY}


# --- 페이크 ---


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

    def upload_wav(self, key: str, pcm: bytes) -> str | None:
        self.uploads.append((key, pcm))
        return f"https://bucket.s3.test/{key}"


class _FakeDB:
    def __init__(self, row: object | None = None) -> None:
        self._row = row
        self.commits = 0

    async def get(self, _model: object, _pk: int) -> object | None:
        return self._row

    async def commit(self) -> None:
        self.commits += 1


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


# --- 프리셋: 생성/캐시 ---


async def test_preset_first_call_generates_audio(monkeypatch) -> None:
    tts, storage = _wire(monkeypatch)
    row = _preset_row()
    db = _FakeDB(row)

    resp = await get_example_conversation(db, 1, user_id=7)

    assert [t.speaker for t in resp.dialogue] == ["ai", "user"]
    assert resp.audio_url and resp.audio_url.startswith("https://")
    assert tts.open_calls == ["voice-default", pick_example_user_voice("voice-default")]
    assert len(storage.uploads) == 1
    key, _pcm = storage.uploads[0]
    assert key.startswith("examples/1-") and key.endswith(".wav")
    assert row.example_audio_url == resp.audio_url
    assert db.commits >= 1


async def test_audio_concatenates_turns_with_silence_gap(monkeypatch) -> None:
    tts = _FakeTTSClient(pcm_per_turn=b"\x01\x02")
    _, storage = _wire(monkeypatch, tts=tts)

    await get_example_conversation(_FakeDB(_preset_row()), 1, user_id=7)

    _key, pcm = storage.uploads[0]
    gap = int(16_000 * 0.4) * 2  # 400ms, 16kHz 16bit mono
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

    assert resp.audio_url
    assert [t.speaker for t in resp.dialogue] == ["ai", "user"]


async def test_s3_unset_returns_dialogue_without_audio(monkeypatch) -> None:
    tts, storage = _wire(monkeypatch, s3=False)

    resp = await get_example_conversation(_FakeDB(_preset_row()), 1, user_id=7)

    assert resp.audio_url is None
    assert [t.text for t in resp.dialogue]
    assert tts.open_calls == []
    assert storage.uploads == []


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
