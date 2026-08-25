import asyncio
import json
import random
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI

import app.core.tts_voices as tts_voices
from app.api.v1 import internal
from app.core.enums import SpeakerAge, SpeakerGender, SpeakerTone
from app.core.security import require_internal_secret
from app.core.tts_voices import VoiceProfile, parse_speaker, pick_voice_id
from app.deps.db import get_db
from app.schemas.llm import AiEmotion, LLMEvent, LLMEventType
from app.schemas.scenario import CustomSessionRequest
from app.services import scenario_service
from app.services.pipeline import VoicePipeline, _State, _TurnTimings
from app.services.scenario_service import create_custom_scenario
from app.services.tts import ElevenLabsTTSClient


def test_pick_voice_matches_tags(monkeypatch) -> None:
    test_registry = [
        VoiceProfile(
            "voice_male_middle", SpeakerGender.MALE, SpeakerAge.MIDDLE,
            SpeakerTone.NEUTRAL, "테스트 보이스",
        ),
        VoiceProfile(
            "voice_female_young", SpeakerGender.FEMALE, SpeakerAge.YOUNG,
            SpeakerTone.NEUTRAL, "다른 보이스",
        ),
    ]
    monkeypatch.setattr(tts_voices, "VOICE_REGISTRY", test_registry)

    result = pick_voice_id(SpeakerGender.MALE, SpeakerAge.MIDDLE)
    assert result == "voice_male_middle"


def test_pick_voice_random_among_candidates(monkeypatch) -> None:
    test_registry = [
        VoiceProfile(
            "voice_female_young_1", SpeakerGender.FEMALE, SpeakerAge.YOUNG,
            SpeakerTone.NEUTRAL, "첫번째",
        ),
        VoiceProfile(
            "voice_female_young_2", SpeakerGender.FEMALE, SpeakerAge.YOUNG,
            SpeakerTone.NEUTRAL, "두번째",
        ),
    ]
    monkeypatch.setattr(tts_voices, "VOICE_REGISTRY", test_registry)

    rng = random.Random(0)
    result1 = pick_voice_id(SpeakerGender.FEMALE, SpeakerAge.YOUNG, rng=rng)

    assert result1 in ["voice_female_young_1", "voice_female_young_2"]

    rng2 = random.Random(0)
    result2 = pick_voice_id(SpeakerGender.FEMALE, SpeakerAge.YOUNG, rng=rng2)
    assert result2 == result1


def test_pick_voice_no_match_returns_none(monkeypatch) -> None:
    monkeypatch.setattr(tts_voices, "VOICE_REGISTRY", [])
    assert pick_voice_id(SpeakerGender.MALE, SpeakerAge.MIDDLE) is None

    test_registry = [
        VoiceProfile(
            "voice_male_old", SpeakerGender.MALE, SpeakerAge.OLD,
            SpeakerTone.NEUTRAL, "늙은 남성",
        ),
    ]
    monkeypatch.setattr(tts_voices, "VOICE_REGISTRY", test_registry)
    result = pick_voice_id(SpeakerGender.FEMALE, SpeakerAge.YOUNG)
    assert result is None


def test_pick_voice_missing_axis_returns_none(monkeypatch) -> None:

    test_registry = [
        VoiceProfile(
            "voice_male_young", SpeakerGender.MALE, SpeakerAge.YOUNG,
            SpeakerTone.NEUTRAL, "테스트",
        ),
    ]
    monkeypatch.setattr(tts_voices, "VOICE_REGISTRY", test_registry)

    assert pick_voice_id(None, SpeakerAge.YOUNG) is None

    assert pick_voice_id(SpeakerGender.MALE, None) is None

    assert pick_voice_id(None, None) is None


def test_parse_speaker_defensive() -> None:
    gender, age, tone = parse_speaker({"gender": "male", "age": "young"})
    assert gender == SpeakerGender.MALE
    assert age == SpeakerAge.YOUNG
    assert tone is None

    gender, age, tone = parse_speaker({"gender": "FEMALE", "age": " Old "})
    assert gender == SpeakerGender.FEMALE
    assert age == SpeakerAge.OLD
    assert tone is None

    gender, age, tone = parse_speaker({"gender": "robot", "age": "young"})
    assert gender is None
    assert age == SpeakerAge.YOUNG
    assert tone is None

    gender, age, tone = parse_speaker({})
    assert gender is None
    assert age is None
    assert tone is None

    gender, age, tone = parse_speaker("male")
    assert gender is None
    assert age is None
    assert tone is None

    gender, age, tone = parse_speaker(None)
    assert gender is None
    assert age is None
    assert tone is None


def test_pick_voice_prefers_exact_tone(monkeypatch) -> None:
    test_registry = [
        VoiceProfile("a", SpeakerGender.FEMALE, SpeakerAge.YOUNG, SpeakerTone.SOFT, "부드러운 젊은 여성"),
        VoiceProfile("b", SpeakerGender.FEMALE, SpeakerAge.YOUNG, SpeakerTone.ROUGH, "거친 젊은 여성"),
    ]
    monkeypatch.setattr(tts_voices, "VOICE_REGISTRY", test_registry)

    result = pick_voice_id(SpeakerGender.FEMALE, SpeakerAge.YOUNG, SpeakerTone.SOFT)
    assert result == "a"


def test_pick_voice_tone_relaxes_to_two_axis(monkeypatch) -> None:
    test_registry = [
        VoiceProfile("n", SpeakerGender.FEMALE, SpeakerAge.YOUNG, SpeakerTone.NEUTRAL, "보통 젊은 여성"),
    ]
    monkeypatch.setattr(tts_voices, "VOICE_REGISTRY", test_registry)

    result = pick_voice_id(SpeakerGender.FEMALE, SpeakerAge.YOUNG, SpeakerTone.ROUGH)
    assert result == "n"


def test_pick_voice_tone_none_uses_full_pool(monkeypatch) -> None:
    test_registry = [
        VoiceProfile("a", SpeakerGender.FEMALE, SpeakerAge.YOUNG, SpeakerTone.SOFT, "부드러운 젊은 여성"),
        VoiceProfile("b", SpeakerGender.FEMALE, SpeakerAge.YOUNG, SpeakerTone.ROUGH, "거친 젊은 여성"),
    ]
    monkeypatch.setattr(tts_voices, "VOICE_REGISTRY", test_registry)

    rng = random.Random(0)
    result1 = pick_voice_id(SpeakerGender.FEMALE, SpeakerAge.YOUNG, None, rng=rng)
    assert result1 in ["a", "b"]

    rng2 = random.Random(0)
    result2 = pick_voice_id(SpeakerGender.FEMALE, SpeakerAge.YOUNG, None, rng=rng2)
    assert result2 == result1


def test_parse_speaker_tone_defensive() -> None:
    gender, age, tone = parse_speaker({"gender": "male", "age": "young", "tone": "soft"})
    assert gender == SpeakerGender.MALE
    assert age == SpeakerAge.YOUNG
    assert tone == SpeakerTone.SOFT

    gender, age, tone = parse_speaker({"gender": "male", "age": "young", "tone": "ROUGH "})
    assert tone == SpeakerTone.ROUGH

    gender, age, tone = parse_speaker({"gender": "male", "age": "young", "tone": "velvet"})
    assert gender == SpeakerGender.MALE
    assert age == SpeakerAge.YOUNG
    assert tone is None

    gender, age, tone = parse_speaker({"gender": "male", "age": "young"})
    assert tone is None


class _FakeWriteDB:
    def __init__(self) -> None:
        self.added: object | None = None

    def add(self, obj: object) -> None:
        self.added = obj

    async def commit(self) -> None:
        pass

    async def refresh(self, obj: object) -> None:
        obj.scenario_id = 123


def _make_fake_anthropic_client(payload: dict) -> type:
    class _FakeAnthropicMessages:
        async def create(self, **_kw) -> SimpleNamespace:
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(payload))])

    class _FakeAnthropicClient:
        def __init__(self, api_key: str) -> None:
            self.messages = _FakeAnthropicMessages()

    return _FakeAnthropicClient


def _request() -> CustomSessionRequest:
    return CustomSessionRequest(
        title="집주인 통화 연습",
        call_target="집주인",
        call_purpose="월세 납부일 조정 문의",
    )


def _apply_settings(monkeypatch) -> None:
    monkeypatch.setattr(
        scenario_service,
        "get_settings",
        lambda: SimpleNamespace(anthropic_api_key="k", llm_analysis_model="m"),
    )


async def test_custom_scenario_assigns_voice(monkeypatch) -> None:
    _apply_settings(monkeypatch)
    monkeypatch.setattr(
        tts_voices,
        "VOICE_REGISTRY",
        [VoiceProfile("v-test", SpeakerGender.MALE, SpeakerAge.MIDDLE, SpeakerTone.NEUTRAL, "테스트")],
    )
    payload = {
        "content": "테스트 시나리오",
        "ai_prompt": "You are a clerk.",
        "script": [{"step": 1, "ai_goal": "인사", "hint": "인사하세요"}],
        "speaker": {"gender": "male", "age": "middle"},
    }
    monkeypatch.setattr(
        scenario_service, "AsyncAnthropic", _make_fake_anthropic_client(payload)
    )
    db = _FakeWriteDB()

    response = await create_custom_scenario(db, _request(), user_id=1)

    assert db.added.tts_voice_id == "v-test"
    assert response.scenario.tts_voice_id == "v-test"


async def test_custom_scenario_assigns_voice_with_tone(monkeypatch) -> None:
    _apply_settings(monkeypatch)
    monkeypatch.setattr(
        tts_voices,
        "VOICE_REGISTRY",
        [
            VoiceProfile(
                "v-rough", SpeakerGender.MALE, SpeakerAge.MIDDLE,
                SpeakerTone.ROUGH, "거친 중년 남성",
            ),
            VoiceProfile(
                "v-soft", SpeakerGender.MALE, SpeakerAge.MIDDLE,
                SpeakerTone.SOFT, "부드러운 중년 남성",
            ),
        ],
    )
    payload = {
        "content": "테스트 시나리오",
        "ai_prompt": "You are a clerk.",
        "script": [{"step": 1, "ai_goal": "인사", "hint": "인사하세요"}],
        "speaker": {"gender": "male", "age": "middle", "tone": "rough"},
    }
    monkeypatch.setattr(
        scenario_service, "AsyncAnthropic", _make_fake_anthropic_client(payload)
    )
    db = _FakeWriteDB()

    response = await create_custom_scenario(db, _request(), user_id=1)

    assert db.added.tts_voice_id == "v-rough"
    assert response.scenario.tts_voice_id == "v-rough"


async def test_custom_scenario_speaker_missing_none(monkeypatch) -> None:
    _apply_settings(monkeypatch)
    monkeypatch.setattr(
        tts_voices,
        "VOICE_REGISTRY",
        [VoiceProfile("v-test", SpeakerGender.MALE, SpeakerAge.MIDDLE, SpeakerTone.NEUTRAL, "테스트")],
    )
    payload = {
        "content": "테스트 시나리오",
        "ai_prompt": "You are a clerk.",
        "script": [{"step": 1, "ai_goal": "인사", "hint": "인사하세요"}],
    }
    monkeypatch.setattr(
        scenario_service, "AsyncAnthropic", _make_fake_anthropic_client(payload)
    )
    db = _FakeWriteDB()

    response = await create_custom_scenario(db, _request(), user_id=1)

    assert db.added.tts_voice_id is None
    assert response.scenario.tts_voice_id is None


async def test_custom_scenario_bad_speaker_none(monkeypatch) -> None:
    _apply_settings(monkeypatch)
    monkeypatch.setattr(
        tts_voices,
        "VOICE_REGISTRY",
        [VoiceProfile("v-test", SpeakerGender.MALE, SpeakerAge.MIDDLE, SpeakerTone.NEUTRAL, "테스트")],
    )
    payload = {
        "content": "테스트 시나리오",
        "ai_prompt": "You are a clerk.",
        "script": [{"step": 1, "ai_goal": "인사", "hint": "인사하세요"}],
        "speaker": {"gender": "robot", "age": 3},
    }
    monkeypatch.setattr(
        scenario_service, "AsyncAnthropic", _make_fake_anthropic_client(payload)
    )
    db = _FakeWriteDB()

    response = await create_custom_scenario(db, _request(), user_id=1)

    assert db.added.tts_voice_id is None
    assert response.scenario.tts_voice_id is None


class _FakeSessionInternal:

    def __init__(self, row: object | None) -> None:
        self._row = row

    async def get(self, _model: object, _pk: int) -> object | None:
        return self._row


def _make_app_internal(db_row: object | None = None) -> FastAPI:
    app = FastAPI()
    app.include_router(internal.router, prefix="/internal/v1")
    app.dependency_overrides[require_internal_secret] = lambda: None

    async def _fake_db():
        yield _FakeSessionInternal(db_row)

    app.dependency_overrides[get_db] = _fake_db
    return app


async def _get_internal(app: FastAPI, path: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path)


async def test_context_custom_includes_tts_voice_id() -> None:
    row = SimpleNamespace(
        title="커스텀 시나리오",
        call_target="역할 담당자",
        ai_prompt="You are a custom role.",
        is_custom=True,
        tts_voice_id="v-custom",
        script=[
            {"step": 1, "ai_goal": "인사", "hint": "인사하세요"},
        ],
    )
    resp = await _get_internal(_make_app_internal(db_row=row), "/internal/v1/scenarios/9999/context")

    assert resp.status_code == 200
    body = resp.json()
    assert body["ttsVoiceId"] == "v-custom"


async def test_context_preset_tts_voice_id_null() -> None:
    resp = await _get_internal(_make_app_internal(db_row=None), "/internal/v1/scenarios/1/context")

    assert resp.status_code == 200
    body = resp.json()
    assert "ttsVoiceId" in body
    assert body["ttsVoiceId"] is None


def _fake_tts_settings() -> SimpleNamespace:
    return SimpleNamespace(
        elevenlabs_api_key="key",
        elevenlabs_voice_id="voice-default",
        elevenlabs_model="eleven_flash_v2_5",
        elevenlabs_output_format="pcm_16000",
        elevenlabs_language_code="ko",
        elevenlabs_apply_text_normalization="on",
        elevenlabs_auto_mode=True,
        elevenlabs_ws_host="wss://api.elevenlabs.io",
        elevenlabs_stability=0.5,
        elevenlabs_similarity_boost=0.75,
        elevenlabs_style=0.0,
        elevenlabs_speaker_boost=False,
        elevenlabs_speed=1.0,
    )


def test_build_uri_override_and_default() -> None:
    client = ElevenLabsTTSClient(_fake_tts_settings())

    overridden = client._build_uri("v-x")
    assert "/text-to-speech/v-x/" in overridden

    default = client._build_uri()
    assert "/text-to-speech/voice-default/" in default


class _FakeTTSSessionVoice:
    async def begin(self, emotion) -> None:
        pass

    async def stream(self, text_source):
        async for _ in text_source:
            pass
        yield b"\x00\x01"

    async def aclose(self) -> None:
        pass


class _RecordingFakeTTSClient:

    def __init__(self) -> None:
        self.open_calls: list[str | None] = []

    async def open(self, voice_id: str | None = None):
        self.open_calls.append(voice_id)
        return _FakeTTSSessionVoice()


class _FailOnOverrideFakeTTSClient:

    def __init__(self) -> None:
        self.open_calls: list[str | None] = []

    async def open(self, voice_id: str | None = None):
        self.open_calls.append(voice_id)
        if voice_id is not None:
            raise RuntimeError("voice_id 연결 실패")
        return _FakeTTSSessionVoice()


class _HappyLLMVoice:

    async def stream(self, ctx):
        yield LLMEvent(type=LLMEventType.EMOTION_RESOLVED, emotion=AiEmotion.NEUTRAL)
        yield LLMEvent(type=LLMEventType.TEXT_DELTA, text="네")
        yield LLMEvent(type=LLMEventType.TURN_END)


def _make_voice_pipeline(session: dict, llm=None, tts=None) -> VoicePipeline:
    p = VoicePipeline.__new__(VoicePipeline)
    p._ws = SimpleNamespace(
        send_json=_noop_send_json, send_bytes=_noop_send_bytes
    )
    p._session_id = "sess-voice"
    p._session = session
    p._llm = llm
    p._tts = tts
    p._state = _State.LISTENING
    p._history = []
    p._current_step = 1
    p._ws_alive = True
    p._time_up = False
    p._closing = asyncio.Event()
    p._turn_task = None
    p._listening_since = None
    p._silence_total = 0.0
    p._tremor_buf = bytearray()
    p._user_turn_intervals = []
    p._turn_open_at = None
    p._script_len = 0
    p._ai_pcm_bytes = 0
    p._server_wait_duration_ms = 0
    p._completed_script_steps = 0

    scenario = session.get("scenario") or {}
    raw_voice = scenario.get("ttsVoiceId") if isinstance(scenario, dict) else None
    p._voice_id_override = (
        raw_voice if isinstance(raw_voice, str) and raw_voice.strip() else None
    )
    return p


async def _noop_send_json(payload: dict) -> None:
    pass


async def _noop_send_bytes(data: bytes) -> None:
    pass


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        ({"ttsVoiceId": "v-abc"}, "v-abc"),
        ({"ttsVoiceId": ""}, None),
        ({"ttsVoiceId": "   "}, None),
        ({"ttsVoiceId": 123}, None),
        ({}, None),
        (None, None),
    ],
)
def test_pipeline_parses_session_tts_voice(scenario, expected) -> None:
    session = {"scenario": scenario} if scenario is not None else {}
    p = _make_voice_pipeline(session)
    assert p._voice_id_override == expected


@pytest.mark.asyncio
async def test_pipeline_passes_override_to_tts_open() -> None:
    tts = _RecordingFakeTTSClient()
    p = _make_voice_pipeline(
        {"scenario": {"ttsVoiceId": "v-abc"}}, llm=_HappyLLMVoice(), tts=tts
    )
    p._state = _State.THINKING

    await asyncio.wait_for(
        p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0
    )

    assert tts.open_calls == ["v-abc"]


@pytest.mark.asyncio
async def test_pipeline_falls_back_to_default_on_open_failure() -> None:
    tts = _FailOnOverrideFakeTTSClient()
    p = _make_voice_pipeline(
        {"scenario": {"ttsVoiceId": "v-abc"}}, llm=_HappyLLMVoice(), tts=tts
    )
    p._state = _State.THINKING

    await asyncio.wait_for(
        p._run_turn("여보세요", _TurnTimings(final_at=0.0)), timeout=2.0
    )

    assert tts.open_calls == ["v-abc", None]
    assert p._history[-1] == {"role": "assistant", "text": "네"}
