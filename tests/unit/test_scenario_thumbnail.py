"""썸네일 생성(저장)부터 목록 조회(presigned URL)까지 외부 호출을 스텁으로 대체해 검증한다."""

from datetime import datetime
from types import SimpleNamespace

import pytest

from app.core.enums import FileType, ScenarioCategory
from app.db.models import FileORM
from app.services import scenario_image_service, scenario_service
from app.services.scenario_image_service import (
    _build_image_prompt,
    _extract_image,
    _image_key,
    generate_scenario_thumbnail,
)
from app.services.scenario_service import get_scenarios

_BUCKET = "test-bucket"
_SCENE = "A friendly cafe owner standing behind a wooden counter, holding a phone"
_IMAGE_BYTES = b"\x89PNG\r\n\x1a\n-fake-image-bytes"


def _scenario_row(scenario_id: int = 101, user_id: int = 1, **kw) -> SimpleNamespace:
    return SimpleNamespace(
        scenario_id=scenario_id,
        title=kw.get("title", "카페 단체 예약 문의"),
        content="단체 예약이 가능한지 물어본다",
        category=kw.get("category", ScenarioCategory.OTHER.value),
        scenario_image=kw.get("scenario_image"),
        tts_voice_id=None,
        ai_prompt="prompt",
        user_id=user_id,
        is_custom=True,
        is_warmup=False,
        call_target="동네 카페 사장님",
        call_purpose="단체 예약 가능 여부 확인",
        script=[],
        created_at=datetime(2026, 1, 1),
    )


# --- 외부 의존성 스텁 ---


class _FakeS3:
    def __init__(self) -> None:
        self.puts: list[dict] = []
        self.signed: list[tuple[str, int]] = []

    def put_object(self, **kwargs) -> dict:
        self.puts.append(kwargs)
        return {}

    def generate_presigned_url(self, _op: str, Params: dict, ExpiresIn: int) -> str:  # noqa: N803
        self.signed.append((Params["Key"], ExpiresIn))
        return f"https://s3.example.com/{Params['Key']}?sig=abc&expires={ExpiresIn}"


class _FakeFileResult:
    def __init__(self, existing: int | None) -> None:
        self._existing = existing

    def scalar_one_or_none(self) -> int | None:
        return self._existing


class _FakeSession:
    """_generate_and_store가 쓰는 get/execute/add/commit만 흉내낸다."""

    def __init__(self, row: SimpleNamespace | None, existing_file_id: int | None = None) -> None:
        self.row = row
        self.existing_file_id = existing_file_id
        self.added: list[object] = []
        self.commits = 0

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *_exc) -> bool:
        return False

    async def get(self, _model: type, _pk: int) -> SimpleNamespace | None:
        return self.row

    async def execute(self, _stmt: object) -> _FakeFileResult:
        return _FakeFileResult(self.existing_file_id)

    def add(self, obj: object) -> None:
        self.added.append(obj)

    async def commit(self) -> None:
        self.commits += 1


def _fake_anthropic(scene: str = _SCENE, calls: list | None = None):
    class _Messages:
        async def create(self, **kwargs):
            if calls is not None:
                calls.append(kwargs)
            return SimpleNamespace(content=[SimpleNamespace(text=scene)])

    class _Client:
        def __init__(self, api_key=None) -> None:
            self.messages = _Messages()

    return _Client


def _fake_genai(mime: str = "image/png", calls: list | None = None, lead_text: bool = True):
    """텍스트 파트가 먼저 오는 실제 응답 형태를 재현한다."""

    parts = []
    if lead_text:
        parts.append(SimpleNamespace(inline_data=None, text="Here is your image"))
    parts.append(SimpleNamespace(inline_data=SimpleNamespace(data=_IMAGE_BYTES, mime_type=mime)))
    response = SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))]
    )

    class _Models:
        async def generate_content(self, **kwargs):
            if calls is not None:
                calls.append(kwargs)
            return response

    class _Client:
        def __init__(self, api_key=None) -> None:
            self.aio = SimpleNamespace(models=_Models())

    return SimpleNamespace(Client=_Client)


@pytest.fixture
def stub_settings(monkeypatch):
    from app.core.config import get_settings

    settings = get_settings().model_copy(update={"s3_bucket": _BUCKET})
    monkeypatch.setattr(scenario_image_service, "get_settings", lambda: settings)
    monkeypatch.setattr(scenario_service, "get_settings", lambda: settings)
    return settings


@pytest.fixture
def fake_s3(monkeypatch) -> _FakeS3:
    client = _FakeS3()
    monkeypatch.setattr(scenario_image_service, "_s3_client", lambda _settings: client)
    return client


# --- 순수 함수 ---


def test_extract_image_skips_leading_text_part() -> None:
    parts = [
        SimpleNamespace(inline_data=None, text="preamble"),
        SimpleNamespace(inline_data=SimpleNamespace(data=_IMAGE_BYTES, mime_type="image/png")),
    ]
    fake = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))])

    data, mime = _extract_image(fake)

    assert data == _IMAGE_BYTES
    assert mime == "image/png"


def test_extract_image_raises_when_no_image_part() -> None:
    parts = [SimpleNamespace(inline_data=None, text="sorry, no image")]
    fake = SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))])

    with pytest.raises(ValueError, match="이미지 파트가 없습니다"):
        _extract_image(fake)


def test_image_key_changes_with_prompt_and_mime() -> None:
    png = _image_key(7, _build_image_prompt("scene A"), "image/png")
    jpg = _image_key(7, _build_image_prompt("scene A"), "image/jpeg")
    other = _image_key(7, _build_image_prompt("scene B"), "image/png")

    assert png.startswith("scenario-images/7-") and png.endswith(".png")
    assert jpg.endswith(".jpg")
    assert png != other, "프롬프트가 바뀌면 키도 바뀌어야 캐시가 무효화된다"


# --- 생성 → 저장 ---


async def test_thumbnail_generation_stores_key_and_registers_file(
    monkeypatch, stub_settings, fake_s3
) -> None:
    row = _scenario_row()
    session = _FakeSession(row)
    image_calls: list = []

    monkeypatch.setattr(scenario_image_service, "AsyncSessionLocal", lambda: session)
    monkeypatch.setattr(scenario_image_service, "AsyncAnthropic", _fake_anthropic())
    monkeypatch.setattr(scenario_image_service, "genai", _fake_genai(calls=image_calls))

    await generate_scenario_thumbnail(row.scenario_id)

    # S3에 실제 바이트가 올라갔는가
    assert len(fake_s3.puts) == 1
    put = fake_s3.puts[0]
    assert put["Bucket"] == _BUCKET
    assert put["Body"] == _IMAGE_BYTES
    assert put["ContentType"] == "image/png"
    assert put["Key"].startswith("scenario-images/101-")

    # 시나리오 행에 key가 저장됐는가
    assert row.scenario_image == put["Key"]
    assert session.commits == 1

    # file 레지스트리에 등록됐는가
    files = [obj for obj in session.added if isinstance(obj, FileORM)]
    assert len(files) == 1
    assert files[0].s3_key == put["Key"]
    assert files[0].file_type == FileType.SCENARIO_PROFILE
    assert files[0].title == row.title

    # 이미지 프롬프트에 스타일이 붙었는가
    assert _SCENE in image_calls[0]["contents"]
    assert "storybook" in image_calls[0]["contents"]
    assert image_calls[0]["model"] == stub_settings.gemini_image_model


async def test_thumbnail_skips_when_image_already_exists(
    monkeypatch, stub_settings, fake_s3
) -> None:
    row = _scenario_row(scenario_image="scenario-images/101-deadbeef.png")
    session = _FakeSession(row)

    monkeypatch.setattr(scenario_image_service, "AsyncSessionLocal", lambda: session)
    monkeypatch.setattr(scenario_image_service, "AsyncAnthropic", _fake_anthropic())
    monkeypatch.setattr(scenario_image_service, "genai", _fake_genai())

    await generate_scenario_thumbnail(row.scenario_id)

    assert fake_s3.puts == []
    assert session.commits == 0
    assert row.scenario_image == "scenario-images/101-deadbeef.png"


async def test_duplicate_s3_key_does_not_insert_second_file_row(
    monkeypatch, stub_settings, fake_s3
) -> None:
    row = _scenario_row()
    session = _FakeSession(row, existing_file_id=42)

    monkeypatch.setattr(scenario_image_service, "AsyncSessionLocal", lambda: session)
    monkeypatch.setattr(scenario_image_service, "AsyncAnthropic", _fake_anthropic())
    monkeypatch.setattr(scenario_image_service, "genai", _fake_genai())

    await generate_scenario_thumbnail(row.scenario_id)

    assert [obj for obj in session.added if isinstance(obj, FileORM)] == []
    assert row.scenario_image is not None
    assert session.commits == 1


async def test_image_model_failure_is_swallowed_and_leaves_row_untouched(
    monkeypatch, stub_settings, fake_s3
) -> None:
    row = _scenario_row()
    session = _FakeSession(row)

    class _BoomModels:
        async def generate_content(self, **_kwargs):
            raise RuntimeError("gemini down")

    class _BoomClient:
        def __init__(self, api_key=None) -> None:
            self.aio = SimpleNamespace(models=_BoomModels())

    monkeypatch.setattr(scenario_image_service, "AsyncSessionLocal", lambda: session)
    monkeypatch.setattr(scenario_image_service, "AsyncAnthropic", _fake_anthropic())
    monkeypatch.setattr(scenario_image_service, "genai", SimpleNamespace(Client=_BoomClient))

    await generate_scenario_thumbnail(row.scenario_id)  # 예외가 밖으로 새지 않아야 한다

    assert row.scenario_image is None
    assert fake_s3.puts == []
    assert session.commits == 0


async def test_no_bucket_skips_generation_entirely(monkeypatch) -> None:
    from app.core.config import get_settings

    settings = get_settings().model_copy(update={"s3_bucket": None})
    monkeypatch.setattr(scenario_image_service, "get_settings", lambda: settings)

    def _boom(*_a, **_kw):
        raise AssertionError("버킷이 없으면 세션조차 열지 않아야 한다")

    monkeypatch.setattr(scenario_image_service, "AsyncSessionLocal", _boom)

    await generate_scenario_thumbnail(101)


# --- 조회 → presigned URL ---


class _FakeDB:
    def __init__(self, rows: list) -> None:
        self._rows = rows

    async def execute(self, _stmt: object):
        rows = self._rows

        class _R:
            def scalars(self):
                return self

            def all(self):
                return rows

        return _R()


def _patch_storage(monkeypatch, url: str | None = "https://signed", boom: bool = False):
    calls: list = []

    class _Storage:
        def __init__(self, _settings) -> None:
            calls.append("constructed")

        def presigned_url(self, key: str, expires_in: int = 600) -> str | None:
            if boom:
                raise RuntimeError("signature failed")
            calls.append((key, expires_in))
            return url

    monkeypatch.setattr(scenario_service, "RecordingStorageService", _Storage)
    return calls


async def test_list_returns_presigned_url_for_stored_key(monkeypatch, stub_settings) -> None:
    key = "scenario-images/101-abcd1234.png"
    calls = _patch_storage(monkeypatch, url=f"https://s3.example.com/{key}?sig=x")
    rows = [_scenario_row(scenario_image=key)]

    resp = await get_scenarios(_FakeDB(rows), None, user_id=1)

    assert len(resp.scenarios) == 1
    assert resp.scenarios[0].scenario_image == f"https://s3.example.com/{key}?sig=x"
    assert (key, scenario_service._IMAGE_URL_TTL_SEC) in calls


async def test_list_returns_null_image_when_key_missing(monkeypatch, stub_settings) -> None:
    calls = _patch_storage(monkeypatch)
    rows = [_scenario_row(scenario_image=None)]

    resp = await get_scenarios(_FakeDB(rows), None, user_id=1)

    assert resp.scenarios[0].scenario_image is None
    assert calls == [], "이미지가 하나도 없으면 S3 클라이언트를 만들지 않아야 한다"


async def test_list_survives_signing_failure(monkeypatch, stub_settings) -> None:
    _patch_storage(monkeypatch, boom=True)
    rows = [_scenario_row(scenario_image="scenario-images/101-abcd1234.png")]

    resp = await get_scenarios(_FakeDB(rows), None, user_id=1)

    assert resp.scenarios[0].scenario_image is None, "서명 실패해도 목록 조회는 살아 있어야 한다"


# --- 생성부터 조회까지 한 흐름 ---


async def test_generated_thumbnail_flows_into_list_response(
    monkeypatch, stub_settings, fake_s3
) -> None:
    row = _scenario_row()
    session = _FakeSession(row)

    monkeypatch.setattr(scenario_image_service, "AsyncSessionLocal", lambda: session)
    monkeypatch.setattr(scenario_image_service, "AsyncAnthropic", _fake_anthropic())
    monkeypatch.setattr(scenario_image_service, "genai", _fake_genai())

    await generate_scenario_thumbnail(row.scenario_id)
    stored_key = row.scenario_image
    assert stored_key is not None

    calls = _patch_storage(monkeypatch, url=f"https://s3.example.com/{stored_key}?sig=x")

    resp = await get_scenarios(_FakeDB([row]), None, user_id=row.user_id)

    assert resp.scenarios[0].scenario_image == f"https://s3.example.com/{stored_key}?sig=x"
    assert (stored_key, scenario_service._IMAGE_URL_TTL_SEC) in calls
