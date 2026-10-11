import json
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import httpx
import jwt
import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.core.config import get_settings
from app.db.external import users_table
from app.deps.db import get_db
from app.deps.inquiry import get_inquiry_service
from app.main import create_app
from app.services.inquiry import InquiryService

_WEBHOOK_URL = "https://discord.com/api/webhooks/123/secret-token"


def _auth_headers(user_id: int = 7) -> dict[str, str]:
    settings = get_settings()
    token = jwt.encode(
        {"sub": str(user_id), "type": "ACCESS", "exp": datetime.now(UTC) + timedelta(minutes=5)},
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
async def inquiry_env():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", poolclass=StaticPool)
    async with engine.begin() as conn:
        await conn.run_sync(users_table.create)
        await conn.execute(
            users_table.insert(),
            [
                {"user_id": 7, "email": "user7@example.com", "status": "ACTIVE"},
                {"user_id": 8, "email": "user8@example.com", "status": "ACTIVE"},
            ],
        )

    sessions = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async def _get_db():
        async with sessions() as session:
            yield session

    requests: list[httpx.Request] = []

    def _receive(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"id": "123"})

    app = create_app()
    app.dependency_overrides[get_db] = _get_db
    service = InquiryService(_WEBHOOK_URL, transport=httpx.MockTransport(_receive))
    app.dependency_overrides[get_inquiry_service] = lambda: service
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            yield SimpleNamespace(client=client, app=app, requests=requests, sessions=sessions)
    finally:
        await engine.dispose()


@pytest.mark.parametrize("user_id", [7, 8])
async def test_inquiry_sends_authenticated_user_id_and_database_email(inquiry_env, user_id) -> None:
    response = await inquiry_env.client.post(
        "/api/v1/inquiries",
        headers=_auth_headers(user_id),
        json={"title": "  로그인 오류  ", "content": "  @everyone\n로그인이 되지 않습니다.  "},
    )

    assert response.status_code == 200
    assert response.json() == {"message": "문의가 전송되었습니다."}
    assert len(inquiry_env.requests) == 1
    request = inquiry_env.requests[0]
    assert request.method == "POST"
    assert request.url.params["wait"] == "true"
    payload = json.loads(request.content)
    assert payload["allowed_mentions"] == {"parse": []}
    embed = payload["embeds"][0]
    assert embed["title"] == "[앱 문의] 로그인 오류"
    assert embed["description"] == "@everyone\n로그인이 되지 않습니다."
    fields = {field["name"]: field["value"] for field in embed["fields"]}
    assert fields["사용자 ID"] == str(user_id)
    assert fields["이메일"] == f"user{user_id}@example.com"


@pytest.mark.parametrize("email", [None, ""])
async def test_missing_email_does_not_prevent_inquiry_delivery(inquiry_env, email) -> None:
    async with inquiry_env.sessions() as session:
        await session.execute(update(users_table).where(users_table.c.user_id == 7).values(email=email))
        await session.commit()

    response = await inquiry_env.client.post(
        "/api/v1/inquiries", headers=_auth_headers(), json={"title": "제목", "content": "내용"}
    )

    assert response.status_code == 200
    assert len(inquiry_env.requests) == 1
    embed = json.loads(inquiry_env.requests[0].content)["embeds"][0]
    fields = {field["name"]: field["value"] for field in embed["fields"]}
    assert fields["사용자 ID"] == "7"
    assert fields["이메일"] == "미등록"


@pytest.mark.parametrize(
    "body",
    [
        {"title": "", "content": "내용"},
        {"title": " \n\t ", "content": "내용"},
        {"title": "가" * 101, "content": "내용"},
        {"title": "제목", "content": ""},
        {"title": "제목", "content": " \n\t "},
        {"title": "제목", "content": "가" * 3_001},
        {"content": "내용"},
        {"title": "제목"},
        {"title": "제목", "content": "내용", "user_id": 8},
        {"title": "제목", "content": "내용", "email": "other@example.com"},
    ],
)
async def test_invalid_inquiry_does_not_send_webhook(inquiry_env, body) -> None:
    response = await inquiry_env.client.post("/api/v1/inquiries", headers=_auth_headers(), json=body)

    assert response.status_code == 422
    assert inquiry_env.requests == []


async def test_maximum_length_inquiry_is_accepted(inquiry_env) -> None:
    response = await inquiry_env.client.post(
        "/api/v1/inquiries",
        headers=_auth_headers(),
        json={"title": "가" * 100, "content": "나" * 3_000},
    )

    assert response.status_code == 200
    embed = json.loads(inquiry_env.requests[0].content)["embeds"][0]
    assert embed["description"] == "나" * 3_000


async def test_unauthenticated_inquiry_does_not_send_webhook(inquiry_env) -> None:
    response = await inquiry_env.client.post("/api/v1/inquiries", json={"title": "제목", "content": "내용"})

    assert response.status_code in (401, 403)
    assert inquiry_env.requests == []


async def test_deleted_user_inquiry_is_rejected_before_webhook_delivery(inquiry_env) -> None:
    headers = _auth_headers()
    async with inquiry_env.sessions() as session:
        await session.execute(users_table.delete().where(users_table.c.user_id == 7))
        await session.commit()

    response = await inquiry_env.client.post(
        "/api/v1/inquiries", headers=headers, json={"title": "제목", "content": "내용"}
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "USER_NOT_FOUND"
    assert inquiry_env.requests == []


@pytest.mark.parametrize("webhook_url", [None, "", "   "])
async def test_unconfigured_webhook_returns_failure(inquiry_env, webhook_url) -> None:
    inquiry_env.app.dependency_overrides[get_inquiry_service] = lambda: InquiryService(webhook_url)

    response = await inquiry_env.client.post(
        "/api/v1/inquiries", headers=_auth_headers(), json={"title": "제목", "content": "내용"}
    )

    assert response.status_code == 503
    assert inquiry_env.requests == []


@pytest.mark.parametrize("discord_status", [400, 404, 429, 500])
async def test_discord_error_returns_failure_without_logging_private_data(inquiry_env, caplog, discord_status) -> None:
    service = InquiryService(
        _WEBHOOK_URL, transport=httpx.MockTransport(lambda _: httpx.Response(discord_status))
    )
    inquiry_env.app.dependency_overrides[get_inquiry_service] = lambda: service

    with caplog.at_level(logging.ERROR, logger="app.services.inquiry"):
        response = await inquiry_env.client.post(
            "/api/v1/inquiries", headers=_auth_headers(), json={"title": "비공개 제목", "content": "비공개 내용"}
        )

    assert response.status_code == 503
    assert "문의 전송 실패" in caplog.text
    for private_value in (_WEBHOOK_URL, "secret-token", "user7@example.com", "비공개 제목", "비공개 내용"):
        assert private_value not in caplog.text
        assert private_value not in response.text


async def test_discord_timeout_returns_failure(inquiry_env) -> None:
    def _timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("응답 시간 초과", request=request)

    service = InquiryService(_WEBHOOK_URL, transport=httpx.MockTransport(_timeout))
    inquiry_env.app.dependency_overrides[get_inquiry_service] = lambda: service

    response = await inquiry_env.client.post(
        "/api/v1/inquiries", headers=_auth_headers(), json={"title": "제목", "content": "내용"}
    )

    assert response.status_code == 503
