from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import httpx
from fastapi import FastAPI
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.api.v1.community import router as community_router
from app.api.v1.community_admin import router as community_admin_router
from app.db.base import Base
from app.db.external import external_metadata, users_table
from app.deps.auth import get_current_user_id
from app.deps.community_moderation import get_community_content_moderator
from app.deps.community_report_alert import get_community_report_alert_service
from app.deps.db import get_db
from app.deps.spring import get_spring_client
from app.services.community_content_moderation import (
    ContentModerationUnavailableError,
    ModerationCategory,
    ObjectionableContentError,
)

DEFAULT_USERS = (
    {
        "user_id": 7,
        "name": "사용자1",
        "profile_image": "profiles/7.png",
        "role": "USER",
        "status": "ACTIVE",
    },
    {"user_id": 8, "name": "사용자2", "profile_image": None, "role": "USER", "status": "ACTIVE"},
    {"user_id": 9, "name": "운영자", "profile_image": None, "role": "ADMIN", "status": "ACTIVE"},
)


class FakeRedis:
    def __init__(self, *, fail: bool = False) -> None:
        self.keys: dict[str, str] = {}
        self.ttls: dict[str, int | None] = {}
        self.fail = fail

    async def set(
        self, name: str, value: str, ex: int | None = None, nx: bool = False
    ) -> bool | None:
        if self.fail:
            raise ConnectionError("redis down")
        if nx and name in self.keys:
            return None
        self.keys[name] = value
        self.ttls[name] = ex
        return True


@dataclass
class FakeSpringClient:
    notifications: list[dict] = field(default_factory=list)
    moderation_updates: list[dict] = field(default_factory=list)
    moderation_succeeds: bool = True

    async def notify_community_notification(self, **notification) -> None:
        self.notifications.append(notification)

    async def update_user_moderation_status(self, user_id: int, **moderation) -> bool:
        self.moderation_updates.append({"user_id": user_id, **moderation})
        return self.moderation_succeeds


@dataclass
class FakeContentModerator:
    objectionable: bool = False
    unavailable: bool = False
    calls: list[dict[str, str | None]] = field(default_factory=list)

    async def moderate(self, *, title: str | None = None, content: str | None = None) -> None:
        self.calls.append({"title": title, "content": content})
        if self.unavailable:
            raise ContentModerationUnavailableError
        if self.objectionable:
            raise ObjectionableContentError(ModerationCategory.ABUSE)


@dataclass
class FakeCommunityReportAlertService:
    reports: list = field(default_factory=list)

    async def notify_report_created(self, report) -> bool:  # noqa: ANN001
        self.reports.append(report)
        return True


@dataclass
class Env:
    client: httpx.AsyncClient
    sessions: async_sessionmaker[AsyncSession]
    redis: FakeRedis
    spring: FakeSpringClient
    moderator: FakeContentModerator
    report_alerts: FakeCommunityReportAlertService
    queries: list[str] = field(default_factory=list)
    _current: dict = field(default_factory=dict)

    def login(self, user_id: int) -> None:
        """요청 주체를 바꾼다. 어드민 여부는 users 픽스처의 role 이 정한다."""
        self._current["user_id"] = user_id


@asynccontextmanager
async def community_app(
    *,
    user_id: int = 7,
    redis: FakeRedis | None = None,
    moderator: FakeContentModerator | None = None,
    report_alerts: FakeCommunityReportAlertService | None = None,
    users: tuple[dict, ...] = DEFAULT_USERS,
) -> AsyncIterator[Env]:
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(external_metadata.create_all)
        # file 은 모델 쪽이 먼저 만들어서 Spring 이 붙이는 user_id 칸을 따로 흉내낸다
        await conn.execute(text("ALTER TABLE file ADD COLUMN user_id BIGINT"))
        for row in users:
            await conn.execute(users_table.insert().values(**row))

    queries: list[str] = []

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def _record_query(conn, cursor, statement, parameters, context, executemany) -> None:  # noqa: ANN001
        queries.append(statement)

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    async def _get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    current = {"user_id": user_id}
    fake_redis = redis or FakeRedis()
    fake_spring = FakeSpringClient()
    fake_moderator = moderator or FakeContentModerator()
    fake_report_alerts = report_alerts or FakeCommunityReportAlertService()

    app = FastAPI()
    app.include_router(community_router)
    app.include_router(community_admin_router)
    app.state.redis = fake_redis
    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_current_user_id] = lambda: current["user_id"]
    app.dependency_overrides[get_spring_client] = lambda: fake_spring
    app.dependency_overrides[get_community_content_moderator] = lambda: fake_moderator
    app.dependency_overrides[get_community_report_alert_service] = lambda: fake_report_alerts

    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield Env(
                client=client,
                sessions=session_factory,
                redis=fake_redis,
                spring=fake_spring,
                moderator=fake_moderator,
                report_alerts=fake_report_alerts,
                queries=queries,
                _current=current,
            )
    finally:
        await engine.dispose()


async def create_post(env: Env, title: str = "제목", content: str = "내용") -> int:
    resp = await env.client.post(
        "/api/v1/community/posts", json={"title": title, "content": content}
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["post_id"]
