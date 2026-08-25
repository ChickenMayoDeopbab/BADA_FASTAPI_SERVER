from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import httpx
from fastapi import FastAPI
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.api.v1.community import router as community_router
from app.db.base import Base
from app.db.external import external_metadata, users_table
from app.deps.auth import get_current_user_id
from app.deps.db import get_db

DEFAULT_USERS = (
    {"user_id": 7, "name": "사용자1", "profile_image": "profiles/7.png", "role": "USER"},
    {"user_id": 8, "name": "사용자2", "profile_image": None, "role": "USER"},
    {"user_id": 9, "name": "운영자", "profile_image": None, "role": "ADMIN"},
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
class Env:
    client: httpx.AsyncClient
    sessions: async_sessionmaker[AsyncSession]
    redis: FakeRedis
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

    app = FastAPI()
    app.include_router(community_router)
    app.state.redis = fake_redis
    app.dependency_overrides[get_db] = _get_db
    app.dependency_overrides[get_current_user_id] = lambda: current["user_id"]

    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield Env(
                client=client,
                sessions=session_factory,
                redis=fake_redis,
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
