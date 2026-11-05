import httpx
import pytest
from fastapi import FastAPI
from sqlalchemy import select

from app.api.v1 import internal
from app.db.models import CallBiosignalORM
from app.deps.db import get_db
from app.services import biosignal_service as svc
from tests.unit.test_biosignal_api import Env, _env, _seed, _Storage

INTERNAL_HEADERS = {"X-Internal-Secret": "test-internal-secret"}


class _BrokenDelete(_Storage):
    def delete(self, key: str) -> bool:
        raise RuntimeError("s3 down")


async def _sessions_left(env: Env) -> list[str]:
    async with env.sessions() as session:
        return sorted((await session.execute(select(CallBiosignalORM.session_id))).scalars())


async def test_purge_removes_only_that_users_rows_and_raw_objects() -> None:
    async with _env() as env:
        await _seed(env, "sess-1", status="ANALYZED", metrics={}, raw_key="biosignal-raw/sess-1.json.gz")
        await _seed(env, "sess-2")
        await _seed(env, "sess-9", 9, status="ANALYZED", metrics={}, raw_key="biosignal-raw/sess-9.json.gz")

        async with env.sessions() as session:
            purged = await svc.delete_user_biosignals(session, 7, storage=env.storage)
        async with env.sessions() as session:
            again = await svc.delete_user_biosignals(session, 7, storage=env.storage)

        assert (purged, again) == (2, 0)
        assert env.storage.deleted == ["biosignal-raw/sess-1.json.gz"]
        assert await _sessions_left(env) == ["sess-9"]


async def test_purge_keeps_the_rows_when_raw_deletion_fails() -> None:
    async with _env() as env:
        await _seed(env, "sess-1", status="ANALYZED", metrics={}, raw_key="biosignal-raw/sess-1.json.gz")

        async with env.sessions() as session:
            with pytest.raises(RuntimeError):
                await svc.delete_user_biosignals(session, 7, storage=_BrokenDelete())

        assert await _sessions_left(env) == ["sess-1"]


async def test_internal_purge_endpoint_requires_the_secret() -> None:
    async with _env() as env:
        await _seed(env, "sess-1")
        await _seed(env, "sess-9", 9)
        app = FastAPI()
        app.include_router(internal.router, prefix="/internal/v1")

        async def _get_db():
            async with env.sessions() as session:
                yield session

        app.dependency_overrides[get_db] = _get_db
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            denied = await client.delete("/internal/v1/users/7/biosignals")
            left_after_denied = await _sessions_left(env)
            allowed = await client.delete("/internal/v1/users/7/biosignals", headers=INTERNAL_HEADERS)

        assert (denied.status_code, left_after_denied) == (403, ["sess-1", "sess-9"])
        assert allowed.status_code == 204
        assert await _sessions_left(env) == ["sess-9"]
