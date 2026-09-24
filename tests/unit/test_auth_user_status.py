from datetime import timedelta

import httpx
from fastapi import Depends, FastAPI
from sqlalchemy import update

import app.deps.auth as auth
from app.core.timeutil import now_utc
from app.db.external import users_table
from app.deps.auth import get_current_user_id
from app.deps.db import get_db
from tests.unit.community_env import community_app


async def _request_with_real_auth(env, monkeypatch) -> httpx.Response:
    monkeypatch.setattr(auth, "decode_access_token", lambda _: {"sub": "7"})
    app = FastAPI()

    @app.get("/protected")
    async def protected(user_id: int = Depends(get_current_user_id)) -> dict[str, int]:
        return {"user_id": user_id}

    async def _get_db():
        async with env.sessions() as session:
            yield session

    app.dependency_overrides[get_db] = _get_db
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get("/protected", headers={"Authorization": "Bearer token"})


async def _set_status(env, status: str, suspended_until=None) -> None:
    async with env.sessions() as session:
        await session.execute(
            update(users_table)
            .where(users_table.c.user_id == 7)
            .values(status=status, suspended_until=suspended_until)
        )
        await session.commit()


async def test_active_user_can_call_authenticated_api(monkeypatch) -> None:
    async with community_app() as env:
        resp = await _request_with_real_auth(env, monkeypatch)

    assert resp.status_code == 200
    assert resp.json() == {"user_id": 7}


async def test_banned_user_is_rejected(monkeypatch) -> None:
    async with community_app() as env:
        await _set_status(env, "BANNED")
        resp = await _request_with_real_auth(env, monkeypatch)

    assert resp.status_code == 403
    assert resp.json()["detail"] == "USER_BANNED"


async def test_active_suspension_is_rejected(monkeypatch) -> None:
    async with community_app() as env:
        await _set_status(env, "SUSPENDED", now_utc() + timedelta(hours=1))
        resp = await _request_with_real_auth(env, monkeypatch)

    assert resp.status_code == 403
    assert resp.json()["detail"] == "USER_SUSPENDED"


async def test_expired_suspension_is_allowed(monkeypatch) -> None:
    async with community_app() as env:
        await _set_status(env, "SUSPENDED", now_utc() - timedelta(seconds=1))
        resp = await _request_with_real_auth(env, monkeypatch)

    assert resp.status_code == 200


async def test_deleted_user_is_rejected(monkeypatch) -> None:
    async with community_app() as env:
        async with env.sessions() as session:
            await session.execute(users_table.delete().where(users_table.c.user_id == 7))
            await session.commit()
        resp = await _request_with_real_auth(env, monkeypatch)

    assert resp.status_code == 401
    assert resp.json()["detail"] == "USER_NOT_FOUND"
