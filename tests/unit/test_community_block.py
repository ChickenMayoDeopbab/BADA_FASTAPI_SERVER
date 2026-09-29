from datetime import UTC, datetime

from sqlalchemy import delete, func, select, update

from app.db.external import users_table
from app.db.models import CommunityUserBlockORM
from tests.unit.community_env import community_app


async def _block_count(env) -> int:
    async with env.sessions() as session:
        return (await session.execute(select(func.count()).select_from(CommunityUserBlockORM))).scalar_one()


async def test_block_user_creates_relation() -> None:
    async with community_app(user_id=7) as env:
        resp = await env.client.put("/api/v1/community/users/8/block")
        async with env.sessions() as session:
            row = (await session.execute(select(CommunityUserBlockORM))).scalar_one()

    assert resp.status_code == 204
    assert resp.content == b""
    assert row.blocker_user_id == 7
    assert row.blocked_user_id == 8


async def test_repeated_block_is_idempotent() -> None:
    async with community_app(user_id=7) as env:
        first = await env.client.put("/api/v1/community/users/8/block")
        second = await env.client.put("/api/v1/community/users/8/block")
        block_count = await _block_count(env)

    assert first.status_code == 204
    assert second.status_code == 204
    assert block_count == 1


async def test_unblock_removes_relation() -> None:
    async with community_app(user_id=7) as env:
        await env.client.put("/api/v1/community/users/8/block")
        resp = await env.client.delete("/api/v1/community/users/8/block")
        block_count = await _block_count(env)

    assert resp.status_code == 204
    assert block_count == 0


async def test_repeated_unblock_is_idempotent() -> None:
    async with community_app(user_id=7) as env:
        first = await env.client.delete("/api/v1/community/users/8/block")
        second = await env.client.delete("/api/v1/community/users/8/block")

    assert first.status_code == 204
    assert second.status_code == 204


async def test_self_block_and_unblock_are_rejected() -> None:
    async with community_app(user_id=7) as env:
        block_resp = await env.client.put("/api/v1/community/users/7/block")
        unblock_resp = await env.client.delete("/api/v1/community/users/7/block")

    assert block_resp.status_code == 400
    assert unblock_resp.status_code == 400


async def test_missing_user_block_and_unblock_return_404() -> None:
    async with community_app(user_id=7) as env:
        block_resp = await env.client.put("/api/v1/community/users/999/block")
        unblock_resp = await env.client.delete("/api/v1/community/users/999/block")

    assert block_resp.status_code == 404
    assert unblock_resp.status_code == 404


async def test_blocked_user_list_is_empty_before_blocking() -> None:
    async with community_app(user_id=7) as env:
        resp = await env.client.get("/api/v1/community/me/blocked-users")

    assert resp.status_code == 200
    assert resp.json() == {"blocked_users": []}


async def test_blocked_user_list_is_scoped_to_current_user_and_sorted_newest_first() -> None:
    async with community_app(user_id=7) as env:
        await env.client.put("/api/v1/community/users/8/block")
        await env.client.put("/api/v1/community/users/9/block")
        env.login(8)
        await env.client.put("/api/v1/community/users/7/block")

        async with env.sessions() as session:
            await session.execute(
                update(CommunityUserBlockORM)
                .where(CommunityUserBlockORM.blocker_user_id == 7, CommunityUserBlockORM.blocked_user_id == 8)
                .values(created_at=datetime(2026, 1, 1, tzinfo=UTC))
            )
            await session.execute(
                update(CommunityUserBlockORM)
                .where(CommunityUserBlockORM.blocker_user_id == 7, CommunityUserBlockORM.blocked_user_id == 9)
                .values(created_at=datetime(2026, 1, 2, tzinfo=UTC))
            )
            await session.commit()

        env.login(7)
        resp = await env.client.get("/api/v1/community/me/blocked-users")

    assert resp.status_code == 200
    assert resp.json() == {
        "blocked_users": [
            {
                "user_id": 9,
                "name": "운영자",
                "profile_image_url": None,
                "blocked_at": "2026-01-02T09:00:00+09:00",
            },
            {
                "user_id": 8,
                "name": "사용자2",
                "profile_image_url": None,
                "blocked_at": "2026-01-01T09:00:00+09:00",
            },
        ]
    }


async def test_blocked_user_list_reflects_unblock_and_keeps_missing_profile() -> None:
    async with community_app(user_id=7) as env:
        await env.client.put("/api/v1/community/users/8/block")
        await env.client.put("/api/v1/community/users/9/block")
        await env.client.delete("/api/v1/community/users/8/block")
        async with env.sessions() as session:
            await session.execute(delete(users_table).where(users_table.c.user_id == 9))
            await session.commit()

        resp = await env.client.get("/api/v1/community/me/blocked-users")

    assert resp.status_code == 200
    assert len(resp.json()["blocked_users"]) == 1
    assert resp.json()["blocked_users"][0]["user_id"] == 9
    assert resp.json()["blocked_users"][0]["name"] is None
    assert resp.json()["blocked_users"][0]["profile_image_url"] is None


async def test_insert_collision_converges_to_existing_block(monkeypatch) -> None:
    from app.services import community_block as svc

    async with community_app(user_id=7) as env:
        await env.client.put("/api/v1/community/users/8/block")
        original = svc._block_row
        missed = {"once": False}

        async def blind_once(db, blocker_user_id: int, blocked_user_id: int):
            if not missed["once"]:
                missed["once"] = True
                return None
            return await original(db, blocker_user_id, blocked_user_id)

        monkeypatch.setattr(svc, "_block_row", blind_once)
        async with env.sessions() as session:
            await svc.block_user(session, blocker_user_id=7, blocked_user_id=8)
        block_count = await _block_count(env)

    assert missed["once"]
    assert block_count == 1
