from datetime import datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.db.models import PostReactionORM
from tests.unit.community_env import community_app, create_post

_STAMP = datetime(2026, 8, 25, 12, 0, 0)


async def test_reaction_is_recorded_and_reflected_in_counts() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)

        resp = await env.client.put(
            f"/api/v1/community/posts/{post_id}/reaction", json={"kind": "CHEER"}
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["reactions"] == {"cheer": 1, "relate": 0, "like": 0, "total": 1}
    assert body["my_reaction"] == "CHEER"


async def _reaction_row_count(env, post_id: int) -> int:
    async with env.sessions() as session:
        stmt = select(func.count()).select_from(PostReactionORM).where(
            PostReactionORM.post_id == post_id
        )
        return (await session.execute(stmt)).scalar_one()


async def test_switching_kind_replaces_the_single_row() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        await env.client.put(
            f"/api/v1/community/posts/{post_id}/reaction", json={"kind": "CHEER"}
        )

        resp = await env.client.put(
            f"/api/v1/community/posts/{post_id}/reaction", json={"kind": "LIKE"}
        )
        rows = await _reaction_row_count(env, post_id)

    assert resp.json()["reactions"] == {"cheer": 0, "relate": 0, "like": 1, "total": 1}
    assert resp.json()["my_reaction"] == "LIKE"
    assert rows == 1, "사용자당 1행이어야 한다"


async def test_pressing_the_same_kind_twice_is_idempotent() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        await env.client.put(
            f"/api/v1/community/posts/{post_id}/reaction", json={"kind": "RELATE"}
        )

        resp = await env.client.put(
            f"/api/v1/community/posts/{post_id}/reaction", json={"kind": "RELATE"}
        )
        rows = await _reaction_row_count(env, post_id)

    assert resp.json()["reactions"]["relate"] == 1
    assert rows == 1


async def test_each_user_keeps_their_own_reaction() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        await env.client.put(
            f"/api/v1/community/posts/{post_id}/reaction", json={"kind": "CHEER"}
        )
        env.login(8)

        resp = await env.client.put(
            f"/api/v1/community/posts/{post_id}/reaction", json={"kind": "LIKE"}
        )

    assert resp.json()["reactions"] == {"cheer": 1, "relate": 0, "like": 1, "total": 2}
    assert resp.json()["my_reaction"] == "LIKE"


async def test_unknown_reaction_kind_is_rejected() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)

        resp = await env.client.put(
            f"/api/v1/community/posts/{post_id}/reaction", json={"kind": "ANGRY"}
        )

    assert resp.status_code == 422


async def test_reacting_to_missing_post_returns_404() -> None:
    async with community_app(user_id=7) as env:
        resp = await env.client.put(
            "/api/v1/community/posts/999/reaction", json={"kind": "CHEER"}
        )

    assert resp.status_code == 404


async def test_reacting_to_deleted_post_returns_404() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        await env.client.delete(f"/api/v1/community/posts/{post_id}")

        resp = await env.client.put(
            f"/api/v1/community/posts/{post_id}/reaction", json={"kind": "CHEER"}
        )

    assert resp.status_code == 404


async def test_cancelling_removes_the_reaction() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        await env.client.put(
            f"/api/v1/community/posts/{post_id}/reaction", json={"kind": "CHEER"}
        )

        resp = await env.client.delete(f"/api/v1/community/posts/{post_id}/reaction")
        listed = await env.client.get("/api/v1/community/posts")
        rows = await _reaction_row_count(env, post_id)

    assert resp.status_code == 204
    assert rows == 0
    post = listed.json()["posts"][0]
    assert post["reactions"] == {"cheer": 0, "relate": 0, "like": 0, "total": 0}
    assert post["my_reaction"] is None


async def test_cancelling_without_a_reaction_is_idempotent() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)

        resp = await env.client.delete(f"/api/v1/community/posts/{post_id}/reaction")

    assert resp.status_code == 204


async def test_cancelling_on_missing_post_returns_404() -> None:
    async with community_app(user_id=7) as env:
        resp = await env.client.delete("/api/v1/community/posts/999/reaction")

    assert resp.status_code == 404


async def test_schema_itself_blocks_two_reactions_from_one_user() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)

        async with env.sessions() as session:
            session.add_all([
                PostReactionORM(
                    post_id=post_id, user_id=7, kind="CHEER", created_at=_STAMP
                ),
                PostReactionORM(
                    post_id=post_id, user_id=7, kind="LIKE", created_at=_STAMP
                ),
            ])
            with pytest.raises(IntegrityError):
                await session.commit()


async def test_concurrent_first_reaction_does_not_blow_up(monkeypatch) -> None:
    from app.core.enums import ReactionKind
    from app.services import community_reaction as svc

    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        await env.client.put(
            f"/api/v1/community/posts/{post_id}/reaction", json={"kind": "CHEER"}
        )

        original = svc._my_reaction_row
        missed = {"once": False}

        async def blind_once(db, pid, uid):
            if not missed["once"]:
                missed["once"] = True
                return None
            return await original(db, pid, uid)

        monkeypatch.setattr(svc, "_my_reaction_row", blind_once)

        async with env.sessions() as session:
            state = await svc.set_reaction(
                session, post_id, user_id=7, kind=ReactionKind.LIKE
            )

    assert missed["once"], "경합 창이 실제로 재현되지 않았다"
    assert state.reactions.total == 1, "행이 두 개로 늘면 안 된다"
    assert state.my_reaction == ReactionKind.LIKE, "나중 요청의 종류로 수렴해야 한다"
