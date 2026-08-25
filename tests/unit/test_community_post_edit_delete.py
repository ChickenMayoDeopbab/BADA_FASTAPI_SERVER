from datetime import datetime

from app.db.models import PostORM
from tests.unit.community_env import community_app, create_post

_STAMP = datetime(2026, 8, 25, 12, 0, 0)


async def test_author_updates_own_post() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env, "원래 제목", "원래 내용")

        resp = await env.client.patch(
            f"/api/v1/community/posts/{post_id}",
            json={"title": "고친 제목", "content": "고친 내용"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["title"] == "고친 제목"
    assert body["content"] == "고친 내용"


async def test_other_user_cannot_update_someones_post() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env, "원래 제목", "원래 내용")
        env.login(8)

        resp = await env.client.patch(
            f"/api/v1/community/posts/{post_id}", json={"title": "남의 글 고침"}
        )

    assert resp.status_code == 403


async def test_updating_missing_post_returns_404() -> None:
    async with community_app(user_id=7) as env:
        resp = await env.client.patch(
            "/api/v1/community/posts/999", json={"title": "없는 글"}
        )

    assert resp.status_code == 404


async def test_updating_deleted_post_returns_404() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        async with env.sessions() as session:
            row = await session.get(PostORM, post_id)
            row.deleted_at = _STAMP
            await session.commit()

        resp = await env.client.patch(
            f"/api/v1/community/posts/{post_id}", json={"title": "지운 글 고침"}
        )

    assert resp.status_code == 404


async def test_partial_update_keeps_untouched_field() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env, "원래 제목", "원래 내용")

        resp = await env.client.patch(
            f"/api/v1/community/posts/{post_id}", json={"title": "제목만 변경"}
        )

    assert resp.json()["title"] == "제목만 변경"
    assert resp.json()["content"] == "원래 내용"


async def test_author_soft_deletes_own_post() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env, "지울 글")

        resp = await env.client.delete(f"/api/v1/community/posts/{post_id}")

        listed = await env.client.get("/api/v1/community/posts")
        detail = await env.client.get(f"/api/v1/community/posts/{post_id}")
        async with env.sessions() as session:
            row = await session.get(PostORM, post_id)
            still_there = row is not None and row.deleted_at is not None

    assert resp.status_code == 204
    assert listed.json()["posts"] == []
    assert detail.status_code == 404
    assert still_there, "soft delete 라 행은 남고 deleted_at 만 찍혀야 한다"


async def test_other_user_cannot_delete_someones_post() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        env.login(8)

        resp = await env.client.delete(f"/api/v1/community/posts/{post_id}")

    assert resp.status_code == 403


async def test_admin_can_delete_any_post() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env, "남의 글")
        env.login(9)

        resp = await env.client.delete(f"/api/v1/community/posts/{post_id}")
        listed = await env.client.get("/api/v1/community/posts")

    assert resp.status_code == 204
    assert listed.json()["posts"] == []


async def test_deleting_missing_post_returns_404() -> None:
    async with community_app(user_id=7) as env:
        resp = await env.client.delete("/api/v1/community/posts/999")

    assert resp.status_code == 404


async def test_deleting_already_deleted_post_returns_404() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        await env.client.delete(f"/api/v1/community/posts/{post_id}")

        resp = await env.client.delete(f"/api/v1/community/posts/{post_id}")

    assert resp.status_code == 404


async def test_admin_cannot_edit_someones_post() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env, "남의 글")
        env.login(9)

        resp = await env.client.patch(
            f"/api/v1/community/posts/{post_id}", json={"title": "어드민이 고침"}
        )

    assert resp.status_code == 403, "어드민 권한은 삭제까지다. 남의 글 내용을 바꾸면 안 된다"


async def test_empty_patch_does_not_mark_post_as_edited() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env, "제목", "내용")
        before = (await env.client.get(f"/api/v1/community/posts/{post_id}")).json()

        resp = await env.client.patch(f"/api/v1/community/posts/{post_id}", json={})

    assert resp.json()["updated_at"] == before["updated_at"]
