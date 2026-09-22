from sqlalchemy import select

from app.db.models import PostORM
from tests.unit.community_env import community_app, create_post


async def test_blocked_author_is_excluded_from_list_search_and_total() -> None:
    async with community_app(user_id=8) as env:
        blocked_post_id = await create_post(env, title="공통 검색어 차단", content="차단 작성자")
        env.login(7)
        visible_post_id = await create_post(env, title="공통 검색어 공개", content="공개 작성자")
        await env.client.put("/api/v1/community/users/8/block")

        resp = await env.client.get("/api/v1/community/posts", params={"q": "공통 검색어"})

    assert resp.status_code == 200
    assert [post["post_id"] for post in resp.json()["posts"]] == [visible_post_id]
    assert resp.json()["total"] == 1
    assert blocked_post_id != visible_post_id


async def test_blocked_post_detail_returns_404_without_counting_view() -> None:
    async with community_app(user_id=8) as env:
        post_id = await create_post(env)
        env.login(7)
        await env.client.put("/api/v1/community/users/8/block")

        resp = await env.client.get(f"/api/v1/community/posts/{post_id}")
        async with env.sessions() as session:
            stmt = select(PostORM.view_count).where(PostORM.post_id == post_id)
            view_count = (await session.execute(stmt)).scalar_one()

    assert resp.status_code == 404
    assert view_count == 0


async def test_blocked_comments_are_excluded_from_threads_and_counts() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        env.login(8)
        blocked_parent = (
            await env.client.post(f"/api/v1/community/posts/{post_id}/comments", json={"content": "차단 댓글"})
        ).json()
        env.login(9)
        await env.client.post(
            f"/api/v1/community/posts/{post_id}/comments",
            json={"content": "차단 댓글의 답글", "parent_comment_id": blocked_parent["comment_id"]},
        )
        visible_comment = (
            await env.client.post(f"/api/v1/community/posts/{post_id}/comments", json={"content": "공개 댓글"})
        ).json()
        env.login(7)
        await env.client.put("/api/v1/community/users/8/block")

        comments_resp = await env.client.get(f"/api/v1/community/posts/{post_id}/comments")
        post_resp = await env.client.get(f"/api/v1/community/posts/{post_id}")

    assert [comment["comment_id"] for comment in comments_resp.json()["comments"]] == [visible_comment["comment_id"]]
    assert post_resp.json()["comment_count"] == 1


async def test_comments_for_blocked_authors_post_return_404() -> None:
    async with community_app(user_id=8) as env:
        post_id = await create_post(env)
        env.login(7)
        await env.client.put("/api/v1/community/users/8/block")

        resp = await env.client.get(f"/api/v1/community/posts/{post_id}/comments")

    assert resp.status_code == 404


async def test_unblock_restores_post_and_comments() -> None:
    async with community_app(user_id=8) as env:
        post_id = await create_post(env)
        await env.client.post(f"/api/v1/community/posts/{post_id}/comments", json={"content": "댓글"})
        env.login(7)
        await env.client.put("/api/v1/community/users/8/block")
        await env.client.delete("/api/v1/community/users/8/block")

        post_resp = await env.client.get(f"/api/v1/community/posts/{post_id}")
        comments_resp = await env.client.get(f"/api/v1/community/posts/{post_id}/comments")

    assert post_resp.status_code == 200
    assert len(comments_resp.json()["comments"]) == 1


async def test_blocked_actor_does_not_create_comment_reply_or_reaction_notifications() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        parent = (
            await env.client.post(f"/api/v1/community/posts/{post_id}/comments", json={"content": "작성자 댓글"})
        ).json()
        await env.client.put("/api/v1/community/users/8/block")
        env.spring.notifications.clear()
        env.login(8)

        await env.client.post(f"/api/v1/community/posts/{post_id}/comments", json={"content": "차단 사용자 댓글"})
        await env.client.post(
            f"/api/v1/community/posts/{post_id}/comments",
            json={"content": "차단 사용자 답글", "parent_comment_id": parent["comment_id"]},
        )
        await env.client.put(f"/api/v1/community/posts/{post_id}/reaction", json={"kind": "LIKE"})

    assert env.spring.notifications == []
