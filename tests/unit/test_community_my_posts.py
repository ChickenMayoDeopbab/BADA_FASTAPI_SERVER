from tests.unit.community_env import community_app, create_post


async def test_only_my_own_posts_are_returned() -> None:
    async with community_app(user_id=7) as env:
        await create_post(env, "내 글 1")
        env.login(8)
        await create_post(env, "남의 글")
        env.login(7)
        await create_post(env, "내 글 2")

        resp = await env.client.get("/api/v1/community/me/posts")

    assert resp.status_code == 200
    assert [p["title"] for p in resp.json()["posts"]] == ["내 글 2", "내 글 1"]


async def test_my_deleted_posts_are_excluded() -> None:
    async with community_app(user_id=7) as env:
        await create_post(env, "남는 글")
        gone = await create_post(env, "지운 글")
        await env.client.delete(f"/api/v1/community/posts/{gone}")

        resp = await env.client.get("/api/v1/community/me/posts")

    body = resp.json()
    assert [p["title"] for p in body["posts"]] == ["남는 글"]
    assert body["total"] == 1


async def test_my_posts_paginate_with_total_and_has_next() -> None:
    async with community_app(user_id=7) as env:
        for i in range(3):
            await create_post(env, f"내 글 {i}")
        env.login(8)
        await create_post(env, "남의 글")
        env.login(7)

        first = await env.client.get("/api/v1/community/me/posts", params={"page": 1, "size": 2})
        second = await env.client.get("/api/v1/community/me/posts", params={"page": 2, "size": 2})

    assert first.json()["total"] == 3, "남의 글은 total 에도 안 잡혀야 한다"
    assert first.json()["has_next"] is True
    assert len(second.json()["posts"]) == 1
    assert second.json()["has_next"] is False


async def test_my_posts_carry_the_same_aggregates_as_the_feed() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env, "내 글")
        await env.client.post(
            f"/api/v1/community/posts/{post_id}/comments", json={"content": "댓글"}
        )
        await env.client.put(
            f"/api/v1/community/posts/{post_id}/reaction", json={"kind": "CHEER"}
        )
        await env.client.get(f"/api/v1/community/posts/{post_id}")

        resp = await env.client.get("/api/v1/community/me/posts")

    post = resp.json()["posts"][0]
    assert post["comment_count"] == 1
    assert post["reactions"] == {"cheer": 1, "relate": 0, "like": 0, "total": 1}
    assert post["my_reaction"] == "CHEER"
    assert post["view_count"] == 1


async def test_user_without_posts_gets_empty_list() -> None:
    async with community_app(user_id=7) as env:
        env.login(8)
        await create_post(env, "남의 글")
        env.login(7)

        resp = await env.client.get("/api/v1/community/me/posts")

    body = resp.json()
    assert body["posts"] == []
    assert body["total"] == 0
    assert body["has_next"] is False


async def test_my_posts_query_count_does_not_grow_with_post_count() -> None:
    async with community_app(user_id=7) as env:
        for i in range(5):
            post_id = await create_post(env, f"내 글 {i}")
            await env.client.post(
                f"/api/v1/community/posts/{post_id}/comments", json={"content": "댓글"}
            )

        env.queries.clear()
        resp = await env.client.get("/api/v1/community/me/posts")

    assert len(resp.json()["posts"]) == 5
    assert len(env.queries) == 3, "\n---\n".join(env.queries)
