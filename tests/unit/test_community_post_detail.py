from tests.unit.community_env import community_app, create_post


async def test_detail_carries_comment_count_and_reactions() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        parent_id = (
            await env.client.post(
                f"/api/v1/community/posts/{post_id}/comments", json={"content": "댓글"}
            )
        ).json()["comment_id"]
        await env.client.post(
            f"/api/v1/community/posts/{post_id}/comments",
            json={"content": "답글", "parent_comment_id": parent_id},
        )
        await env.client.put(
            f"/api/v1/community/posts/{post_id}/reaction", json={"kind": "CHEER"}
        )
        env.login(8)
        await env.client.put(
            f"/api/v1/community/posts/{post_id}/reaction", json={"kind": "LIKE"}
        )
        env.login(7)

        resp = await env.client.get(f"/api/v1/community/posts/{post_id}")

    body = resp.json()
    assert body["comment_count"] == 2
    assert body["reactions"] == {"cheer": 1, "relate": 0, "like": 1, "total": 2}
    assert body["my_reaction"] == "CHEER"


async def test_freshly_created_post_reports_empty_aggregates() -> None:
    async with community_app(user_id=7) as env:
        resp = await env.client.post(
            "/api/v1/community/posts", json={"title": "새 글", "content": "내용"}
        )

    body = resp.json()
    assert body["comment_count"] == 0
    assert body["reactions"] == {"cheer": 0, "relate": 0, "like": 0, "total": 0}
    assert body["my_reaction"] is None


async def test_update_response_keeps_the_aggregates() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env, "원래 제목")
        await env.client.post(
            f"/api/v1/community/posts/{post_id}/comments", json={"content": "댓글"}
        )
        await env.client.put(
            f"/api/v1/community/posts/{post_id}/reaction", json={"kind": "RELATE"}
        )

        resp = await env.client.patch(
            f"/api/v1/community/posts/{post_id}", json={"title": "고친 제목"}
        )

    body = resp.json()
    assert body["title"] == "고친 제목"
    assert body["comment_count"] == 1
    assert body["reactions"]["relate"] == 1
    assert body["my_reaction"] == "RELATE"


async def test_deleted_comments_are_excluded_from_detail_count() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        keep = await env.client.post(
            f"/api/v1/community/posts/{post_id}/comments", json={"content": "남는 댓글"}
        )
        drop = await env.client.post(
            f"/api/v1/community/posts/{post_id}/comments", json={"content": "지울 댓글"}
        )
        await env.client.delete(
            f"/api/v1/community/comments/{drop.json()['comment_id']}"
        )

        resp = await env.client.get(f"/api/v1/community/posts/{post_id}")

    assert keep.status_code == 201
    assert resp.json()["comment_count"] == 1


async def test_detail_query_count_does_not_grow_with_comments_or_reactions() -> None:
    async with community_app(user_id=7) as env:
        quiet = await create_post(env, "조용한 글")
        busy = await create_post(env, "붐비는 글")

        for i in range(6):
            await env.client.post(
                f"/api/v1/community/posts/{busy}/comments", json={"content": f"댓글 {i}"}
            )
        for uid in (7, 8, 9):
            env.login(uid)
            await env.client.put(
                f"/api/v1/community/posts/{busy}/reaction", json={"kind": "LIKE"}
            )
        env.login(7)

        env.queries.clear()
        await env.client.get(f"/api/v1/community/posts/{quiet}")
        quiet_queries = len(env.queries)

        env.queries.clear()
        await env.client.get(f"/api/v1/community/posts/{busy}")
        busy_queries = len(env.queries)

    assert quiet_queries == busy_queries, (
        f"조용한 글 {quiet_queries}회 vs 붐비는 글 {busy_queries}회 — 댓글·공감 수에 비례하면 안 된다"
    )
