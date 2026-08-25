from tests.unit.community_env import community_app, create_post


async def test_comment_is_created_on_a_post() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)

        resp = await env.client.post(
            f"/api/v1/community/posts/{post_id}/comments",
            json={"content": "저도 그랬어요"},
        )

    assert resp.status_code == 201
    body = resp.json()
    assert body["comment_id"] > 0
    assert body["content"] == "저도 그랬어요"
    assert body["parent_comment_id"] is None
    assert body["author"]["name"] == "사용자1"


async def _comment(env, post_id: int, content: str, parent: int | None = None) -> int:
    payload = {"content": content}
    if parent is not None:
        payload["parent_comment_id"] = parent
    resp = await env.client.post(f"/api/v1/community/posts/{post_id}/comments", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()["comment_id"]


async def test_reply_is_attached_to_its_parent_comment() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        parent_id = await _comment(env, post_id, "댓글")

        resp = await env.client.post(
            f"/api/v1/community/posts/{post_id}/comments",
            json={"content": "답글", "parent_comment_id": parent_id},
        )

    assert resp.status_code == 201
    assert resp.json()["parent_comment_id"] == parent_id


async def test_reply_to_a_reply_is_rejected() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        parent_id = await _comment(env, post_id, "댓글")
        reply_id = await _comment(env, post_id, "답글", parent=parent_id)

        resp = await env.client.post(
            f"/api/v1/community/posts/{post_id}/comments",
            json={"content": "답글의 답글", "parent_comment_id": reply_id},
        )

    assert resp.status_code == 400


async def test_parent_from_another_post_is_rejected() -> None:
    async with community_app(user_id=7) as env:
        post_a = await create_post(env, "글 A")
        post_b = await create_post(env, "글 B")
        parent_id = await _comment(env, post_a, "A의 댓글")

        resp = await env.client.post(
            f"/api/v1/community/posts/{post_b}/comments",
            json={"content": "엉뚱한 부모", "parent_comment_id": parent_id},
        )

    assert resp.status_code == 400


async def test_missing_parent_comment_is_rejected() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)

        resp = await env.client.post(
            f"/api/v1/community/posts/{post_id}/comments",
            json={"content": "없는 부모", "parent_comment_id": 999},
        )

    assert resp.status_code == 400


async def test_comments_come_back_as_a_two_level_tree_oldest_first() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        first = await _comment(env, post_id, "첫 댓글")
        await _comment(env, post_id, "첫 댓글의 답글", parent=first)
        await _comment(env, post_id, "둘째 댓글")

        resp = await env.client.get(f"/api/v1/community/posts/{post_id}/comments")

    assert resp.status_code == 200
    comments = resp.json()["comments"]
    assert [c["content"] for c in comments] == ["첫 댓글", "둘째 댓글"]
    assert [r["content"] for r in comments[0]["replies"]] == ["첫 댓글의 답글"]
    assert comments[1]["replies"] == []
    assert comments[0]["author"]["name"] == "사용자1"


async def test_comments_of_missing_post_return_404() -> None:
    async with community_app(user_id=7) as env:
        resp = await env.client.get("/api/v1/community/posts/999/comments")

    assert resp.status_code == 404


async def test_comments_of_deleted_post_return_404() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        await _comment(env, post_id, "댓글")
        await env.client.delete(f"/api/v1/community/posts/{post_id}")

        resp = await env.client.get(f"/api/v1/community/posts/{post_id}/comments")

    assert resp.status_code == 404


async def test_author_updates_own_comment() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        comment_id = await _comment(env, post_id, "원래 댓글")

        resp = await env.client.patch(
            f"/api/v1/community/comments/{comment_id}", json={"content": "고친 댓글"}
        )

    assert resp.status_code == 200
    assert resp.json()["content"] == "고친 댓글"


async def test_other_user_cannot_update_someones_comment() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        comment_id = await _comment(env, post_id, "댓글")
        env.login(8)

        resp = await env.client.patch(
            f"/api/v1/community/comments/{comment_id}", json={"content": "남의 댓글"}
        )

    assert resp.status_code == 403


async def test_updating_missing_comment_returns_404() -> None:
    async with community_app(user_id=7) as env:
        resp = await env.client.patch(
            "/api/v1/community/comments/999", json={"content": "없는 댓글"}
        )

    assert resp.status_code == 404


async def test_author_deletes_own_comment_and_it_leaves_the_tree() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        keep = await _comment(env, post_id, "남는 댓글")
        drop = await _comment(env, post_id, "지울 댓글")

        resp = await env.client.delete(f"/api/v1/community/comments/{drop}")
        listed = await env.client.get(f"/api/v1/community/posts/{post_id}/comments")

    assert resp.status_code == 204
    assert [c["comment_id"] for c in listed.json()["comments"]] == [keep]


async def test_admin_can_delete_any_comment() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        comment_id = await _comment(env, post_id, "남의 댓글")
        env.login(9)

        resp = await env.client.delete(f"/api/v1/community/comments/{comment_id}")

    assert resp.status_code == 204


async def test_other_user_cannot_delete_someones_comment() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        comment_id = await _comment(env, post_id, "댓글")
        env.login(8)

        resp = await env.client.delete(f"/api/v1/community/comments/{comment_id}")

    assert resp.status_code == 403


async def test_deleting_a_parent_hides_its_replies_too() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        parent_id = await _comment(env, post_id, "부모 댓글")
        await _comment(env, post_id, "딸린 답글", parent=parent_id)

        await env.client.delete(f"/api/v1/community/comments/{parent_id}")
        listed = await env.client.get(f"/api/v1/community/posts/{post_id}/comments")
        counted = await env.client.get("/api/v1/community/posts")

    assert listed.json()["comments"] == []
    assert counted.json()["posts"][0]["comment_count"] == 0, (
        "트리에서 안 보이는 답글이 카운트에만 남으면 목록과 상세가 어긋난다"
    )


async def test_deleting_already_deleted_comment_returns_404() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        comment_id = await _comment(env, post_id, "댓글")
        await env.client.delete(f"/api/v1/community/comments/{comment_id}")

        resp = await env.client.delete(f"/api/v1/community/comments/{comment_id}")

    assert resp.status_code == 404


async def test_comment_listing_query_count_does_not_grow_with_comment_count() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        for i in range(5):
            parent_id = await _comment(env, post_id, f"댓글 {i}")
            await _comment(env, post_id, f"답글 {i}", parent=parent_id)

        env.queries.clear()
        resp = await env.client.get(f"/api/v1/community/posts/{post_id}/comments")

    assert len(resp.json()["comments"]) == 5
    assert sum(len(c["replies"]) for c in resp.json()["comments"]) == 5
    assert len(env.queries) == 2, "\n---\n".join(env.queries)


async def test_deleting_a_reply_leaves_its_parent_alone() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        parent_id = await _comment(env, post_id, "부모 댓글")
        keep_id = await _comment(env, post_id, "남길 답글", parent=parent_id)
        drop_id = await _comment(env, post_id, "지울 답글", parent=parent_id)

        resp = await env.client.delete(f"/api/v1/community/comments/{drop_id}")
        tree = (await env.client.get(f"/api/v1/community/posts/{post_id}/comments")).json()
        counted = await env.client.get("/api/v1/community/posts")

    assert resp.status_code == 204
    assert [c["comment_id"] for c in tree["comments"]] == [parent_id], "부모는 남아야 한다"
    assert [r["comment_id"] for r in tree["comments"][0]["replies"]] == [keep_id]
    assert counted.json()["posts"][0]["comment_count"] == 2, "부모 1 + 남은 답글 1"


async def test_other_user_cannot_delete_someones_reply() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        parent_id = await _comment(env, post_id, "댓글")
        reply_id = await _comment(env, post_id, "답글", parent=parent_id)
        env.login(8)

        resp = await env.client.delete(f"/api/v1/community/comments/{reply_id}")

    assert resp.status_code == 403


async def test_admin_can_delete_a_reply() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        parent_id = await _comment(env, post_id, "댓글")
        reply_id = await _comment(env, post_id, "답글", parent=parent_id)
        env.login(9)

        resp = await env.client.delete(f"/api/v1/community/comments/{reply_id}")

    assert resp.status_code == 204


async def test_author_updates_own_reply() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        parent_id = await _comment(env, post_id, "댓글")
        reply_id = await _comment(env, post_id, "원래 답글", parent=parent_id)

        resp = await env.client.patch(
            f"/api/v1/community/comments/{reply_id}", json={"content": "고친 답글"}
        )

    assert resp.status_code == 200
    assert resp.json()["content"] == "고친 답글"
    assert resp.json()["parent_comment_id"] == parent_id


async def test_deleting_already_deleted_reply_returns_404() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        parent_id = await _comment(env, post_id, "댓글")
        reply_id = await _comment(env, post_id, "답글", parent=parent_id)
        await env.client.delete(f"/api/v1/community/comments/{reply_id}")

        resp = await env.client.delete(f"/api/v1/community/comments/{reply_id}")

    assert resp.status_code == 404
