from tests.unit.community_env import community_app, create_post


async def test_root_comment_notifies_post_author() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        env.login(8)

        response = await env.client.post(
            f"/api/v1/community/posts/{post_id}/comments",
            json={"content": "새 댓글"},
        )

        assert response.status_code == 201
        assert env.spring.notifications == [
            {
                "notification_type": "COMMENT",
                "recipient_user_id": 7,
                "actor_user_id": 8,
                "post_id": post_id,
                "comment_id": response.json()["comment_id"],
            }
        ]


async def test_reply_notifies_parent_comment_author() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        env.login(8)
        parent = await env.client.post(
            f"/api/v1/community/posts/{post_id}/comments",
            json={"content": "부모 댓글"},
        )
        parent_id = parent.json()["comment_id"]
        env.spring.notifications.clear()
        env.login(9)

        response = await env.client.post(
            f"/api/v1/community/posts/{post_id}/comments",
            json={"content": "새 답글", "parent_comment_id": parent_id},
        )

        assert response.status_code == 201
        assert env.spring.notifications == [
            {
                "notification_type": "REPLY",
                "recipient_user_id": 8,
                "actor_user_id": 9,
                "post_id": post_id,
                "comment_id": response.json()["comment_id"],
            }
        ]


async def test_self_comment_and_reply_do_not_create_notifications() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        comment = await env.client.post(
            f"/api/v1/community/posts/{post_id}/comments",
            json={"content": "내 글의 내 댓글"},
        )
        await env.client.post(
            f"/api/v1/community/posts/{post_id}/comments",
            json={
                "content": "내 댓글의 내 답글",
                "parent_comment_id": comment.json()["comment_id"],
            },
        )

        assert env.spring.notifications == []
