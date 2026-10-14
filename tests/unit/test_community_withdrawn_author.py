"""탈퇴한 회원의 글·댓글은 남고 작성자는 "삭제된 사용자"로 보인다."""

from app.db.external import users_table
from app.services.user_data_cleanup import delete_user_data
from tests.unit.community_env import Env, community_app, create_post

C = "/api/v1/community"

WITHDRAWN = {"user_id": 8, "name": "삭제된 사용자", "profile_image_url": None, "is_withdrawn": True}


async def _withdraw(env: Env, user_id: int) -> None:
    """Spring 탈퇴 순서 흉내: FastAPI 데이터 정리 후 회원 행 삭제"""
    async with env.sessions() as db:
        await delete_user_data(db, user_id)
        await db.execute(users_table.delete().where(users_table.c.user_id == user_id))
        await db.commit()


async def test_withdrawn_authors_posts_and_comments_stay_with_placeholder_name() -> None:
    async with community_app(user_id=8) as env:
        post_8 = await create_post(env, title="탈퇴할 사람의 글")
        env.login(7)
        post_7 = await create_post(env, title="남은 사람의 글")
        await env.client.put(f"{C}/posts/{post_8}/reaction", json={"kind": "CHEER"})

        env.login(8)
        comment_8 = (
            await env.client.post(f"{C}/posts/{post_7}/comments", json={"content": "탈퇴할 사람의 댓글"})
        ).json()["comment_id"]
        await env.client.put(f"{C}/posts/{post_7}/reaction", json={"kind": "LIKE"})

        env.login(7)
        await env.client.post(
            f"{C}/posts/{post_7}/comments",
            json={"content": "남은 사람의 답글", "parent_comment_id": comment_8},
        )

        await _withdraw(env, 8)

        feed = (await env.client.get(f"{C}/posts")).json()
        detail = (await env.client.get(f"{C}/posts/{post_8}")).json()
        tree = (await env.client.get(f"{C}/posts/{post_7}/comments")).json()

    posts = {post["post_id"]: post for post in feed["posts"]}
    assert set(posts) == {post_7, post_8}
    assert posts[post_8]["author"] == WITHDRAWN
    assert posts[post_8]["reactions"]["total"] == 1, "남이 누른 공감은 남는다"
    assert posts[post_7]["reactions"]["total"] == 0, "탈퇴자가 누른 공감은 빠진다"
    assert posts[post_7]["author"]["is_withdrawn"] is False

    assert detail["author"] == WITHDRAWN
    assert detail["title"] == "탈퇴할 사람의 글"

    [thread] = tree["comments"]
    assert thread["comment_id"] == comment_8
    assert thread["author"] == WITHDRAWN
    assert [reply["content"] for reply in thread["replies"]] == ["남은 사람의 답글"]


async def test_existing_user_without_name_is_not_marked_withdrawn() -> None:
    users = (
        {"user_id": 7, "name": None, "profile_image": None, "role": "USER", "status": "ACTIVE"},
    )
    async with community_app(user_id=7, users=users) as env:
        post_id = await create_post(env)
        detail = (await env.client.get(f"{C}/posts/{post_id}")).json()

    assert detail["author"] == {"user_id": 7, "name": None, "profile_image_url": None, "is_withdrawn": False}


async def test_blocker_still_does_not_see_withdrawn_authors_posts() -> None:
    async with community_app(user_id=8) as env:
        post_id = await create_post(env)
        env.login(7)
        await env.client.put(f"{C}/users/8/block")

        await _withdraw(env, 8)

        blocker_feed = (await env.client.get(f"{C}/posts")).json()
        blocker_detail = await env.client.get(f"{C}/posts/{post_id}")
        env.login(9)
        other_feed = (await env.client.get(f"{C}/posts")).json()

    assert blocker_feed["posts"] == []
    assert blocker_detail.status_code == 404
    assert [post["author"] for post in other_feed["posts"]] == [WITHDRAWN]


async def test_no_notifications_are_sent_to_withdrawn_author() -> None:
    async with community_app(user_id=8) as env:
        post_id = await create_post(env)
        comment_id = (
            await env.client.post(f"{C}/posts/{post_id}/comments", json={"content": "내 댓글"})
        ).json()["comment_id"]

        await _withdraw(env, 8)

        env.login(7)
        comment = await env.client.post(f"{C}/posts/{post_id}/comments", json={"content": "댓글"})
        reply = await env.client.post(
            f"{C}/posts/{post_id}/comments", json={"content": "답글", "parent_comment_id": comment_id}
        )
        reaction = await env.client.put(f"{C}/posts/{post_id}/reaction", json={"kind": "LIKE"})

    assert (comment.status_code, reply.status_code, reaction.status_code) == (201, 201, 200)
    assert env.spring.notifications == []


async def test_admin_resolves_report_on_withdrawn_author_without_sanction() -> None:
    async with community_app(user_id=8) as env:
        post_id = await create_post(env)
        env.login(7)
        report_id = (
            await env.client.post(f"{C}/posts/{post_id}/reports", json={"reason": "ABUSE"})
        ).json()["report_id"]

        await _withdraw(env, 8)

        env.login(9)
        resolved = await env.client.post(
            f"/api/v1/admin/community/reports/{report_id}/resolve", json={"action": "BAN", "note": "욕설"}
        )
        env.login(7)
        after = await env.client.get(f"{C}/posts/{post_id}")

    assert resolved.status_code == 200, resolved.text
    assert resolved.json()["status"] == "RESOLVED"
    assert after.status_code == 404
    assert env.spring.moderation_updates == [], "탈퇴한 사람은 Spring 에 제재를 보내지 않는다"
