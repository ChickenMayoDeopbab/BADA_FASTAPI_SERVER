from app.db.external import users_table
from tests.unit.community_env import community_app

C = "/api/v1/community"

_USERS = (
    {"user_id": 7, "name": "사용자1", "profile_image": "profiles/7.png", "role": "USER"},
    {"user_id": 8, "name": "사용자2", "profile_image": None, "role": "USER"},
    {"user_id": 9, "name": "운영자", "profile_image": None, "role": "ADMIN"},
)


async def test_full_community_journey() -> None:
    async with community_app(user_id=7, users=_USERS) as env:
        created = await env.client.post(
            f"{C}/posts",
            json={"title": "병원 예약 전화 성공", "content": "3번 연습하고 걸었더니 됐어요"},
        )
        assert created.status_code == 201
        post_id = created.json()["post_id"]
        assert created.json()["author"]["name"] == "사용자1"

        feed = (await env.client.get(f"{C}/posts")).json()
        assert feed["total"] == 1
        assert feed["posts"][0]["title"] == "병원 예약 전화 성공"
        assert feed["posts"][0]["view_count"] == 0
        assert feed["posts"][0]["comment_count"] == 0

        env.login(8)
        detail = (await env.client.get(f"{C}/posts/{post_id}")).json()
        assert detail["view_count"] == 1
        assert detail["content"] == "3번 연습하고 걸었더니 됐어요"

        again = (await env.client.get(f"{C}/posts/{post_id}")).json()
        assert again["view_count"] == 1

        comment_id = (
            await env.client.post(
                f"{C}/posts/{post_id}/comments", json={"content": "저도 해봐야겠어요"}
            )
        ).json()["comment_id"]

        env.login(7)
        reply = await env.client.post(
            f"{C}/posts/{post_id}/comments",
            json={"content": "꼭 해보세요", "parent_comment_id": comment_id},
        )
        assert reply.status_code == 201
        reply_id = reply.json()["comment_id"]

        blocked = await env.client.post(
            f"{C}/posts/{post_id}/comments",
            json={"content": "3뎁스 시도", "parent_comment_id": reply_id},
        )
        assert blocked.status_code == 400

        env.login(8)
        await env.client.put(f"{C}/posts/{post_id}/reaction", json={"kind": "CHEER"})
        env.login(7)
        await env.client.put(f"{C}/posts/{post_id}/reaction", json={"kind": "LIKE"})
        switched = await env.client.put(
            f"{C}/posts/{post_id}/reaction", json={"kind": "CHEER"}
        )
        assert switched.json()["reactions"] == {
            "cheer": 2, "relate": 0, "like": 0, "total": 2
        }
        assert switched.json()["my_reaction"] == "CHEER"

        feed = (await env.client.get(f"{C}/posts")).json()["posts"][0]
        detail = (await env.client.get(f"{C}/posts/{post_id}")).json()
        assert feed["comment_count"] == detail["comment_count"] == 2
        assert feed["reactions"] == detail["reactions"]
        assert feed["my_reaction"] == detail["my_reaction"] == "CHEER"

        tree = (await env.client.get(f"{C}/posts/{post_id}/comments")).json()["comments"]
        assert [c["content"] for c in tree] == ["저도 해봐야겠어요"]
        assert [r["content"] for r in tree[0]["replies"]] == ["꼭 해보세요"]
        assert tree[0]["author"]["name"] == "사용자2"

        await env.client.post(f"{C}/posts", json={"title": "배달 주문", "content": "무관"})
        hits = (await env.client.get(f"{C}/posts", params={"q": "병원"})).json()
        assert hits["total"] == 1
        assert hits["posts"][0]["post_id"] == post_id

        mine = (await env.client.get(f"{C}/me/posts")).json()
        assert mine["total"] == 2
        env.login(8)
        assert (await env.client.get(f"{C}/me/posts")).json()["total"] == 0

        assert (
            await env.client.patch(f"{C}/posts/{post_id}", json={"title": "가로채기"})
        ).status_code == 403
        assert (await env.client.delete(f"{C}/posts/{post_id}")).status_code == 403

        env.login(7)
        edited = await env.client.patch(
            f"{C}/posts/{post_id}", json={"title": "병원 예약 전화 성공 (수정)"}
        )
        assert edited.status_code == 200
        assert edited.json()["comment_count"] == 2
        assert edited.json()["reactions"]["cheer"] == 2

        env.login(9)
        assert (await env.client.delete(f"{C}/posts/{post_id}")).status_code == 204

        env.login(7)
        assert (await env.client.get(f"{C}/posts/{post_id}")).status_code == 404
        assert (await env.client.get(f"{C}/posts/{post_id}/comments")).status_code == 404
        remaining = (await env.client.get(f"{C}/posts")).json()
        assert [p["title"] for p in remaining["posts"]] == ["배달 주문"]
        assert remaining["total"] == 1
        assert (await env.client.get(f"{C}/me/posts")).json()["total"] == 1


async def test_posts_survive_when_their_author_withdraws() -> None:
    async with community_app(user_id=7, users=_USERS) as env:
        post_id = (
            await env.client.post(
                f"{C}/posts", json={"title": "탈퇴 예정자의 글", "content": "내용"}
            )
        ).json()["post_id"]
        await env.client.post(f"{C}/posts/{post_id}/comments", json={"content": "댓글"})

        async with env.sessions() as session:
            await session.execute(users_table.delete().where(users_table.c.user_id == 7))
            await session.commit()

        env.login(8)
        feed = await env.client.get(f"{C}/posts")
        detail = await env.client.get(f"{C}/posts/{post_id}")
        tree = await env.client.get(f"{C}/posts/{post_id}/comments")

    assert feed.status_code == 200
    assert detail.status_code == 200
    assert tree.status_code == 200
    assert feed.json()["posts"][0]["author"] == {
        "user_id": 7, "name": None, "profile_image_url": None
    }
    assert detail.json()["author"]["name"] is None
    assert tree.json()["comments"][0]["author"]["name"] is None
