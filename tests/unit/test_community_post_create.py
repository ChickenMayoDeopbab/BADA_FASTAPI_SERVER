from tests.unit.community_env import FakeRedis, community_app


async def test_create_post_returns_201_with_author_and_zero_views() -> None:
    async with community_app(user_id=7) as env:
        resp = await env.client.post(
            "/api/v1/community/posts",
            json={"title": "떨지 않고 병원 예약했어요", "content": "3번 연습하니까 됐습니다"},
        )

    assert resp.status_code == 201
    body = resp.json()
    assert body["post_id"] > 0
    assert body["title"] == "떨지 않고 병원 예약했어요"
    assert body["content"] == "3번 연습하니까 됐습니다"
    assert body["author"] == {
        "user_id": 7,
        "name": "사용자1",
        "profile_image_url": "profiles/7.png",
    }
    assert body["view_count"] == 0


async def test_get_post_returns_content_and_author() -> None:
    async with community_app(user_id=7) as env:
        created = await env.client.post(
            "/api/v1/community/posts",
            json={"title": "면접 전화 받았습니다", "content": "손이 떨렸지만 끝까지 했어요"},
        )
        post_id = created.json()["post_id"]

        resp = await env.client.get(f"/api/v1/community/posts/{post_id}")

    assert resp.status_code == 200
    body = resp.json()
    assert body["post_id"] == post_id
    assert body["title"] == "면접 전화 받았습니다"
    assert body["content"] == "손이 떨렸지만 끝까지 했어요"
    assert body["author"]["name"] == "사용자1"


async def test_view_count_increments_on_first_read() -> None:
    async with community_app(user_id=7) as env:
        created = await env.client.post(
            "/api/v1/community/posts",
            json={"title": "제목", "content": "내용"},
        )
        assert created.json()["view_count"] == 0
        post_id = created.json()["post_id"]

        resp = await env.client.get(f"/api/v1/community/posts/{post_id}")

    assert resp.json()["view_count"] == 1


async def test_same_user_rereading_does_not_inflate_view_count() -> None:
    redis = FakeRedis()
    async with community_app(user_id=7, redis=redis) as env:
        post_id = (
            await env.client.post(
                "/api/v1/community/posts", json={"title": "제목", "content": "내용"}
            )
        ).json()["post_id"]

        await env.client.get(f"/api/v1/community/posts/{post_id}")
        resp = await env.client.get(f"/api/v1/community/posts/{post_id}")

    assert resp.json()["view_count"] == 1
    assert list(redis.ttls.values()) == [86_400]


async def test_missing_post_returns_404() -> None:
    async with community_app() as env:
        resp = await env.client.get("/api/v1/community/posts/999")

    assert resp.status_code == 404


async def test_different_user_still_counts_a_view() -> None:
    async with community_app(user_id=7) as env:
        post_id = (
            await env.client.post(
                "/api/v1/community/posts", json={"title": "제목", "content": "내용"}
            )
        ).json()["post_id"]

        await env.client.get(f"/api/v1/community/posts/{post_id}")
        env.login(8)
        resp = await env.client.get(f"/api/v1/community/posts/{post_id}")

    assert resp.json()["view_count"] == 2


async def test_blank_title_is_rejected() -> None:
    async with community_app() as env:
        resp = await env.client.post(
            "/api/v1/community/posts", json={"title": "   ", "content": "내용"}
        )

    assert resp.status_code == 422


async def test_view_still_counted_when_redis_is_down() -> None:
    async with community_app(user_id=7, redis=FakeRedis(fail=True)) as env:
        post_id = (
            await env.client.post(
                "/api/v1/community/posts", json={"title": "제목", "content": "내용"}
            )
        ).json()["post_id"]

        resp = await env.client.get(f"/api/v1/community/posts/{post_id}")

    assert resp.status_code == 200
    assert resp.json()["view_count"] == 1
