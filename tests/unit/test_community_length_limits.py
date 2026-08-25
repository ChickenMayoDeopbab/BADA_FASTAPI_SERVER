"""본문 길이 상한 — 제목만 묶고 본문을 열어두면 3MB 글도 통과한다.

길이는 NFC 정규화와 공백 제거가 끝난 뒤 센다. NFD 는 같은 글자를 여러
코드포인트로 쓰므로, 정규화 전에 세면 한글이 부당하게 짧게 잘리거나
반대로 우회 수단이 된다.
"""

import unicodedata

from tests.unit.community_env import community_app, create_post

C = "/api/v1/community"
POST_MAX = 5_000
COMMENT_MAX = 1_000


async def test_post_body_at_the_limit_is_accepted() -> None:
    async with community_app(user_id=7) as env:
        resp = await env.client.post(
            f"{C}/posts", json={"title": "제목", "content": "가" * POST_MAX}
        )

    assert resp.status_code == 201
    assert len(resp.json()["content"]) == POST_MAX


async def test_post_body_over_the_limit_is_rejected() -> None:
    async with community_app(user_id=7) as env:
        resp = await env.client.post(
            f"{C}/posts", json={"title": "제목", "content": "가" * (POST_MAX + 1)}
        )

    assert resp.status_code == 422


async def test_post_update_obeys_the_same_limit() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)

        ok = await env.client.patch(f"{C}/posts/{post_id}", json={"content": "가" * POST_MAX})
        over = await env.client.patch(f"{C}/posts/{post_id}", json={"content": "가" * (POST_MAX + 1)})

    assert ok.status_code == 200
    assert over.status_code == 422


async def test_comment_at_the_limit_is_accepted() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)

        resp = await env.client.post(
            f"{C}/posts/{post_id}/comments", json={"content": "가" * COMMENT_MAX}
        )

    assert resp.status_code == 201


async def test_comment_over_the_limit_is_rejected() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)

        resp = await env.client.post(
            f"{C}/posts/{post_id}/comments", json={"content": "가" * (COMMENT_MAX + 1)}
        )

    assert resp.status_code == 422


async def test_reply_and_comment_update_obey_the_comment_limit() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        parent_id = (
            await env.client.post(f"{C}/posts/{post_id}/comments", json={"content": "부모"})
        ).json()["comment_id"]

        reply_over = await env.client.post(
            f"{C}/posts/{post_id}/comments",
            json={"content": "가" * (COMMENT_MAX + 1), "parent_comment_id": parent_id},
        )
        update_over = await env.client.patch(
            f"{C}/comments/{parent_id}", json={"content": "가" * (COMMENT_MAX + 1)}
        )

    assert reply_over.status_code == 422
    assert update_over.status_code == 422


async def test_post_body_is_measured_after_nfc_normalization() -> None:
    """NFD 한글은 코드포인트가 3배지만, 정규화 후 5000자면 통과해야 한다."""
    nfd_body = unicodedata.normalize("NFD", "각" * POST_MAX)
    assert len(nfd_body) > POST_MAX, "전제: NFD 는 코드포인트가 더 많다"

    async with community_app(user_id=7) as env:
        resp = await env.client.post(
            f"{C}/posts", json={"title": "제목", "content": nfd_body}
        )

    assert resp.status_code == 201
    assert len(resp.json()["content"]) == POST_MAX


async def test_trailing_whitespace_does_not_count_toward_the_limit() -> None:
    async with community_app(user_id=7) as env:
        resp = await env.client.post(
            f"{C}/posts", json={"title": "제목", "content": "가" * POST_MAX + "   \n  "}
        )

    assert resp.status_code == 201
