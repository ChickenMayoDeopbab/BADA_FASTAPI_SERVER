from datetime import datetime

from app.db.models import PostCommentORM, PostORM, PostReactionORM
from tests.unit.community_env import community_app

_STAMP = datetime(2026, 8, 25, 12, 0, 0)


async def _seed_posts(env, titles: tuple[str, ...]) -> list[int]:
    ids = []
    for title in titles:
        resp = await env.client.post(
            "/api/v1/community/posts", json={"title": title, "content": f"{title} 본문"}
        )
        ids.append(resp.json()["post_id"])
    return ids


async def test_list_returns_posts_newest_first() -> None:
    async with community_app(user_id=7) as env:
        await _seed_posts(env, ("첫 글", "둘째 글", "셋째 글"))

        resp = await env.client.get("/api/v1/community/posts")

    assert resp.status_code == 200
    assert [p["title"] for p in resp.json()["posts"]] == ["셋째 글", "둘째 글", "첫 글"]


async def test_search_matches_title_or_content() -> None:
    async with community_app(user_id=7) as env:
        await env.client.post(
            "/api/v1/community/posts",
            json={"title": "병원 예약 성공", "content": "떨렸지만 끝까지 했다"},
        )
        await env.client.post(
            "/api/v1/community/posts",
            json={"title": "면접 전화", "content": "병원 얘기는 아니지만 비슷했다"},
        )
        await env.client.post(
            "/api/v1/community/posts",
            json={"title": "배달 주문", "content": "완전히 무관한 글"},
        )

        resp = await env.client.get("/api/v1/community/posts", params={"q": "병원"})

    assert [p["title"] for p in resp.json()["posts"]] == ["면접 전화", "병원 예약 성공"]


async def test_search_treats_like_wildcards_as_literal_text() -> None:
    async with community_app(user_id=7) as env:
        await env.client.post(
            "/api/v1/community/posts",
            json={"title": "성공률 100% 달성", "content": "본문"},
        )
        await env.client.post(
            "/api/v1/community/posts",
            json={"title": "1000원 배달 주문", "content": "본문"},
        )

        resp = await env.client.get("/api/v1/community/posts", params={"q": "100%"})

    assert [p["title"] for p in resp.json()["posts"]] == ["성공률 100% 달성"]


async def test_comment_count_counts_replies_and_skips_deleted() -> None:
    async with community_app(user_id=7) as env:
        [post_id] = await _seed_posts(env, ("글",))

        async with env.sessions() as session:
            comment = PostCommentORM(
                post_id=post_id, user_id=8, content="댓글",
                created_at=_STAMP, updated_at=_STAMP,
            )
            session.add(comment)
            await session.flush()
            session.add_all([
                PostCommentORM(
                    post_id=post_id, parent_comment_id=comment.comment_id, user_id=7,
                    content="답글", created_at=_STAMP, updated_at=_STAMP,
                ),
                PostCommentORM(
                    post_id=post_id, user_id=8, content="지운 댓글",
                    created_at=_STAMP, updated_at=_STAMP, deleted_at=_STAMP,
                ),
            ])
            await session.commit()

        resp = await env.client.get("/api/v1/community/posts")

    assert resp.json()["posts"][0]["comment_count"] == 2


async def test_reaction_counts_and_my_reaction() -> None:
    async with community_app(user_id=7) as env:
        [post_id] = await _seed_posts(env, ("글",))

        async with env.sessions() as session:
            session.add_all([
                PostReactionORM(post_id=post_id, user_id=7, kind="CHEER", created_at=_STAMP),
                PostReactionORM(post_id=post_id, user_id=8, kind="CHEER", created_at=_STAMP),
                PostReactionORM(post_id=post_id, user_id=9, kind="LIKE", created_at=_STAMP),
            ])
            await session.commit()

        resp = await env.client.get("/api/v1/community/posts")

    post = resp.json()["posts"][0]
    assert post["reactions"] == {"cheer": 2, "relate": 0, "like": 1, "total": 3}
    assert post["my_reaction"] == "CHEER"


async def test_my_reaction_is_null_when_viewer_has_not_reacted() -> None:
    async with community_app(user_id=7) as env:
        [post_id] = await _seed_posts(env, ("글",))

        async with env.sessions() as session:
            session.add(
                PostReactionORM(post_id=post_id, user_id=8, kind="RELATE", created_at=_STAMP)
            )
            await session.commit()

        resp = await env.client.get("/api/v1/community/posts")

    post = resp.json()["posts"][0]
    assert post["reactions"] == {"cheer": 0, "relate": 1, "like": 0, "total": 1}
    assert post["my_reaction"] is None


async def test_list_query_count_does_not_grow_with_post_count() -> None:
    async with community_app(user_id=7) as env:
        post_ids = await _seed_posts(env, tuple(f"글 {i}" for i in range(5)))

        async with env.sessions() as session:
            for post_id in post_ids:
                session.add_all([
                    PostCommentORM(
                        post_id=post_id, user_id=8, content="댓글",
                        created_at=_STAMP, updated_at=_STAMP,
                    ),
                    PostReactionORM(
                        post_id=post_id, user_id=8, kind="LIKE", created_at=_STAMP
                    ),
                ])
            await session.commit()

        env.queries.clear()
        resp = await env.client.get("/api/v1/community/posts")

    assert len(resp.json()["posts"]) == 5
    assert len(env.queries) == 3, "\n---\n".join(env.queries)


async def test_soft_deleted_post_disappears_from_list() -> None:
    async with community_app(user_id=7) as env:
        post_ids = await _seed_posts(env, ("남는 글", "지운 글"))

        async with env.sessions() as session:
            row = await session.get(PostORM, post_ids[1])
            row.deleted_at = _STAMP
            await session.commit()

        resp = await env.client.get("/api/v1/community/posts")

    assert [p["title"] for p in resp.json()["posts"]] == ["남는 글"]


async def test_preview_truncates_long_content_to_100_chars() -> None:
    async with community_app(user_id=7) as env:
        await env.client.post(
            "/api/v1/community/posts", json={"title": "긴 글", "content": "가" * 150}
        )

        resp = await env.client.get("/api/v1/community/posts")

    assert resp.json()["posts"][0]["content_preview"] == "가" * 100


async def test_paging_splits_results_and_reports_has_next() -> None:
    async with community_app(user_id=7) as env:
        await _seed_posts(env, ("하나", "둘", "셋"))

        first = await env.client.get("/api/v1/community/posts", params={"page": 1, "size": 2})
        second = await env.client.get("/api/v1/community/posts", params={"page": 2, "size": 2})

    assert [p["title"] for p in first.json()["posts"]] == ["셋", "둘"]
    assert first.json()["has_next"] is True
    assert [p["title"] for p in second.json()["posts"]] == ["하나"]
    assert second.json()["has_next"] is False


async def test_total_reports_full_match_count_beyond_current_page() -> None:
    async with community_app(user_id=7) as env:
        await _seed_posts(env, ("하나", "둘", "셋"))

        resp = await env.client.get("/api/v1/community/posts", params={"page": 1, "size": 2})

    body = resp.json()
    assert body["total"] == 3
    assert len(body["posts"]) == 2
    assert body["has_next"] is True


async def test_total_respects_search_and_soft_delete() -> None:
    async with community_app(user_id=7) as env:
        post_ids = await _seed_posts(env, ("병원 예약", "병원 문의", "배달 주문"))

        async with env.sessions() as session:
            row = await session.get(PostORM, post_ids[1])
            row.deleted_at = _STAMP
            await session.commit()

        resp = await env.client.get("/api/v1/community/posts", params={"q": "병원"})

    assert resp.json()["total"] == 1


async def test_total_is_zero_when_nothing_matches() -> None:
    async with community_app(user_id=7) as env:
        await _seed_posts(env, ("병원 예약",))

        resp = await env.client.get("/api/v1/community/posts", params={"q": "없는말"})

    body = resp.json()
    assert body["total"] == 0
    assert body["posts"] == []


async def test_total_survives_a_page_past_the_end() -> None:
    async with community_app(user_id=7) as env:
        await _seed_posts(env, ("하나", "둘", "셋"))

        resp = await env.client.get("/api/v1/community/posts", params={"page": 3, "size": 2})

    body = resp.json()
    assert body["posts"] == []
    assert body["has_next"] is False
    assert body["total"] == 3


async def test_comment_count_ignores_replies_whose_parent_is_gone() -> None:

    async with community_app(user_id=7) as env:
        [post_id] = await _seed_posts(env, ("글",))

        async with env.sessions() as session:
            parent = PostCommentORM(
                post_id=post_id, user_id=7, content="부모",
                created_at=_STAMP, updated_at=_STAMP,
            )
            session.add(parent)
            await session.flush()
            session.add(PostCommentORM(
                post_id=post_id, parent_comment_id=parent.comment_id, user_id=8,
                content="고아 답글", created_at=_STAMP, updated_at=_STAMP,
            ))
            parent.deleted_at = _STAMP
            await session.commit()

        listed = await env.client.get(f"/api/v1/community/posts/{post_id}/comments")
        feed = await env.client.get("/api/v1/community/posts")

    assert listed.json()["comments"] == []
    assert feed.json()["posts"][0]["comment_count"] == 0, (
        "트리에 안 보이는 답글이 카운트에만 남으면 목록과 상세가 어긋난다"
    )
