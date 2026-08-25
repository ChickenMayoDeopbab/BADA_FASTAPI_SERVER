import logging
import unicodedata
from datetime import datetime

from app.db.models import PostORM, PostReactionORM
from tests.unit.community_env import community_app, create_post

C = "/api/v1/community"
_STAMP = datetime(2026, 8, 25, 12, 0, 0)

NFC = unicodedata.normalize("NFC", "병원 예약")
NFD = unicodedata.normalize("NFD", "병원 예약")


def test_fixtures_really_differ() -> None:
    assert NFC != NFD
    assert len(NFC.encode()) != len(NFD.encode())


async def test_nfd_search_finds_nfc_stored_post() -> None:
    async with community_app(user_id=7) as env:
        await env.client.post(f"{C}/posts", json={"title": NFC, "content": "본문"})

        found = await env.client.get(f"{C}/posts", params={"q": NFD})

    assert found.json()["total"] == 1


async def test_nfc_search_finds_nfd_stored_post() -> None:
    async with community_app(user_id=7) as env:
        await env.client.post(f"{C}/posts", json={"title": NFD, "content": "본문"})

        found = await env.client.get(f"{C}/posts", params={"q": NFC})

    assert found.json()["total"] == 1


async def test_stored_title_and_content_are_normalized_to_nfc() -> None:
    async with community_app(user_id=7) as env:
        post_id = (
            await env.client.post(f"{C}/posts", json={"title": NFD, "content": NFD})
        ).json()["post_id"]

        async with env.sessions() as session:
            row = await session.get(PostORM, post_id)
            stored_title, stored_content = row.title, row.content

    assert stored_title == NFC, "DB 에는 NFC 로 저장돼야 한다"
    assert stored_content == NFC


async def test_comment_content_is_normalized_too() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)

        resp = await env.client.post(
            f"{C}/posts/{post_id}/comments", json={"content": NFD}
        )

    assert resp.json()["content"] == NFC


async def test_updated_title_is_normalized() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)

        await env.client.patch(f"{C}/posts/{post_id}", json={"title": NFD})
        found = await env.client.get(f"{C}/posts", params={"q": NFC})

    assert found.json()["total"] == 1


async def test_unknown_reaction_kind_is_logged(caplog) -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env)
        async with env.sessions() as session:
            session.add(
                PostReactionORM(post_id=post_id, user_id=8, kind="ANGRY", created_at=_STAMP)
            )
            await session.commit()

        with caplog.at_level(logging.WARNING):
            await env.client.get(f"{C}/posts")

    assert "ANGRY" in caplog.text, "정의 밖 kind 가 조용히 사라지면 안 된다"
    assert any(r.levelno >= logging.WARNING for r in caplog.records)
