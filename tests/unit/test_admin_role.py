"""어드민 판정 — 권한의 근거는 DB users.role 이지 JWT 클레임이 아니다.

JWT 의 role 은 로그인 시점 값이라, 어드민을 해임해도 그 토큰이 만료될 때까지
어드민으로 남는다. 판정을 DB 로 옮기면 해임이 즉시 반영된다.
"""

import pytest
from sqlalchemy import update

from app.core.security import is_admin
from app.db.external import users_table
from tests.unit.community_env import community_app, create_post

C = "/api/v1/community"


@pytest.mark.parametrize("role,expected", [
    ("ADMIN", True),
    ("USER", False),
    ("admin", False),      # DB 는 @Enumerated(STRING) 이라 대문자 한 표현뿐
    ("ROLE_ADMIN", False), # JWT 형식을 넣으면 통과하면 안 된다 (잘못된 경로)
    ("", False),
    (None, False),
])
def test_is_admin_reads_db_representation_only(role, expected) -> None:
    assert is_admin(role) is expected


async def test_admin_can_delete_others_post() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env, "남의 글")
        env.login(9)                       # 9번은 DB role=ADMIN

        resp = await env.client.delete(f"{C}/posts/{post_id}")

    assert resp.status_code == 204


async def test_demoted_admin_loses_privilege_immediately() -> None:
    """토큰을 새로 받지 않아도 해임 즉시 막혀야 한다."""
    async with community_app(user_id=7) as env:
        first = await create_post(env, "글 1")
        second = await create_post(env, "글 2")
        env.login(9)

        before = await env.client.delete(f"{C}/posts/{first}")

        async with env.sessions() as session:      # 해임 — 토큰은 그대로
            await session.execute(
                update(users_table).where(users_table.c.user_id == 9).values(role="USER")
            )
            await session.commit()

        after = await env.client.delete(f"{C}/posts/{second}")

    assert before.status_code == 204
    assert after.status_code == 403, "DB 에서 해임했는데 옛 토큰으로 계속 지울 수 있다"


async def test_promoted_user_gains_privilege_without_new_token() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env, "남의 글")
        env.login(8)

        before = await env.client.delete(f"{C}/posts/{post_id}")

        async with env.sessions() as session:
            await session.execute(
                update(users_table).where(users_table.c.user_id == 8).values(role="ADMIN")
            )
            await session.commit()

        after = await env.client.delete(f"{C}/posts/{post_id}")

    assert before.status_code == 403
    assert after.status_code == 204


async def test_user_without_db_row_is_not_admin() -> None:
    async with community_app(user_id=7) as env:
        post_id = await create_post(env, "남의 글")
        env.login(77)                      # users 에 없는 사용자

        resp = await env.client.delete(f"{C}/posts/{post_id}")

    assert resp.status_code == 403
