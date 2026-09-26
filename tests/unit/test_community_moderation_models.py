from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from app.db.models import CommunityReportORM, CommunityUserBlockORM
from tests.unit.community_env import community_app

_STAMP = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


def _report(**overrides) -> CommunityReportORM:
    values = {
        "reporter_user_id": 7,
        "reported_user_id": 8,
        "target_type": "POST",
        "target_id": 1,
        "reason": "ABUSE",
        "content_snapshot": {"title": "제목", "content": "내용", "attachments": []},
        "created_at": _STAMP,
        "due_at": _STAMP + timedelta(hours=24),
    }
    values.update(overrides)
    return CommunityReportORM(**values)


async def test_moderation_tables_are_created() -> None:
    async with community_app() as env, env.sessions() as session:
        rows = await session.execute(text("SELECT name FROM sqlite_master WHERE type = 'table'"))
        table_names = set(rows.scalars())

    assert {"community_report", "community_user_block"} <= table_names


async def test_schema_blocks_duplicate_reporter_target() -> None:
    async with community_app() as env, env.sessions() as session:
        session.add_all([_report(), _report(reason="SPAM")])
        with pytest.raises(IntegrityError):
            await session.commit()


async def test_schema_allows_different_users_to_report_same_target() -> None:
    async with community_app() as env, env.sessions() as session:
        session.add_all([_report(), _report(reporter_user_id=9)])
        await session.commit()


async def test_schema_blocks_duplicate_user_block() -> None:
    async with community_app() as env, env.sessions() as session:
        session.add_all([
            CommunityUserBlockORM(blocker_user_id=7, blocked_user_id=8, created_at=_STAMP),
            CommunityUserBlockORM(blocker_user_id=7, blocked_user_id=8, created_at=_STAMP),
        ])
        with pytest.raises(IntegrityError):
            await session.commit()


async def test_schema_blocks_self_block() -> None:
    async with community_app() as env, env.sessions() as session:
        session.add(CommunityUserBlockORM(blocker_user_id=7, blocked_user_id=7, created_at=_STAMP))
        with pytest.raises(IntegrityError):
            await session.commit()
