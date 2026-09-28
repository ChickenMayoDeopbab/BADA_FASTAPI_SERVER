import pytest
from sqlalchemy import func, select

from app.core.enums import AttachmentKind
from app.core.timeutil import now_utc
from app.db.external import training_records_table
from app.db.models import (
    FeedbackORM,
    PostAttachmentORM,
    PostORM,
    VoiceTremorMetricORM,
)
from app.services.morphed_recording import morphed_key
from app.services.training_data_cleanup import delete_training_data
from tests.unit.community_env import community_app


class _Storage:
    def __init__(self, *, fail: bool = False) -> None:
        self.deleted: list[str] = []
        self.fail = fail

    def delete(self, key: str) -> bool:
        if self.fail:
            raise RuntimeError("s3 down")
        self.deleted.append(key)
        return True


async def _count(session, model) -> int:
    return (await session.execute(select(func.count()).select_from(model))).scalar_one()


async def test_delete_training_data_removes_session_rows_attachment_and_morphed_audio() -> None:
    storage = _Storage()
    recording_key = "recordings/sess-1/source.wav"

    async with community_app() as env, env.sessions() as session:
        await session.execute(
            training_records_table.insert().values(
                record_id=42,
                session_id="sess-1",
                user_id=7,
                recording_key=recording_key,
            )
        )
        session.add_all(
            [
                FeedbackORM(
                    feedback_id=1,
                    session_id="sess-1",
                    user_id=7,
                    scenario_id=1,
                    shake_count=2,
                    silence_duration=3,
                    highlights="[]",
                    created_at=now_utc(),
                ),
                VoiceTremorMetricORM(
                    metric_id=1,
                    session_id="sess-1",
                    part_index=0,
                    start_sec=0,
                    end_sec=1,
                    status="PASS",
                    script_version="v1",
                    created_at=now_utc(),
                ),
                PostORM(
                    post_id=10,
                    user_id=7,
                    title="공유 글",
                    content="내용",
                    view_count=0,
                    created_at=now_utc(),
                    updated_at=now_utc(),
                ),
                PostAttachmentORM(
                    attachment_id=20,
                    post_id=10,
                    kind=AttachmentKind.TRAINING_RECORD.value,
                    ref_id=42,
                    created_at=now_utc(),
                ),
            ]
        )
        await session.commit()

        await delete_training_data(session, "sess-1", storage=storage)

        assert await _count(session, FeedbackORM) == 0
        assert await _count(session, VoiceTremorMetricORM) == 0
        assert await _count(session, PostAttachmentORM) == 0

    assert storage.deleted == [morphed_key(recording_key)]


async def test_delete_training_data_is_idempotent_when_session_is_missing() -> None:
    storage = _Storage()

    async with community_app() as env, env.sessions() as session:
        await delete_training_data(session, "missing", storage=storage)
        await delete_training_data(session, "missing", storage=storage)

    assert storage.deleted == []


async def test_s3_failure_keeps_training_data_for_retry() -> None:
    storage = _Storage(fail=True)
    recording_key = "recordings/sess-1/source.wav"

    async with community_app() as env, env.sessions() as session:
        await session.execute(
            training_records_table.insert().values(
                record_id=42,
                session_id="sess-1",
                user_id=7,
                recording_key=recording_key,
            )
        )
        session.add_all(
            [
                FeedbackORM(
                    feedback_id=1,
                    session_id="sess-1",
                    user_id=7,
                    scenario_id=1,
                    shake_count=2,
                    silence_duration=3,
                    highlights="[]",
                    created_at=now_utc(),
                ),
                VoiceTremorMetricORM(
                    metric_id=1,
                    session_id="sess-1",
                    part_index=0,
                    start_sec=0,
                    end_sec=1,
                    status="PASS",
                    script_version="v1",
                    created_at=now_utc(),
                ),
                PostORM(
                    post_id=10,
                    user_id=7,
                    title="공유 글",
                    content="내용",
                    view_count=0,
                    created_at=now_utc(),
                    updated_at=now_utc(),
                ),
                PostAttachmentORM(
                    attachment_id=20,
                    post_id=10,
                    kind=AttachmentKind.TRAINING_RECORD.value,
                    ref_id=42,
                    created_at=now_utc(),
                ),
            ]
        )
        await session.commit()

        with pytest.raises(RuntimeError, match="s3 down"):
            await delete_training_data(session, "sess-1", storage=storage)

        assert await _count(session, FeedbackORM) == 1
        assert await _count(session, VoiceTremorMetricORM) == 1
        assert await _count(session, PostAttachmentORM) == 1
