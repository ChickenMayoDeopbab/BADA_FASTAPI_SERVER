import pytest
from sqlalchemy import func, select

from app.core.enums import AttachmentKind
from app.core.timeutil import now_utc
from app.db.external import files_table, training_records_table
from app.db.models import (
    CommunityReportORM,
    CommunityUserBlockORM,
    FeedbackORM,
    FileORM,
    PostAttachmentORM,
    PostCommentORM,
    PostORM,
    PostReactionORM,
    ScenarioORM,
    UsageEventORM,
    VoiceTremorMetricORM,
)
from app.services.morphed_recording import morphed_key
from app.services.user_data_cleanup import delete_user_data
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


async def _seed_user_data(session) -> None:
    now = now_utc()
    await session.execute(
        training_records_table.insert().values(
            record_id=42,
            session_id="sess-1",
            user_id=7,
            recording_key="recordings/sess-1/source.wav",
        )
    )
    await session.execute(
        files_table.insert().values(
            file_id=40,
            file_type="COMMUNITY_IMAGE",
            s3_key="community/user-7.png",
            title="사진",
            user_id=7,
        )
    )
    session.add_all(
        [
            ScenarioORM(
                scenario_id=30,
                title="내 시나리오",
                content="내용",
                category="daily",
                scenario_image="scenario-images/30.png",
                ai_prompt="prompt",
                user_id=7,
                is_custom=True,
                is_warmup=False,
                call_target="병원",
                call_purpose="예약",
                example_audio_url="example_audio/30.wav",
                created_at=now,
            ),
            FileORM(
                file_id=41,
                file_type="SCENARIO_PROFILE",
                s3_key="scenario-images/30.png",
                title="내 시나리오",
            ),
            PostORM(
                post_id=10,
                user_id=7,
                title="내 글",
                content="내용",
                view_count=0,
                created_at=now,
                updated_at=now,
            ),
            PostORM(
                post_id=11,
                user_id=8,
                title="다른 글",
                content="내용",
                view_count=0,
                created_at=now,
                updated_at=now,
            ),
            PostCommentORM(
                comment_id=21,
                post_id=11,
                user_id=7,
                content="탈퇴 회원 댓글",
                created_at=now,
                updated_at=now,
            ),
            PostCommentORM(
                comment_id=22,
                post_id=11,
                parent_comment_id=21,
                user_id=8,
                content="다른 회원 답글",
                created_at=now,
                updated_at=now,
            ),
            PostCommentORM(
                comment_id=23,
                post_id=10,
                user_id=8,
                content="탈퇴 회원 글의 댓글",
                created_at=now,
                updated_at=now,
            ),
            PostReactionORM(
                reaction_id=50,
                post_id=11,
                user_id=7,
                kind="LIKE",
                created_at=now,
            ),
            PostReactionORM(
                reaction_id=51,
                post_id=10,
                user_id=8,
                kind="LIKE",
                created_at=now,
            ),
            PostAttachmentORM(
                attachment_id=60,
                post_id=10,
                kind=AttachmentKind.TRAINING_RECORD.value,
                ref_id=42,
                created_at=now,
            ),
            PostAttachmentORM(
                attachment_id=61,
                post_id=10,
                kind=AttachmentKind.SCENARIO.value,
                ref_id=30,
                created_at=now,
            ),
            PostAttachmentORM(
                attachment_id=62,
                post_id=10,
                kind=AttachmentKind.FILE.value,
                ref_id=40,
                created_at=now,
            ),
            FeedbackORM(
                feedback_id=70,
                session_id="sess-1",
                user_id=7,
                scenario_id=30,
                shake_count=1,
                silence_duration=2,
                highlights="[]",
                created_at=now,
            ),
            VoiceTremorMetricORM(
                metric_id=71,
                session_id="sess-1",
                part_index=0,
                start_sec=0,
                end_sec=1,
                status="PASS",
                script_version="v1",
                created_at=now,
            ),
            CommunityReportORM(
                report_id=80,
                reporter_user_id=7,
                reported_user_id=8,
                target_type="POST",
                target_id=11,
                reason="OTHER",
                content_snapshot={"title": "신고"},
                status="PENDING",
                created_at=now,
                due_at=now,
            ),
            CommunityUserBlockORM(
                block_id=81,
                blocker_user_id=7,
                blocked_user_id=8,
                created_at=now,
            ),
            UsageEventORM(
                event_id=82,
                kind="session",
                session_id="usage-sess-1",
                user_id=7,
                payload={},
                created_at=now,
            ),
        ]
    )
    await session.commit()


async def test_delete_user_data_purges_owned_content_and_keeps_policy_records() -> None:
    storage = _Storage()

    async with community_app() as env, env.sessions() as session:
        await _seed_user_data(session)

        await delete_user_data(session, 7, storage=storage)

        assert await _count(session, ScenarioORM) == 0
        assert await _count(session, FeedbackORM) == 0
        assert await _count(session, VoiceTremorMetricORM) == 0
        assert await _count(session, PostAttachmentORM) == 0
        assert await _count(session, PostReactionORM) == 0
        assert await _count(session, PostORM) == 1
        assert await _count(session, PostCommentORM) == 1
        remaining_comment = await session.get(PostCommentORM, 22)
        assert remaining_comment is not None
        assert remaining_comment.parent_comment_id is None

        assert await _count(session, CommunityReportORM) == 1
        assert await _count(session, CommunityUserBlockORM) == 1
        assert await _count(session, UsageEventORM) == 1

        remaining_file_ids = set(
            (await session.execute(select(FileORM.file_id))).scalars()
        )
        assert remaining_file_ids == {40}

    assert set(storage.deleted) == {
        "example_audio/30.wav",
        "scenario-images/30.png",
        morphed_key("recordings/sess-1/source.wav"),
    }


async def test_s3_failure_keeps_database_rows_for_retry() -> None:
    storage = _Storage(fail=True)

    async with community_app() as env, env.sessions() as session:
        await _seed_user_data(session)

        with pytest.raises(RuntimeError, match="s3 down"):
            await delete_user_data(session, 7, storage=storage)

        assert await _count(session, ScenarioORM) == 1
        assert await _count(session, PostORM) == 2
        assert await _count(session, FeedbackORM) == 1
