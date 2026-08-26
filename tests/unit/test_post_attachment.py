from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from app.core.enums import AttachmentKind
from app.db.external import training_records_table
from app.db.models import PostORM, ScenarioORM
from app.schemas.community import AttachmentRequest, PostCreateRequest
from app.services.community_post import create_post as svc_create_post
from app.services.post_attachment import AttachmentInvalidError
from tests.unit.community_env import Env, community_app, create_post


def _scenario(
    user_id: int = 7,
    *,
    is_custom: bool = True,
    is_warmup: bool = False,
    deleted: bool = False,
) -> ScenarioORM:
    now = datetime.now(UTC).replace(tzinfo=None)
    return ScenarioORM(
        title="병원 예약 전화",
        content="처음 거는 예약 전화",
        category="OTHER",
        ai_prompt="You are a hospital desk clerk.",
        call_target="○○병원 접수",
        call_purpose="진료 예약",
        script=[{"step": 1, "ai_goal": "인사", "hint": "예약하려고요"}],
        user_id=user_id,
        is_custom=is_custom,
        is_warmup=is_warmup,
        created_at=now,
        deleted_at=now - timedelta(minutes=1) if deleted else None,
    )


async def _add_scenario(env: Env, row: ScenarioORM) -> int:
    async with env.sessions() as db:
        db.add(row)
        await db.commit()
        await db.refresh(row)
        return row.scenario_id


@pytest.mark.asyncio
async def test_scenario_row_gets_an_id_without_being_told_one() -> None:
    async with community_app() as env:
        scenario_id = await _add_scenario(env, _scenario())

        assert scenario_id is not None
        async with env.sessions() as db:
            stored = await db.scalar(select(func.count()).select_from(ScenarioORM))
        assert stored == 1


@pytest.mark.asyncio
async def test_post_carries_my_custom_scenario() -> None:
    async with community_app() as env:
        scenario_id = await _add_scenario(env, _scenario(user_id=7))

        resp = await env.client.post(
            "/api/v1/community/posts",
            json={
                "title": "드디어 예약 성공",
                "content": "손이 떨렸지만 끝까지 말했어요",
                "attachments": [{"kind": "SCENARIO", "ref_id": scenario_id}],
            },
        )

        assert resp.status_code == 201, resp.text
        detail = await env.client.get(f"/api/v1/community/posts/{resp.json()['post_id']}")
        attachments = detail.json()["attachments"]
        assert [a["kind"] for a in attachments] == ["SCENARIO"]
        assert attachments[0]["scenario"]["title"] == "병원 예약 전화"


@pytest.mark.asyncio
async def test_post_without_attachments_still_works() -> None:
    async with community_app() as env:
        post_id = await create_post(env)

        detail = await env.client.get(f"/api/v1/community/posts/{post_id}")

        assert detail.status_code == 200
        assert detail.json()["attachments"] == []


@pytest.mark.parametrize(
    ("label", "make"),
    [
        ("기본 시나리오", lambda: _scenario(user_id=7, is_custom=False)),
        ("워밍업", lambda: _scenario(user_id=7, is_warmup=True)),
        ("삭제됨", lambda: _scenario(user_id=7, deleted=True)),
        ("남의 것", lambda: _scenario(user_id=8)),
    ],
)
@pytest.mark.asyncio
async def test_these_cannot_be_attached(label: str, make) -> None:
    async with community_app() as env:
        scenario_id = await _add_scenario(env, make())

        resp = await env.client.post(
            "/api/v1/community/posts",
            json={
                "title": "제목",
                "content": "내용",
                "attachments": [{"kind": "SCENARIO", "ref_id": scenario_id}],
            },
        )

        assert resp.status_code == 400, f"{label}: {resp.status_code} {resp.text}"


@pytest.mark.asyncio
async def test_missing_scenario_is_rejected() -> None:
    async with community_app() as env:
        resp = await env.client.post(
            "/api/v1/community/posts",
            json={
                "title": "제목",
                "content": "내용",
                "attachments": [{"kind": "SCENARIO", "ref_id": 99999}],
            },
        )
        assert resp.status_code == 400


@pytest.mark.asyncio
async def test_same_kind_twice_is_rejected() -> None:
    async with community_app() as env:
        first = await _add_scenario(env, _scenario())
        second = await _add_scenario(env, _scenario())

        resp = await env.client.post(
            "/api/v1/community/posts",
            json={
                "title": "제목",
                "content": "내용",
                "attachments": [
                    {"kind": "SCENARIO", "ref_id": first},
                    {"kind": "SCENARIO", "ref_id": second},
                ],
            },
        )

        assert resp.status_code == 400


@pytest.mark.asyncio
async def test_rejected_attachment_leaves_no_orphan_post() -> None:
    async with community_app() as env:
        someone_elses = await _add_scenario(env, _scenario(user_id=8))

        resp = await env.client.post(
            "/api/v1/community/posts",
            json={
                "title": "남을까",
                "content": "내용",
                "attachments": [{"kind": "SCENARIO", "ref_id": someone_elses}],
            },
        )
        assert resp.status_code == 400

        listing = await env.client.get("/api/v1/community/posts")
        assert listing.json()["total"] == 0, "실패한 요청의 글이 남았다"


@pytest.mark.asyncio
async def test_service_rolls_back_the_flushed_post_itself() -> None:
    async with community_app() as env:
        someone_elses = await _add_scenario(env, _scenario(user_id=8))

        async with env.sessions() as db:
            body = PostCreateRequest(
                title="남을까",
                content="내용",
                attachments=[
                    AttachmentRequest(kind=AttachmentKind.SCENARIO, ref_id=someone_elses)
                ],
            )
            with pytest.raises(AttachmentInvalidError):
                await svc_create_post(db, body, user_id=7)

            remaining = await db.scalar(select(func.count()).select_from(PostORM))
            assert remaining == 0, "검증 실패 후에도 게시글이 세션에 남아 있다"


@pytest.mark.asyncio
async def test_list_shows_which_kinds_are_attached_without_extra_queries() -> None:
    async with community_app() as env:
        scenario_id = await _add_scenario(env, _scenario())
        await env.client.post(
            "/api/v1/community/posts",
            json={
                "title": "첨부 있음",
                "content": "내용",
                "attachments": [{"kind": "SCENARIO", "ref_id": scenario_id}],
            },
        )
        await create_post(env, title="첨부 없음")

        env.queries.clear()
        resp = await env.client.get("/api/v1/community/posts")

        selects = [q for q in env.queries if q.strip().upper().startswith("SELECT")]
        assert len(selects) <= 4, f"쿼리가 {len(selects)}개로 늘었다:\n" + "\n".join(selects)

        posts = {p["title"]: p["attachment_kinds"] for p in resp.json()["posts"]}
        assert posts["첨부 있음"] == ["SCENARIO"]
        assert posts["첨부 없음"] == []


async def _add_training_record(env: Env, record_id: int, user_id: int) -> int:
    async with env.sessions() as db:
        await db.execute(
            training_records_table.insert().values(
                record_id=record_id,
                user_id=user_id,
                scenario_name="병원 예약 전화",
                session_type="PRACTICE",
                started_at=datetime.now(UTC).replace(tzinfo=None),
                duration_seconds=132,
                anxiety_score=41,
            )
        )
        await db.commit()
    return record_id


@pytest.mark.asyncio
async def test_post_carries_my_training_record() -> None:
    async with community_app() as env:
        record_id = await _add_training_record(env, 501, user_id=7)

        resp = await env.client.post(
            "/api/v1/community/posts",
            json={
                "title": "이렇게 연습했어요",
                "content": "내용",
                "attachments": [{"kind": "TRAINING_RECORD", "ref_id": record_id}],
            },
        )
        assert resp.status_code == 201, resp.text

        detail = await env.client.get(f"/api/v1/community/posts/{resp.json()['post_id']}")
        block = detail.json()["attachments"][0]["training_record"]
        assert block["scenario_name"] == "병원 예약 전화"
        assert block["anxiety_score"] == 41


@pytest.mark.asyncio
async def test_someone_elses_training_record_is_rejected() -> None:
    async with community_app() as env:
        record_id = await _add_training_record(env, 502, user_id=8)

        resp = await env.client.post(
            "/api/v1/community/posts",
            json={
                "title": "제목",
                "content": "내용",
                "attachments": [{"kind": "TRAINING_RECORD", "ref_id": record_id}],
            },
        )
        assert resp.status_code == 400


@pytest.mark.asyncio
async def test_both_kinds_fit_on_one_post() -> None:
    async with community_app() as env:
        scenario_id = await _add_scenario(env, _scenario())
        record_id = await _add_training_record(env, 503, user_id=7)

        resp = await env.client.post(
            "/api/v1/community/posts",
            json={
                "title": "둘 다",
                "content": "내용",
                "attachments": [
                    {"kind": "SCENARIO", "ref_id": scenario_id},
                    {"kind": "TRAINING_RECORD", "ref_id": record_id},
                ],
            },
        )
        assert resp.status_code == 201, resp.text

        detail = await env.client.get(f"/api/v1/community/posts/{resp.json()['post_id']}")
        assert {a["kind"] for a in detail.json()["attachments"]} == {
            "SCENARIO", "TRAINING_RECORD"
        }


@pytest.mark.asyncio
async def test_patch_without_the_field_leaves_attachments_alone() -> None:
    async with community_app() as env:
        scenario_id = await _add_scenario(env, _scenario())
        created = await env.client.post(
            "/api/v1/community/posts",
            json={
                "title": "제목",
                "content": "내용",
                "attachments": [{"kind": "SCENARIO", "ref_id": scenario_id}],
            },
        )
        post_id = created.json()["post_id"]

        resp = await env.client.patch(
            f"/api/v1/community/posts/{post_id}", json={"title": "제목 고침"}
        )

        assert resp.status_code == 200, resp.text
        assert [a["kind"] for a in resp.json()["attachments"]] == ["SCENARIO"]


@pytest.mark.asyncio
async def test_patch_with_empty_list_clears_attachments() -> None:
    async with community_app() as env:
        scenario_id = await _add_scenario(env, _scenario())
        created = await env.client.post(
            "/api/v1/community/posts",
            json={
                "title": "제목",
                "content": "내용",
                "attachments": [{"kind": "SCENARIO", "ref_id": scenario_id}],
            },
        )
        post_id = created.json()["post_id"]

        resp = await env.client.patch(
            f"/api/v1/community/posts/{post_id}", json={"attachments": []}
        )

        assert resp.status_code == 200, resp.text
        assert resp.json()["attachments"] == []


@pytest.mark.asyncio
async def test_patch_replaces_attachments_wholesale() -> None:
    async with community_app() as env:
        first = await _add_scenario(env, _scenario())
        second = await _add_scenario(env, _scenario())
        created = await env.client.post(
            "/api/v1/community/posts",
            json={
                "title": "제목",
                "content": "내용",
                "attachments": [{"kind": "SCENARIO", "ref_id": first}],
            },
        )
        post_id = created.json()["post_id"]

        resp = await env.client.patch(
            f"/api/v1/community/posts/{post_id}",
            json={"attachments": [{"kind": "SCENARIO", "ref_id": second}]},
        )

        assert resp.status_code == 200, resp.text
        assert resp.json()["attachments"][0]["ref_id"] == second
