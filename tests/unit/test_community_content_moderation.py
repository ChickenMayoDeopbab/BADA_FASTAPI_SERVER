import json
from types import SimpleNamespace

import pytest
from google.genai import types
from sqlalchemy import func, select

from app.db.models import PostCommentORM, PostORM
from app.services.community_content_moderation import (
    CommunityContentModerator,
    ContentModerationUnavailableError,
    ObjectionableContentError,
    normalize_for_moderation,
)
from tests.unit.community_env import FakeContentModerator, community_app, create_post


async def _count_rows(env, model) -> int:
    async with env.sessions() as session:
        return await session.scalar(select(func.count()).select_from(model))


async def test_post_create_rejects_objectionable_content_without_saving() -> None:
    moderator = FakeContentModerator(objectionable=True)
    async with community_app(moderator=moderator) as env:
        resp = await env.client.post("/api/v1/community/posts", json={"title": "문제 제목", "content": "문제 본문"})
        post_count = await _count_rows(env, PostORM)

    assert resp.status_code == 422
    assert resp.json()["detail"] == "커뮤니티 운영정책에 위반되는 내용은 등록할 수 없습니다."
    assert moderator.calls == [{"title": "문제 제목", "content": "문제 본문"}]
    assert post_count == 0


async def test_post_update_checks_resulting_text_and_preserves_original_on_rejection() -> None:
    moderator = FakeContentModerator()
    async with community_app(moderator=moderator) as env:
        post_id = await create_post(env, title="원래 제목", content="원래 본문")
        moderator.objectionable = True
        resp = await env.client.patch(f"/api/v1/community/posts/{post_id}", json={"title": "문제 제목"})
        detail = await env.client.get(f"/api/v1/community/posts/{post_id}")

    assert resp.status_code == 422
    assert moderator.calls[-1] == {"title": "문제 제목", "content": "원래 본문"}
    assert detail.json()["title"] == "원래 제목"


async def test_comment_create_and_update_reject_before_saving() -> None:
    moderator = FakeContentModerator()
    async with community_app(moderator=moderator) as env:
        post_id = await create_post(env)
        moderator.objectionable = True
        create_resp = await env.client.post(
            f"/api/v1/community/posts/{post_id}/comments", json={"content": "문제 댓글"}
        )
        comment_count = await _count_rows(env, PostCommentORM)

        moderator.objectionable = False
        original = await env.client.post(f"/api/v1/community/posts/{post_id}/comments", json={"content": "원래 댓글"})
        moderator.objectionable = True
        comment_id = original.json()["comment_id"]
        update_resp = await env.client.patch(f"/api/v1/community/comments/{comment_id}", json={"content": "문제 수정"})
        comments = await env.client.get(f"/api/v1/community/posts/{post_id}/comments")

    assert create_resp.status_code == 422
    assert comment_count == 0
    assert update_resp.status_code == 422
    assert comments.json()["comments"][0]["content"] == "원래 댓글"


async def test_moderation_failure_returns_503_without_saving() -> None:
    moderator = FakeContentModerator(unavailable=True)
    async with community_app(moderator=moderator) as env:
        resp = await env.client.post("/api/v1/community/posts", json={"title": "제목", "content": "본문"})
        post_count = await _count_rows(env, PostORM)

    assert resp.status_code == 503
    assert post_count == 0


async def test_attachment_only_update_skips_redundant_moderation() -> None:
    async with community_app() as env:
        post_id = await create_post(env)
        calls_after_create = len(env.moderator.calls)
        resp = await env.client.patch(f"/api/v1/community/posts/{post_id}", json={"attachments": []})

    assert resp.status_code == 200
    assert len(env.moderator.calls) == calls_after_create


async def test_unauthorized_update_does_not_call_moderation() -> None:
    async with community_app() as env:
        post_id = await create_post(env)
        calls_after_create = len(env.moderator.calls)
        env.login(8)
        resp = await env.client.patch(f"/api/v1/community/posts/{post_id}", json={"content": "남의 글 수정"})

    assert resp.status_code == 403
    assert len(env.moderator.calls) == calls_after_create


class _CapturingModels:
    def __init__(self, response=None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.kwargs: dict | None = None

    async def generate_content(self, **kwargs):
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        return self.response


def _service(models: _CapturingModels) -> CommunityContentModerator:
    moderator = CommunityContentModerator.__new__(CommunityContentModerator)
    moderator._client = SimpleNamespace(aio=SimpleNamespace(models=models))
    moderator._model = "test-model"
    moderator._timeout_seconds = 1.0
    return moderator


def _response(*, allowed: bool = True, category: str = "SAFE", finish_reason=None):
    candidates = [] if finish_reason is None else [SimpleNamespace(finish_reason=finish_reason)]
    return SimpleNamespace(
        parsed={"allowed": allowed, "category": category},
        candidates=candidates,
        prompt_feedback=None,
        usage_metadata=None,
    )


def test_normalization_removes_compatibility_and_invisible_character_obfuscation() -> None:
    assert normalize_for_moderation("  Ａ\u200bＢ\nＣ  ") == "AB C"


async def test_gemini_classifier_receives_normalized_data_and_structured_config() -> None:
    models = _CapturingModels(_response())
    moderator = _service(models)

    await moderator.moderate(title=" Ｔｅｓｔ\u200b ", content="본문\n 내용")

    assert json.loads(models.kwargs["contents"]) == {"title": "Test", "content": "본문 내용"}
    config = models.kwargs["config"]
    assert config.temperature == 0
    assert config.response_mime_type == "application/json"
    assert config.response_schema is not None


async def test_classifier_rejection_and_provider_safety_block_are_objectionable() -> None:
    with pytest.raises(ObjectionableContentError):
        await _service(_CapturingModels(_response(allowed=False, category="ABUSE"))).moderate(content="문제")

    safety_block = _response(finish_reason=types.FinishReason.SAFETY)
    safety_block.parsed = None
    with pytest.raises(ObjectionableContentError):
        await _service(_CapturingModels(safety_block)).moderate(content="차단")


async def test_provider_or_invalid_response_failure_is_unavailable() -> None:
    with pytest.raises(ContentModerationUnavailableError):
        await _service(_CapturingModels(error=RuntimeError("provider down"))).moderate(content="본문")

    invalid = _response()
    invalid.parsed = None
    with pytest.raises(ContentModerationUnavailableError):
        await _service(_CapturingModels(invalid)).moderate(content="본문")
