from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import Any

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam
from google import genai
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.concurrency import loop_semaphore
from app.core.config import Settings, get_settings
from app.core.enums import FileType
from app.db.base import AsyncSessionLocal
from app.db.models import FileORM, ScenarioORM, is_deleted

logger = logging.getLogger(__name__)

_PROMPT_MAX_TOKENS = 300
_DEFAULT_MIME = "image/png"
_IMAGE_MAX_CONCURRENCY = 2
_TIMEOUT_SEC = 90.0

_image_semaphore = loop_semaphore(_IMAGE_MAX_CONCURRENCY)

_THUMBNAIL_PROMPT_SYSTEM = """You write image-generation prompts for thumbnails in a Korean \
phone call training app.
Given a scenario (title, call target, call purpose, description), describe the VISUAL SCENE
of the call target — the person or place being called — in 1-2 English sentences.
Rules:
- Describe only WHAT is in the picture: the character, their role, setting, props, expression.
- Do NOT mention art style, colors, rendering, camera, or quality words. Those are added later.
- Do NOT include any text, letters, numbers, logos, or UI in the scene.
- Focus on the callee (e.g. hospital desk staff, restaurant owner, delivery driver),
  not on the trainee making the call.
- Keep it concrete and single-subject so it reads well at thumbnail size.
Output the description only. No quotes, no preamble, no markdown.
"""

_THUMBNAIL_STYLE = (
    "Charming children's storybook cartoon illustration, rounded chibi-leaning "
    "character proportions with a large head and small body, soft crayon and "
    "gouache texture, warm cozy color palette, hand-drawn wobbly outlines, "
    "gentle shading, simple background, centered single subject, "
    "heartwarming friendly mood. "
    "No text, no letters, no watermark, no photorealism."
)


def _build_image_prompt(scene: str) -> str:
    return f"{scene.strip()}\n\n{_THUMBNAIL_STYLE}"


def _image_key(scenario_id: int, prompt: str, mime: str) -> str:
    """프롬프트가 바뀌면 키가 달라져 캐시가 자동 무효화"""
    digest = hashlib.sha1(prompt.encode()).hexdigest()[:8]
    ext = "jpg" if mime == "image/jpeg" else "png"
    return f"scenario-images/{scenario_id}-{digest}.{ext}"


def _extract_image(response: object) -> tuple[bytes, str]:
    """이미지 응답에서 첫 이미지 파트를 찾는다. 텍스트 파트가 먼저 올 수 있어 순회한다."""
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and inline.data:
                return inline.data, inline.mime_type or _DEFAULT_MIME
    raise ValueError("이미지 응답에 이미지 파트가 없습니다.")


def _s3_client(settings: Settings) -> Any:
    import boto3

    if settings.aws_access_key and settings.aws_secret_key:
        return boto3.client(
            "s3",
            aws_access_key_id=settings.aws_access_key,
            aws_secret_access_key=settings.aws_secret_key,
            region_name=settings.aws_region,
        )
    return boto3.client("s3", region_name=settings.aws_region)


def _upload_image(settings: Settings, key: str, data: bytes, mime: str) -> str | None:
    """S3에 썸네일을 올리고 key를 반환한다. 버킷 미설정이면 None."""
    if not settings.s3_bucket or not data:
        return None

    _s3_client(settings).put_object(
        Bucket=settings.s3_bucket,
        Key=key,
        Body=data,
        ContentType=mime,
    )
    return key


async def _write_scene_prompt(settings: Settings, row: ScenarioORM) -> str:
    """시나리오 정보를 이미지 모델이 이해할 장면 묘사로 번역한다."""
    user_msg = (
        f"Scenario title: {row.title}\n"
        f"Call target: {row.call_target}\n"
        f"Call purpose: {row.call_purpose}\n"
        f"Description: {row.content}"
    )
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)
    ai_response = await client.messages.create(
        model=settings.llm_analysis_model,
        max_tokens=_PROMPT_MAX_TOKENS,
        system=_THUMBNAIL_PROMPT_SYSTEM,
        messages=[MessageParam(role="user", content=user_msg)],
    )
    scene = ai_response.content[0].text.strip()
    if not scene:
        raise ValueError("썸네일 장면 묘사가 비어 있습니다.")
    return scene


async def _generate_image(settings: Settings, prompt: str) -> tuple[bytes, str]:
    client = genai.Client(api_key=settings.gemini_api_key)
    response = await client.aio.models.generate_content(
        model=settings.gemini_image_model,
        contents=prompt,
    )
    return _extract_image(response)


async def _register_file(db: AsyncSession, key: str, title: str) -> None:
    """file 테이블에 레지스트리 행을 남긴다. s3_key가 unique라 중복이면 건너뛴다."""
    result = await db.execute(select(FileORM.file_id).where(FileORM.s3_key == key))
    if result.scalar_one_or_none() is not None:
        return
    db.add(FileORM(file_type=FileType.SCENARIO_PROFILE, s3_key=key, title=title))


async def _generate_and_store(scenario_id: int, settings: Settings) -> None:
    async with AsyncSessionLocal() as db:
        row = await db.get(ScenarioORM, scenario_id)
        if row is None or is_deleted(row) or row.scenario_image:
            return

        scene = await _write_scene_prompt(settings, row)
        prompt = _build_image_prompt(scene)
        image, mime = await _generate_image(settings, prompt)

        key = _image_key(scenario_id, prompt, mime)
        stored_key = await asyncio.to_thread(_upload_image, settings, key, image, mime)
        if stored_key is None:
            return

        await _register_file(db, stored_key, row.title)
        row.scenario_image = stored_key
        await db.commit()


async def generate_scenario_thumbnail(scenario_id: int) -> None:
    settings = get_settings()
    if not settings.s3_bucket:
        return

    try:
        async with asyncio.timeout(_TIMEOUT_SEC), _image_semaphore():
            await _generate_and_store(scenario_id, settings)
    except Exception:
        logger.exception("시나리오 %s 썸네일 생성 실패", scenario_id)
