import asyncio
import logging
import re

from google import genai
from google.genai import errors, types

from app.core.config import get_settings
from app.schemas.llm import AiEmotion, LLMEvent, LLMEventType, TurnContext
from app.services.llm_prompt import build_contents, build_system_prompt

logger = logging.getLogger(__name__)

_EMOTION_PATTERN = re.compile(r"\[EMOTION:\s*([A-Z_]+)\s*\]", re.IGNORECASE)
_EMOTION_PREFIX = "[EMOTION"

_CONTROL_TAGS: dict[str, LLMEventType] = {
    "[STEP_DONE]": LLMEventType.STEP_DONE,
    "[END_CALL]": LLMEventType.END_CALL,
}


def _find_first_tag(buffer: str) -> tuple[int, str] | None:
    best_idx: int | None = None
    best_tag = ""
    for tag in _CONTROL_TAGS:
        idx = buffer.find(tag)
        if idx != -1 and (best_idx is None or idx < best_idx):
            best_idx, best_tag = idx, tag
    if best_idx is None:
        return None
    return best_idx, best_tag


def _trailing_partial_len(buffer: str) -> int:
    max_hold = 0
    for tag in _CONTROL_TAGS:
        limit = min(len(buffer), len(tag) - 1)
        for k in range(limit, 0, -1):
            if tag.startswith(buffer[-k:]):
                max_hold = max(max_hold, k)
                break
    return max_hold


def _drain_pending(pending: str) -> tuple[str, list[LLMEventType], str]:
    emit_parts: list[str] = []
    controls: list[LLMEventType] = []

    while True:
        found = _find_first_tag(pending)
        if found is None:
            break
        idx, tag = found
        if idx > 0:
            emit_parts.append(pending[:idx])
        controls.append(_CONTROL_TAGS[tag])
        pending = pending[idx + len(tag):]

    hold = _trailing_partial_len(pending)
    if hold:
        emit_parts.append(pending[:-hold])
        pending = pending[-hold:]
    else:
        emit_parts.append(pending)
        pending = ""
    return "".join(emit_parts), controls, pending

# 혐오, 증오, 성적 표현 막기
_SAFETY_SETTINGS = [
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
    types.SafetySetting(
        category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        threshold=types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    ),
]

class LLMClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.llm_realtime_model

    def _build_gen_config(self, system_prompt: str) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            system_instruction=system_prompt,
            safety_settings=_SAFETY_SETTINGS,
            temperature=0.8, # LLM 내부적으로 확률 분포 넓게 하거나 작게하는거, 최솟값:0, 최댓값:2
            max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(thinking_budget=0),  # 추론 비활성화
        )

    @staticmethod
    def _emit_emotion_head(buffer: str) -> tuple[AiEmotion, str] | None:
        """감정 태그 찾으면 태그 제외해서 반환 못 찾으면 None 반환"""
        match = _EMOTION_PATTERN.search(buffer)
        if match is None:
            return None
        raw = match.group(1).upper()
        try:
            emotion = AiEmotion(raw)
        except ValueError:
            logger.warning("알 수 없는 emotion 값 수신, NEUTRAL로 대체: %s", raw)
            emotion = AiEmotion.NEUTRAL
        remainder = buffer[match.end():].lstrip("\n ")
        return emotion, remainder

    @staticmethod
    def _looks_like_partial_head(buffer: str) -> bool:
        """무슨 태그를 받는 중인지 판별."""
        head = buffer.lstrip("\n ")
        if len(head) <= len(_EMOTION_PREFIX):
            return _EMOTION_PREFIX.startswith(head)
        return head.startswith(_EMOTION_PREFIX) and "]" not in head

    async def warmup(self) -> None:
        """첫 턴 콜드 TTFT 완화용"""
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self._model,
                contents="안녕",
                config=types.GenerateContentConfig(
                    max_output_tokens=1,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            async for _ in stream:
                break
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.debug("LLM 워밍업 실패(무시)", exc_info=True)

    async def stream(self, ctx: TurnContext):
        """진입점임 async for로 LLMEvent를 yield"""
        system_prompt = build_system_prompt(ctx)
        contents = build_contents(ctx)
        config = self._build_gen_config(system_prompt)

        head_buffer = ""
        emotion_resolved = False
        pending = ""
        usage = None

        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config=config,
            )
            async for chunk in stream:
                chunk_usage = getattr(chunk, "usage_metadata", None)
                if chunk_usage is not None:
                    usage = chunk_usage
                if self._is_blocked(chunk):
                    logger.warning("LLM 응답 차단됨 (session role=%s)", ctx.scenario_role)
                    yield LLMEvent(type=LLMEventType.SAFETY_BLOCK)
                    return

                delta = chunk.text or ""
                if not delta:
                    continue

                if not emotion_resolved:
                    head_buffer += delta
                    parsed = self._emit_emotion_head(head_buffer)
                    if parsed is None:
                        if self._looks_like_partial_head(head_buffer):
                            continue
                        yield LLMEvent(
                            type=LLMEventType.EMOTION_RESOLVED, emotion=AiEmotion.NEUTRAL
                        )
                        emotion_resolved = True
                        pending = head_buffer
                        head_buffer = ""
                    else:
                        emotion, remainder = parsed
                        emotion_resolved = True
                        head_buffer = ""
                        yield LLMEvent(
                            type=LLMEventType.EMOTION_RESOLVED, emotion=emotion
                        )
                        pending = remainder
                else:
                    pending += delta

                if emotion_resolved:
                    text, controls, pending = _drain_pending(pending)
                    if text:
                        yield LLMEvent(type=LLMEventType.TEXT_DELTA, text=text)
                    for control in controls:
                        yield LLMEvent(type=control)

            if not emotion_resolved:
                yield LLMEvent(
                    type=LLMEventType.EMOTION_RESOLVED, emotion=AiEmotion.NEUTRAL
                )
                if head_buffer and not self._looks_like_partial_head(head_buffer):
                    text, controls, _ = _drain_pending(head_buffer)
                    if text:
                        yield LLMEvent(type=LLMEventType.TEXT_DELTA, text=text)
                    for control in controls:
                        yield LLMEvent(type=control)
                elif head_buffer:
                    logger.warning("미완성 emotion 태그로 응답 종료, 대사 없음: %r", head_buffer)
            elif pending:
                if _trailing_partial_len(pending) == len(pending):
                    logger.debug("미완성 제어 태그로 종료, 누락: %r", pending)
                else:
                    yield LLMEvent(type=LLMEventType.TEXT_DELTA, text=pending)

            if usage is not None:
                yield LLMEvent(
                    type=LLMEventType.TURN_END,
                    prompt_tokens=getattr(usage, "prompt_token_count", None),
                    cached_tokens=getattr(usage, "cached_content_token_count", None) or 0,
                )
            else:
                yield LLMEvent(type=LLMEventType.TURN_END)

        except asyncio.CancelledError:
            logger.debug("LLM 스트림 취소(barge-in 등)")
            raise
        except errors.APIError:
            logger.exception("LLM API 에러(rate limit/네트워크 등)")
            yield LLMEvent(type=LLMEventType.ERROR)
        except Exception:
            logger.exception("LLM 스트리밍 중 예상치 못한 예외")
            yield LLMEvent(type=LLMEventType.ERROR)

    @staticmethod
    def _is_blocked(chunk) -> bool:
        """불건전한 사유로 응답이 막혔는지 확인"""
        feedback = getattr(chunk, "prompt_feedback", None)
        if feedback is not None and getattr(feedback, "block_reason", None):
            return True
        candidates = getattr(chunk, "candidates", None) or []
        for cand in candidates:
            finish = getattr(cand, "finish_reason", None)
            if finish is not None and str(finish).upper().endswith("SAFETY"):
                return True
        return False
