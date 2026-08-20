import asyncio
import logging
import re

from google import genai
from google.genai import errors, types

from app.core.config import get_settings
from app.schemas.llm import AiEmotion, LLMEvent, LLMEventType, TurnContext
from app.services.feedback_points import split_title
from app.services.llm_prompt import build_contents, build_system_prompt

logger = logging.getLogger(__name__)

_EMOTION_PATTERN = re.compile(r"\[EMOTION:\s*([A-Z_]+)\s*\]", re.IGNORECASE)
_EMOTION_PREFIX = "[EMOTION"

_CONTROL_TAGS: dict[str, LLMEventType] = {
    "[STEP_DONE]": LLMEventType.STEP_DONE,
    "[END_CALL]": LLMEventType.END_CALL,
}

_SUGGEST_TAG = "[SUGGEST]"
_ALL_TAGS = (*_CONTROL_TAGS, _SUGGEST_TAG)

# 사용자 발화 꼬리표 '(지금 단계: N)'를 모델이 따라 말하면 TTS 전에 제거
_STEP_ECHO_PATTERN = re.compile(r"\s*\(\s*지금\s*단계\s*:?\s*\d+\s*\)")
# 지시문 없애기
# 발화성 괄호(예: (창가 자리))는 유지해야 하므로 지문 어휘 기반으로 선별한다.
_STAGE_DIRECTION_PATTERN = re.compile(
    r"\s*\([^()]*(?:한숨|웃음|웃으|피식|킥킥|침묵|콧방귀|헛기침|하품|흐느|훌쩍|울먹|"
    r"중얼|속삭|잠시|한참|멈칫|멈추|목소리|말투|어이없|짜증|빈정|비웃|숨을|숨소리|"
    r"혀를 차|딴청|톤)[^()]*\)"
    r"|\s*\([^()]{0,20}[며듯]\)"
)
# 청크 경계에 걸쳐 쓰다 만 괄호(스텝 에코·지문 공통, 예: '...(길게 한'). 닫힐 때까지 방출 보류
_OPEN_PAREN_TAIL = re.compile(r"\([^()]*$")


def _strip_step_echo(text: str) -> str:
    return _STEP_ECHO_PATTERN.sub("", text)


def _sanitize_speech(text: str) -> str:
    """TTS로 새면 안 되는 비발화 텍스트(스텝 에코·지문성 괄호) 제거"""
    return _STAGE_DIRECTION_PATTERN.sub("", _strip_step_echo(text))


def _find_first_tag(buffer: str) -> tuple[int, str] | None:
    best_idx: int | None = None
    best_tag = ""
    for tag in _ALL_TAGS:
        idx = buffer.find(tag)
        if idx != -1 and (best_idx is None or idx < best_idx):
            best_idx, best_tag = idx, tag
    if best_idx is None:
        return None
    return best_idx, best_tag


def _trailing_partial_len(buffer: str) -> int:
    max_hold = 0
    for tag in _ALL_TAGS:
        limit = min(len(buffer), len(tag) - 1)
        for k in range(limit, 0, -1):
            if tag.startswith(buffer[-k:]):
                max_hold = max(max_hold, k)
                break
    return max_hold


def _drain_pending(pending: str) -> tuple[str, list[LLMEventType], str, bool]:
    emit_parts: list[str] = []
    controls: list[LLMEventType] = []

    while True:
        found = _find_first_tag(pending)
        if found is None:
            break
        idx, tag = found
        if idx > 0:
            emit_parts.append(pending[:idx])
        pending = pending[idx + len(tag):]
        if tag == _SUGGEST_TAG:
            return _sanitize_speech("".join(emit_parts)), controls, pending, True
        controls.append(_CONTROL_TAGS[tag])

    hold = _trailing_partial_len(pending)
    partial = _OPEN_PAREN_TAIL.search(pending)
    if partial is not None:
        hold = max(hold, len(pending) - partial.start())
    if hold:
        emit_parts.append(pending[:-hold])
        pending = pending[-hold:]
    else:
        emit_parts.append(pending)
        pending = ""
    return _sanitize_speech("".join(emit_parts)), controls, pending, False

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

    async def segment_feedback(
        self,
        items: list[dict],
        *,
        scenario_title: str = "",
        call_target: str = "",
        call_purpose: str = "",
    ) -> list[tuple[str, str]]:
        """구간마다 (제목, 내용) 한 쌍. 입력 순서 그대로 같은 개수를 돌려준다.

        items[i] = {"type": GOOD|IMPROVE, "utterance": 그 구간 발화,
                    "avti": 그 턴의 떨림 값 or None, "reason": voice|clean|fallback}

        GOOD 구간은 발화 내용을 칭찬하고, IMPROVE 구간은 목소리가 흔들렸다는 것을
        말해야 한다. 그래서 여기서는 음성 얘기를 금지하지 않는다 —
        구간을 고른 근거가 그것이기 때문이다.
        """
        if not items:
            return []

        situation = " / ".join(
            part for part in (scenario_title, call_target, call_purpose) if part
        ) or "일반 통화"

        lines = []
        for i, item in enumerate(items, start=1):
            kind = "칭찬할 구간" if item["type"] == "GOOD" else "짚어줄 구간"
            measured = (
                f", 떨림 {item['avti']:.1f}/10" if item.get("avti") is not None else ""
            )
            said = (item.get("utterance") or "").strip() or "(발화 인식 안 됨)"
            lines.append(f"{i}. [{kind}{measured}] \"{said}\"")

        system_prompt = (
            "너는 전화 통화를 무서워하는 사람을 돕는 코치야.\n"
            f"이번 통화 상황: {situation}\n"
            "통화에서 뽑아낸 구간들에 대해 한 구간씩 피드백을 써.\n"
            "\n"
            "구간 종류:\n"
            "- [칭찬할 구간]: 그 발화에서 잘한 점을 짚어준다.\n"
            "- [짚어줄 구간]: 이 구간에서 목소리가 흔들렸다. "
            "그 얘기를 하고 다음에 어떻게 할지 알려준다.\n"
            "- '떨림 N/10' 이 붙어 있으면 그 구간의 측정값이다. "
            "낮으면 안정적, 높으면 흔들린 것. 판단에만 쓰고 숫자는 쓰지 마.\n"
            "\n"
            "형식:\n"
            "- 번호마다 한 줄, '번호. 제목 | 내용' 으로 쓴다.\n"
            "- 제목은 20자 안쪽, 내용은 두 문장 안쪽. 둘이 같은 말이면 안 된다.\n"
            "- 입력한 번호 개수만큼만, 순서대로.\n"
            "\n"
            "제목 쓰는 법 — 그 구간에서 무슨 일이 있었는지를 제목에 담는다:\n"
            "  좋은 예: '용건을 먼저 말했어요' / '주소를 말할 때 흔들렸어요'\n"
            "           '날짜를 정해서 갔어요' / '되묻는 말에 바로 답했어요'\n"
            "  나쁜 예: '칭찬' / '아쉬움' / '잘했어요' / '아쉬웠어요'\n"
            "           → 무엇에 대한 얘긴지 알 수 없다. 절대 쓰지 마.\n"
            "\n"
            "내용 규칙:\n"
            "- 그 구간에서 실제로 한 말에 근거해서 써. 없는 장면을 지어내지 마.\n"
            "- 발화를 그대로 따라 읽지 마. 무엇이 좋았는지/아쉬웠는지를 써.\n"
            "- 짚어줄 구간은 '다음엔 ~해봐요' 로 끝낸다.\n"
            "- 목소리 얘기는 '상대방이 ~하게 느꼈을 것 같아요' 처럼 "
            "듣는 사람 입장으로. 몸 상태를 단정하지 마.\n"
            "- 숫자, 지표 이름, 점수를 쓰지 마.\n"
            "- 어려운 말 없이 초등학생도 이해할 단어만. 겁주는 말투 금지.\n"
        )

        try:
            resp = await self._client.aio.models.generate_content(
                model=self._model,
                contents="\n".join(lines),
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    safety_settings=_SAFETY_SETTINGS,
                    temperature=0.6,
                    max_output_tokens=512,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("구간 피드백 생성 실패", exc_info=True)
            return []

        return self._parse_numbered_pairs(resp.text or "", len(items))

    @staticmethod
    def _parse_numbered_pairs(raw: str, n: int) -> list[tuple[str, str]]:
        """'1. 제목 | 내용' 줄들을 순서대로. 부족하면 빈 쌍으로 채운다."""
        pairs: list[tuple[str, str]] = []
        for line in raw.splitlines():
            text = line.strip()
            if not text:
                continue
            text = re.sub(r"^[-*•]\s*", "", text)
            text = re.sub(r"^\d+\s*[.)]\s*", "", text).strip()
            if not text:
                continue
            if "|" in text:
                title, _, content = text.partition("|")
                title, content = title.strip().strip('"“”'), content.strip().strip('"“”')
                if not title or not content:
                    title, content = split_title(text.replace("|", " ").strip())
            else:
                title, content = split_title(text.strip('"“”'))
            pairs.append((title, content))
        pairs = pairs[:n]
        pairs.extend([("", "")] * (n - len(pairs)))
        return pairs

    @staticmethod
    def _parse_numbered(raw: str, n: int) -> list[str]:
        """'1. 칭찬' 형식 응답을 파싱해 길이 n 리스트로 정렬(부족분은 빈 문자열)."""
        items: list[str] = []
        for line in raw.splitlines():
            s = line.strip().lstrip("-*• ").strip()
            s = re.sub(r"^\d+\s*[.)]\s*", "", s).strip().strip('"“”')
            if s:
                items.append(s)
        items = items[:n]
        items.extend([""] * (n - len(items)))
        return items

    async def stream(self, ctx: TurnContext):
        """진입점임 async for로 LLMEvent를 yield"""
        system_prompt = build_system_prompt(ctx)
        contents = build_contents(ctx)
        config = self._build_gen_config(system_prompt)

        head_buffer = ""
        emotion_resolved = False
        pending = ""
        usage = None
        suggest_mode = False
        suggest_parts: list[str] = []

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
                elif suggest_mode:
                    suggest_parts.append(delta)
                else:
                    pending += delta

                if emotion_resolved and not suggest_mode:
                    text, controls, pending, suggest_started = _drain_pending(pending)
                    if text:
                        yield LLMEvent(type=LLMEventType.TEXT_DELTA, text=text)
                    for control in controls:
                        yield LLMEvent(type=control)
                    if suggest_started:
                        suggest_mode = True
                        suggest_parts.append(pending)
                        pending = ""

            if not emotion_resolved:
                yield LLMEvent(
                    type=LLMEventType.EMOTION_RESOLVED, emotion=AiEmotion.NEUTRAL
                )
                if head_buffer and not self._looks_like_partial_head(head_buffer):
                    text, controls, rest, suggest_started = _drain_pending(head_buffer)
                    if text:
                        yield LLMEvent(type=LLMEventType.TEXT_DELTA, text=text)
                    for control in controls:
                        yield LLMEvent(type=control)
                    if suggest_started:
                        suggest_mode = True
                        suggest_parts.append(rest)
                elif head_buffer:
                    logger.warning("미완성 emotion 태그로 응답 종료, 대사 없음: %r", head_buffer)
            elif pending:
                if _trailing_partial_len(pending) == len(pending):
                    logger.debug("미완성 제어 태그로 종료, 누락: %r", pending)
                else:
                    text = _OPEN_PAREN_TAIL.sub("", _sanitize_speech(pending))
                    if text:
                        yield LLMEvent(type=LLMEventType.TEXT_DELTA, text=text)

            if suggest_mode:
                suggestion = "".join(suggest_parts)
                for tag, control in _CONTROL_TAGS.items():
                    if tag in suggestion:
                        suggestion = suggestion.replace(tag, "")
                        yield LLMEvent(type=control)
                suggestion = suggestion.strip()
                if suggestion:
                    yield LLMEvent(type=LLMEventType.SUGGESTION, text=suggestion)

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
