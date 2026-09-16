from __future__ import annotations

from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass, field

from app.core.metrics import log_metric

PCM_BYTES_PER_SECOND = 16_000 * 2  # 16 kHz * 16-bit * mono

# 로그 컬럼을 고정
_KNOWN_TTS_ENGINES = ("eleven", "qwen")


def _int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


@dataclass
class SessionUsage:
    turns: int = 0
    llm_prompt_tokens: int = 0
    llm_cached_tokens: int = 0
    llm_output_tokens: int = 0
    llm_thought_tokens: int = 0
    feedback_prompt_tokens: int = 0
    feedback_cached_tokens: int = 0
    feedback_output_tokens: int = 0
    feedback_thought_tokens: int = 0
    tts_chars: dict[str, int] = field(default_factory=dict)
    tts_pcm_bytes: dict[str, int] = field(default_factory=dict)
    stt_bytes: int = 0

    def add_llm_turn(self, usage: Mapping[str, object]) -> None:
        """실시간 턴 하나의 토큰"""
        self.turns += 1
        self.llm_prompt_tokens += _int(usage.get("prompt"))
        self.llm_cached_tokens += _int(usage.get("cached"))
        self.llm_output_tokens += _int(usage.get("output"))
        self.llm_thought_tokens += _int(usage.get("thought"))

    def add_feedback(self, usage_metadata: object) -> None:
        """구간 피드백 호출의 google-genai usage_metadata"""
        self.feedback_prompt_tokens += _int(getattr(usage_metadata, "prompt_token_count", None))
        self.feedback_cached_tokens += _int(
            getattr(usage_metadata, "cached_content_token_count", None)
        )
        self.feedback_output_tokens += _int(
            getattr(usage_metadata, "candidates_token_count", None)
        )
        self.feedback_thought_tokens += _int(getattr(usage_metadata, "thoughts_token_count", None))

    def add_tts(self, engine: str, *, chars: object = 0, pcm_bytes: object = 0) -> None:
        key = str(engine or "eleven")
        self.tts_chars[key] = self.tts_chars.get(key, 0) + _int(chars)
        self.tts_pcm_bytes[key] = self.tts_pcm_bytes.get(key, 0) + _int(pcm_bytes)

    def as_metrics(self) -> dict[str, object]:
        """session_usage dict"""
        out: dict[str, object] = {
            "turns": self.turns,
            "stt_sec": round(self.stt_bytes / PCM_BYTES_PER_SECOND, 3),
            "llm_prompt_tokens": self.llm_prompt_tokens,
            "llm_cached_tokens": self.llm_cached_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "llm_thought_tokens": self.llm_thought_tokens,
            "feedback_prompt_tokens": self.feedback_prompt_tokens,
            "feedback_cached_tokens": self.feedback_cached_tokens,
            "feedback_output_tokens": self.feedback_output_tokens,
            "feedback_thought_tokens": self.feedback_thought_tokens,
        }
        engines = list(_KNOWN_TTS_ENGINES) + sorted(
            set(self.tts_chars) | set(self.tts_pcm_bytes) - set(_KNOWN_TTS_ENGINES)
        )
        for engine in engines:
            out[f"tts_chars_{engine}"] = self.tts_chars.get(engine, 0)
            out[f"tts_audio_sec_{engine}"] = round(
                self.tts_pcm_bytes.get(engine, 0) / PCM_BYTES_PER_SECOND, 3
            )
        return out

_LLM_ZERO = {
    "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
    "cache_write_tokens": 0, "thought_tokens": 0,
}


def llm_usage_fields(provider: str, usage: object) -> dict[str, int]:
    """Anthropic Usage 와 Gemini usage_metadata 같은 키로 맞추기"""
    if usage is None:
        return dict(_LLM_ZERO)
    if provider == "anthropic":
        return {
            "input_tokens": _int(getattr(usage, "input_tokens", None)),
            "output_tokens": _int(getattr(usage, "output_tokens", None)),
            "cache_read_tokens": _int(getattr(usage, "cache_read_input_tokens", None)),
            "cache_write_tokens": _int(getattr(usage, "cache_creation_input_tokens", None)),
            "thought_tokens": 0,
        }
    return {
        "input_tokens": _int(getattr(usage, "prompt_token_count", None)),
        "output_tokens": _int(getattr(usage, "candidates_token_count", None)),
        "cache_read_tokens": _int(getattr(usage, "cached_content_token_count", None)),
        "cache_write_tokens": 0,
        "thought_tokens": _int(getattr(usage, "thoughts_token_count", None)),
    }


def log_llm_usage(
    provider: str,
    model: object,
    purpose: str,
    *,
    usage: object,
    user_id: object = None,
    scenario_id: object = None,
    attempt: int = 1,
    ok: bool = True,
    images: int = 0,
    duration_ms: object = None,
) -> None:
    """llm_usage 지표"""
    with suppress(Exception):
        log_metric(
            "llm_usage",
            provider=provider,
            model=model,
            purpose=purpose,
            user_id=user_id,
            scenario_id=scenario_id,
            attempt=attempt,
            ok=ok,
            images=images,
            duration_ms=duration_ms,
            **llm_usage_fields(provider, usage),
        )


def log_tts_usage(
    engine: str,
    model: object,
    purpose: str,
    *,
    chars: object,
    pcm_bytes: object,
    turns: object,
    user_id: object = None,
    scenario_id: object = None,
    trigger: object = None,
    ok: bool = True,
) -> None:
    """tts_usage 지표"""
    with suppress(Exception):
        log_metric(
            "tts_usage",
            engine=engine,
            model=model,
            purpose=purpose,
            user_id=user_id,
            scenario_id=scenario_id,
            trigger=trigger,
            ok=ok,
            chars=_int(chars),
            audio_sec=round(_int(pcm_bytes) / PCM_BYTES_PER_SECOND, 3),
            turns=_int(turns),
        )
