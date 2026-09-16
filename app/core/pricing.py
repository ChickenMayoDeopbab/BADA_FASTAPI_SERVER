from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

CHECKED_ON = "2026-09-16"
GEMINI_PRICING_URL = "https://ai.google.dev/gemini-api/docs/pricing"
ANTHROPIC_PRICING_URL = "https://platform.claude.com/docs/en/about-claude/pricing"
STT_PRICING_URL = "https://cloud.google.com/speech-to-text/pricing"
ELEVEN_API_PRICING_URL = "https://elevenlabs.io/pricing/api"
ELEVEN_PLAN_PRICING_URL = "https://elevenlabs.io/pricing"

TOKENS_PER_M = 1_000_000


@dataclass(frozen=True)
class Price:
    usd: float       # per 단위당 USD
    per: float       # 예: 1_000_000 토큰, 1_000 문자, 60 초, 1 장
    unit: str        # tokens | chars | sec | image
    source: str
    checked_on: str
    note: str = ""

    def cost(self, quantity: float) -> float:
        return float(quantity) / self.per * self.usd


PRICES: dict[tuple[str, str, str], Price] = {
    ("gemini", "gemini-3.5-flash-lite", "input_tokens"): Price(
        0.30, TOKENS_PER_M, "tokens", GEMINI_PRICING_URL, CHECKED_ON, "text/image/video/audio 입력"
    ),
    ("gemini", "gemini-3.5-flash-lite", "cached_tokens"): Price(
        0.03, TOKENS_PER_M, "tokens", GEMINI_PRICING_URL, CHECKED_ON, "implicit 캐시 히트, 저장료 없음"
    ),
    ("gemini", "gemini-3.5-flash-lite", "output_tokens"): Price(
        2.50, TOKENS_PER_M, "tokens", GEMINI_PRICING_URL, CHECKED_ON, "thinking 토큰 포함"
    ),
    ("anthropic", "claude-sonnet-4-6", "input_tokens"): Price(
        3.0, TOKENS_PER_M, "tokens", ANTHROPIC_PRICING_URL, CHECKED_ON
    ),
    ("anthropic", "claude-sonnet-4-6", "output_tokens"): Price(
        15.0, TOKENS_PER_M, "tokens", ANTHROPIC_PRICING_URL, CHECKED_ON
    ),
    ("anthropic", "claude-sonnet-4-6", "cache_read_tokens"): Price(
        0.30, TOKENS_PER_M, "tokens", ANTHROPIC_PRICING_URL, CHECKED_ON
    ),
    ("anthropic", "claude-sonnet-4-6", "cache_write_tokens"): Price(
        3.75, TOKENS_PER_M, "tokens", ANTHROPIC_PRICING_URL, CHECKED_ON, "5분 캐시 쓰기"
    ),
    ("anthropic", "claude-sonnet-4-20250514", "input_tokens"): Price(
        3.0, TOKENS_PER_M, "tokens", ANTHROPIC_PRICING_URL, CHECKED_ON, "Sonnet 4 (페이지상 retired)"
    ),
    ("anthropic", "claude-sonnet-4-20250514", "output_tokens"): Price(
        15.0, TOKENS_PER_M, "tokens", ANTHROPIC_PRICING_URL, CHECKED_ON, "Sonnet 4 (페이지상 retired)"
    ),
    ("anthropic", "claude-sonnet-4-20250514", "cache_read_tokens"): Price(
        0.30, TOKENS_PER_M, "tokens", ANTHROPIC_PRICING_URL, CHECKED_ON, "Sonnet 4 (페이지상 retired)"
    ),
    ("anthropic", "claude-sonnet-4-20250514", "cache_write_tokens"): Price(
        3.75, TOKENS_PER_M, "tokens", ANTHROPIC_PRICING_URL, CHECKED_ON, "5분 캐시 쓰기, Sonnet 4"
    ),
    ("gemini", "gemini-2.5-flash-image", "image"): Price(
        0.039, 1, "image", GEMINI_PRICING_URL, CHECKED_ON, "standard; ≤1024px = 1290 토큰"
    ),
    ("google_stt", "chirp_3", "stt_sec"): Price(
        0.016, 60, "sec", STT_PRICING_URL, CHECKED_ON,
        "사용자 확인 $0.016/분; 초당 = 분당 ÷ 60, 반올림·최소 과금 미반영",
    ),
    ("gemini_live", "gemini-3.5-transcribe-live", "stt_sec"): Price(
        0.005, 60, "sec", GEMINI_PRICING_URL, CHECKED_ON, "참고; 운영 미사용"
    ),
    ("eleven", "eleven_flash_v2_5", "chars_payg"): Price(
        0.05, 1_000, "chars", ELEVEN_API_PRICING_URL, CHECKED_ON, "종량제 참고값"
    ),
}


@dataclass(frozen=True)
class ElevenPlan:
    name: str
    usd_per_month: float
    credits_per_month: int
    credits_per_char_low: float
    credits_per_char_high: float
    source: str
    checked_on: str
    note: str

    @property
    def usd_per_credit(self) -> float:
        return self.usd_per_month / self.credits_per_month

    def cost_range(self, chars: float) -> tuple[float, float]:
        chars = float(chars)
        return (
            chars * self.credits_per_char_low * self.usd_per_credit,
            chars * self.credits_per_char_high * self.usd_per_credit,
        )


ELEVEN_PLAN = ElevenPlan(
    name="Starter",
    usd_per_month=6.0,
    credits_per_month=30_000,
    credits_per_char_low=0.5,
    credits_per_char_high=1.0,
    source=ELEVEN_PLAN_PRICING_URL,
    checked_on=CHECKED_ON,
    note="플랜은 사용자 확인. 문자당 크레딧은 공식 원문 '0.5~1 크레딧' 범위, 한도 초과 요금 미확인",
)


@dataclass(frozen=True)
class Fx:
    krw_per_usd: float
    basis: str
    checked_on: str


FX = Fx(krw_per_usd=1368.0, basis="사용자 지정", checked_on=CHECKED_ON)

# stt_engine -> 단가표 키
STT_ENGINE_MODEL: dict[str, tuple[str, str]] = {
    "chirp": ("google_stt", "chirp_3"),
    "gemini_live": ("gemini_live", "gemini-3.5-transcribe-live"),
}

# 자체 호스팅이라 0 원으로 TTS 엔진 표기
ZERO_COST_TTS_ENGINES: dict[str, str] = {"qwen": "자체 호스팅(학교 GPU), 0원 표기 — 사용자 결정 2026-09-16"}

DEFAULT_ELEVEN_MODEL = "eleven_flash_v2_5"


@dataclass(frozen=True)
class CostLine:
    item: str
    quantity: float
    unit: str
    usd_low: float | None
    usd_high: float | None
    priced: bool
    note: str = ""
    reference: bool = False  # 참고선(합계 미포함)

    @property
    def krw_low(self) -> float | None:
        return None if self.usd_low is None else self.usd_low * FX.krw_per_usd

    @property
    def krw_high(self) -> float | None:
        return None if self.usd_high is None else self.usd_high * FX.krw_per_usd


def get_price(provider: str | None, model: str | None, item: str) -> Price | None:
    return PRICES.get((str(provider or ""), str(model or ""), item))


def _qty(payload: Mapping[str, object], key: str) -> float:
    try:
        return float(payload.get(key) or 0)
    except (TypeError, ValueError):
        return 0.0


def _line(item: str, quantity: float, unit: str, price: Price | None, note: str = "") -> CostLine:
    if price is None:
        return CostLine(item, quantity, unit, None, None, False, note or "단가 미설정")
    usd = price.cost(quantity)
    return CostLine(item, quantity, unit, usd, usd, True, note or price.note)


def gemini_text_lines(
    prefix: str, model: str | None, *, prompt: float, cached: float, output: float, thought: float
) -> list[CostLine]:
    """Gemini 텍스트 과금 3줄. 입력 = prompt − cached, 캐시분 = cached, 출력 = candidates + thoughts"""
    billable_input = max(prompt - cached, 0.0)
    return [
        _line(f"{prefix}_input_tokens", billable_input, "tokens",
              get_price("gemini", model, "input_tokens"), f"prompt−cached, {model}"),
        _line(f"{prefix}_cached_tokens", cached, "tokens",
              get_price("gemini", model, "cached_tokens"), f"캐시 히트, {model}"),
        _line(f"{prefix}_output_tokens", output + thought, "tokens",
              get_price("gemini", model, "output_tokens"), f"candidates+thoughts, {model}"),
    ]


def tts_engine_lines(
    engine: str, chars: float, model: str | None = None, *, label: str = "tts_chars"
) -> list[CostLine]:
    engine = engine or "eleven"
    chars = float(chars or 0)
    item = f"{label}[{engine}]"
    if engine in ZERO_COST_TTS_ENGINES:
        return [CostLine(item, chars, "chars", 0.0, 0.0, True, ZERO_COST_TTS_ENGINES[engine])]
    if engine == "eleven":
        low, high = ELEVEN_PLAN.cost_range(chars)
        lines = [CostLine(
            item, chars, "chars", low, high, True,
            f"{ELEVEN_PLAN.name} 플랜 평균 단가, 문자당 "
            f"{ELEVEN_PLAN.credits_per_char_low}~{ELEVEN_PLAN.credits_per_char_high} 크레딧",
        )]
        payg = get_price("eleven", model or DEFAULT_ELEVEN_MODEL, "chars_payg")
        if payg is not None:
            usd = payg.cost(chars)
            lines.append(CostLine(f"{item} PAYG 참고", chars, "chars", usd, usd, True,
                                  "종량제였다면 — 합계 미포함", reference=True))
        return lines
    return [CostLine(item, chars, "chars", None, None, False, f"TTS 엔진 {engine} 단가 없음")]


def session_cost_lines(payload: Mapping[str, object]) -> list[CostLine]:
    """session_usage payload"""
    lines: list[CostLine] = []
    stt_engine = str(payload.get("stt_engine") or "chirp")
    key = STT_ENGINE_MODEL.get(stt_engine)
    stt_price = get_price(key[0], key[1], "stt_sec") if key else None
    lines.append(_line(f"stt_sec[{stt_engine}]", _qty(payload, "stt_sec"), "sec", stt_price,
                       "" if stt_price else f"STT 엔진 {stt_engine} 단가 없음"))
    model = payload.get("llm_model")
    model_s = str(model) if model is not None else None
    for prefix in ("llm", "feedback"):
        lines.extend(gemini_text_lines(
            prefix, model_s,
            prompt=_qty(payload, f"{prefix}_prompt_tokens"),
            cached=_qty(payload, f"{prefix}_cached_tokens"),
            output=_qty(payload, f"{prefix}_output_tokens"),
            thought=_qty(payload, f"{prefix}_thought_tokens"),
        ))
    tts_model = payload.get("tts_model")
    for key_name in sorted(payload):
        if key_name.startswith("tts_chars_"):
            engine = key_name[len("tts_chars_"):]
            lines.extend(tts_engine_lines(engine, _qty(payload, key_name),
                                          str(tts_model) if tts_model else None))
    return lines


def llm_cost_lines(payload: Mapping[str, object]) -> list[CostLine]:
    """llm_usage payload"""
    provider = str(payload.get("provider") or "")
    model = payload.get("model")
    model_s = str(model) if model is not None else None
    purpose = str(payload.get("purpose") or "")
    tag = f"[{purpose}]" if purpose else ""
    images = _qty(payload, "images")
    if provider == "gemini" and (images or purpose == "thumbnail_image"):
        lines = [_line(f"image{tag}", images, "image", get_price("gemini", model_s, "image"))]
        inp = _qty(payload, "input_tokens")
        if inp:
            lines.append(CostLine(f"input_tokens{tag}", inp, "tokens", None, None, False,
                                  f"{model_s} 입력 토큰 단가 미조회"))
        return lines
    if provider == "gemini":
        return gemini_text_lines(
            f"gen{tag}", model_s,
            prompt=_qty(payload, "input_tokens"), cached=_qty(payload, "cache_read_tokens"),
            output=_qty(payload, "output_tokens"), thought=_qty(payload, "thought_tokens"),
        )
    if provider == "anthropic":
        lines = []
        for item in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
            qty = _qty(payload, item)
            if item.startswith("cache") and qty == 0:
                continue
            lines.append(_line(f"{item}{tag}", qty, "tokens", get_price("anthropic", model_s, item),
                               f"{model_s}"))
        return lines
    return [CostLine(f"llm[{provider}/{model_s}]{tag}", 0.0, "", None, None, False,
                     "단가표에 없는 공급자")]


def tts_cost_lines(payload: Mapping[str, object]) -> list[CostLine]:
    """tts_usage payload"""
    purpose = str(payload.get("purpose") or "tts")
    model = payload.get("model")
    return tts_engine_lines(str(payload.get("engine") or "eleven"), _qty(payload, "chars"),
                            str(model) if model else None, label=f"{purpose}_chars")


def cost_lines(kind: str, payload: Mapping[str, object]) -> list[CostLine]:
    if kind == "session_usage":
        return session_cost_lines(payload)
    if kind == "llm_usage":
        return llm_cost_lines(payload)
    if kind == "tts_usage":
        return tts_cost_lines(payload)
    raise ValueError(f"알 수 없는 사용량 종류: {kind}")


def price_sources() -> list[tuple[str, str]]:
    """실측 확인용"""
    seen: dict[str, str] = {}
    for price in PRICES.values():
        seen.setdefault(price.source, price.checked_on)
    seen.setdefault(ELEVEN_PLAN.source, ELEVEN_PLAN.checked_on)
    return sorted(seen.items())
