import asyncio
import logging
import sys
import time

from app.schemas.llm import (
    AiPersonality,
    LLMEventType,
    ScenarioTurn,
    TurnContext,
)
from app.services.llm import LLMClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

_SCENARIO_TITLE = "병원 예약 변경"
_SCENARIO_ROLE = "병원 접수 직원"
_SCRIPT = [
    ScenarioTurn(step=1, ai_goal="전화를 받고 어느 병원인지 밝히며 용건을 묻는다"),
    ScenarioTurn(step=2, ai_goal="기존 예약자 본인 확인(이름, 생년월일)을 요청한다"),
    ScenarioTurn(step=3, ai_goal="변경을 원하는 날짜와 시간을 물어본다"),
    ScenarioTurn(step=4, ai_goal="변경 가능 여부를 안내하고 확정한다"),
    ScenarioTurn(step=5, ai_goal="마무리 인사를 한다"),
]

_PERSONALITY = AiPersonality.RUDE

async def run_turn(client: LLMClient, ctx: TurnContext) -> str:
    first_token_at: float | None = None
    started = time.perf_counter()
    collected: list[str] = []

    async for event in client.stream(ctx):
        if event.type == LLMEventType.EMOTION_RESOLVED:
            print(f"\n[emotion] {event.emotion.value if event.emotion else '?'}")
            print("AI: ", end="", flush=True)
        elif event.type == LLMEventType.TEXT_DELTA:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            print(event.text, end="", flush=True)
            collected.append(event.text)
        elif event.type == LLMEventType.SAFETY_BLOCK:
            print("\n[SAFETY_BLOCK] 응답이 차단됨. 오케스트레이터가 fallback 멘트로 대체해야 함.")
        elif event.type == LLMEventType.TURN_END:
            total = (time.perf_counter() - started) * 1000
            ttft = (first_token_at - started) * 1000 if first_token_at else -1
            print(f"\n  (TTFT {ttft:.0f}ms / 턴 전체 {total:.0f}ms)")

    return "".join(collected)


async def main() -> None:
    client = LLMClient()
    history: list[dict] = []
    current_step = 1

    print(f"=== LLM 단독 테스트 (성격={_PERSONALITY.value}, 시나리오={_SCENARIO_TITLE}) ===")
    print("사용자 발화를 입력하세요. 빈 줄이면 종료.\n")

    while True:
        try:
            user_utterance = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료.")
            break
        if not user_utterance:
            print("종료.")
            break

        ctx = TurnContext(
            personality=_PERSONALITY,
            scenario_title=_SCENARIO_TITLE,
            scenario_role=_SCENARIO_ROLE,
            script=_SCRIPT,
            current_step=current_step,
            history=history,
            user_utterance=user_utterance,
        )

        ai_text = await run_turn(client, ctx)

        # 오케스트레이터 역할 대행: 히스토리 누적 + step 진행
        history.append({"role": "user", "text": user_utterance})
        history.append({"role": "assistant", "text": ai_text})
        if current_step < len(_SCRIPT):
            current_step += 1
        print()


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())