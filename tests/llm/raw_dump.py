import asyncio

from google.genai import types

from app.schemas.llm import AiPersonality, ScenarioTurn, TurnContext
from app.services.llm import LLMClient
from app.services.llm_prompt import build_contents, build_system_prompt

_CTX = TurnContext(
    personality=AiPersonality.RUDE,
    scenario_title="병원 예약 변경",
    scenario_role="병원 접수 직원",
    script=[ScenarioTurn(step=1, ai_goal="전화를 받고 어느 병원인지 밝히며 용건을 묻는다")],
    current_step=1,
    history=[
        {"role": "user", "text": "예약 변경하려고요"},
        {"role": "assistant", "text": "본인 확인해야 하니까 이름이랑 생년월일 말씀해주세요."},
        {"role": "user", "text": "싫어요"},
        {"role": "assistant", "text": "본인 확인 안 되면 예약 변경 안 된다고요."},
    ],
    user_utterance="어차피 못 끊으면서",
)


async def main() -> None:
    client = LLMClient()
    system_prompt = build_system_prompt(_CTX)
    contents = build_contents(_CTX)

    config = client._build_gen_config(system_prompt)
    print("=== CONFIG ===")
    print("max_output_tokens:", config.max_output_tokens)
    print("thinking_config:", config.thinking_config)
    print("\n=== RAW CHUNKS ===")

    stream = await client._client.aio.models.generate_content_stream(
        model=client._model,
        contents=contents,
        config=config,
    )
    full = ""
    last_usage = None
    async for chunk in stream:
        text = chunk.text
        print(f"[chunk] text={text!r}")
        cands = getattr(chunk, "candidates", None) or []
        for c in cands:
            fr = getattr(c, "finish_reason", None)
            if fr:
                print(f"  finish_reason={fr}")
        um = getattr(chunk, "usage_metadata", None)
        if um:
            last_usage = um
        if text:
            full += text

    print(f"\n=== 누적 텍스트 ===\n{full!r}")
    print("\n=== TOKEN USAGE (마지막 청크) ===")
    if last_usage:
        print("prompt_token_count    :", last_usage.prompt_token_count)
        print("thoughts_token_count  :", last_usage.thoughts_token_count)
        print("candidates_token_count:", last_usage.candidates_token_count)
        print("total_token_count     :", last_usage.total_token_count)
    else:
        print("usage_metadata 없음")


if __name__ == "__main__":
    asyncio.run(main())