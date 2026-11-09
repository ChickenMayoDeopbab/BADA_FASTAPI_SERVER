"""기본 시나리오 등록을 위한 운영 명령."""

import argparse
import asyncio

from app.core.config import get_settings
from app.db.base import AsyncSessionLocal
from app.db.seed import next_preset_start_id
from app.services.scenario_image_service import generate_pizza_thumbnail


async def _next_id() -> int:
    async with AsyncSessionLocal() as db:
        return await next_preset_start_id(db)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("next-id", "pizza-thumbnail"))
    args = parser.parse_args()

    if args.action == "next-id":
        print(asyncio.run(_next_id()))
        return

    scenario_id = get_settings().default_scenario_start_id
    if scenario_id is None:
        parser.error("DEFAULT_SCENARIO_START_ID is required")
    print(asyncio.run(generate_pizza_thumbnail(scenario_id)))


if __name__ == "__main__":
    main()
