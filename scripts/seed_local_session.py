import json
import os
import sys
import time

import jwt
import redis


def _env(key: str) -> str:
    path = os.environ.get("ENV_FILE", ".env")
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(f"{key}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(f"{key} 가 {path} 에 없다")


_GOALS = [
    "전화를 받고 어느 병원인지 밝히며 용건을 묻는다",
    "기존 예약자 본인 확인(이름, 생년월일)을 요청한다",
    "변경을 원하는 날짜와 시간을 물어본다",
    "변경 가능 여부를 안내하고 대안을 제시한다",
    "추가로 궁금한 점이 있는지 묻는다",
]


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    sid = sys.argv[1]
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    ttl = int(os.environ.get("SESSION_TTL", "3600"))

    url = os.environ.get("REDIS_URL", "redis://localhost:6379/15")
    client = redis.Redis.from_url(url)
    db = client.connection_pool.connection_kwargs.get("db", 0)
    if not db:
        raise SystemExit(f"REDIS_URL 의 db 가 0 이다({url}) — 작업용 Redis 에는 쓰지 않는다")

    session = {
        "userId": 1,
        "type": "SCENARIO",
        "aiPersonality": "NORMAL",
        "maxDurationSeconds": 900,
        "scenario": {
            "title": "병원 예약 변경",
            "aiRole": "병원 접수 직원",
            # 측정 중 SCENARIO_DONE 으로 끊기지 않게 스텝을 길게 둔다
            "script": [{"step": i + 1, "aiGoal": _GOALS[i % len(_GOALS)]} for i in range(steps)],
        },
    }
    client.set(f"session:{sid}", json.dumps(session, ensure_ascii=False), ex=ttl)
    token = jwt.encode(
        {"sub": "1", "type": "ACCESS", "role": "USER", "exp": int(time.time()) + ttl},
        _env("JWT_SECRET"),
        algorithm=_env("JWT_ALGORITHM"),
    )
    print(sid)
    print(token)


if __name__ == "__main__":
    main()
