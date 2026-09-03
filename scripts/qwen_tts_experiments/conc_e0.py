"""E0 — 현행 서버의 threading.Lock 이 동시 요청에서 프로세스 abort 를 막는지 확인한다.

계획 0047. 모델을 직접 올리지 않고 실행 중인 server.py 에 HTTP 로만 부하를 준다.
판정: 3건을 동시에 던진 뒤에도 /health 가 응답하면 락이 abort 를 막은 것이다.
사용법: python conc_e0.py [포트]  (기본 8010)
"""

import sys
import threading
import time

import requests

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8010
BASE = f"http://localhost:{PORT}"
TEXT = "네, 확인해 보고 바로 안내해 드리겠습니다."
N = 3


def call(idx: int, out: dict) -> None:
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{BASE}/v1/tts/stream",
            json={"voice": "ai", "text": TEXT},
            stream=True,
            timeout=120,
        )
        first = None
        total_bytes = 0
        for chunk in resp.iter_content(4096):
            if chunk:
                if first is None:
                    first = (time.perf_counter() - t0) * 1000
                total_bytes += len(chunk)
        elapsed = time.perf_counter() - t0
        head = f"#{idx} status={resp.status_code} 총={elapsed:5.2f}s 바이트={total_bytes}"
        out[idx] = f"{head} 첫청크={first:6.0f}ms" if first else f"{head} (바디 없음)"
    except Exception as exc:  # noqa: BLE001 - 실패 모드 자체가 관측 대상
        out[idx] = f"#{idx} 예외 {type(exc).__name__}: {exc}"


def main() -> None:
    print(f"=== E0: {BASE} 에 동시 {N}건 ===", flush=True)
    print("사전 /health:", requests.get(f"{BASE}/health", timeout=10).text, flush=True)

    out: dict[int, str] = {}
    threads = [threading.Thread(target=call, args=(i, out)) for i in range(N)]
    wall = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"벽시계 총 {time.perf_counter() - wall:.2f}s", flush=True)
    for i in range(N):
        print(" ", out.get(i, f"#{i} 결과 없음"), flush=True)

    try:
        alive = requests.get(f"{BASE}/health", timeout=10).text
        print("사후 /health:", alive, flush=True)
        print("판정: 서버 생존 — 락이 abort 를 막았다", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"사후 /health 실패: {type(exc).__name__}: {exc}", flush=True)
        print("판정: 서버 사망 — 락이 새거나 부족하다", flush=True)


if __name__ == "__main__":
    main()
