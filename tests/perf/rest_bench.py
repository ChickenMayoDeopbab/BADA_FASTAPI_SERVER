import argparse
import json
import time
from pathlib import Path

import httpx

from tests.perf._stats import print_table, summarize

_DEFAULT_SPEC = [
    {"name": "openapi.json", "method": "GET", "path": "/openapi.json"},
    {"name": "docs", "method": "GET", "path": "/docs"},
]


def _load_spec(path: str | None) -> list[dict]:
    if path is None:
        return _DEFAULT_SPEC
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("--spec 파일은 JSON 배열이어야 합니다.")
    return data


def _measure(
    client: httpx.Client, ep: dict, count: int, base_headers: dict
) -> tuple[list[float], list[int], float | None]:
    """엔드포인트를 count회 호출. (warm 지연들, 상태코드들, cold 지연) 반환."""
    method = ep.get("method", "GET").upper()
    path = ep["path"]
    headers = {**base_headers, **ep.get("headers", {})}
    body = ep.get("json")

    durations: list[float] = []
    statuses: list[int] = []
    for _ in range(count):
        start = time.perf_counter()
        try:
            resp = client.request(method, path, json=body, headers=headers)
            status = resp.status_code
        except httpx.HTTPError as exc:
            print(f"  [warn] {method} {path} 요청 실패: {exc}")
            status = -1
        durations.append((time.perf_counter() - start) * 1000.0)
        statuses.append(status)

    cold = durations[0] if durations else None
    warm = durations[1:] if len(durations) > 1 else durations
    return warm, statuses, cold


def main() -> None:
    parser = argparse.ArgumentParser(
        description="REST 엔드포인트 지연 벤치마크",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--count", type=int, default=30, help="엔드포인트당 호출 횟수")
    parser.add_argument("--spec", default=None, help="엔드포인트 스펙 JSON 파일")
    parser.add_argument("--token", default=None, help="Bearer 토큰(모든 요청에 적용)")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--json", action="store_true", help="결과를 JSON으로 출력")
    args = parser.parse_args()

    spec = _load_spec(args.spec)
    base_headers = {"Authorization": f"Bearer {args.token}"} if args.token else {}

    rows: list[tuple[str, dict | None]] = []
    raw: dict[str, dict] = {}
    with httpx.Client(base_url=args.base_url, timeout=args.timeout) as client:
        for ep in spec:
            name = ep.get("name", ep["path"])
            warm, statuses, cold = _measure(client, ep, args.count, base_headers)
            summ = summarize(warm)
            rows.append((name, summ))
            ok = sum(1 for s in statuses if 200 <= s < 400)
            raw[name] = {
                "summary": summ,
                "cold_ms": round(cold, 1) if cold is not None else None,
                "calls": len(statuses),
                "ok": ok,
                "statuses": sorted(set(statuses)),
            }

    if args.json:
        print(json.dumps({"base_url": args.base_url, "endpoints": raw}, ensure_ascii=False, indent=2))
        return

    print(f"대상: {args.base_url} · 엔드포인트당 {args.count}회 (콜드스타트 제외 통계)\n")
    print_table(rows)
    print("\n콜드스타트 / 성공률:")
    for name, info in raw.items():
        cold = info["cold_ms"]
        print(f"  {name}: cold={cold}ms · ok={info['ok']}/{info['calls']} · status={info['statuses']}")


if __name__ == "__main__":
    main()
