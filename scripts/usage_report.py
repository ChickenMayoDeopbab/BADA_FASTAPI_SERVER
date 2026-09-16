"""사용자당 월 API 사용량·비용 리포트 — 계획 0057 F94. 읽기 전용.

원천: DB `usage_event`(F93) 또는 텍스트 로그(`metric=session_usage|llm_usage|tts_usage` 줄, .gz 가능).
단가는 app/core/pricing.py 의 승인표. 단가 없는 항목은 "단가 미설정" 으로 수량만 낸다.

예)
  .venv/bin/python scripts/usage_report.py --source db --month 2026-09
  .venv/bin/python scripts/usage_report.py --source log --log app.log app-2026-09-15.log.gz --month 2026-09 --csv out/
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import json
import os
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.pricing import ELEVEN_PLAN, FX, CostLine, cost_lines, price_sources  # noqa: E402

KINDS = ("session_usage", "llm_usage", "tts_usage")
_SHORT_TO_KIND = {"session": "session_usage", "llm": "llm_usage", "tts": "tts_usage"}
_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})(?:[.,]\d{3})?\S*\s.*?"
    r"metric=(?P<kind>session_usage|llm_usage|tts_usage)\s*(?P<rest>.*)$"
)
_KV = re.compile(r"(\w+)=(\S+)")
UNATTRIBUTED = "미귀속"


@dataclass
class Event:
    kind: str
    ts_utc: datetime
    payload: dict[str, object]

    @property
    def user_key(self) -> object:
        raw = self.payload.get("user_id")
        if raw is None or raw == "None":
            return UNATTRIBUTED
        try:
            return int(raw)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return UNATTRIBUTED


def _coerce(value: str) -> object:
    if value == "None":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _open_text(path: str | os.PathLike):
    p = Path(path)
    if p.suffix == ".gz":
        return gzip.open(p, "rt", encoding="utf-8", errors="replace")
    return open(p, encoding="utf-8", errors="replace")


def parse_log_line(line: str, log_tz: ZoneInfo | None) -> Event | None:
    line = line.strip()
    if not line:
        return None
    if line.startswith("{"):
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            return None
        kind = d.get("metric")
        if kind not in KINDS:
            return None
        ts = d.get("asctime") or d.get("timestamp")
        if not ts:
            return None
        skip = {"asctime", "timestamp", "levelname", "name", "message", "metric"}
        payload = {k: v for k, v in d.items() if k not in skip}
        return Event(kind, _to_utc(str(ts), log_tz), payload)
    m = _LINE.search(line)
    if not m:
        return None
    payload = {k: _coerce(v) for k, v in _KV.findall(m.group("rest"))}
    return Event(m.group("kind"), _to_utc(m.group("ts"), log_tz), payload)


def _to_utc(ts: str, log_tz: ZoneInfo | None) -> datetime:
    naive = datetime.strptime(ts[:19].replace("T", " "), "%Y-%m-%d %H:%M:%S")
    return naive.replace(tzinfo=log_tz or UTC).astimezone(UTC)


def load_from_logs(paths: Iterable[str], log_tz: ZoneInfo | None = None) -> list[Event]:
    events: list[Event] = []
    for path in paths:
        with _open_text(path) as f:
            for line in f:
                ev = parse_log_line(line, log_tz)
                if ev is not None:
                    events.append(ev)
    return events


def month_bounds_utc(month: str, tz: ZoneInfo) -> tuple[datetime, datetime]:
    """`YYYY-MM` 을 tz 기준 [월초, 다음 월초) 의 UTC 경계로."""
    year, mon = (int(x) for x in month.split("-"))
    start = datetime(year, mon, 1, tzinfo=tz)
    end = datetime(year + (mon == 12), 1 if mon == 12 else mon + 1, 1, tzinfo=tz)
    return start.astimezone(UTC), end.astimezone(UTC)


def load_from_db(
    database_url: str,
    *,
    month: str | None = None,
    tz: ZoneInfo | None = None,
    user: int | None = None,
) -> list[Event]:
    """월·사용자 필터는 DB 에서 건다 — 테이블은 통화·호출마다 쌓이므로 전체를 올리지 않는다(리뷰 지적)."""
    async def _load() -> list[Event]:
        from sqlalchemy import select
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

        from app.db.models import UsageEventORM

        stmt = select(UsageEventORM)
        if month:
            lo, hi = month_bounds_utc(month, tz or ZoneInfo("Asia/Seoul"))
            stmt = stmt.where(UsageEventORM.created_at >= lo, UsageEventORM.created_at < hi)
        if user is not None:
            stmt = stmt.where(UsageEventORM.user_id == user)
        stmt = stmt.order_by(UsageEventORM.event_id)

        engine = create_async_engine(database_url)
        try:
            sessions = async_sessionmaker(engine, expire_on_commit=False)
            async with sessions() as db:
                rows = (await db.execute(stmt)).scalars().all()
        finally:
            await engine.dispose()
        out: list[Event] = []
        for row in rows:
            ts = row.created_at if row.created_at.tzinfo else row.created_at.replace(tzinfo=UTC)
            out.append(Event(_SHORT_TO_KIND.get(row.kind, row.kind), ts.astimezone(UTC), dict(row.payload or {})))
        return out

    return asyncio.run(_load())


# --- 집계 ------------------------------------------------------------------------


@dataclass
class Group:
    user: object
    month: str
    sessions: int = 0
    quantities: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    lines: dict[str, CostLine] = field(default_factory=dict)

    def add_quantity(self, key: str, value: object) -> None:
        with suppress(TypeError, ValueError):
            self.quantities[key] += float(value or 0)

    def add_line(self, line: CostLine) -> None:
        cur = self.lines.get(line.item)
        if cur is None:
            self.lines[line.item] = line
            return
        self.lines[line.item] = CostLine(
            line.item, cur.quantity + line.quantity, line.unit,
            None if cur.usd_low is None or line.usd_low is None else cur.usd_low + line.usd_low,
            None if cur.usd_high is None or line.usd_high is None else cur.usd_high + line.usd_high,
            cur.priced and line.priced, cur.note, cur.reference,
        )

    def totals(self) -> tuple[float, float]:
        counted = [ln for ln in self.lines.values() if ln.priced and not ln.reference]
        low = sum(ln.usd_low for ln in counted if ln.usd_low is not None)
        high = sum(ln.usd_high for ln in counted if ln.usd_high is not None)
        return low, high

    def unpriced(self) -> list[CostLine]:
        return [ln for ln in self.lines.values() if not ln.priced]


_SESSION_QTY_KEYS = (
    "turns", "stt_sec", "llm_prompt_tokens", "llm_cached_tokens", "llm_output_tokens", "llm_thought_tokens",
    "feedback_prompt_tokens", "feedback_output_tokens",
)


def aggregate(
    events: Iterable[Event], *, tz: ZoneInfo, month: str | None = None, user: int | None = None
) -> dict[tuple[object, str], Group]:
    groups: dict[tuple[object, str], Group] = {}
    for ev in events:
        ev_month = ev.ts_utc.astimezone(tz).strftime("%Y-%m")
        if month and ev_month != month:
            continue
        key_user = ev.user_key
        if user is not None and key_user != user:
            continue
        g = groups.setdefault((key_user, ev_month), Group(key_user, ev_month))
        p = ev.payload
        if ev.kind == "session_usage":
            g.sessions += 1
            for k in _SESSION_QTY_KEYS:
                g.add_quantity(k, p.get(k))
            for k, v in p.items():
                if k.startswith("tts_chars_") or k.startswith("tts_audio_sec_"):
                    g.add_quantity(k, v)
        elif ev.kind == "llm_usage":
            g.add_quantity("gen_calls", 1)
            g.add_quantity("gen_input_tokens", p.get("input_tokens"))
            g.add_quantity("gen_output_tokens", p.get("output_tokens"))
            g.add_quantity("images", p.get("images"))
        elif ev.kind == "tts_usage":
            engine = p.get("engine") or "eleven"
            g.add_quantity(f"example_chars_{engine}", p.get("chars"))
            g.add_quantity(f"example_audio_sec_{engine}", p.get("audio_sec"))
        for line in cost_lines(ev.kind, p):
            g.add_line(line)
    return groups


# --- 출력 ------------------------------------------------------------------------


def _fmt_usd(v: float | None) -> str:
    return "미설정" if v is None else f"{v:,.4f}"


def _fmt_krw(v: float | None) -> str:
    return "미설정" if v is None else f"{v:,.0f}"


def _fmt_qty(v: float) -> str:
    return f"{v:,.0f}" if float(v).is_integer() else f"{v:,.2f}"


def render_text(groups: dict[tuple[object, str], Group]) -> str:
    out: list[str] = []
    ordered = sorted(groups.values(), key=lambda g: (g.month, str(g.user)))
    out.append("① 수량표 (사용자 × 월)")
    for g in ordered:
        out.append(f"- {g.month} user={g.user}: 세션 {g.sessions}, 턴 {_fmt_qty(g.quantities['turns'])}, "
                   f"STT {_fmt_qty(g.quantities['stt_sec'])}s, "
                   f"LLM 입력 {_fmt_qty(g.quantities['llm_prompt_tokens'])}"
                   f"(캐시 {_fmt_qty(g.quantities['llm_cached_tokens'])}) / "
                   f"출력 {_fmt_qty(g.quantities['llm_output_tokens'])}"
                   f"(+thought {_fmt_qty(g.quantities['llm_thought_tokens'])}), "
                   f"피드백 {_fmt_qty(g.quantities['feedback_prompt_tokens'])}/"
                   f"{_fmt_qty(g.quantities['feedback_output_tokens'])}, "
                   f"TTS 문자 eleven {_fmt_qty(g.quantities['tts_chars_eleven'])} / "
                   f"qwen {_fmt_qty(g.quantities['tts_chars_qwen'])}, "
                   f"생성 호출 {_fmt_qty(g.quantities['gen_calls'])}"
                   f"(토큰 {_fmt_qty(g.quantities['gen_input_tokens'])}/"
                   f"{_fmt_qty(g.quantities['gen_output_tokens'])}), 이미지 {_fmt_qty(g.quantities['images'])}")
    out.append("")
    out.append(f"② 비용표 (USD low~high, KRW @ {FX.krw_per_usd:,.0f} {FX.basis} {FX.checked_on})")
    for g in ordered:
        low, high = g.totals()
        out.append(f"- {g.month} user={g.user}: 합계 USD {_fmt_usd(low)}~{_fmt_usd(high)} "
                   f"/ KRW {_fmt_krw(low * FX.krw_per_usd)}~{_fmt_krw(high * FX.krw_per_usd)}")
        for ln in sorted(g.lines.values(), key=lambda x: x.item):
            if not ln.priced:
                continue
            ref = " (참고, 합계 미포함)" if ln.reference else ""
            ranged = ln.usd_high != ln.usd_low
            usd = _fmt_usd(ln.usd_low) + (f"~{_fmt_usd(ln.usd_high)}" if ranged else "")
            krw = _fmt_krw(ln.krw_low) + (f"~{_fmt_krw(ln.krw_high)}" if ranged else "")
            note = f"  # {ln.note}" if ln.note else ""
            out.append(f"    {ln.item}: {_fmt_qty(ln.quantity)} {ln.unit} → USD {usd} / KRW {krw}{ref}{note}")
    out.append("")
    out.append("③ 단가 미설정 (수량만)")
    any_unpriced = False
    for g in ordered:
        for ln in g.unpriced():
            any_unpriced = True
            out.append(f"- {g.month} user={g.user}: {ln.item} {_fmt_qty(ln.quantity)} {ln.unit}  # {ln.note}")
    if not any_unpriced:
        out.append("- 없음")
    out.append("")
    out.append("④ 고정비 (사용자 배분 안 함)")
    months = sorted({g.month for g in ordered})
    for m in months:
        out.append(f"- {m}: ElevenLabs {ELEVEN_PLAN.name} 플랜 USD {ELEVEN_PLAN.usd_per_month:,.2f}/월 "
                   f"(포함 크레딧 {ELEVEN_PLAN.credits_per_month:,}) — {ELEVEN_PLAN.note}")
    if not months:
        out.append("- (해당 월 데이터 없음)")
    out.append("")
    out.append("⑤ 단가 출처")
    for source, checked in price_sources():
        out.append(f"- {source} (확인 {checked})")
    return "\n".join(out)


def write_csv(groups: dict[tuple[object, str], Group], out_dir: str | os.PathLike) -> list[Path]:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    qty_keys = sorted({k for g in groups.values() for k in g.quantities})
    files = [d / "usage_quantities.csv", d / "usage_costs.csv", d / "usage_unpriced.csv"]
    with open(files[0], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["month", "user", "sessions", *qty_keys])
        for g in sorted(groups.values(), key=lambda g: (g.month, str(g.user))):
            w.writerow([g.month, g.user, g.sessions, *[g.quantities.get(k, 0) for k in qty_keys]])
    with open(files[1], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["month", "user", "item", "quantity", "unit", "usd_low", "usd_high",
                    "krw_low", "krw_high", "reference", "note"])
        for g in sorted(groups.values(), key=lambda g: (g.month, str(g.user))):
            for ln in sorted(g.lines.values(), key=lambda x: x.item):
                if ln.priced:
                    w.writerow([g.month, g.user, ln.item, ln.quantity, ln.unit, ln.usd_low, ln.usd_high,
                                ln.krw_low, ln.krw_high, ln.reference, ln.note])
    with open(files[2], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["month", "user", "item", "quantity", "unit", "note"])
        for g in sorted(groups.values(), key=lambda g: (g.month, str(g.user))):
            for ln in g.unpriced():
                w.writerow([g.month, g.user, ln.item, ln.quantity, ln.unit, ln.note])
    return files


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="사용자당 월 API 사용량·비용 리포트 (계획 0057)")
    p.add_argument("--source", choices=("db", "log"), default=None,
                   help="기본: --database-url 또는 DATABASE_URL 이 있으면 db, 아니면 log")
    p.add_argument("--database-url", default=None, help="예: postgresql+asyncpg://... / sqlite+aiosqlite:///x.db")
    p.add_argument("--log", nargs="*", default=[], help="텍스트 로그 파일(.gz 가능)")
    p.add_argument("--log-tz", default="UTC", help="로그 줄 시각의 시간대(운영 컨테이너는 UTC)")
    p.add_argument("--month", default=None, help="YYYY-MM (--tz 기준)")
    p.add_argument("--tz", default="Asia/Seoul", help="월 경계 시간대")
    p.add_argument("--user", type=int, default=None)
    p.add_argument("--csv", default=None, help="CSV 3개를 쓸 디렉터리")
    p.add_argument("--json", action="store_true", help="집계를 JSON 으로 출력")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    database_url = args.database_url or os.environ.get("DATABASE_URL")
    source = args.source or ("db" if database_url else "log")
    if source == "db":
        if not database_url:
            print("DB 원천인데 --database-url 도 DATABASE_URL 도 없다", file=sys.stderr)
            return 2
        events = load_from_db(database_url, month=args.month, tz=ZoneInfo(args.tz), user=args.user)
    else:
        if not args.log:
            print("--source log 에는 --log 파일이 필요하다", file=sys.stderr)
            return 2
        events = load_from_logs(args.log, ZoneInfo(args.log_tz))
    groups = aggregate(events, tz=ZoneInfo(args.tz), month=args.month, user=args.user)
    if args.json:
        payload = [{
            "month": g.month, "user": g.user, "sessions": g.sessions,
            "quantities": dict(g.quantities),
            "usd_low": g.totals()[0], "usd_high": g.totals()[1],
            "unpriced": [ln.item for ln in g.unpriced()],
        } for g in sorted(groups.values(), key=lambda g: (g.month, str(g.user)))]
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    else:
        print(render_text(groups))
    if args.csv:
        for f in write_csv(groups, args.csv):
            print(f"CSV: {f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
