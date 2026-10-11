from datetime import UTC, datetime

from app.core.timeutil import KST, from_spring_local


def test_value_before_switch_is_utc_wall_clock() -> None:
    assert from_spring_local(datetime(2026, 9, 3, 14, 20, 30)) == datetime(2026, 9, 3, 14, 20, 30, tzinfo=UTC)


def test_value_after_switch_is_kst_wall_clock() -> None:
    assert from_spring_local(datetime(2026, 9, 3, 19, 34, 35)) == datetime(2026, 9, 3, 19, 34, 35, tzinfo=KST)


def test_oldest_prod_record_lands_on_demo_morning() -> None:
    resolved = from_spring_local(datetime(2026, 7, 6, 22, 29, 40))
    assert resolved.astimezone(KST) == datetime(2026, 7, 7, 7, 29, 40, tzinfo=KST)


def test_aware_value_is_left_alone() -> None:
    aware = datetime(2026, 8, 1, 12, 0, tzinfo=KST)
    assert from_spring_local(aware) is aware
