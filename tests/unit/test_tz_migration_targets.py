from sqlalchemy import DateTime

from app.db import models  # noqa: F401  (metadata 등록)
from app.db.base import Base, tz_migration_targets
from app.db.external import external_metadata


def _declared_in_models() -> set[tuple[str, str]]:
    return {
        (table.name, column.name)
        for table in Base.metadata.tables.values()
        for column in table.columns
        if isinstance(column.type, DateTime) and column.type.timezone
    }


def test_targets_match_the_models_exactly() -> None:
    declared = _declared_in_models()
    targets = set(tz_migration_targets())

    assert targets == declared, (
        f"모델에만 있음: {sorted(declared - targets)} / 대상에만 있음: {sorted(targets - declared)}"
    )
    assert declared, "tz 컬럼이 하나도 안 잡혔다. 도출 로직이 깨졌다."


def test_targets_never_touch_spring_owned_tables() -> None:
    tables = {table for table, _ in tz_migration_targets()}

    assert tables.isdisjoint(external_metadata.tables), (
        f"Spring 소유 테이블이 대상에 들어왔다: {sorted(tables & set(external_metadata.tables))}"
    )


def test_targets_are_deterministic() -> None:
    assert tz_migration_targets() == tz_migration_targets()
    assert list(tz_migration_targets()) == sorted(tz_migration_targets())
