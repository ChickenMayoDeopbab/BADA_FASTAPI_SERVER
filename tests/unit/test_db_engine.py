from app.db.base import engine


def test_pool_pre_ping_enabled() -> None:
    assert engine.sync_engine.pool._pre_ping is True


def test_pool_recycle_one_hour() -> None:
    assert engine.sync_engine.pool._recycle == 3600
