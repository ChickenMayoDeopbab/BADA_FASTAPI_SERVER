import inspect

import pytest
from fastapi import HTTPException

from app.core import security


@pytest.mark.asyncio
async def test_valid_secret_passes() -> None:
    await security.require_internal_secret("test-internal-secret")  # conftest env


@pytest.mark.asyncio
async def test_missing_secret_403() -> None:
    with pytest.raises(HTTPException) as exc:
        await security.require_internal_secret(None)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_wrong_secret_403() -> None:
    with pytest.raises(HTTPException) as exc:
        await security.require_internal_secret("wrong-secret")
    assert exc.value.status_code == 403


def test_uses_constant_time_compare() -> None:
    src = inspect.getsource(security.require_internal_secret)
    assert "compare_digest" in src
