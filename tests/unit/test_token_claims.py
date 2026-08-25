"""토큰 필수 클레임 — PyJWT 는 exp 가 "있으면" 검사하고 "없으면" 통과시킨다.

exp 없는 토큰은 영원히 유효하다. Spring 은 항상 넣지만, 시크릿이 새거나
누가 손으로 만든 토큰이 들어오면 만료가 영영 안 온다. 필수로 요구해 막는다.
"""

from datetime import UTC, datetime, timedelta

import jwt
import pytest
from fastapi import HTTPException

from app.core.config import get_settings
from app.core.security import decode_access_token


def _token(**claims) -> str:
    s = get_settings()
    return jwt.encode(claims, s.jwt_secret, algorithm=s.jwt_algorithm)


def _valid_claims(**overrides) -> dict:
    base = {
        "sub": "7",
        "type": "ACCESS",
        "exp": datetime.now(UTC) + timedelta(minutes=30),
    }
    base.update(overrides)
    return base


def test_complete_token_is_accepted() -> None:
    payload = decode_access_token(_token(**_valid_claims()))
    assert payload["sub"] == "7"


@pytest.mark.parametrize("missing", ["exp", "sub", "type"])
def test_token_missing_a_required_claim_is_rejected(missing: str) -> None:
    claims = _valid_claims()
    del claims[missing]

    with pytest.raises(HTTPException) as e:
        decode_access_token(_token(**claims))

    assert e.value.status_code == 401


def test_expired_token_is_still_rejected_with_its_own_message() -> None:
    claims = _valid_claims(exp=datetime.now(UTC) - timedelta(seconds=1))

    with pytest.raises(HTTPException) as e:
        decode_access_token(_token(**claims))

    assert e.value.status_code == 401
    assert e.value.detail == "TOKEN_EXPIRED"


def test_refresh_token_cannot_be_used_as_access() -> None:
    with pytest.raises(HTTPException) as e:
        decode_access_token(_token(**_valid_claims(type="REFRESH")))

    assert e.value.status_code == 401
    assert e.value.detail == "NOT_ACCESS_TOKEN"


def test_token_signed_with_another_secret_is_rejected() -> None:
    bad = jwt.encode(_valid_claims(), "another-secret-entirely",
                     algorithm=get_settings().jwt_algorithm)

    with pytest.raises(HTTPException) as e:
        decode_access_token(bad)

    assert e.value.status_code == 401
    assert e.value.detail == "TOKEN_INVALID"
