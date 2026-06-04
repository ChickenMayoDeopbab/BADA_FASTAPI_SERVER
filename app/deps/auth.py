from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.security import decode_access_token

_bearer = HTTPBearer()

def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer)
) -> int:
    payload = decode_access_token(credentials.credentials)
    try:
        return int(payload["sub"])
    except(KeyError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail = "유효하지 않은 토큰"
        ) from e
