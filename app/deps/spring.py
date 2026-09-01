from fastapi import Depends

from app.core.config import Settings, get_settings
from app.services.spring_client import SpringInternalClient


def get_spring_client(
    settings: Settings = Depends(get_settings),
) -> SpringInternalClient:
    return SpringInternalClient(settings)
