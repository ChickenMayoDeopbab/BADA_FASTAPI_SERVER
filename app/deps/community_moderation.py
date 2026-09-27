from functools import lru_cache

from app.core.config import get_settings
from app.services.community_content_moderation import CommunityContentModerator


@lru_cache
def get_community_content_moderator() -> CommunityContentModerator:
    return CommunityContentModerator(get_settings())
