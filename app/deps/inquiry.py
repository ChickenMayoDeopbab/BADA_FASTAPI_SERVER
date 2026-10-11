from functools import lru_cache

from app.core.config import get_settings
from app.services.inquiry import InquiryService


@lru_cache
def get_inquiry_service() -> InquiryService:
    return InquiryService(get_settings().inquiry_webhook_url)
