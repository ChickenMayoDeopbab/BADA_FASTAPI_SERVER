from functools import lru_cache

from app.core.config import get_settings
from app.services.community_report_alert import CommunityReportAlertService


@lru_cache
def get_community_report_alert_service() -> CommunityReportAlertService:
    return CommunityReportAlertService(get_settings().community_report_alert_webhook_url)
