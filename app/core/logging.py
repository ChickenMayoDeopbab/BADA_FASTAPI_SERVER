import logging
import re
import sys
from logging.config import dictConfig

from app.core.config import get_settings


_SECRET_PATTERNS = (
    (re.compile(r"([?&]token=)[^&\s\"'\\]+", re.IGNORECASE), r"\1[REDACTED]"),
    (re.compile(r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*"), "[REDACTED]"),
)


class RedactSecretsFilter(logging.Filter):
    """완성된 메시지에서 쿼리 token 값과 JWT를 가린다."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        redacted = message
        for pattern, replacement in _SECRET_PATTERNS:
            redacted = pattern.sub(replacement, redacted)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


def setup_logging() -> None:
    """
    uvicorn 로거 통합
    사용 패턴:
    import logging
    logger = logging.getLogger(__name__)
    prod에서는 json 형식으로 출력
    """
    settings = get_settings()
    is_prod = settings.env == "prod"

    level = "INFO" if is_prod else "DEBUG"
    formatter_key = "json" if is_prod else "text"

    config: dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "text": {
                "format": "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
            "json": {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
                "datefmt": "%Y-%m-%dT%H:%M:%S",
            },
        },
        "filters": {
            "redact_secrets": {"()": RedactSecretsFilter},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": formatter_key,
                "filters": ["redact_secrets"],
            },
        },
        "loggers": {
            "app": {
                "handlers": ["console"],
                "level": level,
                "propagate": False,
            },
            "uvicorn": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["console"],
                "level": "INFO",
                "propagate": False,
            },
            "httpx": {"level": "WARNING", "handlers": ["console"], "propagate": False},
            "httpcore": {"level": "WARNING", "handlers": ["console"], "propagate": False},
        },
        "root": {
            "handlers": ["console"],
            "level": "WARNING",
        },
    }

    dictConfig(config)
