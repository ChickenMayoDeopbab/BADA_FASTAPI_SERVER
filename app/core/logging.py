import sys
from logging.config import dictConfig

from app.core.config import get_settings


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
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": sys.stdout,
                "formatter": formatter_key,
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
