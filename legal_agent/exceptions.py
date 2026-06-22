"""Retry helpers for HTTP errors."""

import logging
from collections.abc import Callable
from typing import Any

import aiohttp
import backoff
import httpx

logger = logging.getLogger(__name__)

# Default retry budget for generic tools (Tavily, parse_html_page).
_DEFAULT_MAX_TRIES = 4
_DEFAULT_MAX_TIME = 60
_DEFAULT_MAX_VALUE = 15

# CourtListener shared API keys hit 429 often under parallel eval load.
COURTLISTENER_MAX_TRIES = 20
COURTLISTENER_MAX_VALUE = 120


class RetryExhaustedError(Exception):
    """Raised when retries for a retryable HTTP status are exhausted."""


def retry_http_errors(
    *status_codes: int,
    max_tries: int = _DEFAULT_MAX_TRIES,
    max_time: float | None = _DEFAULT_MAX_TIME,
    max_value: float = _DEFAULT_MAX_VALUE,
) -> Callable:
    def should_retry(exception: Exception) -> bool:
        for code in status_codes:
            if (
                isinstance(exception, aiohttp.ClientResponseError)
                and exception.status == code
            ):
                logger.warning("HTTP %s (will retry): %s", code, exception)
                return True
            if (
                isinstance(exception, httpx.HTTPStatusError)
                and exception.response.status_code == code
            ):
                logger.warning("HTTP %s (will retry): %s", code, exception)
                return True
        return False

    def decorator(func: Callable) -> Callable:
        @backoff.on_exception(
            backoff.expo,
            Exception,
            max_tries=max_tries,
            max_time=max_time,
            max_value=max_value,
            base=2,
            factor=3,
            jitter=backoff.full_jitter,
            giveup=lambda e: not should_retry(e),
        )
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            return await func(*args, **kwargs)

        return wrapper

    return decorator
