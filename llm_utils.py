from __future__ import annotations

import os
import time
from typing import Dict, List

from groq import (
    APIConnectionError,
    APITimeoutError,
    Groq,
    InternalServerError,
    RateLimitError,
)

from logging_config import get_logger

logger = get_logger(__name__)

RETRYABLE_EXCEPTIONS = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)


class GroqDailyQuotaExceededError(RuntimeError):
    """Groq hesabının günlük (TPD - tokens per day) kotası tükendiğinde
    fırlatılır.
    """
def _is_daily_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "tokens per day" in text or " tpd" in text or "tpd)" in text


def reasoning_kwargs_to_hide_thinking(model: str) -> Dict[str, object]:
    name = model.lower()
    if "gpt-oss" in name:
        return {"include_reasoning": False, "reasoning_effort": "low"}
    if "qwen" in name:
        return {"reasoning_effort": "none", "reasoning_format": "hidden"}
    return {}


_client: "Groq | None" = None


def get_groq_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY ortam değişkeni tanımlı değil."
            )
        _client = Groq(api_key=api_key, max_retries=0)
    return _client


def call_groq(
    messages: List[Dict[str, str]],
    model: str = "qwen/qwen3.6-27b",
    max_tokens: int = 1024,
    temperature: float = 0.2,
    max_retries: int = 3,
    base_delay: float = 1.0,
    extra_kwargs: Dict[str, object] | None = None,
) -> str:
    client = get_groq_client()
    last_exc: Exception | None = None

    call_kwargs: Dict[str, object] = reasoning_kwargs_to_hide_thinking(model)
    if extra_kwargs:
        call_kwargs.update(extra_kwargs)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                **call_kwargs,
            )
            return response.choices[0].message.content
        except RateLimitError as e:
            if _is_daily_quota_error(e):
                logger.error(f"Groq günlük token kotası (TPD) doldu: {e}")
                raise GroqDailyQuotaExceededError(str(e)) from e
            last_exc = e
            wait = base_delay * (2 ** (attempt - 1))
            logger.warning(
                f"Groq çağrısı geçici bir hatayla başarısız oldu "
                f"(deneme {attempt}/{max_retries}): {type(e).__name__}: {e}"
            )
            if attempt < max_retries:
                logger.info(f"{wait:.1f} sn sonra tekrar denenecek...")
                time.sleep(wait)
        except RETRYABLE_EXCEPTIONS as e:
            last_exc = e
            wait = base_delay * (2 ** (attempt - 1))
            logger.warning(
                f"Groq çağrısı geçici bir hatayla başarısız oldu "
                f"(deneme {attempt}/{max_retries}): {type(e).__name__}: {e}"
            )
            if attempt < max_retries:
                logger.info(f"{wait:.1f} sn sonra tekrar denenecek...")
                time.sleep(wait)
        except Exception as e:
            logger.error(
                f"Groq çağrısı kalıcı bir hatayla başarısız oldu, "
                f"tekrar denenmiyor: {type(e).__name__}: {e}"
            )
            raise

    raise RuntimeError(
        f"Groq API {max_retries} denemeden sonra hâlâ başarısız oldu: {last_exc}"
    ) from last_exc