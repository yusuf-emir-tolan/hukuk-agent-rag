from __future__ import annotations

import json
import logging
import os
import time
from typing import List, Tuple

from groq import RateLimitError
from retriever import RetrievedChunk
from llm_utils import get_groq_client, reasoning_kwargs_to_hide_thinking, GroqDailyQuotaExceededError, _is_daily_quota_error

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("GRADER_MODEL_NAME", "qwen/qwen3.6-27b")

MAX_ATTEMPTS = 3
RETRY_BASE_DELAY_SN = 2.0

GRADER_PROMPT = """
Sen bir hukuk denetçisisin. Verilen CEVAP içerisindeki iddiaların, BAĞLAM (Kanun Maddeleri) ile uyumlu olup olmadığını doğrula.

SORU: {query}
BAĞLAM: {context}
CEVAP: {answer}

Sadece aşağıdaki JSON formatında yanıt ver. "reason" alanını tek cümle tut:
{{
  "grounded": true/false,
  "reason": "Kısa açıklama"
}}
"""


def _extract_text(chunk: RetrievedChunk) -> str:
    for attr in ("parent_text", "metin", "icerik", "text", "madde_metni", "content", "page_content"):
        if hasattr(chunk, attr):
            value = getattr(chunk, attr)
            if value:
                return value
    if isinstance(chunk, dict):
        for key in ("parent_text", "metin", "icerik", "text", "madde_metni", "content", "page_content"):
            if key in chunk and chunk[key]:
                return chunk[key]
    return str(chunk)


def check_groundedness(
    query: str, results: List[RetrievedChunk], answer: str
) -> Tuple[bool, str]:
    if not results or not answer:
        return False, "Eksik girdi veya yanıt."

    context_blocks = []
    for r in results:
        kanun = getattr(r, "kanun_adi", "Kanun")
        madde = getattr(r, "madde_no", "Madde")
        metin = _extract_text(r)
        context_blocks.append(f"[{kanun} Madde {madde}]: {metin}")

    context_str = "\n\n".join(context_blocks)[:3000]

    prompt = GRADER_PROMPT.format(query=query, context=context_str, answer=answer)

    client = get_groq_client()
    reasoning_kwargs = reasoning_kwargs_to_hide_thinking(MODEL_NAME)
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200,
                response_format={"type": "json_object"},
                **reasoning_kwargs,
            )

            raw_output = response.choices[0].message.content.strip()

            if not raw_output:
                raise ValueError("LLM boş yanıt döndürdü.")

            data = json.loads(raw_output)
            is_grounded = bool(data.get("grounded", False))
            reason = str(data.get("reason", ""))
            return is_grounded, reason

        except RateLimitError as e:
            if _is_daily_quota_error(e):
                logger.error(f"grader | Groq günlük token kotası (TPD) doldu: {e}")
                raise GroqDailyQuotaExceededError(str(e)) from e
            last_error = e
            logger.warning(
                f"grader | Rate limit (deneme {attempt}/{MAX_ATTEMPTS}): {e}"
            )
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BASE_DELAY_SN * attempt)
                continue

        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            logger.warning(
                f"grader | JSON parse hatası (deneme {attempt}/{MAX_ATTEMPTS}): {e}"
            )
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BASE_DELAY_SN)
                continue

        except Exception as e:
            last_error = e
            logger.warning(
                f"grader | Beklenmeyen hata (deneme {attempt}/{MAX_ATTEMPTS}): {e}"
            )
            if attempt < MAX_ATTEMPTS:
                time.sleep(RETRY_BASE_DELAY_SN)
                continue

    logger.warning(f"grader | {MAX_ATTEMPTS} denemeden sonra başarısız, fail-closed uygulanıyor: {last_error}")
    return False, f"Hata: {last_error}"