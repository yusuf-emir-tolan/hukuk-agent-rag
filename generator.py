import time
from dotenv import load_dotenv
load_dotenv()
from retriever import dedupe_by_madde
from llm_utils import call_groq
from logging_config import get_logger

logger = get_logger(__name__)


def generate_answer(query: str, chunks: list) -> str:

    unique_chunks = dedupe_by_madde(chunks)

    context = "\n\n".join(
        f"[{c.context_path}]\n{c.parent_text}" for c in unique_chunks
    )

    system_prompt = (
        "Sen bir Türk hukuku asistanısın. Sana verilen kanun maddeleri "
        "bağlamına dayanarak kullanıcının sorusunu açık, anlaşılır ve "
        "doğru bir şekilde cevapla. Sadece verilen bağlamdaki bilgileri "
        "kullan; bağlamda olmayan bir şey varsa 'bu konuda elimde yeterli "
        "bilgi yok' de. Cevabında ilgili madde numaralarına atıfta bulun."
    )

    user_message = f"""Bağlam (ilgili kanun maddeleri):
{context}

Soru: {query}

Yukarıdaki bağlama dayanarak soruyu cevapla."""

    t0 = time.perf_counter()
    answer = call_groq(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        model="openai/gpt-oss-20b",
        max_tokens=800,
        temperature=0.2,
    )
    logger.debug(f"Groq üretimi: {time.perf_counter() - t0:.2f} sn")
    return answer