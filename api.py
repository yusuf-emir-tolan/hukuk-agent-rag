from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from retriever import QueryEmbedder, dedupe_by_madde, wait_for_pool, close_pool
from graph import compiled_graph
from rate_limiter import RateLimiter
from logging_config import get_logger
from llm_utils import GroqDailyQuotaExceededError

logger = get_logger(__name__)

sys.stdout.reconfigure(encoding="utf-8")

DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql://hukuk:hukuk@localhost:5432/hukuk_db"
)

per_ip_limiter = RateLimiter(max_requests=5, window_seconds=60)

global_minute_limiter = RateLimiter(max_requests=20, window_seconds=60)
global_daily_limiter = RateLimiter(max_requests=800, window_seconds=86400)


TRUST_PROXY_HEADERS = os.environ.get("TRUST_PROXY_HEADERS", "false").lower() == "true"


def get_client_ip(request: Request) -> str:
    if TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "bilinmeyen"


class AppState:
    embedder: QueryEmbedder | None = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("BGE-M3 yükleniyor...")
    t0 = time.perf_counter()
    state.embedder = QueryEmbedder()
    logger.info(f"Model yüklendi: {time.perf_counter() - t0:.2f} sn - sunucu hazır.")

    wait_for_pool(DB_URL)
    logger.info("Postgres bağlantı havuzu hazır.")

    yield

    close_pool()
    logger.info("Postgres bağlantı havuzu kapatıldı. Sunucu kapatılıyor.")


app = FastAPI(title="Türk Hukuku RAG API", lifespan=lifespan)

class SoruRequest(BaseModel):
    soru: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    kanun_adi: str | None = None  # ör. "5237 TCK" - verilmezse tüm kanunlarda arar


class Kaynak(BaseModel):
    context_path: str
    mesafe: float


class CevapResponse(BaseModel):
    cevap: str
    kaynaklar: list[Kaynak]


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_yuklu": state.embedder is not None,
    }

@app.post("/soru", response_model=CevapResponse)
def soru_sor(req: SoruRequest, request: Request) -> CevapResponse:
    client_ip = get_client_ip(request)

    if not global_daily_limiter.allow("global"):
        raise HTTPException(
            status_code=429,
            detail="Uygulama bugünkü soru kotasına ulaştı. Lütfen yarın tekrar deneyin.",
        )
    if not global_minute_limiter.allow("global"):
        raise HTTPException(
            status_code=429,
            detail="Şu an çok fazla istek var, lütfen birkaç saniye sonra tekrar deneyin.",
        )
    if not per_ip_limiter.allow(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Çok sık soru soruyorsun, lütfen bir dakika bekleyip tekrar dene.",
        )

    t0 = time.perf_counter()

    initial_state = {
        "query": req.soru,
        "kanun_adi": req.kanun_adi,
        "top_k": req.top_k,
        "embedder": state.embedder,
        "results": [],
        "answer": "",
        "is_grounded": False,
        "grading_reason": "",
        "retry_count": 0,
        "max_retries": 2,
    }

    try:
        final_state = compiled_graph.invoke(initial_state)
    except GroqDailyQuotaExceededError:
        logger.error("Groq günlük token kotası doldu.")
        raise HTTPException(
            status_code=503,
            detail=(
                "Servis şu anda günlük kullanım kotasına ulaştı, lütfen "
                "daha sonra tekrar deneyin."
            ),
        )
    except Exception:
        logger.exception("Soru işlenirken beklenmeyen bir hata oluştu.")
        raise HTTPException(
            status_code=500,
            detail="Sorunuz işlenirken bir hata oluştu. Lütfen tekrar deneyin.",
        )

    kaynaklar = [
        Kaynak(context_path=r.context_path, mesafe=r.distance)
        for r in dedupe_by_madde(final_state["results"])
    ]

    logger.debug(f"Toplam istek süresi: {time.perf_counter() - t0:.2f} sn")

    return CevapResponse(cevap=final_state["answer"], kaynaklar=kaynaklar)


_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(_STATIC_DIR):
    app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")
else:
    logger.warning(
        f"'{_STATIC_DIR}' bulunamadı; statik dosyalar sunulmuyor "
        "(sadece /health ve /soru uç noktaları aktif)."
    )