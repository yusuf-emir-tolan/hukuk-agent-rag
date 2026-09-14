from __future__ import annotations
import os
import sys
import time
from dataclasses import dataclass
from typing import List

import psycopg2
from psycopg2 import pool as pg_pool

from logging_config import get_logger

logger = get_logger(__name__)

_pool: "pg_pool.ThreadedConnectionPool | None" = None
_pool_db_url: str | None = None


def get_pool(db_url: str, minconn: int = 1, maxconn: int = 10) -> "pg_pool.ThreadedConnectionPool":

    global _pool, _pool_db_url
    if _pool is None or _pool_db_url != db_url:
        if _pool is not None:
            _pool.closeall()
        _pool = pg_pool.ThreadedConnectionPool(minconn, maxconn, db_url)
        _pool_db_url = db_url
    return _pool


def close_pool() -> None:
    global _pool, _pool_db_url
    if _pool is not None:
        _pool.closeall()
        _pool = None
        _pool_db_url = None


def wait_for_pool(
    db_url: str, max_attempts: int = 10, delay: float = 2.0
) -> "pg_pool.ThreadedConnectionPool":
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return get_pool(db_url)
        except psycopg2.OperationalError as e:
            last_exc = e
            logger.warning(
                f"Postgres'e henüz bağlanılamadı (deneme {attempt}/{max_attempts}): {e}"
            )
            if attempt < max_attempts:
                time.sleep(delay)
    raise RuntimeError(
        f"Postgres'e {max_attempts} denemeden sonra hâlâ bağlanılamadı: {last_exc}"
    ) from last_exc


class QueryEmbedder:

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool | None = None):
        from FlagEmbedding import BGEM3FlagModel

        if use_fp16 is None:
            import torch
            use_fp16 = torch.cuda.is_available()

        t0 = time.perf_counter()
        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        logger.info(
            f"BGE-M3 modeli yüklendi ({'GPU/fp16' if use_fp16 else 'CPU/fp32'}): "
            f"{time.perf_counter() - t0:.2f} sn"
        )

    def embed_query(self, query: str) -> List[float]:
        t0 = time.perf_counter()
        out = self.model.encode(
            [query],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        logger.debug(f"Sorgu embed edildi: {time.perf_counter() - t0:.2f} sn")
        return out["dense_vecs"][0].tolist()


@dataclass
class RetrievedChunk:
    kanun_adi: str
    madde_no: str
    fikra_no: str | None
    bent_no: str | None
    context_path: str
    parent_text: str
    distance: float


def dedupe_by_madde(chunks: List["RetrievedChunk"]) -> List["RetrievedChunk"]:
    seen = set()
    unique_chunks = []
    for c in chunks:
        key = (c.kanun_adi, c.madde_no)
        if key in seen:
            continue
        seen.add(key)
        unique_chunks.append(c)
    return unique_chunks


def search(
    db_url: str,
    query_embedding: List[float],
    top_k: int = 5,
    kanun_adi: str | None = None,
    max_retries: int = 2,
) -> List[RetrievedChunk]:
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    where_clause = ""
    params: list = [embedding_str]
    if kanun_adi:
        where_clause = "WHERE kanun_adi = %s"
        params.append(kanun_adi)
    params.append(embedding_str)
    params.append(top_k)

    sql = f"""
        SELECT
            kanun_adi, madde_no, fikra_no, bent_no,
            context_path, parent_text,
            embedding <=> %s AS distance
        FROM madde_chunks
        {where_clause}
        ORDER BY embedding <=> %s
        LIMIT %s;
    """

    pool = get_pool(db_url)
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        conn = pool.getconn()
        try:
            with conn.cursor() as cur:
                t0 = time.perf_counter()
                cur.execute(sql, params)
                rows = cur.fetchall()
                logger.debug(f"pgvector sorgusu: {time.perf_counter() - t0:.2f} sn")
            pool.putconn(conn)
            break
        except psycopg2.OperationalError as e:
            last_exc = e
            pool.putconn(conn, close=True)
            logger.warning(
                f"pgvector sorgusu başarısız (deneme {attempt}/{max_retries}): {e}"
            )
            if attempt == max_retries:
                raise
    else:
        raise RuntimeError(f"pgvector sorgusu {max_retries} denemeden sonra başarısız") from last_exc

    return [
        RetrievedChunk(
            kanun_adi=row[0],
            madde_no=row[1],
            fikra_no=row[2],
            bent_no=row[3],
            context_path=row[4],
            parent_text=row[5],
            distance=row[6],
        )
        for row in rows
    ]

def retrieve(
    query: str,
    db_url: str,
    top_k: int = 5,
    kanun_adi: str | None = None,
    embedder: "QueryEmbedder | None" = None,
) -> List[RetrievedChunk]:
    if embedder is None:
        embedder = QueryEmbedder()
    query_vec = embedder.embed_query(query)
    return search(db_url, query_vec, top_k=top_k, kanun_adi=kanun_adi)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit('Kullanım: python retriever.py "soru metni"')

    query = sys.argv[1]
    db_url = os.environ.get(
        "DATABASE_URL", "postgresql://hukuk:hukuk@localhost:5432/hukuk_db"
    )

    print(f"Soru: {query}\n")
    results = retrieve(query, db_url, top_k=5)

    for i, r in enumerate(results, 1):
        print(f"--- Sonuç {i} (mesafe: {r.distance:.4f}) ---")
        print(f"Yol: {r.context_path}")
        print(f"Madde metni:\n{r.parent_text[:400]}...\n")