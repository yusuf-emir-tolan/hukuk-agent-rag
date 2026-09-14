from __future__ import annotations

import argparse
import os
import re
from typing import Any, Dict, List

import psycopg2
from psycopg2.extras import execute_values
import pymupdf

from madde_chunker import TurkishLegislationChunker  # v2 chunker (düzeltilmiş)


def pdf_to_text(pdf_path: str) -> str:

    doc = pymupdf.open(pdf_path)
    try:
        return "\n".join(page.get_text() for page in doc)
    finally:
        doc.close()

def validate_coverage(
    chunks: List[Dict[str, Any]],
    kanun_adi: str,
    max_missing_ratio: float = 0.02,
) -> None:
    if not chunks:
        raise RuntimeError(
            f"[{kanun_adi}] Hiç chunk üretilmedi - PDF çıkarımı veya "
            "chunker regex'leri kontrol edilmeli."
        )
    normal_chunks = [
        ch for ch in chunks
        if ch["metadata"].get("madde_tipi", "MADDE") == "MADDE"
    ]
    ozel_chunks = [
        ch for ch in chunks
        if ch["metadata"].get("madde_tipi", "MADDE") != "MADDE"
    ]

    empty = [
        ch["metadata"]["madde_no"]
        for ch in chunks
        if len(ch["page_content"].strip()) < 20
    ]

    if not normal_chunks:
        print(
            f"[{kanun_adi}] chunk: {len(chunks)}  "
            f"(normal numaralı 'MADDE' bulunamadı, sadece geçici/ek madde var mı kontrol edin)"
        )
        missing_ratio = 0.0
    else:
        nums = sorted(
            int(re.match(r"\d+", ch["metadata"]["madde_no"]).group())
            for ch in normal_chunks
        )
        missing = [n for n in range(nums[0], nums[-1] + 1) if n not in nums]

        print(
            f"[{kanun_adi}] chunk: {len(chunks)}  "
            f"madde aralığı: {nums[0]}-{nums[-1]}  "
            f"eşsiz madde: {len(set(nums))}  "
            f"geçici/ek madde chunk'ı: {len(ozel_chunks)}"
        )

        if missing:
            print(
                f"  ⚠️  {len(missing)} madde numarası tespit edilemedi: {missing}\n"
            )

        missing_ratio = len(missing) / len(nums)

    if empty:
        print(f"  ⚠️  {len(empty)} chunk şüpheli derecede kısa (<20 karakter): {empty[:10]}")

    if missing_ratio > max_missing_ratio:
        raise RuntimeError(
            f"[{kanun_adi}] Kayıp madde oranı %{100 * missing_ratio:.1f} - "
            f"eşiği (%{100 * max_missing_ratio:.1f}) aştı. Embedding'e devam "
            "edilmiyor; önce metin çıkarımını/chunker'ı incele."
        )

class Embedder:

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True):
        from FlagEmbedding import BGEM3FlagModel  # gecikmeli import

        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def embed(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        out = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=1024,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return out["dense_vecs"].tolist()


def ensure_schema(conn) -> None:
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    with open(schema_path, encoding="utf-8") as f:
        sql = f.read()
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def delete_existing(conn, kanun_adi: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM madde_chunks WHERE kanun_adi = %s", (kanun_adi,))
        deleted = cur.rowcount
    conn.commit()
    if deleted:
        print(f"  ↺ {deleted} eski kayıt silindi ({kanun_adi})")


def _to_vector_literal(vec: List[float]) -> str:
    return "[" + ",".join(str(x) for x in vec) + "]"


def insert_chunks(
    conn, chunks: List[Dict[str, Any]], embeddings: List[List[float]]
) -> None:
    rows = [
        (
            ch["metadata"]["kanun_adi"],
            ch["metadata"]["madde_no"],
            ch["metadata"].get("madde_tipi", "MADDE"),
            ch["metadata"]["fikra_no"],
            ch["metadata"]["bent_no"],
            ch["metadata"]["context_path"],
            ch["page_content"],
            ch["metadata"]["parent_text"],
            _to_vector_literal(emb),
        )
        for ch, emb in zip(chunks, embeddings)
    ]
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO madde_chunks
                (kanun_adi, madde_no, madde_tipi, fikra_no, bent_no,
                 context_path, page_content, parent_text, embedding)
            VALUES %s
            """,
            rows,
        )
    conn.commit()


def ingest(
    pdf_path: str,
    kanun_adi: str,
    db_url: str,
    batch_size: int = 16,
    replace: bool = True,
) -> None:
    print(f"[1/5] PDF okunuyor: {pdf_path}")
    text = pdf_to_text(pdf_path)

    print("[2/5] Chunking yapılıyor...")
    chunker = TurkishLegislationChunker()
    chunks = chunker.process_legislation(text, kanun_adi=kanun_adi)

    print("[3/5] Kapsama doğrulanıyor...")
    validate_coverage(chunks, kanun_adi)

    print(f"[4/5] {len(chunks)} chunk embed ediliyor")
    embedder = Embedder()
    texts = [ch["page_content"] for ch in chunks]
    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embeddings.extend(embedder.embed(batch, batch_size=batch_size))
        print(f"    {min(i + batch_size, len(texts))}/{len(texts)}")

    print("[5/5] pgvector'a yazılıyor...")
    conn = psycopg2.connect(db_url)
    try:
        ensure_schema(conn)
        if replace:
            delete_existing(conn, kanun_adi)
        insert_chunks(conn, chunks, embeddings)
    finally:
        conn.close()

    print(f"✅ {kanun_adi}: {len(chunks)} chunk başarıyla yazıldı.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, help="Kanun PDF yolu")
    parser.add_argument("--kanun-adi", required=True, help='Örn: "5237 TCK"')
    parser.add_argument(
        "--db-url",
        default=os.environ.get("DATABASE_URL"),
        help="postgresql://user:pass@host:port/db (veya DATABASE_URL env)",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--no-replace",
        action="store_true",
        help="Aynı kanun_adi'ye ait eski kayıtları SİLME (varsayılan: siler)",
    )
    args = parser.parse_args()

    if not args.db_url:
        raise SystemExit("--db-url ver ya da DATABASE_URL ortam değişkenini ayarla")

    ingest(
        args.pdf,
        args.kanun_adi,
        args.db_url,
        batch_size=args.batch_size,
        replace=not args.no_replace,
    )