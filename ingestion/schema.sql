-- madde_chunks tablosu ve ilgili indeksler.
-- ingestion_pipeline.py::ensure_schema() bu dosyayı her ingestion
-- çalıştırıldığında "CREATE ... IF NOT EXISTS" mantığıyla uygular,
-- bu yüzden idempotent olacak şekilde yazılmıştır.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS madde_chunks (
    id           BIGSERIAL PRIMARY KEY,
    kanun_adi    TEXT NOT NULL,
    madde_no     TEXT NOT NULL,
    -- 'MADDE' | 'GEÇİCİ MADDE' | 'EK MADDE' (bkz. madde_chunker.py:MADDE_PATTERN)
    madde_tipi   TEXT NOT NULL DEFAULT 'MADDE',
    fikra_no     TEXT,
    bent_no      TEXT,
    context_path TEXT NOT NULL,
    page_content TEXT NOT NULL,
    parent_text  TEXT NOT NULL,
    -- BGE-M3 dense embedding boyutu (bkz. retriever.py / ingestion_pipeline.py)
    embedding    vector(1024) NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- retriever.py::search() kanun_adi'ye göre filtreleyebiliyor (WHERE kanun_adi = %s)
CREATE INDEX IF NOT EXISTS madde_chunks_kanun_adi_idx
    ON madde_chunks (kanun_adi);

-- evaluation.py, (kanun_adi, madde_no) çiftine göre retrieval isabetini kontrol ediyor
CREATE INDEX IF NOT EXISTS madde_chunks_kanun_madde_idx
    ON madde_chunks (kanun_adi, madde_no);

-- README'de belirtilen "pgvector - cosine similarity + HNSW index" mimarisi.
-- retriever.py'deki `embedding <=> %s` operatörü cosine mesafesi kullanır,
-- bu yüzden index de vector_cosine_ops ile oluşturulmalı.
CREATE INDEX IF NOT EXISTS madde_chunks_embedding_hnsw_idx
    ON madde_chunks
    USING hnsw (embedding vector_cosine_ops);