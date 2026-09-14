# Türk Hukuku AI Asistanı

TCK (5237 – Türk Ceza Kanunu), TMK (4721 – Türk Medeni Kanunu) ve TBK
(6098 – Türk Borçlar Kanunu) maddelerine dayalı, agent mantığıyla çalışan
bir RAG (Retrieval-Augmented Generation) prototipi.

Kapsam bilinçli olarak sadece bu üç kanunla sınırlıdır: sistem, bağlamda
karşılığı olmayan bir soruyla karşılaştığında (örn. İş Kanunu, Vergi Usul
Kanunu ya da hukukla hiç ilgisi olmayan bir soru) uydurma bir cevap
üretmek yerine dürüstçe "bu konuda elimde yeterli bilgi yok" der.

> ⚠️ Bu bir portfolyo/prototip projesidir. Verilen cevaplar hukuki tavsiye
> yerine geçmez; gerçek bir hukuki durumda mutlaka bir uzmana danışın.

## Mimari

```
PDF (kanun metni)
   │
   ▼
[ingestion_pipeline.py] ── madde_chunker.py ile hiyerarşik chunking
   │                        (KİTAP > KISIM > BÖLÜM > Madde > Fıkra > Bent)
   ▼
BGE-M3 embedding (1024 boyut) ──► pgvector (Postgres, HNSW + cosine)
                                          │
                                          ▼
                                    [api.py / main.py]
                                          │
                                          ▼
                              ┌── LangGraph akışı (graph.py) ──┐
                              │  retrieve → generate → grade   │
                              │  grounded ise bitir             │
                              │  değilse retry (max N kez)      │
                              │  hâlâ desteklenmiyorsa dürüst    │
                              │  "bilmiyorum" cevabı             │
                              └──────────────────────────────────┘
```

- **Chunking:** Madde/fıkra/bent seviyesinde hiyerarşik ayrıştırma
  (`madde_chunker.py` – `TurkishLegislationChunker`), custom regex tabanlı
  parser. KİTAP/KISIM/BÖLÜM başlıklarını, roma rakamı/harf alt başlıklarını
  ve GEÇİCİ/EK madde varyantlarını ayrı ayrı tanır; her chunk'a tam
  `context_path` (örn. `5237 TCK > KİTAP İKİNCİ: ... > Madde 81`) ve
  `parent_text` (maddenin tam metni) eklenir.
- **Ingestion doğrulama:** `ingestion_pipeline.py::validate_coverage`,
  chunk'lanan madde numaralarını sıraya dizip aradaki boşlukları tespit
  eder; kayıp madde oranı eşiği (%2) aşarsa embedding adımına hiç
  geçmeden işlemi durdurur.
- **Embedding:** BGE-M3 (çok dilli, Türkçe destekli, 1024 boyut) —
  ingestion'da `FlagEmbedding.BGEM3FlagModel`, sorgu tarafında
  `retriever.QueryEmbedder` ile aynı model kullanılır.
- **Vektör DB:** pgvector (Postgres) — cosine similarity + HNSW index
  (bkz. `schema.sql`).
- **Orkestrasyon:** LangGraph (`graph.py`)
  - `retrieve` → `generate` → `grade` (halüsinasyon denetimi, ayrı bir
    LLM çağrısıyla, `grader.py`) → destekliyse (`is_grounded`) bitir,
    desteksizse tekrar dene (`max_retries`), hâlâ desteklenmiyorsa
    `fallback` düğümünde dürüstçe "bilmiyorum" de.
  - Mesafe eşik-filtresi (`DISTANCE_THRESHOLD = 0.40`): alakasız
    sorularda `generate`/`grade`'e hiç girmeden `no_context` ile erken
    çıkış.
- **LLM:** Groq — cevap üretimi `openai/gpt-oss-20b`, grounding kontrolü
  ise ayrı ve varsayılan olarak daha ucuz bir model
  (`GRADER_MODEL_NAME`, varsayılan `qwen/qwen3.6-27b`).
- **Dayanıklılık:**
  - Postgres connection pooling (`ThreadedConnectionPool`) + geçici
    hatalarda otomatik retry / exponential backoff (`retriever.py`).
  - Groq çağrılarında retry + backoff, günlük token kotası (TPD)
    dolduğunda özel bir hata (`GroqDailyQuotaExceededError`) ile
    kullanıcıya anlamlı bir 503 mesajı (`llm_utils.py`, `api.py`).
  - Rate limiting: IP başına (`per_ip_limiter`) + uygulama geneli
    dakikalık ve günlük limitler (`rate_limiter.py`, `api.py`).

## Proje yapısı

```
.
├── api.py                  # FastAPI servisi (/health, /soru)
├── main.py                 # CLI: tek seferlik soru-cevap
├── graph.py                # LangGraph akışı (retrieve/generate/grade/fallback)
├── retriever.py            # pgvector sorgusu, connection pool, QueryEmbedder
├── generator.py            # Groq ile cevap üretimi
├── grader.py                # Groundedness (halüsinasyon) denetimi
├── llm_utils.py             # Groq client, retry/backoff, kota yönetimi
├── rate_limiter.py          # Basit thread-safe sliding-window rate limiter
├── logging_config.py        # Ortak logging kurulumu
│
├── madde_chunker.py         # Hiyerarşik madde/fıkra/bent parser'ı
├── ingestion_pipeline.py    # PDF → chunk → embed → pgvector ingestion CLI
├── schema.sql                # madde_chunks tablosu + index'ler (idempotent)
│
├── evaluation.py             # Test seti üzerinde uçtan uca değerlendirme
├── eval_results*.json        # Örnek değerlendirme çıktıları
│
├── requirements.txt          # Geliştirme (torch + FlagEmbedding dahil)
├── requirements-deploy.txt   # Dağıtım (hafif, fastembed tabanlı)
├── Dockerfile
├── docker-compose.yml        # Postgres (pgvector) + api servisi
└── .env                       # GROQ_API_KEY, DATABASE_URL vb. (repoya girmez)
```

## Kurulum

### 1. Ortam değişkenleri

`.env` dosyası oluşturun:

```env
GROQ_API_KEY=...
DATABASE_URL=postgresql://hukuk:hukuk@localhost:5432/hukuk_db
```

### 2. Veritabanını ayağa kaldırma

```bash
docker-compose up -d postgres
```

Bu adım `pgvector/pgvector:pg16` imajıyla Postgres'i ayağa kaldırır;
`schema.sql` ise ingestion sırasında otomatik uygulanır (`ensure_schema`).

### 3. Bağımlılıkları kurma (yerel geliştirme)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> Not: `requirements.txt` PDF okuma (`pymupdf`) ve embedding
> (`torch`, `FlagEmbedding`) için gereken ağır bağımlılıkları içerir;
> bunlar sadece ingestion ve yerel geliştirme için gereklidir.
> Prod dağıtımı `requirements-deploy.txt`'teki hafif sürümü kullanır
> (bkz. `Dockerfile`).

### 4. Kanunları içe aktarma (ingestion)

Her kanun PDF'i için ayrı ayrı çalıştırılır:

```bash
python ingestion_pipeline.py --pdf tck_5237.pdf --kanun-adi "5237 TCK" --db-url "$DATABASE_URL"
python ingestion_pipeline.py --pdf tmk_4721.pdf --kanun-adi "4721 TMK" --db-url "$DATABASE_URL"
python ingestion_pipeline.py --pdf tbk_6098.pdf --kanun-adi "6098 TBK" --db-url "$DATABASE_URL"
```

Pipeline sırasıyla: PDF'ten metin çıkarır (`pymupdf`) → hiyerarşik
chunk'lara ayırır (`TurkishLegislationChunker`) → madde kapsamını
doğrular → BGE-M3 ile embed eder → pgvector'a yazar. `--no-replace`
verilmezse aynı `kanun_adi`'ye ait eski kayıtlar otomatik silinip
yeniden yazılır.

### 5. Servisi çalıştırma

**CLI (tek soru):**

```bash
python main.py "Kasten adam öldürmenin cezası nedir?" "5237 TCK"
```

`kanun_adi` argümanı opsiyoneldir; verilmezse üç kanun genelinde arama
yapılır.

**API (FastAPI):**

```bash
docker-compose up --build
```

veya yerelde:

```bash
uvicorn api:app --host 0.0.0.0 --port 10000
```

## API

### `GET /health`

```json
{ "status": "ok", "model_yuklu": true }
```

### `POST /soru`

```json
{
  "soru": "Bir kişinin hak ehliyeti ne zaman başlar?",
  "top_k": 5,
  "kanun_adi": null
}
```

`kanun_adi` verilmezse üç kanun genelinde arama yapılır (örn. `"5237 TCK"`,
`"4721 TMK"`, `"6098 TBK"` değerlerinden biri verilebilir).

Yanıt:

```json
{
  "cevap": "...",
  "kaynaklar": [
    { "context_path": "4721 TMK > ... > Madde 28", "mesafe": 0.21 }
  ]
}
```

Rate limit aşıldığında veya Groq günlük kotası dolduğunda sırasıyla
`429` ve `503` durum kodları, kullanıcı dostu Türkçe hata mesajlarıyla
döner.

## Değerlendirme (evaluation)

```bash
python evaluation.py --testset testset.json --output eval_results.json --top-k 5
```

Her test sorusu için kapsam içi sorularda **retrieval isabet oranı**
(beklenen maddenin getirilip getirilmediği), kapsam dışı sorularda ise
**doğru red oranı** (sistemin "bilmiyorum" deyip demediği) hesaplanır.
Sonuçlar her soru sonrası diske yazılır; Groq günlük kotası dolarsa
koşu o ana kadarki sonuçlarla birlikte güvenli şekilde durur.

## Notlar / Bilinen sınırlar

- Kapsam sadece TCK, TMK ve TBK ile sınırlıdır; başka kanunlarla ilgili
  sorularda sistem bilinçli olarak reddeder.
- `grader.py`'deki groundedness kontrolü ayrı bir LLM çağrısı olduğundan
  ek gecikme ve token maliyeti getirir; bunun karşılığında halüsinasyon
  riski önemli ölçüde azalır.
- Groq ücretsiz kotası (TPD) sınırlıdır; yoğun test/eval koşularında
  `503` hatasıyla karşılaşmak normaldir (bkz. `eval_results_50.json`
  örneği).
