# Türk Hukuku AI Asistanı

TCK (5237), TMK (4721) ve TBK (6098) maddelerine dayalı, agent mantığıyla
çalışan bir RAG (Retrieval-Augmented Generation) prototipi.

## Mimari

- **Chunking:** Madde/fıkra/bent seviyesinde hiyerarşik ayrıştırma (KİTAP
  > KISIM > BÖLÜM > Madde > Fıkra > Bent), custom regex tabanlı parser
- **Embedding:** BGE-M3 (çok dilli, Türkçe destekli, 1024 boyut)
- **Vektör DB:** pgvector (Postgres) - cosine similarity + HNSW index
- **Orkestrasyon:** LangGraph
  - `retrieve` → `generate` → `grade` (halüsinasyon denetimi, ayrı bir
    LLM çağrısıyla) → destekliyse bitir, desteksizse tekrar dene, hiç
    desteklenmiyorsa dürüstçe "bilmiyorum" de
  - Mesafe eşik-filtresi: alakasız sorularda generate/grade'e hiç
    girmeden erken çıkış
- **LLM:** Groq (`openai/gpt-oss-20b`)
- **Dayanıklılık:** connection pooling, geçici hatalarda otomatik retry
  (exponential backoff), rate limiting (IP başına + uygulama geneli)

## Not

Bu bir portfolyo/prototip projesidir. Verilen cevaplar hukuki tavsiye
yerine geçmez.
