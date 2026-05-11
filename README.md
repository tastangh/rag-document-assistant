# rag-document-assistant

Yerel (offline/air-gapped uyumlu) RAG tabanli belge analiz ve soru-cevap sistemi.

## Kurulum (Poetry)

```bash
poetry install
poetry run streamlit run src/ui_streamlit.py
```

## Ortam Degiskenleri

`.env.example` dosyasini kopyalayip `.env` olusturabilirsiniz.

- `RAG_RESULTS_DIR` (default: `src/results`)
- `RAG_RUNTIME_ROOT` (default: `src/results/ui_simple_runtime`)
- `RAG_CHUNK_ARTIFACTS_DIR` (default: `src/results/chunkEmbeddings`)
- `RAG_VECTOR_PERSIST_DIR` (default: `src/results/vectorStore/chroma`)
- `RAG_EVAL_DIR` (default: `src/results/eval`)
- `RAG_OCR_CACHE_DIR` (default: `src/results/cache/paddlex`)
- `RAG_COLLECTION_NAME` (default: `rag_chunks_v1`)
- `RAG_EMBED_MODEL_NAME` (default: `newmindai/Mursit-Large-TR-Retrieval`)
- `RAG_RERANK_MODEL_NAME` (default: `BAAI/bge-reranker-v2-m3`)
- `RAG_OLLAMA_URL` (default: `http://localhost:11434/api/generate`)
- `RAG_OLLAMA_MODEL` (default: `qwen3:8b`)
- `RAG_GUARDRAIL_MODEL` (default: `newmindai/ettin-encoder-150M-TR-HD`)
- `RAG_GUARDRAIL_THRESHOLD` (default: `0.55`)
- `RAG_OCR_USE_GPU` (default: `0`)
- `RAG_EMBED_DEVICE` (default: `cpu`)
- `RAG_RETRIEVAL_DEVICE` (default: `cpu`)

## Docker

```bash
copy .env.example .env

docker compose up --build
```

Servis varsayilan olarak `http://localhost:8501` adresinde calisir.

Notlar:
- Compose varsayilanlari CPU odaklidir.
- Air-gapped ortamlarda Python paketleri, model dosyalari ve container base image onceden mirror edilmelidir.

## Faz 6 Degerlendirme (RAG Triad)

```bash
poetry run python src/faz6_eval.py \
  --questions-file src/results/eval/faz4_smoke_questions.jsonl \
  --persist-dir src/results/vectorStore/chroma \
  --collection rag_chunks_v1 \
  --strict-guardrail \
  --pass-threshold 0.55
```

Rapor cikti dosyasi:
- `src/results/eval/faz6_triad_report.json`

CI/offline fixture modu:

```bash
poetry run python src/faz6_eval.py \
  --answers-file src/results/eval/faz6_answers_fixture.jsonl \
  --pass-threshold 0.55 \
  --output src/results/eval/faz6_triad_report_ci.json
```

Bu modda Ollama cagrisi yapilmaz; mevcut cevap kayitlari uzerinden triad metrikleri hesaplanir.
