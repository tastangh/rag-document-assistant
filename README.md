# rag-document-assistant

Yerel (offline/air-gapped uyumlu) RAG tabanli belge analiz ve soru-cevap sistemi.

## Hızlı Başlangıç (Poetry)

```bash
poetry install
poetry run streamlit run src/ui_streamlit.py
```

## Ortam Degiskenleri

`.env` veya shell env ile ayarlanabilir:

- `RAG_RESULTS_DIR` (default: `src/results`)
- `RAG_RUNTIME_ROOT` (default: `src/results/ui_simple_runtime`)
- `RAG_CHUNK_ARTIFACTS_DIR` (default: `src/results/chunkEmbeddings`)
- `RAG_VECTOR_PERSIST_DIR` (default: `src/results/vectorStore/chroma`)
- `RAG_COLLECTION_NAME` (default: `rag_chunks_v1`)
- `RAG_EMBED_MODEL_NAME` (default: `newmindai/Mursit-Large-TR-Retrieval`)
- `RAG_RERANK_MODEL_NAME` (default: `BAAI/bge-reranker-v2-m3`)
- `RAG_OLLAMA_URL` (default: `http://localhost:11434/api/generate`)
- `RAG_OLLAMA_MODEL` (default: `qwen3:8b`)
- `RAG_GUARDRAIL_MODEL` (default: `newmindai/ettin-encoder-150M-TR-HD`)
- `RAG_GUARDRAIL_THRESHOLD` (default: `0.55`)
- `RAG_OCR_USE_GPU` (default: `1`, Paddle CUDA yoksa otomatik CPU fallback)
- `RAG_EMBED_DEVICE` (default: `cuda`)
- `RAG_RETRIEVAL_DEVICE` (default: `cuda`)

## Windows Notu

- Windows'ta `paddlepaddle` cogu senaryoda CPU calisir.
- Tam OCR GPU hizlandirma gerekiyorsa Linux/WSL2 uzerinde Paddle GPU wheel kurulumu onerilir.

## Docker

```bash
docker compose up --build
```

Varsayilan olarak Streamlit `8501` portundan servis verir.

Smoke test icin Docker varsayilanlari CPU olarak ayarlidir
(`RAG_OCR_USE_GPU=0`, `RAG_EMBED_DEVICE=cpu`, `RAG_RETRIEVAL_DEVICE=cpu`).
GPU denemek istersen compose komutunda env override verebilirsin.

Not: Air-gapped ortamlarda image/model artifactleri onceden mirror edilmelidir.

## Faz 6 Degerlendirme (RAG Triad)

```bash
python src/faz6_eval.py \
  --questions-file src/results/eval/faz4_smoke_questions.jsonl \
  --persist-dir src/results/vectorStore/chroma \
  --collection rag_chunks_v1 \
  --strict-guardrail \
  --pass-threshold 0.55
```

Rapor cikti dosyasi:
- `src/results/eval/faz6_triad_report.json`
