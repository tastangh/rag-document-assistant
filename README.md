# rag-document-assistant

Yerel (offline/air-gapped uyumlu) RAG tabanli belge analiz ve soru-cevap sistemi.

## 1) Yeni Bilgisayarda Sifirdan Kurulum (Windows)

### Adim 1: Repo'yu klonla

```powershell
git clone https://github.com/tastangh/rag-document-assistant.git
cd rag-document-assistant
```

### Adim 2: Python 3.11 ile venv olustur

```powershell
py -3.11 -m venv .venv
```

### Adim 3: Venv aktive et

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

CMD:

```bat
.\.venv\Scripts\activate.bat
```

### Adim 4: Pip ve bagimliliklari kur

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Not:
- OCR tarafi Windows'ta Paddle CPU calisir.
- Embedding/retrieval tarafi Torch CUDA kuruluysa GPU kullanir.

### Adim 5: Ollama hazirligi

Ollama'nin ayakta oldugunu kontrol et:

```powershell
ollama list
```

Model yoksa cek:

```powershell
ollama pull qwen3:8b
```

### Adim 6: (Opsiyonel ama onerilir) OCR model cache yolunu sabitle

```powershell
$env:RAG_OCR_CACHE_DIR="E:\Workspace\rag-document-assistant\models\paddlex"
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True"
```

### Adim 7: Uygulamayi baslat

```powershell
.\run.ps1
```

UI:
- `http://localhost:8501`

## 2) Ilk Calistirmada Beklenenler

- Ilk dokuman hazirlamada OCR/layout modelleri indirilebilir (Paddle resmi modelleri).
- Bu sadece ilk seferdir; sonra cache'den kullanilir.
- Ilk embedding yuklemesi de model cache olusana kadar daha uzun surebilir.

## 3) Hizli Smoke Test

1. UI'dan `TEST.pdf` yukle.
2. `Belgeleri Hazirla` tikla.
3. Su sorulari dene:
   - `Teslimatta istenen dosyalar nelerdir?`
   - `Bu dokumanda projenin ozeti nedir?`
   - `Teslim yontemi nedir?`

## 4) Ortam Degiskenleri

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

## 5) Sorun Giderme

- `Torch not compiled with CUDA enabled`:
  - GPU yerine CPU Torch kurulu. CUDA wheel kur veya `RAG_EMBED_DEVICE=cpu` ile devam et.
- `pyarrow access violation`:
  - Bu projede OCR import zinciri icin koruyucu shim eklendi; yine olursa venv'i temiz kur.
- `Baglamda yeterli bilgi bulunamadi`:
  - Dokuman `ready` mi kontrol et, sonra soruyu tekrar dene (sistem otomatik retry yapiyor).

# KRİTİK NOT yeni bir belge veya çalışmada reset index + chat e tıklamayı unutmayınız .