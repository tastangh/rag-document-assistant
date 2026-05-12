from __future__ import annotations

from pathlib import Path
from typing import Dict, List, TypedDict


class ModelPreset(TypedDict):
    id: str
    label: str
    ocr_lang: str
    ocr_profile: str
    embedding_model: str
    reranker_model: str
    llm_model: str

# 5 adet, bu proje icin pratik preset (TR agirlikli RAG + cok dilli alternatifler)
MODEL_PRESETS: List[ModelPreset] = [
    {
        "id": "tr_balanced",
        "label": "TR Balanced (Recommended)",
        "ocr_lang": "tr",
        "ocr_profile": "default",
        "embedding_model": "newmindai/Mursit-Large-TR-Retrieval",
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "llm_model": "qwen3:8b",
    },
    {
        "id": "tr_fast",
        "label": "TR Fast",
        "ocr_lang": "tr",
        "ocr_profile": "lightweight",
        "embedding_model": "newmindai/Mursit-Base-TR-Retrieval",
        "reranker_model": "BAAI/bge-reranker-base",
        "llm_model": "qwen2.5:7b",
    },
    {
        "id": "multilingual_quality",
        "label": "Multilingual Quality",
        "ocr_lang": "en",
        "ocr_profile": "default",
        "embedding_model": "intfloat/multilingual-e5-large-instruct",
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "llm_model": "llama3.1:8b",
    },
    {
        "id": "multilingual_fast",
        "label": "Multilingual Fast",
        "ocr_lang": "en",
        "ocr_profile": "lightweight",
        "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "reranker_model": "BAAI/bge-reranker-base",
        "llm_model": "gemma3:12b",
    },
    {
        "id": "high_precision",
        "label": "High Precision",
        "ocr_lang": "tr",
        "ocr_profile": "default",
        "embedding_model": "BAAI/bge-m3",
        "reranker_model": "BAAI/bge-reranker-v2-m3",
        "llm_model": "mistral-nemo:12b",
    },
]


EMBEDDING_MODELS: List[str] = sorted({row["embedding_model"] for row in MODEL_PRESETS})
RERANK_MODELS: List[str] = sorted({row["reranker_model"] for row in MODEL_PRESETS})
LLM_MODELS: List[str] = sorted({row["llm_model"] for row in MODEL_PRESETS})
OCR_LANG_OPTIONS: List[str] = ["tr", "en"]
OCR_PROFILE_OPTIONS: List[str] = ["default", "lightweight"]
OCR_BACKEND_OPTIONS: List[str] = ["paddle"]
OCR_DEVICE_MODE_OPTIONS: List[str] = ["cpu"]


def get_preset(preset_id: str) -> ModelPreset:
    for row in MODEL_PRESETS:
        if row["id"] == preset_id:
            return row
    return MODEL_PRESETS[0]


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def resolve_local_hf_model(model_id: str, model_type: str, models_root: str = "models") -> str:
    base = Path(models_root)
    sub = "embeddings" if model_type == "embedding" else "rerankers"
    local_path = base / sub / model_slug(model_id)
    if local_path.exists() and local_path.is_dir():
        return str(local_path)
    return model_id
