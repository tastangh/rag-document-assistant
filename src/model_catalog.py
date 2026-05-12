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


def _looks_like_hf_model_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    marker_files = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "sentence_bert_config.json",
        "modules.json",
        "model.safetensors",
        "pytorch_model.bin",
    }
    names = {p.name for p in path.iterdir() if p.is_file()}
    return bool(names & marker_files)


def resolve_local_hf_model(model_id: str, model_type: str, models_root: str = "models") -> str:
    # 1) Dogrudan path verildiyse ve model klasoru ise oldugu gibi kullan.
    direct = Path(model_id)
    if _looks_like_hf_model_dir(direct):
        return str(direct.resolve())

    # 2) models root'u hem CWD'ye gore hem de repo kokune gore dene.
    repo_root = Path(__file__).resolve().parent.parent
    root_candidates = [Path(models_root), repo_root / models_root]
    base = next((p for p in root_candidates if p.exists()), root_candidates[0])
    sub = "embeddings" if model_type == "embedding" else "rerankers"
    sub_root = base / sub
    search_roots = [sub_root] if sub_root.exists() else []
    # Bazi kurulumlarda modeller dogrudan models/ altina atilabiliyor.
    if base.exists() and base not in search_roots:
        search_roots.append(base)

    # Tolerant fallback: alt klasorlerde model id / slug ile eslesen dizin ara.
    needle_slug = model_slug(model_id).lower()
    needle_leaf = model_id.split("/")[-1].lower()
    for root in search_roots:
        candidates = [
            root / model_slug(model_id),      # newmindai__Mursit-Large-TR-Retrieval
            root / model_id,                  # newmindai/Mursit-Large-TR-Retrieval
            root / model_id.split("/")[-1],   # Mursit-Large-TR-Retrieval
        ]
        for path in candidates:
            if _looks_like_hf_model_dir(path):
                return str(path.resolve())

        for d in root.rglob("*"):
            if not d.is_dir():
                continue
            name_l = d.name.lower()
            full_l = str(d).replace("\\", "/").lower()
            if needle_slug in name_l or needle_leaf in name_l or needle_slug in full_l:
                if _looks_like_hf_model_dir(d):
                    return str(d.resolve())

    return model_id
