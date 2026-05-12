from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Dict, List

from huggingface_hub import snapshot_download

from model_catalog import EMBEDDING_MODELS, LLM_MODELS, RERANK_MODELS


def _slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def _download_hf_models(models: List[str], out_dir: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for model_id in models:
        local_dir = out_dir / _slug(model_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(repo_id=model_id, local_dir=str(local_dir), local_dir_use_symlinks=False)
            rows.append({"model": model_id, "status": "ok", "path": str(local_dir)})
        except Exception as exc:
            rows.append({"model": model_id, "status": "error", "error": str(exc)})
    return rows


def _pull_ollama_models(models: List[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for model in models:
        try:
            proc = subprocess.run(
                ["ollama", "pull", model],
                capture_output=True,
                text=True,
                check=False,
            )
            if proc.returncode == 0:
                rows.append({"model": model, "status": "ok"})
            else:
                rows.append({"model": model, "status": "error", "error": (proc.stderr or proc.stdout).strip()})
        except Exception as exc:
            rows.append({"model": model, "status": "error", "error": str(exc)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Preset model setlerini models/ altina indirir.")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--skip-ollama", action="store_true")
    args = parser.parse_args()

    root = Path(args.models_dir)
    emb_dir = root / "embeddings"
    rerank_dir = root / "rerankers"
    emb_dir.mkdir(parents=True, exist_ok=True)
    rerank_dir.mkdir(parents=True, exist_ok=True)

    embedding_results = _download_hf_models(EMBEDDING_MODELS, emb_dir)
    rerank_results = _download_hf_models(RERANK_MODELS, rerank_dir)
    ollama_results = [] if args.skip_ollama else _pull_ollama_models(LLM_MODELS)

    report = {
        "models_dir": str(root),
        "embedding_results": embedding_results,
        "rerank_results": rerank_results,
        "ollama_results": ollama_results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

