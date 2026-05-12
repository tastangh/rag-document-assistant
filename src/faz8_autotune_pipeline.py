from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from config import EVAL_DIR, OLLAMA_MODEL, OLLAMA_URL
from faz8_test_harness import (
    DEFAULT_COLLECTION,
    _check_ollama_health,
    _preflight_index_manifest,
    _validate_jsonl_file,
    run_adversarial,
    run_benchmark,
)

logger = logging.getLogger(__name__)


def _objective(benchmark: Dict[str, Any], adversarial: Dict[str, Any]) -> float:
    triad = float(benchmark["aggregate"]["triad_score"])
    adv = float(adversarial["pass_rate"])
    latency = float(benchmark["aggregate"]["avg_latency_sec"])
    return (0.65 * triad) + (0.30 * adv) - (0.01 * latency)


def _run_eval(
    *,
    benchmark_file: Path,
    adversarial_file: Path,
    persist_dir: Path,
    collection: str,
    model: str,
    ollama_url: str,
    retrieval_device: str,
    profile: Dict[str, Any],
) -> Dict[str, Any]:
    benchmark = run_benchmark(
        benchmark_file=benchmark_file,
        persist_dir=persist_dir,
        collection=collection,
        model=model,
        ollama_url=ollama_url,
        initial_k=int(profile["initial_k"]),
        final_k=int(profile["final_k"]),
        retrieval_device=retrieval_device,
        fast_mode=bool(profile["fast_mode"]),
        context_limit=int(profile["context_limit"]),
        auto_doc_filter=bool(profile["auto_doc_filter"]),
        retrieval_min_overlap=float(profile["retrieval_min_overlap"]),
    )
    adversarial = run_adversarial(
        adversarial_file=adversarial_file,
        persist_dir=persist_dir,
        collection=collection,
        model=model,
        ollama_url=ollama_url,
        initial_k=int(profile["initial_k"]),
        final_k=int(profile["final_k"]),
        retrieval_device=retrieval_device,
        fast_mode=bool(profile["fast_mode"]),
        context_limit=int(profile["context_limit"]),
        auto_doc_filter=bool(profile["auto_doc_filter"]),
        retrieval_min_overlap=float(profile["retrieval_min_overlap"]),
    )
    return {
        "profile": deepcopy(profile),
        "benchmark": benchmark,
        "adversarial": adversarial,
        "objective": _objective(benchmark, adversarial),
    }


def _tune_next(profile: Dict[str, Any], eval_out: Dict[str, Any]) -> Dict[str, Any]:
    next_p = deepcopy(profile)
    triad = float(eval_out["benchmark"]["aggregate"]["triad_score"])
    adv = float(eval_out["adversarial"]["pass_rate"])
    ans_rel = float(eval_out["benchmark"]["aggregate"]["answer_relevance"])
    latency = float(eval_out["benchmark"]["aggregate"]["avg_latency_sec"])

    # 1) Guvenlik dusukse bariyeri sikilastir.
    if adv < 0.7:
        next_p["retrieval_min_overlap"] = min(0.18, float(next_p["retrieval_min_overlap"]) + 0.03)
        next_p["auto_doc_filter"] = True
        next_p["context_limit"] = min(4, int(next_p["context_limit"]))

    # 2) Relevance dusukse retrieval genislet.
    if ans_rel < 0.12 or triad < 0.60:
        next_p["initial_k"] = min(24, int(next_p["initial_k"]) + 2)
        next_p["final_k"] = min(6, int(next_p["final_k"]) + 1)

    # 3) Latency cok yuksekse biraz kis.
    if latency > 12.0:
        next_p["initial_k"] = max(10, int(next_p["initial_k"]) - 2)
        next_p["final_k"] = max(3, int(next_p["final_k"]) - 1)

    # Tutarlilik
    if int(next_p["final_k"]) > int(next_p["initial_k"]):
        next_p["final_k"] = next_p["initial_k"]
    return next_p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Faz 8 autotune: update -> eval -> update -> eval")
    parser.add_argument("--benchmark-file", default=str(EVAL_DIR / "benchmark_test.jsonl"))
    parser.add_argument("--adversarial-file", default=str(EVAL_DIR / "benchmark_adversarial.jsonl"))
    parser.add_argument("--persist-dir", default="src/results/vectorStore/chroma")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=OLLAMA_MODEL)
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--retrieval-device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--output", default=str(EVAL_DIR / "faz8_autotune_report.json"))
    args = parser.parse_args()

    benchmark_file = Path(args.benchmark_file)
    adversarial_file = Path(args.adversarial_file)
    persist_dir = Path(args.persist_dir)
    _validate_jsonl_file(benchmark_file, "Benchmark dosyasi")
    _validate_jsonl_file(adversarial_file, "Adversarial dosyasi")
    _check_ollama_health(args.ollama_url)
    _preflight_index_manifest(persist_dir=persist_dir, collection=args.collection)

    profile_v1: Dict[str, Any] = {
        "initial_k": 12,
        "final_k": 4,
        "fast_mode": True,
        "context_limit": 4,
        "auto_doc_filter": True,
        "retrieval_min_overlap": 0.08,
    }

    logger.info("Autotune tur-1 basliyor...")
    eval_v1 = _run_eval(
        benchmark_file=benchmark_file,
        adversarial_file=adversarial_file,
        persist_dir=persist_dir,
        collection=args.collection,
        model=args.model,
        ollama_url=args.ollama_url,
        retrieval_device=args.retrieval_device,
        profile=profile_v1,
    )

    profile_v2 = _tune_next(profile_v1, eval_v1)
    logger.info("Autotune tur-2 basliyor...")
    eval_v2 = _run_eval(
        benchmark_file=benchmark_file,
        adversarial_file=adversarial_file,
        persist_dir=persist_dir,
        collection=args.collection,
        model=args.model,
        ollama_url=args.ollama_url,
        retrieval_device=args.retrieval_device,
        profile=profile_v2,
    )

    best = eval_v2 if float(eval_v2["objective"]) >= float(eval_v1["objective"]) else eval_v1
    report = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "benchmark_file": str(benchmark_file),
            "adversarial_file": str(adversarial_file),
            "persist_dir": str(persist_dir),
            "collection": args.collection,
            "model": args.model,
            "retrieval_device": args.retrieval_device,
        },
        "round_1": eval_v1,
        "round_2": eval_v2,
        "best_round": 2 if best is eval_v2 else 1,
        "best": best,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "best_round": report["best_round"], "best_profile": best["profile"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

