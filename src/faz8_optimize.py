from __future__ import annotations

import argparse
import itertools
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def _score(report: Dict[str, Any], latency_weight: float) -> float:
    triad = float(report["benchmark"]["aggregate"]["triad_score"])
    adv = float(report["adversarial"]["pass_rate"])
    lat = float(report["benchmark"]["aggregate"]["avg_latency_sec"])
    # Buyuk daha iyi: triad + adversarial, gecikme cezasi
    return (0.7 * triad) + (0.3 * adv) - (latency_weight * lat)


def _evaluate_one(
    *,
    benchmark_file: Path,
    adversarial_file: Path,
    persist_dir: Path,
    collection: str,
    model: str,
    ollama_url: str,
    retrieval_device: str,
    initial_k: int,
    final_k: int,
    pass_threshold: float,
) -> Dict[str, Any]:
    benchmark = run_benchmark(
        benchmark_file=benchmark_file,
        persist_dir=persist_dir,
        collection=collection,
        model=model,
        ollama_url=ollama_url,
        initial_k=initial_k,
        final_k=final_k,
        retrieval_device=retrieval_device,
    )
    adversarial = run_adversarial(
        adversarial_file=adversarial_file,
        persist_dir=persist_dir,
        collection=collection,
        model=model,
        ollama_url=ollama_url,
        initial_k=initial_k,
        final_k=final_k,
        retrieval_device=retrieval_device,
    )
    triad = float(benchmark["aggregate"]["triad_score"])
    status = "PASS" if triad >= pass_threshold else "FAIL"
    return {
        "params": {
            "initial_k": initial_k,
            "final_k": final_k,
            "retrieval_device": retrieval_device,
        },
        "summary": {
            "triad_score": triad,
            "status": status,
            "adversarial_pass_rate": float(adversarial["pass_rate"]),
            "avg_latency_sec": float(benchmark["aggregate"]["avg_latency_sec"]),
        },
        "benchmark": benchmark,
        "adversarial": adversarial,
    }


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Faz 8 parametre optimizasyonu")
    parser.add_argument("--benchmark-file", default=str(EVAL_DIR / "benchmark_test.jsonl"))
    parser.add_argument("--adversarial-file", default=str(EVAL_DIR / "benchmark_adversarial.jsonl"))
    parser.add_argument("--persist-dir", default="src/results/vectorStore/chroma")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=OLLAMA_MODEL)
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--retrieval-device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--initial-k-grid", default="8,10,12,16,20,24")
    parser.add_argument("--final-k-grid", default="3,4,5")
    parser.add_argument("--pass-threshold", type=float, default=0.60)
    parser.add_argument("--latency-weight", type=float, default=0.01)
    parser.add_argument("--output", default=str(EVAL_DIR / "faz8_optimization_report.json"))
    args = parser.parse_args()

    benchmark_file = Path(args.benchmark_file)
    adversarial_file = Path(args.adversarial_file)
    persist_dir = Path(args.persist_dir)

    _validate_jsonl_file(benchmark_file, "Benchmark dosyasi")
    _validate_jsonl_file(adversarial_file, "Adversarial dosyasi")
    _check_ollama_health(args.ollama_url)
    _preflight_index_manifest(persist_dir=persist_dir, collection=args.collection)

    initial_grid = _parse_int_list(args.initial_k_grid)
    final_grid = _parse_int_list(args.final_k_grid)
    combos: List[Tuple[int, int]] = [(i, f) for (i, f) in itertools.product(initial_grid, final_grid) if i >= f]
    if not combos:
        raise RuntimeError("Gecerli kombinasyon yok (initial_k >= final_k olmali).")

    logger.info("Toplam kombinasyon: %d", len(combos))
    runs: List[Dict[str, Any]] = []
    for idx, (initial_k, final_k) in enumerate(combos, start=1):
        logger.info("Optimizasyon denemesi %d/%d | initial_k=%d | final_k=%d", idx, len(combos), initial_k, final_k)
        run = _evaluate_one(
            benchmark_file=benchmark_file,
            adversarial_file=adversarial_file,
            persist_dir=persist_dir,
            collection=args.collection,
            model=args.model,
            ollama_url=args.ollama_url,
            retrieval_device=args.retrieval_device,
            initial_k=initial_k,
            final_k=final_k,
            pass_threshold=args.pass_threshold,
        )
        run["objective_score"] = _score(run, latency_weight=args.latency_weight)
        runs.append(run)

    runs_sorted = sorted(runs, key=lambda x: float(x["objective_score"]), reverse=True)
    best = runs_sorted[0]
    report = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "benchmark_file": str(benchmark_file),
            "adversarial_file": str(adversarial_file),
            "persist_dir": str(persist_dir),
            "collection": args.collection,
            "model": args.model,
            "retrieval_device": args.retrieval_device,
            "initial_k_grid": initial_grid,
            "final_k_grid": final_grid,
            "pass_threshold": args.pass_threshold,
            "latency_weight": args.latency_weight,
        },
        "best": best,
        "top3": runs_sorted[:3],
        "all_runs": runs_sorted,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"best": best["params"], "summary": best["summary"], "output": str(out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

