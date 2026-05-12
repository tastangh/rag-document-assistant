from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from config import OLLAMA_MODEL, OLLAMA_URL
from generation_pipeline import ask_question


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            rows.append(json.loads(t))
    return rows


def _retrieved_doc_hit(pred_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str]) -> float:
    rel = {str(x).strip() for x in relevant_doc_ids if str(x).strip()}
    if not rel:
        return 1.0
    pred = {str(x).strip() for x in pred_doc_ids if str(x).strip()}
    return 1.0 if pred.intersection(rel) else 0.0


def _citation_parse_success(verification: Dict[str, Any]) -> float:
    claims = list(verification.get("claims", []) or [])
    if not claims:
        return 1.0
    ok = 0
    for c in claims:
        citations = list(c.get("citations", []) or [])
        if len(citations) > 0:
            ok += 1
    return ok / max(1, len(claims))


def main() -> None:
    parser = argparse.ArgumentParser(description="Faz8 diagnosis: retrieval/guardrail/citation metrikleri")
    parser.add_argument("--benchmark-file", default="src/data/benchmark_test.jsonl")
    parser.add_argument("--adversarial-file", default="src/results/eval/benchmark_adversarial.jsonl")
    parser.add_argument("--persist-dir", default="src/results/vectorStore/chroma")
    parser.add_argument("--collection", default="rag_chunks_v1")
    parser.add_argument("--model", default=OLLAMA_MODEL)
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--retrieval-device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--strict-guardrail", action="store_true")
    parser.add_argument("--output", default="src/results/eval/faz8_diagnose_report.json")
    args = parser.parse_args()

    benchmark_rows = _load_jsonl(Path(args.benchmark_file))
    adversarial_rows = _load_jsonl(Path(args.adversarial_file))

    merged: List[Dict[str, Any]] = []
    for r in benchmark_rows:
        merged.append(
            {
                "type": "benchmark",
                "question": str(r.get("question", "")).strip(),
                "relevant_doc_ids": list(r.get("relevant_doc_ids", []) or []),
            }
        )
    for r in adversarial_rows:
        merged.append(
            {
                "type": "adversarial",
                "question": str(r.get("question", "")).strip(),
                "relevant_doc_ids": list(r.get("relevant_doc_ids", []) or []),
            }
        )

    merged = [x for x in merged if x["question"]][: max(1, args.limit)]

    per_q: List[Dict[str, Any]] = []
    for idx, row in enumerate(merged, start=1):
        question = row["question"]
        rel = list(row.get("relevant_doc_ids", []) or [])
        result = ask_question(
            question=question,
            persist_dir=Path(args.persist_dir),
            collection_name=args.collection,
            initial_k=12,
            final_k=4,
            device=args.retrieval_device,
            model_name=args.model,
            ollama_url=args.ollama_url,
            fast_mode=True,
            context_limit=4,
            auto_doc_filter=True,
            retrieval_min_overlap=0.08,
            strict_guardrail=bool(args.strict_guardrail),
            doc_id=(rel[0] if rel else None),
        )
        pred_doc_ids = [str(s.get("doc_id", "")) for s in list(result.get("sources", []) or [])]
        verification = result.get("verification", {}) or {}
        cfg = result.get("config", {}) or {}
        best_overlap = float(cfg.get("best_overlap", 0.0))

        per_q.append(
            {
                "idx": idx,
                "type": row["type"],
                "question": question,
                "relevant_doc_ids": rel,
                "pred_doc_ids": pred_doc_ids,
                "retrieved_doc_hit_at_k": _retrieved_doc_hit(pred_doc_ids, rel),
                "best_overlap": best_overlap,
                "citation_parse_success": _citation_parse_success(verification),
                "guardrail_available": verification.get("guardrail_available", None),
                "supported_ratio": float(verification.get("supported_ratio", 0.0)),
                "fallback_used": bool(verification.get("fallback_used", False)),
                "answer": str(result.get("answer", "")),
            }
        )

    def _avg(key: str) -> float:
        return sum(float(x.get(key, 0.0)) for x in per_q) / max(1, len(per_q))

    guardrail_known = [x for x in per_q if x.get("guardrail_available") is not None]
    guardrail_available_rate = (
        sum(1 for x in guardrail_known if bool(x.get("guardrail_available"))) / max(1, len(guardrail_known))
        if guardrail_known
        else 0.0
    )

    report = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "count": len(per_q),
            "strict_guardrail": bool(args.strict_guardrail),
        },
        "aggregate": {
            "retrieved_doc_hit_at_k_avg": _avg("retrieved_doc_hit_at_k"),
            "best_overlap_avg": _avg("best_overlap"),
            "citation_parse_success_avg": _avg("citation_parse_success"),
            "supported_ratio_avg": _avg("supported_ratio"),
            "fallback_rate": sum(1 for x in per_q if bool(x.get("fallback_used"))) / max(1, len(per_q)),
            "guardrail_available_rate": guardrail_available_rate,
        },
        "questions": per_q,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "aggregate": report["aggregate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

