"""Faz 6: RAG Triad Otomatik Degerlendirme Pipeline.

Metrikler:
- Context Relevance
- Faithfulness / Groundedness
- Answer Relevance (+ citation bonus)

Cikti:
- src/results/eval/triad_report.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from config import EVAL_DIR, OLLAMA_MODEL, OLLAMA_URL
from generation_pipeline import FALLBACK_ANSWER, ask_question
from retrieval_pipeline import DEFAULT_COLLECTION, DEFAULT_PERSIST_DIR

PASS_THRESHOLD_DEFAULT = 0.60
CITATION_BONUS = 0.05


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                rows.append(json.loads(payload))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL parse hatasi ({path}:{line_no}): {exc}") from exc
    return rows


def _confidence_to_score(confidence_level: str) -> float:
    c = (confidence_level or "").strip().lower()
    if c == "high":
        return 1.0
    if c == "medium":
        return 0.6
    return 0.0


def _context_relevance_score(sources: Sequence[Dict[str, Any]], relevant_doc_ids: Sequence[str]) -> float:
    if not sources:
        return 0.0

    expected = {str(x).strip() for x in relevant_doc_ids if str(x).strip()}
    if not expected:
        return 1.0

    hit = 0
    for s in sources:
        if str(s.get("doc_id", "")).strip() in expected:
            hit += 1
    return hit / max(len(sources), 1)


def _answer_relevance_score(answer: str, confidence_level: str, sources: Sequence[Dict[str, Any]]) -> float:
    a = (answer or "").strip()
    if not a or a == FALLBACK_ANSWER or a == "Veri bulunamadı.":
        return 0.0

    conf_score = _confidence_to_score(confidence_level)
    len_bonus = 0.2 if len(a) >= 40 else 0.0
    citation_bonus = CITATION_BONUS if sources else 0.0
    return min(1.0, conf_score * 0.75 + len_bonus + citation_bonus)


def run_triad_eval(
    questions_file: Path,
    output_file: Path,
    persist_dir: Path,
    collection_name: str,
    initial_k: int,
    final_k: int,
    model_name: str,
    ollama_url: str,
    pass_threshold: float,
    strict_guardrail: bool,
) -> Dict[str, Any]:
    rows = _load_jsonl(questions_file)
    if not rows:
        raise RuntimeError("Soru seti bos.")

    per_question: List[Dict[str, Any]] = []

    for row in rows:
        question = str(row.get("question", "")).strip()
        if not question:
            continue

        relevant_doc_ids = list(row.get("relevant_doc_ids", []) or [])
        result = ask_question(
            question=question,
            persist_dir=persist_dir,
            collection_name=collection_name,
            initial_k=initial_k,
            final_k=final_k,
            model_name=model_name,
            ollama_url=ollama_url,
            strict_guardrail=strict_guardrail,
        )

        answer = str(result.get("answer", "")).strip()
        sources = list(result.get("sources", []) or [])
        verification = result.get("verification", {}) or {}

        confidence_level = str(verification.get("confidence_level", verification.get("confidence", "low")))
        faithfulness = float(verification.get("supported_ratio", 0.0))
        context_relevance = _context_relevance_score(sources=sources, relevant_doc_ids=relevant_doc_ids)
        answer_relevance = _answer_relevance_score(
            answer=answer,
            confidence_level=confidence_level,
            sources=sources,
        )
        triad_score = (context_relevance + faithfulness + answer_relevance) / 3.0

        per_question.append(
            {
                "question": question,
                "relevant_doc_ids": relevant_doc_ids,
                "answer": answer,
                "sources": sources,
                "verification": verification,
                "scores": {
                    "context_relevance": context_relevance,
                    "faithfulness": faithfulness,
                    "answer_relevance": answer_relevance,
                    "triad_score": triad_score,
                },
            }
        )

    if not per_question:
        raise RuntimeError("Degerlendirilecek gecerli soru bulunamadi.")

    avg_context = sum(x["scores"]["context_relevance"] for x in per_question) / len(per_question)
    avg_faith = sum(x["scores"]["faithfulness"] for x in per_question) / len(per_question)
    avg_answer = sum(x["scores"]["answer_relevance"] for x in per_question) / len(per_question)
    avg_triad = sum(x["scores"]["triad_score"] for x in per_question) / len(per_question)

    status = "PASS" if avg_triad >= pass_threshold else "FAIL"

    report: Dict[str, Any] = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "questions_file": str(questions_file),
            "output_file": str(output_file),
            "persist_dir": str(persist_dir),
            "collection_name": collection_name,
            "initial_k": initial_k,
            "final_k": final_k,
            "model_name": model_name,
            "strict_guardrail": strict_guardrail,
            "pass_threshold": pass_threshold,
            "citation_bonus": CITATION_BONUS,
        },
        "aggregate": {
            "context_relevance": avg_context,
            "faithfulness": avg_faith,
            "answer_relevance": avg_answer,
            "triad_score": avg_triad,
            "status": status,
        },
        "questions": per_question,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG Triad otomatik degerlendirme pipeline")
    parser.add_argument("--questions-file", default=str(EVAL_DIR / "faz4_smoke_questions.jsonl"))
    parser.add_argument("--output", default=str(EVAL_DIR / "triad_report.json"))
    parser.add_argument("--persist-dir", default=str(DEFAULT_PERSIST_DIR))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--initial-k", type=int, default=24)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--model", default=OLLAMA_MODEL)
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--pass-threshold", type=float, default=PASS_THRESHOLD_DEFAULT)
    parser.add_argument("--strict-guardrail", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_triad_eval(
        questions_file=Path(args.questions_file),
        output_file=Path(args.output),
        persist_dir=Path(args.persist_dir),
        collection_name=args.collection,
        initial_k=args.initial_k,
        final_k=args.final_k,
        model_name=args.model,
        ollama_url=args.ollama_url,
        pass_threshold=float(args.pass_threshold),
        strict_guardrail=bool(args.strict_guardrail),
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["aggregate"]["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
