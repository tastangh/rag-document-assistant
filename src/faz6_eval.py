"""Faz 6: RAG Triad degerlendirme scripti.

Metrikler:
- Context Relevance
- Faithfulness
- Answer Relevance

Not:
- RAGAS kuruluysa rapora `ragas_available=true` yansitilir.
- Air-gapped ortamlarda RAGAS olmadan da fallback metriklerle calisir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from config import EVAL_DIR, OLLAMA_MODEL, OLLAMA_URL
from generation_pipeline import FALLBACK_ANSWER, ask_question
from retrieval_pipeline import DEFAULT_COLLECTION, DEFAULT_PERSIST_DIR


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            payload = line.strip()
            if not payload:
                continue
            rows.append(json.loads(payload))
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _confidence_to_score(confidence: str) -> float:
    c = (confidence or "").strip().lower()
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


def _answer_relevance_score(answer: str, confidence: str) -> float:
    a = (answer or "").strip()
    if not a or a == FALLBACK_ANSWER or a == "Veri bulunamadı.":
        return 0.0
    conf_score = _confidence_to_score(confidence)
    # minimum uzunluk sinyali: tek kelimelik anlamsiz ciktilari cezalandir
    len_bonus = 0.2 if len(a) >= 40 else 0.0
    return min(1.0, conf_score * 0.8 + len_bonus)


def run_eval(
    questions_file: Path,
    persist_dir: Path,
    collection_name: str,
    initial_k: int,
    final_k: int,
    model_name: str,
    ollama_url: str,
    strict_guardrail: bool,
) -> Dict[str, Any]:
    rows = _load_jsonl(questions_file)
    results: List[Dict[str, Any]] = []

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

        answer = str(result.get("answer", ""))
        sources = list(result.get("sources", []) or [])
        verification = result.get("verification", {}) or {}
        confidence = str(verification.get("confidence", "low"))
        faithfulness = float(verification.get("supported_ratio", 0.0))
        context_rel = _context_relevance_score(sources=sources, relevant_doc_ids=relevant_doc_ids)
        answer_rel = _answer_relevance_score(answer=answer, confidence=confidence)
        triad = (context_rel + faithfulness + answer_rel) / 3.0

        results.append(
            {
                "question": question,
                "relevant_doc_ids": relevant_doc_ids,
                "answer": answer,
                "confidence": confidence,
                "metrics": {
                    "context_relevance": context_rel,
                    "faithfulness": faithfulness,
                    "answer_relevance": answer_rel,
                    "triad_score": triad,
                },
                "sources": sources,
                "verification": verification,
            }
        )

    if not results:
        raise RuntimeError("Degerlendirme icin gecerli soru bulunamadi.")

    avg_context = sum(r["metrics"]["context_relevance"] for r in results) / len(results)
    avg_faith = sum(r["metrics"]["faithfulness"] for r in results) / len(results)
    avg_answer = sum(r["metrics"]["answer_relevance"] for r in results) / len(results)
    avg_triad = sum(r["metrics"]["triad_score"] for r in results) / len(results)

    try:
        import ragas  # noqa: F401

        ragas_available = True
    except Exception:
        ragas_available = False

    return {
        "count": len(results),
        "ragas_available": ragas_available,
        "aggregate": {
            "context_relevance": avg_context,
            "faithfulness": avg_faith,
            "answer_relevance": avg_answer,
            "triad_score": avg_triad,
        },
        "results": results,
    }


def run_eval_from_answers(answers_file: Path) -> Dict[str, Any]:
    """CI icin offline/fixture bazli triad hesaplama.

    Beklenen JSONL alanlari:
    - question (zorunlu)
    - answer (zorunlu)
    - relevant_doc_ids (opsiyonel)
    - sources (opsiyonel)
    - verification (opsiyonel)
    """
    rows = _load_jsonl(answers_file)
    if not rows:
        raise RuntimeError("Offline cevap dosyasi bos.")

    results: List[Dict[str, Any]] = []
    for row in rows:
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer:
            continue

        relevant_doc_ids = list(row.get("relevant_doc_ids", []) or [])
        sources = list(row.get("sources", []) or [])
        verification = row.get("verification", {}) or {}
        confidence = str(verification.get("confidence", "low"))
        faithfulness = _safe_float(verification.get("supported_ratio", 0.0), default=0.0)

        context_rel = _context_relevance_score(sources=sources, relevant_doc_ids=relevant_doc_ids)
        answer_rel = _answer_relevance_score(answer=answer, confidence=confidence)
        triad = (context_rel + faithfulness + answer_rel) / 3.0

        results.append(
            {
                "question": question,
                "relevant_doc_ids": relevant_doc_ids,
                "answer": answer,
                "confidence": confidence,
                "metrics": {
                    "context_relevance": context_rel,
                    "faithfulness": faithfulness,
                    "answer_relevance": answer_rel,
                    "triad_score": triad,
                },
                "sources": sources,
                "verification": verification,
                "mode": "offline_fixture",
            }
        )

    if not results:
        raise RuntimeError("Offline cevap dosyasinda gecerli kayit bulunamadi.")

    avg_context = sum(r["metrics"]["context_relevance"] for r in results) / len(results)
    avg_faith = sum(r["metrics"]["faithfulness"] for r in results) / len(results)
    avg_answer = sum(r["metrics"]["answer_relevance"] for r in results) / len(results)
    avg_triad = sum(r["metrics"]["triad_score"] for r in results) / len(results)

    return {
        "count": len(results),
        "ragas_available": False,
        "mode": "offline_fixture",
        "aggregate": {
            "context_relevance": avg_context,
            "faithfulness": avg_faith,
            "answer_relevance": avg_answer,
            "triad_score": avg_triad,
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Faz 6 RAG Triad evaluator")
    parser.add_argument("--questions-file", default=str(EVAL_DIR / "faz4_smoke_questions.jsonl"))
    parser.add_argument("--persist-dir", default=str(DEFAULT_PERSIST_DIR))
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--initial-k", type=int, default=16)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--model", default=OLLAMA_MODEL)
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--strict-guardrail", action="store_true")
    parser.add_argument("--pass-threshold", type=float, default=0.55)
    parser.add_argument("--output", default=str(EVAL_DIR / "faz6_triad_report.json"))
    parser.add_argument(
        "--answers-file",
        default=None,
        help="Opsiyonel offline JSONL cevap dosyasi. Verilirse Ollama cagrisi yapilmaz.",
    )
    args = parser.parse_args()

    if args.answers_file:
        report = run_eval_from_answers(Path(args.answers_file))
    else:
        report = run_eval(
            questions_file=Path(args.questions_file),
            persist_dir=Path(args.persist_dir),
            collection_name=args.collection,
            initial_k=args.initial_k,
            final_k=args.final_k,
            model_name=args.model,
            ollama_url=args.ollama_url,
            strict_guardrail=args.strict_guardrail,
        )
    accepted = float(report["aggregate"]["triad_score"]) >= float(args.pass_threshold)
    report["pass_threshold"] = args.pass_threshold
    report["accepted"] = accepted

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
