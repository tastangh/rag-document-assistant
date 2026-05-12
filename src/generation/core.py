from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from config import EVAL_DIR, GUARDRAIL_THRESHOLD, OLLAMA_MODEL, OLLAMA_URL
from .ollama import GenerationOptions, call_ollama, coerce_generation_options
from .prompts import (
    FALLBACK_ANSWER,
    build_context_block,
    build_general_chat_prompt,
    build_prompt,
    is_cross_lingual_like,
    overlap_ratio,
    question_doc_hint,
    resolve_fast_retrieval_plan,
)
from .verify import (
    TurkLettuceGuardrail,
    build_extractive_cited_answer,
    filter_supported_claims,
    to_sources,
    verify_answer,
)
from retrieval import (
    DEFAULT_COLLECTION,
    DEFAULT_FINAL_K,
    DEFAULT_INITIAL_K,
    DEFAULT_PERSIST_DIR,
    DEFAULT_RERANK_MODEL,
    retrieve_contexts,
)

DEFAULT_DEVICE = "cuda"


def chat_without_rag(
    question: str,
    model_name: str = OLLAMA_MODEL,
    ollama_url: str = OLLAMA_URL,
    system_instructions: Optional[str] = None,
    thinking_level: Optional[str] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repeat_penalty: Optional[float] = None,
) -> Dict[str, Any]:
    prompt = build_general_chat_prompt(question, system_instructions, thinking_level)
    options = coerce_generation_options(temperature, top_k, top_p, repeat_penalty)
    answer = call_ollama(prompt=prompt, model_name=model_name, ollama_url=ollama_url, options=options)
    return {
        "question": question,
        "answer": answer,
        "sources": [],
        "verification": {
            "claims": [], "claim_count": 0, "supported_count": 0, "supported_ratio": None,
            "citation_coverage": None, "confidence": "n/a", "hallucination_risk": "n/a",
            "fallback_used": False, "mode": "chat_without_rag",
        },
        "config": {
            "model_name": model_name,
            "temperature": options.temperature,
            "top_k": options.top_k,
            "top_p": options.top_p,
            "repeat_penalty": options.repeat_penalty,
            "thinking_level": str(thinking_level or "medium"),
            "system_instructions_enabled": bool((system_instructions or "").strip()),
            "mode": "chat_without_rag",
        },
    }


def ask_question(
    question: str,
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    initial_k: int = DEFAULT_INITIAL_K,
    final_k: int = DEFAULT_FINAL_K,
    device: str = DEFAULT_DEVICE,
    reranker_model: str = DEFAULT_RERANK_MODEL,
    disable_rerank: bool = False,
    model_name: str = OLLAMA_MODEL,
    ollama_url: str = OLLAMA_URL,
    doc_id: Optional[str] = None,
    chunk_type: Optional[str] = None,
    strict_guardrail: bool = True,
    fast_mode: bool = False,
    context_limit: int = 4,
    auto_doc_filter: bool = True,
    retrieval_min_overlap: float = 0.08,
    system_instructions: Optional[str] = None,
    temperature: Optional[float] = None,
    thinking_level: Optional[str] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repeat_penalty: Optional[float] = None,
    guardrail_threshold: Optional[float] = None,
    citation_min_coverage: float = 1.0,
    allow_extractive_on_guardrail_fail: bool = False,
) -> Dict[str, Any]:
    eff_initial, eff_final, eff_disable = (initial_k, final_k, disable_rerank)
    if fast_mode:
        eff_initial, eff_final, eff_disable = resolve_fast_retrieval_plan(question, initial_k, final_k, disable_rerank)

    eff_doc_id = doc_id
    if auto_doc_filter and not eff_doc_id:
        hinted = question_doc_hint(question)
        if hinted:
            eff_doc_id = hinted

    contexts = retrieve_contexts(
        question=question,
        persist_dir=persist_dir,
        collection_name=collection_name,
        initial_k=eff_initial,
        final_k=eff_final,
        device=device,
        doc_id=eff_doc_id,
        chunk_type=chunk_type,
        reranker_model=reranker_model,
        disable_rerank=eff_disable,
    )
    if fast_mode and context_limit > 0:
        contexts = contexts[: min(context_limit, len(contexts))]
    if not contexts:
        return {"question": question, "answer": FALLBACK_ANSWER, "sources": [], "verification": {"claims": [], "claim_count": 0, "supported_count": 0, "supported_ratio": 1.0, "citation_coverage": 1.0, "confidence": "high", "hallucination_risk": "low", "fallback_used": True}, "config": {"initial_k": initial_k, "final_k": final_k, "device": device, "disable_rerank": eff_disable, "model_name": model_name, "strict_guardrail": strict_guardrail, "fast_mode": fast_mode, "auto_doc_filter": auto_doc_filter}}

    best_overlap = max((overlap_ratio(question, str(c.get("text", ""))) for c in contexts), default=0.0)
    avg_vector_score = sum(float(c.get("vector_score", 0.0)) for c in contexts) / max(1, len(contexts))
    cross_lingual = is_cross_lingual_like(question, contexts)
    eff_overlap = float(retrieval_min_overlap)
    if cross_lingual:
        eff_overlap = min(eff_overlap, 0.02)
    if best_overlap < eff_overlap and avg_vector_score < 0.40 and not cross_lingual:
        fallback_sources = to_sources(contexts)
        return {
            "question": question,
            "answer": FALLBACK_ANSWER,
            "sources": [{"doc_id": s.doc_id, "page": s.page, "chunk_id": s.chunk_id, "text_preview": s.text_preview} for s in fallback_sources],
            "verification": {"claims": [], "claim_count": 0, "supported_count": 0, "supported_ratio": 0.0, "citation_coverage": 0.0, "confidence": "low", "hallucination_risk": "high", "fallback_used": True, "reason": "low_retrieval_overlap", "avg_vector_score": avg_vector_score, "cross_lingual_like": cross_lingual},
            "config": {"initial_k": eff_initial, "final_k": eff_final, "device": device, "disable_rerank": eff_disable, "model_name": model_name, "strict_guardrail": strict_guardrail, "fast_mode": fast_mode, "context_limit": context_limit, "auto_doc_filter": auto_doc_filter, "doc_id": eff_doc_id, "retrieval_min_overlap": retrieval_min_overlap, "effective_overlap_threshold": eff_overlap, "avg_vector_score": avg_vector_score, "cross_lingual_like": cross_lingual},
        }

    prompt = build_prompt(question, build_context_block(contexts), system_instructions, thinking_level)
    options = coerce_generation_options(temperature, top_k, top_p, repeat_penalty)
    guardrail = TurkLettuceGuardrail(threshold=float(GUARDRAIL_THRESHOLD if guardrail_threshold is None else guardrail_threshold))
    raw_answer = call_ollama(prompt=prompt, model_name=model_name, ollama_url=ollama_url, options=options)
    verification = verify_answer(raw_answer, contexts, guardrail)
    filtered_answer = filter_supported_claims(raw_answer, verification)
    answer = filtered_answer or raw_answer

    guardrail_available = bool(verification.get("guardrail_available", False))
    threshold = float(GUARDRAIL_THRESHOLD if guardrail_threshold is None else guardrail_threshold)
    if strict_guardrail:
        if guardrail_available:
            if verification.get("confidence_level") == "low" or float(verification.get("supported_ratio", 0.0)) < threshold or float(verification.get("citation_coverage", 0.0)) < float(citation_min_coverage) or not filtered_answer:
                if allow_extractive_on_guardrail_fail:
                    extractive = build_extractive_cited_answer(contexts, max_items=min(4, len(contexts)))
                    if extractive.strip():
                        answer = extractive
                        verification["fallback_used"] = False
                        verification["reason"] = "strict_guardrail_extractive_answer"
                    else:
                        answer = FALLBACK_ANSWER
                        verification["fallback_used"] = True
                        verification["reason"] = "strict_guardrail_blocked_unsupported"
                else:
                    answer = FALLBACK_ANSWER
                    verification["fallback_used"] = True
                    verification["reason"] = "strict_guardrail_blocked_unsupported"
        else:
            answer = FALLBACK_ANSWER
            verification["fallback_used"] = True
            verification["reason"] = "guardrail_unavailable_strict_block"

    sources = to_sources(contexts)
    return {
        "question": question,
        "answer": answer,
        "sources": [{"doc_id": s.doc_id, "page": s.page, "chunk_id": s.chunk_id, "text_preview": s.text_preview} for s in sources],
        "verification": verification,
        "config": {
            "initial_k": eff_initial, "final_k": eff_final, "device": device, "disable_rerank": eff_disable,
            "model_name": model_name, "strict_guardrail": strict_guardrail, "fast_mode": fast_mode,
            "context_limit": context_limit, "auto_doc_filter": auto_doc_filter, "doc_id": eff_doc_id,
            "best_overlap": best_overlap, "retrieval_min_overlap": retrieval_min_overlap,
            "effective_overlap_threshold": eff_overlap, "avg_vector_score": avg_vector_score,
            "cross_lingual_like": cross_lingual, "temperature": options.temperature,
            "thinking_level": str(thinking_level or "medium"),
            "system_instructions_enabled": bool((system_instructions or "").strip()),
            "top_k": options.top_k, "top_p": options.top_p, "repeat_penalty": options.repeat_penalty,
            "guardrail_threshold": threshold, "citation_min_coverage": float(citation_min_coverage),
        },
    }


def load_smoke_questions(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            payload = line.strip()
            if not payload:
                continue
            obj = json.loads(payload)
            question = str(obj.get("question", "")).strip()
            if question:
                rows.append({"question": question, "relevant_doc_ids": list(obj.get("relevant_doc_ids", []) or [])})
    return rows


def evaluate_smoke_result(result: Dict[str, Any], relevant_doc_ids: Sequence[str]) -> Dict[str, Any]:
    answer = str(result.get("answer", "")).strip()
    sources = list(result.get("sources", []) or [])
    doc_ids = {str(s.get("doc_id", "")) for s in sources}
    has_answer = bool(answer)
    has_sources = len(sources) > 0
    source_match = True if not relevant_doc_ids else bool(doc_ids.intersection(set(relevant_doc_ids)))
    no_template_leak = "cikti formati" not in answer.lower()
    passed = has_answer and has_sources and source_match and no_template_leak
    return {"has_answer": has_answer, "has_sources": has_sources, "source_match": source_match, "no_template_leak": no_template_leak, "passed": passed}


def evaluate_adversarial_result(result: Dict[str, Any]) -> Dict[str, Any]:
    answer = str(result.get("answer", "")).strip().lower()
    fallback_like = ("baglamda" in answer and "yeterli" in answer and "bilgi" in answer) or ("veri bulunamad" in answer)
    no_sources = len(result.get("sources", []) or []) == 0
    return {"fallback_like": fallback_like, "no_sources": no_sources, "passed": fallback_like}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask")
    ask.add_argument("--question", required=True)
    ask.add_argument("--model-name", default=OLLAMA_MODEL)
    ask.add_argument("--ollama-url", default=OLLAMA_URL)
    ask.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=DEFAULT_DEVICE)

    smoke = sub.add_parser("smoke-test")
    smoke.add_argument("--questions", type=Path, default=EVAL_DIR / "faz4_smoke_questions.jsonl")
    smoke.add_argument("--model-name", default=OLLAMA_MODEL)
    smoke.add_argument("--ollama-url", default=OLLAMA_URL)
    smoke.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=DEFAULT_DEVICE)

    safety = sub.add_parser("safety-eval")
    safety.add_argument("--questions", type=Path, default=EVAL_DIR / "faz4_smoke_questions.jsonl")
    safety.add_argument("--adversarial", type=Path, default=EVAL_DIR / "faz5_adversarial_questions.jsonl")
    safety.add_argument("--model-name", default=OLLAMA_MODEL)
    safety.add_argument("--ollama-url", default=OLLAMA_URL)
    safety.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=DEFAULT_DEVICE)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "ask":
        result = ask_question(question=args.question, model_name=args.model_name, ollama_url=args.ollama_url, device=args.device)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "smoke-test":
        questions = load_smoke_questions(args.questions)
        results: List[Dict[str, Any]] = []
        passed_count = 0
        for row in questions:
            r = ask_question(question=row["question"], model_name=args.model_name, ollama_url=args.ollama_url, device=args.device)
            checks = evaluate_smoke_result(r, row.get("relevant_doc_ids", []))
            if checks["passed"]:
                passed_count += 1
            results.append({"question": row["question"], "result": r, "checks": checks})
        pass_rate = (passed_count / max(1, len(results)))
        print(json.dumps({"summary": {"question_count": len(results), "passed_count": passed_count, "pass_rate": pass_rate}, "results": results}, ensure_ascii=False, indent=2))
        return

    normal_questions = load_smoke_questions(args.questions)
    normal_eval: List[Dict[str, Any]] = []
    normal_passed = 0
    for row in normal_questions:
        r = ask_question(question=row["question"], model_name=args.model_name, ollama_url=args.ollama_url, device=args.device)
        checks = evaluate_smoke_result(r, row.get("relevant_doc_ids", []))
        if checks["passed"]:
            normal_passed += 1
        normal_eval.append({"question": row["question"], "result": r, "checks": checks})

    adversarial_questions = load_smoke_questions(args.adversarial)
    adversarial_eval: List[Dict[str, Any]] = []
    adversarial_passed = 0
    for row in adversarial_questions:
        r = ask_question(question=row["question"], model_name=args.model_name, ollama_url=args.ollama_url, device=args.device, strict_guardrail=True)
        checks = evaluate_adversarial_result(r)
        if checks["passed"]:
            adversarial_passed += 1
        adversarial_eval.append({"question": row["question"], "result": r, "checks": checks})

    print(json.dumps({
        "summary": {
            "normal_count": len(normal_eval),
            "normal_passed": normal_passed,
            "normal_pass_rate": normal_passed / max(1, len(normal_eval)),
            "adversarial_count": len(adversarial_eval),
            "adversarial_passed": adversarial_passed,
            "adversarial_pass_rate": adversarial_passed / max(1, len(adversarial_eval)),
        },
        "normal_results": normal_eval,
        "adversarial_results": adversarial_eval,
    }, ensure_ascii=False, indent=2))

