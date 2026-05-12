from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from config import GUARDRAIL_MODEL, GUARDRAIL_THRESHOLD
from .prompts import CITATION_RE, FALLBACK_ANSWER, GUARDRAIL_REJECT_ANSWER, preview


@dataclass
class SourceItem:
    doc_id: str
    page: int
    chunk_id: str
    text_preview: str


def to_sources(contexts: Sequence[Dict[str, Any]]) -> List[SourceItem]:
    return [
        SourceItem(
            doc_id=str(row.get("doc_id", "")),
            page=int(row.get("page", 0)),
            chunk_id=str(row.get("chunk_id", "")),
            text_preview=preview(str(row.get("text", ""))),
        )
        for row in contexts
    ]


class TurkLettuceGuardrail:
    def __init__(self, model_name: str = GUARDRAIL_MODEL, threshold: float = GUARDRAIL_THRESHOLD) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self._loaded = False
        self._available = False
        self._pipeline = None
        self._load_error: str | None = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                task="token-classification",
                model=self.model_name,
                tokenizer=self.model_name,
                aggregation_strategy="simple",
            )
            self._available = True
        except Exception as exc:
            self._available = False
            self._load_error = str(exc)

    @staticmethod
    def _is_supported_label(label: str) -> bool | None:
        l = label.strip().lower()
        if l in {"0", "label_0", "class_0"}:
            return True
        if l in {"1", "label_1", "class_1"}:
            return False
        if any(k in l for k in ("support", "entail", "ground", "faithful", "true")):
            return True
        if any(k in l for k in ("halluc", "unsupport", "contradict", "false")):
            return False
        return None

    def verify_claim_semantically(self, claim_text: str, contexts: Sequence[str]) -> Dict[str, Any]:
        self._ensure_loaded()
        if not self._available or self._pipeline is None:
            return {"available": False, "supported": False, "score": 0.0, "reason": f"guardrail_model_unavailable: {self._load_error or 'unknown'}"}
        context_text = " ".join(c.strip() for c in contexts if c and c.strip())
        if not context_text:
            return {"available": True, "supported": False, "score": 0.0, "reason": "empty_context"}
        composed = f"CONTEXT: {context_text}\nANSWER: {claim_text}"
        try:
            preds = self._pipeline(composed)
        except Exception as exc:
            return {"available": False, "supported": False, "score": 0.0, "reason": f"inference_error: {exc}"}
        if not isinstance(preds, list) or not preds:
            return {"available": True, "supported": False, "score": 0.0, "reason": "empty_prediction"}
        supported_scores: List[float] = []
        unsupported_scores: List[float] = []
        for p in preds:
            flag = self._is_supported_label(str(p.get("entity_group") or p.get("entity") or ""))
            score = float(p.get("score", 0.0))
            if flag is True:
                supported_scores.append(score)
            elif flag is False:
                unsupported_scores.append(score)
        if supported_scores:
            sem = float(sum(supported_scores) / max(len(supported_scores), 1))
            return {"available": True, "supported": sem >= self.threshold, "score": sem, "reason": "supported_scores"}
        if unsupported_scores:
            unsup = float(sum(unsupported_scores) / max(len(unsupported_scores), 1))
            sem = max(0.0, 1.0 - unsup)
            return {"available": True, "supported": sem >= self.threshold, "score": sem, "reason": "unsupported_scores"}
        return {"available": True, "supported": False, "score": 0.0, "reason": "unmapped_labels"}

    def verify_claim(self, claim_text: str, contexts: Sequence[str]) -> Dict[str, Any]:
        return self.verify_claim_semantically(claim_text, contexts)


def strip_citations(text: str) -> str:
    return CITATION_RE.sub("", text).strip()


def normalize_citation(doc_id: str, page: int, chunk_raw: str) -> tuple[str, int, str]:
    chunk = chunk_raw.strip().lstrip(":")
    if chunk.startswith("c"):
        return doc_id, page, f"{doc_id}::p{page}::{chunk}"
    if chunk.startswith(f"{doc_id}::p"):
        return doc_id, page, chunk
    short = chunk.split("::")[-1]
    if not short.startswith("c"):
        short = f"c{short}"
    return doc_id, page, f"{doc_id}::p{page}::{short}"


def parse_claims(answer: str) -> List[Dict[str, Any]]:
    claims: List[Dict[str, Any]] = []
    if answer.strip() == FALLBACK_ANSWER:
        return claims
    for ln in answer.splitlines():
        line = ln.strip()
        if not line:
            continue
        if line.startswith("-"):
            line = line[1:].strip()
        if not line:
            continue
        citations = list(CITATION_RE.finditer(line))
        claim = {"text": line, "plain_text": strip_citations(line), "citations": []}
        for c in citations:
            doc_id, page, chunk_id = normalize_citation(c.group("doc_id"), int(c.group("page")), c.group("chunk"))
            claim["citations"].append({"doc_id": doc_id, "page": page, "chunk_id": chunk_id})
        claims.append(claim)
    return claims


def verify_answer(answer: str, contexts: Sequence[Dict[str, Any]], guardrail: TurkLettuceGuardrail) -> Dict[str, Any]:
    source_map = {(str(c.get("doc_id", "")), int(c.get("page", 0)), str(c.get("chunk_id", ""))): str(c.get("text", "")) for c in contexts}
    claims = parse_claims(answer)
    if answer.strip() in {FALLBACK_ANSWER, GUARDRAIL_REJECT_ANSWER}:
        return {"claims": [], "claim_count": 0, "supported_count": 0, "supported_ratio": 1.0, "citation_coverage": 1.0, "confidence": "high", "hallucination_risk": "low", "fallback_used": True, "guardrail_model": GUARDRAIL_MODEL}

    verified_claims: List[Dict[str, Any]] = []
    supported_count = 0
    cited_count = 0
    guardrail_available_any = False
    for claim in claims:
        citations = claim["citations"]
        has_citation = len(citations) > 0
        if has_citation:
            cited_count += 1
        citation_exists = True
        cited_contexts: List[str] = []
        for c in citations:
            source_text = source_map.get((c["doc_id"], c["page"], c["chunk_id"]), "")
            if not source_text:
                citation_exists = False
                continue
            cited_contexts.append(source_text)

        semantic_supported = False
        semantic_score = 0.0
        semantic_reason = "no_citation"
        if has_citation and citation_exists:
            sem = guardrail.verify_claim_semantically(claim_text=claim["plain_text"], contexts=cited_contexts)
            guardrail_available_any = guardrail_available_any or bool(sem.get("available", False))
            semantic_supported = bool(sem.get("supported", False))
            semantic_score = float(sem.get("score", 0.0))
            semantic_reason = str(sem.get("reason", "unknown"))

        supported = has_citation and citation_exists and semantic_supported
        if supported:
            supported_count += 1
        verified_claims.append({"text": claim["text"], "citations": citations, "has_citation": has_citation, "citation_exists": citation_exists, "semantic_supported": semantic_supported, "semantic_score": semantic_score, "semantic_reason": semantic_reason, "supported": supported})

    claim_count = len(claims)
    supported_ratio = (supported_count / claim_count) if claim_count else 0.0
    citation_coverage = (cited_count / claim_count) if claim_count else 0.0
    if not guardrail_available_any and claim_count > 0:
        confidence, risk = "low", "high"
    elif supported_ratio >= 0.8 and citation_coverage >= 0.9:
        confidence, risk = "high", "low"
    elif supported_ratio >= 0.5:
        confidence, risk = "medium", "medium"
    else:
        confidence, risk = "low", "high"

    return {
        "claims": verified_claims,
        "claim_count": claim_count,
        "supported_count": supported_count,
        "supported_ratio": supported_ratio,
        "citation_coverage": citation_coverage,
        "confidence": confidence,
        "confidence_level": confidence,
        "hallucination_risk": risk,
        "fallback_used": False,
        "guardrail_model": GUARDRAIL_MODEL,
        "guardrail_available": guardrail_available_any,
    }


def filter_supported_claims(raw_answer: str, verification: Dict[str, Any]) -> str:
    claims = list(verification.get("claims", []) or [])
    if not claims:
        return raw_answer.strip()
    supported_texts = {str(c.get("text", "")).strip() for c in claims if bool(c.get("supported", False))}
    if not supported_texts:
        return ""
    kept: List[str] = []
    for ln in raw_answer.splitlines():
        line = ln.strip()
        if not line:
            continue
        normalized = line[1:].strip() if line.startswith("-") else line
        if normalized in supported_texts:
            kept.append(f"- {normalized}")
    return "\n".join(kept).strip()


def strip_injected_header(text: str) -> str:
    return re.sub(r"^\[Dokuman:[^\]]+\]\s*", "", text.strip(), flags=re.IGNORECASE)


def build_extractive_cited_answer(contexts: Sequence[Dict[str, Any]], max_items: int = 3) -> str:
    lines: List[str] = []
    for row in list(contexts)[:max_items]:
        doc_id = str(row.get("doc_id", "")).strip()
        page = int(row.get("page", 0))
        chunk_id = str(row.get("chunk_id", "")).strip()
        clean_text = re.sub(r"\s+", " ", strip_injected_header(str(row.get("text", "")).strip()))
        if not clean_text:
            continue
        snippet = clean_text[:220].rstrip() + ("..." if len(clean_text) > 220 else "")
        lines.append(f"- {snippet} [{doc_id}:p{page}:{chunk_id}]")
    return "\n".join(lines).strip()
