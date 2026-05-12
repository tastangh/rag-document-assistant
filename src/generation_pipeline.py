"""Faz 4: Retrieval + local LLM (Ollama) generation pipeline.

Akis:
1) Soru icin retrieval (Faz 3)
2) Donen context'leri prompt'a yerlestirme
3) Ollama ile cevap uretme
4) Kaynakli cikti dondurme
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from retrieval_pipeline import (
    DEFAULT_COLLECTION,
    DEFAULT_FINAL_K,
    DEFAULT_INITIAL_K,
    DEFAULT_PERSIST_DIR,
    DEFAULT_RERANK_MODEL,
    retrieve_contexts,
)
from config import EVAL_DIR, GUARDRAIL_MODEL, GUARDRAIL_THRESHOLD, OLLAMA_MODEL, OLLAMA_URL


DEFAULT_OLLAMA_URL = OLLAMA_URL
DEFAULT_OLLAMA_MODEL = OLLAMA_MODEL
DEFAULT_DEVICE = "cuda"


SYSTEM_RULES = """Sen bir RAG yardimcisisin.
- Sadece verilen BAGLAM'a dayanarak cevap ver.
- BAGLAM'da olmayan bilgi icin tahmin yapma.
- Bilgi yoksa aynen su cumleyi yaz: "Baglamda yeterli bilgi yok."
- Bilgi varsa her maddenin sonuna kaynak etiketi koy: [doc_id:p<page>:<chunk_id>]
- Kaynak etiketi olmayan iddia yazma.
"""


@dataclass
class SourceItem:
    doc_id: str
    page: int
    chunk_id: str
    text_preview: str

FALLBACK_ANSWER = "Bağlamda yeterli bilgi bulunamadı"
GUARDRAIL_REJECT_ANSWER = "Veri bulunamadı."
# Desteklenen alinti ornekleri:
# [doc_id:TEST:p2::c1]
# [TEST:p2::c1]
# [TEST:p2:c1]
CITATION_RE = re.compile(
    r"\[(?:doc_id:)?(?P<doc_id>[^:\]]+):p(?P<page>\d+):(?P<chunk>[^\]]+)\]"
)


def _preview(text: str, max_len: int = 220) -> str:
    clean = " ".join(text.split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + "..."


def build_context_block(contexts: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for idx, row in enumerate(contexts, start=1):
        parts.append(
            "\n".join(
                [
                    f"[KAYNAK {idx}]",
                    f"doc_id: {row.get('doc_id', '')}",
                    f"page: {row.get('page', 0)}",
                    f"chunk_id: {row.get('chunk_id', '')}",
                    f"section: {row.get('section', 'ROOT')}",
                    f"text: {row.get('text', '')}",
                ]
            )
        )
    return "\n\n".join(parts).strip()


def _resolve_thinking_instruction(thinking_level: Optional[str]) -> str:
    level = str(thinking_level or "").strip().lower()
    if level == "low":
        return "Dusuk dusunme seviyesi: kisa, dogrudan ve gereksiz detay vermeden cevapla."
    if level == "high":
        return "Yuksek dusunme seviyesi: adim adim, daha dikkatli ve kaynak-kisitli dogrulama ile cevapla."
    return "Orta dusunme seviyesi: dengeleyici, acik ve oz bir cevap ver."


def build_prompt(
    question: str,
    context_block: str,
    system_instructions: Optional[str] = None,
    thinking_level: Optional[str] = None,
) -> str:
    extra_system = (system_instructions or "").strip()
    extra_block = f"\nEK SISTEM TALIMATI:\n{extra_system}\n" if extra_system else ""
    thinking_block = _resolve_thinking_instruction(thinking_level=thinking_level)
    return (
        f"{SYSTEM_RULES}\n"
        f"{extra_block}"
        f"DUSUNME KILIDI:\n{thinking_block}\n\n"
        f"BAGLAM:\n{context_block}\n\n"
        f"SORU:\n{question}\n\n"
        "CEVAP KURALI:\n"
        "- Eger baglam yeterliyse: 3-6 maddeyle cevap ver.\n"
        "- Her maddenin sonuna [doc_id:p<page>:<chunk_id>] ekle.\n"
        "- Eger baglam yetersizse: sadece 'Baglamda yeterli bilgi yok.' yaz.\n"
    )


def _question_tokens(question: str) -> List[str]:
    return re.findall(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü\-_/\.]+", (question or "").lower())


def _has_technical_code_signal(question: str) -> bool:
    q = question or ""
    patterns = [
        r"\b[a-z]{2,}-\d{2,}\b",   # arinc-429, mil-std-1553
        r"\b[A-Z]{2,}\d{2,}\b",    # F16, A320 vb.
        r"\b\d{2,}[A-Za-z]+\b",
        r"\b[A-Za-z]+\d{2,}\b",
    ]
    return any(re.search(p, q) for p in patterns)


def _resolve_fast_retrieval_plan(
    question: str,
    initial_k: int,
    final_k: int,
    disable_rerank: bool,
) -> tuple[int, int, bool]:
    toks = _question_tokens(question)
    short_query = len(toks) < 8
    code_like = _has_technical_code_signal(question)

    planned_initial = min(initial_k, 12)
    planned_final = min(final_k, 4)
    planned_disable_rerank = disable_rerank or (short_query and code_like)
    if planned_final > planned_initial:
        planned_final = planned_initial
    return planned_initial, planned_final, planned_disable_rerank


def _normalize_tr(text: str) -> str:
    return (text or "").lower().strip()


def _question_doc_hint(question: str) -> Optional[str]:
    q = _normalize_tr(question)
    if any(k in q for k in ("case study", "case-study", "tusaş", "tusas")):
        return "test"
    if any(k in q for k in ("cv", "mehmet taştan", "mehmet tastan")):
        return "mehmet-taştan"
    if "merkez bankası" in q or "merkez bankasi" in q or "cbrt" in q:
        if "ingilizce" in q or "english" in q:
            return "merkezbankası_eng"
        return "merkezbankası"
    return None


def _overlap_ratio(question: str, text: str) -> float:
    q_tokens = set(_question_tokens(question))
    t_tokens = set(_question_tokens(text))
    if not q_tokens or not t_tokens:
        return 0.0
    common = q_tokens.intersection(t_tokens)
    return len(common) / max(1, len(q_tokens))


def _ascii_ratio(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    ascii_count = sum(1 for ch in t if ord(ch) < 128)
    return ascii_count / max(1, len(t))


def _is_cross_lingual_like(question: str, contexts: Sequence[Dict[str, Any]]) -> bool:
    # Pratik sezgi: soru TR karakter agirlikli, context daha ASCII/EN agirlikli ise.
    q = question or ""
    tr_chars = sum(1 for ch in q if ch in "çğıöşüÇĞİÖŞÜ")
    if tr_chars == 0:
        return False
    sample = " ".join(str(c.get("text", ""))[:500] for c in list(contexts)[:2])
    return _ascii_ratio(sample) > 0.92


def clean_answer(answer: str) -> str:
    """Modelin prompt sÄ±zÄ±ntÄ±sÄ± olarak kopyaladÄ±ÄŸÄ± ÅŸablon satÄ±rlarÄ±nÄ± temizler."""
    text = answer.strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned: List[str] = []
    for ln in lines:
        lower = ln.lower()
        if lower.startswith("3)") and "baglamda yeterli bilgi yok" in lower:
            continue
        if lower.startswith("cikti formati"):
            continue
        cleaned.append(ln)

    text = "\n".join(cleaned).strip()
    if not text:
        return FALLBACK_ANSWER
    return text


def build_general_chat_prompt(
    question: str,
    system_instructions: Optional[str] = None,
    thinking_level: Optional[str] = None,
) -> str:
    extra_system = (system_instructions or "").strip()
    extra_block = f"\nEK SISTEM TALIMATI:\n{extra_system}\n" if extra_system else ""
    thinking_block = _resolve_thinking_instruction(thinking_level=thinking_level)
    return (
        "Sen yardimci bir yapay zeka asistansin.\n"
        "Kullaniciyla dogal, kibar ve acik bir sekilde konus.\n"
        "Bilmedigin bir konuda eminmis gibi davranma.\n"
        f"{extra_block}"
        f"DUSUNME KILIDI:\n{thinking_block}\n\n"
        f"KULLANICI MESAJI:\n{question}\n\n"
        "Yanit:"
    )


def chat_without_rag(
    question: str,
    model_name: str = DEFAULT_OLLAMA_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    system_instructions: Optional[str] = None,
    thinking_level: Optional[str] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repeat_penalty: Optional[float] = None,
) -> Dict[str, Any]:
    prompt = build_general_chat_prompt(
        question=question,
        system_instructions=system_instructions,
        thinking_level=thinking_level,
    )
    effective_temperature = 0.0 if temperature is None else float(temperature)
    answer = call_ollama(
        prompt=prompt,
        model_name=model_name,
        ollama_url=ollama_url,
        temperature=effective_temperature,
        top_k=top_k,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
    )
    return {
        "question": question,
        "answer": answer,
        "sources": [],
        "verification": {
            "claims": [],
            "claim_count": 0,
            "supported_count": 0,
            "supported_ratio": None,
            "citation_coverage": None,
            "confidence": "n/a",
            "hallucination_risk": "n/a",
            "fallback_used": False,
            "mode": "chat_without_rag",
        },
        "config": {
            "model_name": model_name,
            "temperature": effective_temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "thinking_level": str(thinking_level or "medium"),
            "system_instructions_enabled": bool((system_instructions or "").strip()),
            "mode": "chat_without_rag",
        },
    }


def call_ollama(
    prompt: str,
    model_name: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repeat_penalty: Optional[float] = None,
    timeout_sec: int = 180,
) -> str:
    options: Dict[str, Any] = {"temperature": temperature}
    if top_k is not None:
        options["top_k"] = int(top_k)
    if top_p is not None:
        options["top_p"] = float(top_p)
    if repeat_penalty is not None:
        options["repeat_penalty"] = float(repeat_penalty)
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": options,
    }
    body = json.dumps(payload).encode("utf-8")
    req = Request(ollama_url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")

    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore") if exc.fp else str(exc)
        raise RuntimeError(f"Ollama HTTP hatasi: {exc.code} {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            "Ollama'ya baglanilamadi. Ollama acik mi? Varsayilan URL: http://localhost:11434"
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama cevabi JSON degil: {raw[:400]}") from exc

    answer = str(data.get("response", "")).strip()
    if not answer:
        raise RuntimeError(f"Ollama bos cevap dondu: {raw[:400]}")
    return clean_answer(answer)


def to_sources(contexts: Sequence[Dict[str, Any]]) -> List[SourceItem]:
    sources: List[SourceItem] = []
    for row in contexts:
        sources.append(
            SourceItem(
                doc_id=str(row.get("doc_id", "")),
                page=int(row.get("page", 0)),
                chunk_id=str(row.get("chunk_id", "")),
                text_preview=_preview(str(row.get("text", ""))),
            )
        )
    return sources


class TurkLettuceGuardrail:
    """Turk-LettuceDetect uyumlu semantik doğrulama katmanı."""

    def __init__(self, model_name: str = GUARDRAIL_MODEL, threshold: float = GUARDRAIL_THRESHOLD) -> None:
        self.model_name = model_name
        self.threshold = threshold
        self._loaded = False
        self._available = False
        self._pipeline = None
        self._load_error: Optional[str] = None

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
        except Exception as exc:  # pragma: no cover
            self._available = False
            self._load_error = str(exc)

    @staticmethod
    def _is_supported_label(label: str) -> Optional[bool]:
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
            return {
                "available": False,
                "supported": False,
                "score": 0.0,
                "reason": f"guardrail_model_unavailable: {self._load_error or 'unknown'}",
            }

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
            label = str(p.get("entity_group") or p.get("entity") or "")
            score = float(p.get("score", 0.0))
            support_flag = self._is_supported_label(label)
            if support_flag is True:
                supported_scores.append(score)
            elif support_flag is False:
                unsupported_scores.append(score)

        if supported_scores:
            sem_score = float(sum(supported_scores) / max(len(supported_scores), 1))
            return {
                "available": True,
                "supported": sem_score >= self.threshold,
                "score": sem_score,
                "reason": "supported_scores",
            }

        if unsupported_scores:
            unsup = float(sum(unsupported_scores) / max(len(unsupported_scores), 1))
            sem_score = max(0.0, 1.0 - unsup)
            return {
                "available": True,
                "supported": sem_score >= self.threshold,
                "score": sem_score,
                "reason": "unsupported_scores",
            }

        return {"available": True, "supported": False, "score": 0.0, "reason": "unmapped_labels"}

    # Geriye uyumluluk
    def verify_claim(self, claim_text: str, contexts: Sequence[str]) -> Dict[str, Any]:
        return self.verify_claim_semantically(claim_text=claim_text, contexts=contexts)


_GUARDRAIL = TurkLettuceGuardrail()
def _strip_citations(text: str) -> str:
    return CITATION_RE.sub("", text).strip()


def _normalize_citation(doc_id: str, page: int, chunk_raw: str) -> tuple[str, int, str]:
    """Modelin farklÄ± chunk gÃ¶sterimlerini canonical chunk_id'ye Ã§evirir.

    Ornek:
    - c19 -> <doc_id>::p<page>::c19
    - ::c19 -> <doc_id>::p<page>::c19
    - <doc_id>::p5::c19 -> oldugu gibi
    """
    chunk = chunk_raw.strip()
    # Model bazen p1::c1 formatinda oldugu icin chunk_raw ':c1' gelebiliyor.
    # Tum bas bastaki ':' karakterlerini temizleyip canonical forma geciyoruz.
    chunk = chunk.lstrip(":")
    if chunk.startswith("c"):
        return doc_id, page, f"{doc_id}::p{page}::{chunk}"
    if chunk.startswith(f"{doc_id}::p"):
        return doc_id, page, chunk
    # Beklenmeyen formatta da yine canonical'a zorla.
    short = chunk.split("::")[-1]
    if not short.startswith("c"):
        short = f"c{short}"
    return doc_id, page, f"{doc_id}::p{page}::{short}"


def _parse_claims(answer: str) -> List[Dict[str, Any]]:
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
        claims.append(
            {
                "text": line,
                "plain_text": _strip_citations(line),
                "citations": [
                    {
                        "doc_id": _normalize_citation(
                            c.group("doc_id"), int(c.group("page")), c.group("chunk")
                        )[0],
                        "page": _normalize_citation(
                            c.group("doc_id"), int(c.group("page")), c.group("chunk")
                        )[1],
                        "chunk_id": _normalize_citation(
                            c.group("doc_id"), int(c.group("page")), c.group("chunk")
                        )[2],
                    }
                    for c in citations
                ],
            }
        )
    return claims


def verify_answer(answer: str, contexts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    source_map = {
        (str(c.get("doc_id", "")), int(c.get("page", 0)), str(c.get("chunk_id", ""))): str(c.get("text", ""))
        for c in contexts
    }
    claims = _parse_claims(answer)
    if answer.strip() in {FALLBACK_ANSWER, GUARDRAIL_REJECT_ANSWER}:
        return {
            "claims": [],
            "claim_count": 0,
            "supported_count": 0,
            "supported_ratio": 1.0,
            "citation_coverage": 1.0,
            "confidence": "high",
            "hallucination_risk": "low",
            "fallback_used": True,
            "guardrail_model": GUARDRAIL_MODEL,
        }

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
            key = (c["doc_id"], c["page"], c["chunk_id"])
            source_text = source_map.get(key, "")
            if not source_text:
                citation_exists = False
                continue
            cited_contexts.append(source_text)

        semantic_supported = False
        semantic_score = 0.0
        semantic_reason = "no_citation"
        if has_citation and citation_exists:
            sem = _GUARDRAIL.verify_claim_semantically(claim_text=claim["plain_text"], contexts=cited_contexts)
            guardrail_available_any = guardrail_available_any or bool(sem.get("available", False))
            semantic_supported = bool(sem.get("supported", False))
            semantic_score = float(sem.get("score", 0.0))
            semantic_reason = str(sem.get("reason", "unknown"))

        supported = has_citation and citation_exists and semantic_supported
        if supported:
            supported_count += 1

        verified_claims.append(
            {
                "text": claim["text"],
                "citations": citations,
                "has_citation": has_citation,
                "citation_exists": citation_exists,
                "semantic_supported": semantic_supported,
                "semantic_score": semantic_score,
                "semantic_reason": semantic_reason,
                "supported": supported,
            }
        )

    claim_count = len(claims)
    supported_ratio = (supported_count / claim_count) if claim_count else 0.0
    citation_coverage = (cited_count / claim_count) if claim_count else 0.0

    if not guardrail_available_any and claim_count > 0:
        confidence = "low"
        risk = "high"
    elif supported_ratio >= 0.8 and citation_coverage >= 0.9:
        confidence = "high"
        risk = "low"
    elif supported_ratio >= 0.5:
        confidence = "medium"
        risk = "medium"
    else:
        confidence = "low"
        risk = "high"

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


def _filter_supported_claims(raw_answer: str, verification: Dict[str, Any]) -> str:
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


def _strip_injected_header(text: str) -> str:
    return re.sub(r"^\[Dokuman:[^\]]+\]\s*", "", text.strip(), flags=re.IGNORECASE)


def _build_extractive_cited_answer(contexts: Sequence[Dict[str, Any]], max_items: int = 3) -> str:
    lines: List[str] = []
    for row in list(contexts)[:max_items]:
        doc_id = str(row.get("doc_id", "")).strip()
        page = int(row.get("page", 0))
        chunk_id = str(row.get("chunk_id", "")).strip()
        raw_text = str(row.get("text", "")).strip()
        clean_text = _strip_injected_header(raw_text)
        clean_text = re.sub(r"\s+", " ", clean_text)
        if not clean_text:
            continue
        snippet = clean_text[:220].rstrip() + ("..." if len(clean_text) > 220 else "")
        lines.append(f"- {snippet} [{doc_id}:p{page}:{chunk_id}]")
    return "\n".join(lines).strip()
def ask_question(
    question: str,
    persist_dir: Path = DEFAULT_PERSIST_DIR,
    collection_name: str = DEFAULT_COLLECTION,
    initial_k: int = DEFAULT_INITIAL_K,
    final_k: int = DEFAULT_FINAL_K,
    device: str = DEFAULT_DEVICE,
    reranker_model: str = DEFAULT_RERANK_MODEL,
    disable_rerank: bool = False,
    model_name: str = DEFAULT_OLLAMA_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
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
    effective_initial_k = initial_k
    effective_final_k = final_k
    effective_disable_rerank = disable_rerank
    if fast_mode:
        effective_initial_k, effective_final_k, effective_disable_rerank = _resolve_fast_retrieval_plan(
            question=question,
            initial_k=initial_k,
            final_k=final_k,
            disable_rerank=disable_rerank,
        )

    effective_doc_id = doc_id
    if auto_doc_filter and not effective_doc_id:
        hinted = _question_doc_hint(question)
        if hinted:
            effective_doc_id = hinted

    contexts = retrieve_contexts(
        question=question,
        persist_dir=persist_dir,
        collection_name=collection_name,
        initial_k=effective_initial_k,
        final_k=effective_final_k,
        device=device,
        doc_id=effective_doc_id,
        chunk_type=chunk_type,
        reranker_model=reranker_model,
        disable_rerank=effective_disable_rerank,
    )
    if fast_mode and context_limit > 0:
        contexts = contexts[: min(context_limit, len(contexts))]
    if not contexts:
        return {
            "question": question,
            "answer": FALLBACK_ANSWER,
            "sources": [],
            "verification": {
                "claims": [],
                "claim_count": 0,
                "supported_count": 0,
                "supported_ratio": 1.0,
                "citation_coverage": 1.0,
                "confidence": "high",
                "hallucination_risk": "low",
                "fallback_used": True,
            },
            "config": {
                "initial_k": initial_k,
                "final_k": final_k,
                "device": device,
                "disable_rerank": effective_disable_rerank,
                "model_name": model_name,
                "strict_guardrail": strict_guardrail,
                "fast_mode": fast_mode,
                "auto_doc_filter": auto_doc_filter,
            },
        }

    # Retrieval kalite kapisi: soru ile baglam token ortusmesi cok dusukse
    # modelin alakasiz dogrulama/distribusyon drift'ine girmesini engelle.
    best_overlap = max((_overlap_ratio(question, str(c.get("text", ""))) for c in contexts), default=0.0)
    avg_vector_score = sum(float(c.get("vector_score", 0.0)) for c in contexts) / max(1, len(contexts))
    cross_lingual_like = _is_cross_lingual_like(question, contexts)
    effective_overlap_threshold = float(retrieval_min_overlap)
    if cross_lingual_like:
        # Dil karisik ise overlap sinyalinin gucu azalir; esigi yumusat.
        effective_overlap_threshold = min(effective_overlap_threshold, 0.02)

    low_overlap_block = (
        best_overlap < effective_overlap_threshold
        and avg_vector_score < 0.40
        and not cross_lingual_like
    )

    if low_overlap_block:
        fallback_sources = to_sources(contexts)
        return {
            "question": question,
            "answer": FALLBACK_ANSWER,
            "sources": [
                {
                    "doc_id": src.doc_id,
                    "page": src.page,
                    "chunk_id": src.chunk_id,
                    "text_preview": src.text_preview,
                }
                for src in fallback_sources
            ],
            "verification": {
                "claims": [],
                "claim_count": 0,
                "supported_count": 0,
                "supported_ratio": 0.0,
                "citation_coverage": 0.0,
                "confidence": "low",
                "hallucination_risk": "high",
                "fallback_used": True,
                "reason": "low_retrieval_overlap",
                "avg_vector_score": avg_vector_score,
                "cross_lingual_like": cross_lingual_like,
            },
            "config": {
                "initial_k": effective_initial_k,
                "final_k": effective_final_k,
                "device": device,
                "disable_rerank": effective_disable_rerank,
                "model_name": model_name,
                "strict_guardrail": strict_guardrail,
                "fast_mode": fast_mode,
                "context_limit": context_limit,
                "auto_doc_filter": auto_doc_filter,
                "doc_id": effective_doc_id,
                "retrieval_min_overlap": retrieval_min_overlap,
                "effective_overlap_threshold": effective_overlap_threshold,
                "avg_vector_score": avg_vector_score,
                "cross_lingual_like": cross_lingual_like,
            },
        }

    prompt = build_prompt(
        question=question,
        context_block=build_context_block(contexts),
        system_instructions=system_instructions,
        thinking_level=thinking_level,
    )
    effective_temperature = 0.0 if temperature is None else float(temperature)
    effective_guardrail_threshold = float(GUARDRAIL_THRESHOLD if guardrail_threshold is None else guardrail_threshold)
    raw_answer = call_ollama(
        prompt=prompt,
        model_name=model_name,
        ollama_url=ollama_url,
        temperature=effective_temperature,
        top_k=top_k,
        top_p=top_p,
        repeat_penalty=repeat_penalty,
    )
    verification = verify_answer(raw_answer, contexts)
    filtered_answer = _filter_supported_claims(raw_answer=raw_answer, verification=verification)
    answer = filtered_answer or raw_answer

    guardrail_available = bool(verification.get("guardrail_available", False))

    # Katı bariyer:
    # - Guardrail modeli aktifse unsupported claim'ler engellenir.
    # - Guardrail modeli yoksa tamamen sessize dusmek yerine baglamdan extractive+citation cevap uretilir.
    if strict_guardrail:
        if guardrail_available:
            if (
                verification.get("confidence_level") == "low"
                or float(verification.get("supported_ratio", 0.0)) < effective_guardrail_threshold
                or float(verification.get("citation_coverage", 0.0)) < float(citation_min_coverage)
                or not filtered_answer
            ):
                if allow_extractive_on_guardrail_fail:
                    extractive = _build_extractive_cited_answer(contexts, max_items=min(4, len(contexts)))
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
            # Faz 8 guvenlik modu: guardrail model yoksa halusinasyon riskini sifira yaklastirmak icin cevap kapat.
            answer = FALLBACK_ANSWER
            verification["fallback_used"] = True
            verification["reason"] = "guardrail_unavailable_strict_block"
    sources = to_sources(contexts)

    return {
        "question": question,
        "answer": answer,
        "sources": [
            {
                "doc_id": src.doc_id,
                "page": src.page,
                "chunk_id": src.chunk_id,
                "text_preview": src.text_preview,
            }
            for src in sources
        ],
        "verification": verification,
        "config": {
            "initial_k": effective_initial_k,
            "final_k": effective_final_k,
            "device": device,
            "disable_rerank": effective_disable_rerank,
            "model_name": model_name,
            "strict_guardrail": strict_guardrail,
            "fast_mode": fast_mode,
            "context_limit": context_limit,
            "auto_doc_filter": auto_doc_filter,
            "doc_id": effective_doc_id,
            "best_overlap": best_overlap,
            "retrieval_min_overlap": retrieval_min_overlap,
            "effective_overlap_threshold": effective_overlap_threshold,
            "avg_vector_score": avg_vector_score,
            "cross_lingual_like": cross_lingual_like,
            "temperature": effective_temperature,
            "thinking_level": str(thinking_level or "medium"),
            "system_instructions_enabled": bool((system_instructions or "").strip()),
            "top_k": top_k,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
            "guardrail_threshold": effective_guardrail_threshold,
            "citation_min_coverage": float(citation_min_coverage),
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
                rows.append(
                    {
                        "question": question,
                        "relevant_doc_ids": list(obj.get("relevant_doc_ids", []) or []),
                    }
                )
    return rows


def _evaluate_smoke_result(result: Dict[str, Any], relevant_doc_ids: Sequence[str]) -> Dict[str, Any]:
    answer = str(result.get("answer", "")).strip()
    sources = list(result.get("sources", []) or [])
    doc_ids = {str(s.get("doc_id", "")) for s in sources}
    expected = set(str(x) for x in relevant_doc_ids if str(x).strip())
    source_match = True if not expected else bool(doc_ids.intersection(expected))
    has_answer = bool(answer)
    no_template_leak = ("3) Baglamda yeterli bilgi yok." not in answer) and ("CIKTI FORMATI" not in answer)
    verification = result.get("verification", {}) or {}
    confidence = str(verification.get("confidence_level", verification.get("confidence", "low")))
    supported_ratio = float(verification.get("supported_ratio", 0.0))
    fallback_used = bool(verification.get("fallback_used", False))
    source_consistent = confidence in {"high", "medium"} and supported_ratio >= 0.5
    passed = has_answer and bool(sources) and source_match and no_template_leak and (source_consistent or fallback_used)
    return {
        "expected_doc_ids": sorted(expected),
        "source_doc_ids": sorted(doc_ids),
        "has_answer": has_answer,
        "has_sources": bool(sources),
        "source_match": source_match,
        "no_template_leak": no_template_leak,
        "confidence": confidence,
        "supported_ratio": supported_ratio,
        "fallback_used": fallback_used,
        "source_consistent": source_consistent,
        "passed": passed,
    }


def _evaluate_adversarial_result(result: Dict[str, Any]) -> Dict[str, Any]:
    answer = str(result.get("answer", "")).strip()
    is_safe_reject = answer in {FALLBACK_ANSWER, GUARDRAIL_REJECT_ANSWER, "Baglamda yeterli bilgi yok."}
    return {
        "answer": answer,
        "is_safe_reject": is_safe_reject,
        "passed": is_safe_reject,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Faz 4 generation pipeline: retrieval + Ollama answer + kaynakli cikti"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_cmd = subparsers.add_parser("ask", help="Tek soru sor ve kaynakli cevap al")
    ask_cmd.add_argument("--question", required=True, help="Kullanici sorusu")
    ask_cmd.add_argument("--persist-dir", default=str(DEFAULT_PERSIST_DIR))
    ask_cmd.add_argument("--collection", default=DEFAULT_COLLECTION)
    ask_cmd.add_argument("--initial-k", type=int, default=DEFAULT_INITIAL_K)
    ask_cmd.add_argument("--final-k", type=int, default=DEFAULT_FINAL_K)
    ask_cmd.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=DEFAULT_DEVICE)
    ask_cmd.add_argument("--reranker-model", default=DEFAULT_RERANK_MODEL)
    ask_cmd.add_argument("--disable-rerank", action="store_true")
    ask_cmd.add_argument("--model", default=DEFAULT_OLLAMA_MODEL, help="Ollama model name")
    ask_cmd.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    ask_cmd.add_argument("--doc-id", default=None)
    ask_cmd.add_argument("--chunk-type", choices=["text", "table"], default=None)

    smoke_cmd = subparsers.add_parser("smoke-test", help="Hazir soru seti ile hizli uc-tan-uca test")
    smoke_cmd.add_argument(
        "--questions-file",
        default=str(EVAL_DIR / "faz4_smoke_questions.jsonl"),
        help="JSONL soru dosyasi",
    )
    smoke_cmd.add_argument("--persist-dir", default=str(DEFAULT_PERSIST_DIR))
    smoke_cmd.add_argument("--collection", default=DEFAULT_COLLECTION)
    smoke_cmd.add_argument("--initial-k", type=int, default=DEFAULT_INITIAL_K)
    smoke_cmd.add_argument("--final-k", type=int, default=DEFAULT_FINAL_K)
    smoke_cmd.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=DEFAULT_DEVICE)
    smoke_cmd.add_argument("--reranker-model", default=DEFAULT_RERANK_MODEL)
    smoke_cmd.add_argument("--disable-rerank", action="store_true")
    smoke_cmd.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    smoke_cmd.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    smoke_cmd.add_argument("--limit", type=int, default=10)
    smoke_cmd.add_argument(
        "--pass-threshold",
        type=float,
        default=0.70,
        help="Kabul esigi. Ornek: 0.70 -> en az %%70 soru pass olmali.",
    )
    smoke_cmd.add_argument("--strict-guardrail", action="store_true", help="Dusuk guvenli cevaplari fallback'e zorla")

    safety_cmd = subparsers.add_parser("safety-eval", help="Faz 5 normal+yaniltici soru guvenlik degerlendirmesi")
    safety_cmd.add_argument("--normal-file", default=str(EVAL_DIR / "faz4_smoke_questions.jsonl"))
    safety_cmd.add_argument("--adversarial-file", default=str(EVAL_DIR / "faz5_adversarial_questions.jsonl"))
    safety_cmd.add_argument("--persist-dir", default=str(DEFAULT_PERSIST_DIR))
    safety_cmd.add_argument("--collection", default=DEFAULT_COLLECTION)
    safety_cmd.add_argument("--initial-k", type=int, default=DEFAULT_INITIAL_K)
    safety_cmd.add_argument("--final-k", type=int, default=DEFAULT_FINAL_K)
    safety_cmd.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default=DEFAULT_DEVICE)
    safety_cmd.add_argument("--reranker-model", default=DEFAULT_RERANK_MODEL)
    safety_cmd.add_argument("--disable-rerank", action="store_true")
    safety_cmd.add_argument("--model", default=DEFAULT_OLLAMA_MODEL)
    safety_cmd.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    safety_cmd.add_argument("--normal-threshold", type=float, default=0.80)
    safety_cmd.add_argument("--adversarial-threshold", type=float, default=0.90)
    safety_cmd.add_argument("--strict-guardrail", action="store_true")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "ask":
        result = ask_question(
            question=args.question,
            persist_dir=Path(args.persist_dir),
            collection_name=args.collection,
            initial_k=args.initial_k,
            final_k=args.final_k,
            device=args.device,
            reranker_model=args.reranker_model,
            disable_rerank=args.disable_rerank,
            model_name=args.model,
            ollama_url=args.ollama_url,
            doc_id=args.doc_id,
            chunk_type=args.chunk_type,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.command == "smoke-test":
        questions = load_smoke_questions(Path(args.questions_file))[: max(1, args.limit)]
        outputs: List[Dict[str, Any]] = []
        checks: List[Dict[str, Any]] = []
        for row in questions:
            question = str(row["question"])
            relevant_doc_ids = list(row.get("relevant_doc_ids", []) or [])
            result = ask_question(
                    question=question,
                    persist_dir=Path(args.persist_dir),
                    collection_name=args.collection,
                    initial_k=args.initial_k,
                    final_k=args.final_k,
                    device=args.device,
                    reranker_model=args.reranker_model,
                    disable_rerank=args.disable_rerank,
                    model_name=args.model,
                    ollama_url=args.ollama_url,
                    doc_id=(relevant_doc_ids[0] if relevant_doc_ids else None),
                    strict_guardrail=args.strict_guardrail,
            )
            outputs.append(result)
            checks.append(_evaluate_smoke_result(result, relevant_doc_ids))

        passed_count = sum(1 for c in checks if c["passed"])
        total = len(checks)
        pass_rate = (passed_count / total) if total else 0.0
        accepted = pass_rate >= float(args.pass_threshold)
        print(
            json.dumps(
                {
                    "count": len(outputs),
                    "pass_threshold": args.pass_threshold,
                    "pass_rate": pass_rate,
                    "passed_count": passed_count,
                    "accepted": accepted,
                    "results": outputs,
                    "checks": checks,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "safety-eval":
        normal_rows = load_smoke_questions(Path(args.normal_file))
        normal_results: List[Dict[str, Any]] = []
        normal_checks: List[Dict[str, Any]] = []
        for row in normal_rows:
            relevant_doc_ids = list(row.get("relevant_doc_ids", []) or [])
            result = ask_question(
                question=str(row["question"]),
                persist_dir=Path(args.persist_dir),
                collection_name=args.collection,
                initial_k=args.initial_k,
                final_k=args.final_k,
                device=args.device,
                reranker_model=args.reranker_model,
                disable_rerank=args.disable_rerank,
                model_name=args.model,
                ollama_url=args.ollama_url,
                doc_id=(relevant_doc_ids[0] if relevant_doc_ids else None),
                strict_guardrail=args.strict_guardrail,
            )
            normal_results.append(result)
            normal_checks.append(_evaluate_smoke_result(result, relevant_doc_ids))

        adv_rows = load_smoke_questions(Path(args.adversarial_file))
        adversarial_results: List[Dict[str, Any]] = []
        adversarial_checks: List[Dict[str, Any]] = []
        for row in adv_rows:
            result = ask_question(
                question=str(row["question"]),
                persist_dir=Path(args.persist_dir),
                collection_name=args.collection,
                initial_k=args.initial_k,
                final_k=args.final_k,
                device=args.device,
                reranker_model=args.reranker_model,
                disable_rerank=args.disable_rerank,
                model_name=args.model,
                ollama_url=args.ollama_url,
                strict_guardrail=True,
            )
            adversarial_results.append(result)
            adversarial_checks.append(_evaluate_adversarial_result(result))

        normal_passed = sum(1 for c in normal_checks if c["passed"])
        normal_total = len(normal_checks)
        normal_rate = (normal_passed / normal_total) if normal_total else 0.0

        adv_passed = sum(1 for c in adversarial_checks if c["passed"])
        adv_total = len(adversarial_checks)
        adv_rate = (adv_passed / adv_total) if adv_total else 0.0

        accepted = normal_rate >= args.normal_threshold and adv_rate >= args.adversarial_threshold
        print(
            json.dumps(
                {
                    "accepted": accepted,
                    "normal_threshold": args.normal_threshold,
                    "adversarial_threshold": args.adversarial_threshold,
                    "normal_pass_rate": normal_rate,
                    "adversarial_pass_rate": adv_rate,
                    "normal_passed_count": normal_passed,
                    "adversarial_passed_count": adv_passed,
                    "normal_total": normal_total,
                    "adversarial_total": adv_total,
                    "normal_checks": normal_checks,
                    "adversarial_checks": adversarial_checks,
                    "normal_results": normal_results,
                    "adversarial_results": adversarial_results,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    raise ValueError(f"Bilinmeyen komut: {args.command}")


if __name__ == "__main__":
    main()


