from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

SYSTEM_RULES = """Sen bir RAG yardimcisisin.
- Sadece verilen BAGLAM'a dayanarak cevap ver.
- BAGLAM'da olmayan bilgi icin tahmin yapma.
- Bilgi yoksa aynen su cumleyi yaz: \"Baglamda yeterli bilgi yok.\"
- Bilgi varsa her maddenin sonuna kaynak etiketi koy: [doc_id:p<page>:<chunk_id>]
- Kaynak etiketi olmayan iddia yazma.
"""

FALLBACK_ANSWER = "Bağlamda yeterli bilgi bulunamadı"
GUARDRAIL_REJECT_ANSWER = "Veri bulunamadı."
CITATION_RE = re.compile(r"\[(?:doc_id:)?(?P<doc_id>[^:\]]+):p(?P<page>\d+):(?P<chunk>[^\]]+)\]")


def preview(text: str, max_len: int = 220) -> str:
    clean = " ".join(text.split())
    return clean if len(clean) <= max_len else clean[: max_len - 3] + "..."


def build_context_block(contexts: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for idx, row in enumerate(contexts, start=1):
        parts.append("\n".join([
            f"[KAYNAK {idx}]",
            f"doc_id: {row.get('doc_id', '')}",
            f"page: {row.get('page', 0)}",
            f"chunk_id: {row.get('chunk_id', '')}",
            f"section: {row.get('section', 'ROOT')}",
            f"text: {row.get('text', '')}",
        ]))
    return "\n\n".join(parts).strip()


def resolve_thinking_instruction(thinking_level: Optional[str]) -> str:
    level = str(thinking_level or "").strip().lower()
    if level == "low":
        return "Dusuk dusunme seviyesi: kisa, dogrudan ve gereksiz detay vermeden cevapla."
    if level == "high":
        return "Yuksek dusunme seviyesi: adim adim, daha dikkatli ve kaynak-kisitli dogrulama ile cevapla."
    return "Orta dusunme seviyesi: dengeleyici, acik ve oz bir cevap ver."


def build_prompt(question: str, context_block: str, system_instructions: Optional[str] = None, thinking_level: Optional[str] = None) -> str:
    extra_system = (system_instructions or "").strip()
    extra_block = f"\nEK SISTEM TALIMATI:\n{extra_system}\n" if extra_system else ""
    thinking_block = resolve_thinking_instruction(thinking_level=thinking_level)
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


def question_tokens(question: str) -> List[str]:
    return re.findall(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü\-_/\.]+", (question or "").lower())


def has_technical_code_signal(question: str) -> bool:
    q = question or ""
    patterns = [r"\b[a-z]{2,}-\d{2,}\b", r"\b[A-Z]{2,}\d{2,}\b", r"\b\d{2,}[A-Za-z]+\b", r"\b[A-Za-z]+\d{2,}\b"]
    return any(re.search(p, q) for p in patterns)


def resolve_fast_retrieval_plan(question: str, initial_k: int, final_k: int, disable_rerank: bool) -> tuple[int, int, bool]:
    short_query = len(question_tokens(question)) < 8
    code_like = has_technical_code_signal(question)
    planned_initial = min(initial_k, 12)
    planned_final = min(final_k, 4)
    planned_disable_rerank = disable_rerank or (short_query and code_like)
    if planned_final > planned_initial:
        planned_final = planned_initial
    return planned_initial, planned_final, planned_disable_rerank


def normalize_tr(text: str) -> str:
    return (text or "").lower().strip()


def question_doc_hint(question: str) -> Optional[str]:
    q = normalize_tr(question)
    if any(k in q for k in ("case study", "case-study", "tusaş", "tusas")):
        return "test"
    if any(k in q for k in ("cv", "mehmet taştan", "mehmet tastan")):
        return "mehmet-taştan-cv-en (1)"
    if "merkez bankası" in q or "merkez bankasi" in q or "cbrt" in q:
        return "merkezbankası_eng" if ("ingilizce" in q or "english" in q) else "merkezbankası"
    return None


def overlap_ratio(question: str, text: str) -> float:
    q_tokens = set(question_tokens(question))
    t_tokens = set(question_tokens(text))
    if not q_tokens or not t_tokens:
        return 0.0
    return len(q_tokens.intersection(t_tokens)) / max(1, len(q_tokens))


def ascii_ratio(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    ascii_count = sum(1 for ch in t if ord(ch) < 128)
    return ascii_count / max(1, len(t))


def is_cross_lingual_like(question: str, contexts: Sequence[Dict[str, Any]]) -> bool:
    tr_chars = sum(1 for ch in (question or "") if ch in "çğıöşüÇĞİÖŞÜ")
    if tr_chars == 0:
        return False
    sample = " ".join(str(c.get("text", ""))[:500] for c in list(contexts)[:2])
    return ascii_ratio(sample) > 0.92


def clean_answer(answer: str) -> str:
    lines = [ln.strip() for ln in answer.strip().splitlines() if ln.strip()]
    cleaned: List[str] = []
    for ln in lines:
        lower = ln.lower()
        if lower.startswith("3)") and "baglamda yeterli bilgi yok" in lower:
            continue
        if lower.startswith("cikti formati"):
            continue
        cleaned.append(ln)
    text = "\n".join(cleaned).strip()
    return text or FALLBACK_ANSWER


def build_general_chat_prompt(question: str, system_instructions: Optional[str] = None, thinking_level: Optional[str] = None) -> str:
    extra_system = (system_instructions or "").strip()
    extra_block = f"\nEK SISTEM TALIMATI:\n{extra_system}\n" if extra_system else ""
    thinking_block = resolve_thinking_instruction(thinking_level=thinking_level)
    return (
        "Sen yardimci bir yapay zeka asistansin.\n"
        "Kullaniciyla dogal, kibar ve acik bir sekilde konus.\n"
        "Bilmedigin bir konuda eminmis gibi davranma.\n"
        f"{extra_block}"
        f"DUSUNME KILIDI:\n{thinking_block}\n\n"
        f"KULLANICI MESAJI:\n{question}\n\n"
        "Yanit:"
    )
