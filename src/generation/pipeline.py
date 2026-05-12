from __future__ import annotations

from .core import (
    ask_question,
    build_parser,
    chat_without_rag,
    evaluate_adversarial_result,
    evaluate_smoke_result,
    load_smoke_questions,
    main,
)
from .ollama import GenerationOptions, call_ollama, coerce_generation_options
from .prompts import FALLBACK_ANSWER, GUARDRAIL_REJECT_ANSWER, CITATION_RE
from .verify import TurkLettuceGuardrail, to_sources, verify_answer

__all__ = [
    "ask_question",
    "chat_without_rag",
    "load_smoke_questions",
    "evaluate_smoke_result",
    "evaluate_adversarial_result",
    "build_parser",
    "main",
    "GenerationOptions",
    "call_ollama",
    "coerce_generation_options",
    "FALLBACK_ANSWER",
    "GUARDRAIL_REJECT_ANSWER",
    "CITATION_RE",
    "TurkLettuceGuardrail",
    "to_sources",
    "verify_answer",
]

if __name__ == "__main__":
    main()
