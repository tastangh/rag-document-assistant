from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .prompts import clean_answer


@dataclass(frozen=True)
class GenerationOptions:
    temperature: float = 0.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    repeat_penalty: Optional[float] = None


def coerce_generation_options(
    temperature: Optional[float],
    top_k: Optional[int],
    top_p: Optional[float],
    repeat_penalty: Optional[float],
) -> GenerationOptions:
    return GenerationOptions(
        temperature=0.0 if temperature is None else float(temperature),
        top_k=None if top_k is None else int(top_k),
        top_p=None if top_p is None else float(top_p),
        repeat_penalty=None if repeat_penalty is None else float(repeat_penalty),
    )


def call_ollama(
    prompt: str,
    model_name: str,
    ollama_url: str,
    options: Optional[GenerationOptions] = None,
    timeout_sec: int = 180,
) -> str:
    generation = options or GenerationOptions()
    ollama_options: Dict[str, Any] = {"temperature": generation.temperature}
    if generation.top_k is not None:
        ollama_options["top_k"] = generation.top_k
    if generation.top_p is not None:
        ollama_options["top_p"] = generation.top_p
    if generation.repeat_penalty is not None:
        ollama_options["repeat_penalty"] = generation.repeat_penalty

    payload = {"model": model_name, "prompt": prompt, "stream": False, "options": ollama_options}
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
        raise RuntimeError("Ollama'ya baglanilamadi. Ollama acik mi? Varsayilan URL: http://localhost:11434") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Ollama cevabi JSON degil: {raw[:400]}") from exc

    answer = str(data.get("response", "")).strip()
    if not answer:
        raise RuntimeError(f"Ollama bos cevap dondu: {raw[:400]}")
    return clean_answer(answer)
