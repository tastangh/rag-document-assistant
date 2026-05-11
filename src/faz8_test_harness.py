from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import math
import os
import socket
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

from config import EVAL_DIR, OLLAMA_MODEL, OLLAMA_URL


SAFE_REJECTS = {"Baglamda yeterli bilgi yok.", "Veri bulunamadı"}
EXPECTED_EMBED_MODEL = "newmindai/Mursit-Large-TR-Retrieval"
logger = logging.getLogger(__name__)
DEFAULT_COLLECTION = "rag_chunks_v1"


def _lazy_runtime_imports() -> Dict[str, Any]:
    from generation_pipeline import FALLBACK_ANSWER, GUARDRAIL_REJECT_ANSWER, ask_question

    SAFE_REJECTS.update({FALLBACK_ANSWER, GUARDRAIL_REJECT_ANSWER})
    return {
        "ask_question": ask_question,
    }


def _lazy_perf_imports() -> Dict[str, Any]:
    from document_processor import process_document_to_markdown
    from chunk_embedding_pipeline import run_pipeline
    from retrieval_pipeline import build_vector_index

    return {
        "process_document_to_markdown": process_document_to_markdown,
        "run_pipeline": run_pipeline,
        "build_vector_index": build_vector_index,
    }


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        msg = f"Dosya bulunamadi: {path}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    rows: List[Dict[str, Any]] = []
    logger.info("JSONL yukleniyor: %s", path)
    with path.open("r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f, start=1):
            t = line.strip()
            if not t:
                continue
            try:
                rows.append(json.loads(t))
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL parse hatasi ({path}:{i}): {exc}") from exc
    return rows


def _validate_jsonl_file(path: Path, label: str) -> None:
    if not os.path.exists(path):
        msg = f"Dosya bulunamadi: {path}"
        logger.error(msg)
        raise FileNotFoundError(msg)
    if os.path.getsize(path) == 0:
        msg = f"Dosya bos: {path}"
        logger.error(msg)
        raise ValueError(msg)
    logger.info("%s dogrulandi: %s (boyut=%d byte)", label, path, os.path.getsize(path))


def _check_ollama_health(ollama_url: str, timeout_sec: int = 5) -> None:
    base = ollama_url.rstrip("/")
    if base.endswith("/api/generate"):
        base = base[: -len("/api/generate")]
    tags_url = base + "/api/tags"
    logger.info("Ollama saglik kontrolu baslatiliyor... url=%s", tags_url)
    try:
        req = urllib.request.Request(tags_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            if status != 200:
                raise RuntimeError(f"Ollama saglik kontrolu basarisiz: HTTP {status}")
    except socket.timeout as exc:
        msg = f"Ollama timeout: {tags_url} | sure={timeout_sec}s"
        logger.error(msg)
        print(msg)
        raise RuntimeError(msg) from exc
    except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
        msg = f"Ollama servisine erisilemedi: {tags_url} | hata={exc}"
        logger.error(msg)
        print(msg)
        raise RuntimeError(msg) from exc
    logger.info("Ollama saglik kontrolu basarili.")


def _tok(s: str) -> List[str]:
    import re

    return re.findall(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü]+", (s or "").lower())


def _token_f1(a: str, b: str) -> float:
    ta = _tok(a)
    tb = _tok(b)
    if not ta or not tb:
        return 0.0
    sa = {}
    for t in ta:
        sa[t] = sa.get(t, 0) + 1
    sb = {}
    for t in tb:
        sb[t] = sb.get(t, 0) + 1
    common = 0
    for k, va in sa.items():
        common += min(va, sb.get(k, 0))
    if common == 0:
        return 0.0
    p = common / max(len(ta), 1)
    r = common / max(len(tb), 1)
    return 2 * p * r / max(p + r, 1e-9)


def _ndcg_at_k(pred_doc_ids: Sequence[str], relevant_doc_ids: Sequence[str], k: int = 5) -> float:
    rel = {str(x).strip() for x in relevant_doc_ids if str(x).strip()}
    if not rel:
        return 0.0
    top = [str(x).strip() for x in pred_doc_ids[:k]]
    dcg = 0.0
    for i, d in enumerate(top, start=1):
        gain = 1.0 if d in rel else 0.0
        if gain > 0:
            dcg += gain / math.log2(i + 1)
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return (dcg / idcg) if idcg > 0 else 0.0


def _answer_relevance(answer: str, gt: str, has_citation: bool) -> float:
    if not answer or answer in SAFE_REJECTS:
        return 0.0
    f1 = _token_f1(answer, gt)
    bonus = 0.05 if has_citation else 0.0
    return min(1.0, f1 + bonus)


def run_benchmark(
    benchmark_file: Path,
    persist_dir: Path,
    collection: str,
    model: str,
    ollama_url: str,
    initial_k: int,
    final_k: int,
    retrieval_device: str,
) -> Dict[str, Any]:
    logger.info("Benchmark baslatiliyor... file=%s", benchmark_file)
    rows = _load_jsonl(benchmark_file)
    per_q: List[Dict[str, Any]] = []
    total = len(rows)

    rt = _lazy_runtime_imports()
    ask_question = rt["ask_question"]
    for idx, row in enumerate(rows, start=1):
        q = str(row.get("question", "")).strip()
        gt = str(row.get("expected_answer_key", "")).strip()
        rel = list(row.get("relevant_doc_ids", []) or [])
        if not q:
            continue

        print(f"Soru {idx} isleniyor: {q}")
        t0 = time.perf_counter()
        logger.info("Soru %d/%d isleniyor...", idx, total)
        logger.info("Ollama yaniti bekleniyor...")
        result = ask_question(
            question=q,
            persist_dir=persist_dir,
            collection_name=collection,
            initial_k=initial_k,
            final_k=final_k,
            model_name=model,
            ollama_url=ollama_url,
            strict_guardrail=True,
            device=retrieval_device,
        )
        t1 = time.perf_counter()

        ans = str(result.get("answer", "")).strip()
        sources = list(result.get("sources", []) or [])
        pred_doc_ids = [str(s.get("doc_id", "")) for s in sources]
        has_citation = len(sources) > 0
        ver = result.get("verification", {}) or {}
        faith = float(ver.get("supported_ratio", 0.0))
        context_rel = _ndcg_at_k(pred_doc_ids=pred_doc_ids, relevant_doc_ids=rel, k=5)
        answer_rel = _answer_relevance(ans, gt=gt, has_citation=has_citation)

        per_q.append(
            {
                "question": q,
                "expected_answer_key": gt,
                "relevant_doc_ids": rel,
                "answer": ans,
                "pred_doc_ids": pred_doc_ids,
                "scores": {
                    "context_relevance_ndcg_at_5": context_rel,
                    "faithfulness": faith,
                    "answer_relevance": answer_rel,
                    "triad_score": (context_rel + faith + answer_rel) / 3.0,
                },
                "latency_sec": t1 - t0,
            }
        )
        logger.info("Soru %d/%d tamamlandi", idx, total)

    if not per_q:
        raise RuntimeError("Benchmark calismasi icin gecerli soru yok.")

    avg = {
        "context_relevance_ndcg_at_5": sum(x["scores"]["context_relevance_ndcg_at_5"] for x in per_q) / len(per_q),
        "faithfulness": sum(x["scores"]["faithfulness"] for x in per_q) / len(per_q),
        "answer_relevance": sum(x["scores"]["answer_relevance"] for x in per_q) / len(per_q),
        "triad_score": sum(x["scores"]["triad_score"] for x in per_q) / len(per_q),
        "avg_latency_sec": sum(x["latency_sec"] for x in per_q) / len(per_q),
    }

    return {"count": len(per_q), "aggregate": avg, "questions": per_q}


def _preflight_index_manifest(persist_dir: Path, collection: str) -> None:
    logger.info("Preflight index manifest kontrolu baslatiliyor...")
    manifest = persist_dir / f"{collection}_index_manifest.json"
    if not manifest.exists():
        logger.error("Index manifest bulunamadi: %s", manifest)
        raise RuntimeError(
            f"Index manifest bulunamadi: {manifest}. Once build-index adimini calistirin."
        )
    data = json.loads(manifest.read_text(encoding="utf-8"))
    model_name = str(data.get("model_name", ""))
    logger.info("Preflight model kontrolu | beklenen=%s | bulunan=%s", EXPECTED_EMBED_MODEL, model_name)
    if model_name != EXPECTED_EMBED_MODEL:
        print(f"HATA: Beklenen model Mursit ama bulunan model {model_name}")
        logger.error(
            "Index modeli uyumsuz | beklenen=%s | bulunan=%s",
            EXPECTED_EMBED_MODEL,
            model_name,
        )
        raise RuntimeError(
            "Index modeli beklenenle uyusmuyor. "
            f"Beklenen={EXPECTED_EMBED_MODEL}, mevcut={model_name}. "
            "Lutfen chunk embedding + build-index adimlarini Mursit modeliyle tekrar calistirin."
        )


def run_adversarial(
    adversarial_file: Path,
    persist_dir: Path,
    collection: str,
    model: str,
    ollama_url: str,
    initial_k: int,
    final_k: int,
    retrieval_device: str,
) -> Dict[str, Any]:
    logger.info("Adversarial test baslatiliyor... file=%s", adversarial_file)
    rows = _load_jsonl(adversarial_file)
    out: List[Dict[str, Any]] = []

    rt = _lazy_runtime_imports()
    ask_question = rt["ask_question"]
    total = len(rows)
    for idx, row in enumerate(rows, start=1):
        q = str(row.get("question", "")).strip()
        if not q:
            continue
        logger.info("Ollama yaniti bekleniyor... (adversarial soru %d/%d)", idx, total)
        res = ask_question(
            question=q,
            persist_dir=persist_dir,
            collection_name=collection,
            initial_k=initial_k,
            final_k=final_k,
            model_name=model,
            ollama_url=ollama_url,
            strict_guardrail=True,
            device=retrieval_device,
        )
        ans = str(res.get("answer", "")).strip()
        passed = ans in SAFE_REJECTS
        out.append({"question": q, "answer": ans, "passed": passed})
        logger.info("Adversarial soru %d/%d tamamlandi", idx, total)

    total = len(out)
    passed_count = sum(1 for x in out if x["passed"])
    return {
        "count": total,
        "passed": passed_count,
        "pass_rate": (passed_count / total) if total else 0.0,
        "results": out,
    }


def run_session_isolation_check() -> Dict[str, Any]:
    candidates = [Path("data/sessions"), Path("src/results/ui_simple_runtime/sessions")]
    roots = [p for p in candidates if p.exists() and p.is_dir()]

    findings: List[Dict[str, Any]] = []
    for root in roots:
        sessions = [p for p in root.iterdir() if p.is_dir()]
        for s in sessions:
            sid = s.name
            state_files = list(s.glob(f"state_{sid}.json")) + list(s.glob("state.json"))
            findings.append(
                {
                    "session_id": sid,
                    "root": str(root),
                    "has_uploads": (s / "uploads").exists(),
                    "has_vector": (s / "vector").exists(),
                    "state_files": [str(x.name) for x in state_files],
                }
            )

    # Basit sizinti kurali: her session kendi klasorunde, state dosyasi sessiona ozel.
    violations = []
    for f in findings:
        if f["root"].endswith("data/sessions") and not any(n.startswith("state_") for n in f["state_files"]):
            violations.append({"session_id": f["session_id"], "issue": "state_session_file_missing"})

    return {
        "roots_checked": [str(r) for r in roots],
        "session_count": len(findings),
        "violations": violations,
        "isolation_ok": len(violations) == 0,
        "sessions": findings,
    }


def run_perf_measure(
    pdf_path: Path,
    runtime_root: Path,
    model: str,
    ollama_url: str,
    embed_device: str,
    retrieval_device: str,
) -> Dict[str, Any]:
    perf_rt = _lazy_perf_imports()
    process_document_to_markdown = perf_rt["process_document_to_markdown"]
    run_pipeline = perf_rt["run_pipeline"]
    build_vector_index = perf_rt["build_vector_index"]
    ask_question = _lazy_runtime_imports()["ask_question"]

    logger.info("Performans olcumu baslatiliyor... pdf=%s", pdf_path)
    session_dir = runtime_root / "perf_session"
    upload_dir = session_dir / "uploads"
    ocr_md_dir = session_dir / "ocr_md"
    chunk_dir = session_dir / "chunks"
    vector_dir = session_dir / "vector"
    for p in (upload_dir, ocr_md_dir, chunk_dir, vector_dir):
        p.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    logger.info("OCR baslatiliyor...")
    md = process_document_to_markdown(pdf_path, use_gpu=(embed_device == "cuda"))
    t1 = time.perf_counter()

    md_path = ocr_md_dir / f"{pdf_path.stem}.md"
    md_path.write_text(md, encoding="utf-8")

    logger.info("Chunk + embedding baslatiliyor...")
    run_pipeline(
        input_dir=ocr_md_dir,
        output_dir=chunk_dir,
        model_name="newmindai/Mursit-Large-TR-Retrieval",
        chunk_size=1200,
        chunk_overlap=200,
        min_chunk_size=120,
        device=embed_device,
        batch_size=32,
    )
    t2 = time.perf_counter()

    logger.info("Vector index build baslatiliyor...")
    build_vector_index(
        artifacts_dir=chunk_dir,
        persist_dir=vector_dir,
        collection_name=DEFAULT_COLLECTION,
        batch_size=256,
    )
    t3 = time.perf_counter()

    q = "Bu doküman ne anlatıyor?"
    g0 = time.perf_counter()
    logger.info("Ollama yaniti bekleniyor... (performans olcumu)")
    _ = ask_question(
        question=q,
        persist_dir=vector_dir,
        collection_name=DEFAULT_COLLECTION,
        model_name=model,
        ollama_url=ollama_url,
        strict_guardrail=True,
        device=retrieval_device,
    )
    g1 = time.perf_counter()

    return {
        "pdf": str(pdf_path),
        "embed_device": embed_device,
        "retrieval_device": retrieval_device,
        "timings_sec": {
            "ocr_only": t1 - t0,
            "chunk_embed": t2 - t1,
            "index_build": t3 - t2,
            "generation_total": g1 - g0,
            "ttft_proxy": g1 - g0,
            "total_e2e": g1 - t0,
        },
    }


def main() -> None:
    faulthandler.enable()
    print("FAZ8: script girisi")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="Faz 8 test ve stres dogrulama harness")
    parser.add_argument("--benchmark-file", default=str(EVAL_DIR / "benchmark_test.jsonl"))
    parser.add_argument("--adversarial-file", default=str(EVAL_DIR / "benchmark_adversarial.jsonl"))
    parser.add_argument("--output", default=str(EVAL_DIR / "faz8_report.json"))
    parser.add_argument("--persist-dir", default="src/results/vectorStore/chroma")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--model", default=OLLAMA_MODEL)
    parser.add_argument("--ollama-url", default=OLLAMA_URL)
    parser.add_argument("--initial-k", type=int, default=24)
    parser.add_argument("--final-k", type=int, default=5)
    parser.add_argument("--pass-threshold", type=float, default=0.60)
    parser.add_argument("--perf-doc", default="")
    parser.add_argument("--run-perf", action="store_true")
    parser.add_argument("--embed-device", default="cuda", choices=["cuda", "cpu", "auto"])
    parser.add_argument("--retrieval-device", default="cuda", choices=["cuda", "cpu", "auto"])
    args = parser.parse_args()
    logger.info("Faz 8 harness baslatildi.")
    logger.info("Benchmark baslatiliyor...")

    _validate_jsonl_file(Path(args.benchmark_file), "Benchmark dosyasi")
    _validate_jsonl_file(Path(args.adversarial_file), "Adversarial dosyasi")
    _check_ollama_health(args.ollama_url)

    _preflight_index_manifest(persist_dir=Path(args.persist_dir), collection=args.collection)

    benchmark = run_benchmark(
        benchmark_file=Path(args.benchmark_file),
        persist_dir=Path(args.persist_dir),
        collection=args.collection,
        model=args.model,
        ollama_url=args.ollama_url,
        initial_k=args.initial_k,
        final_k=args.final_k,
        retrieval_device=args.retrieval_device,
    )

    adversarial = run_adversarial(
        adversarial_file=Path(args.adversarial_file),
        persist_dir=Path(args.persist_dir),
        collection=args.collection,
        model=args.model,
        ollama_url=args.ollama_url,
        initial_k=args.initial_k,
        final_k=args.final_k,
        retrieval_device=args.retrieval_device,
    )

    session_iso = run_session_isolation_check()
    logger.info("Session izolasyon kontrolu tamamlandi. session_count=%d", session_iso.get("session_count", 0))

    perf: Dict[str, Any] = {"skipped": True}
    if args.run_perf and args.perf_doc:
        perf = run_perf_measure(
            pdf_path=Path(args.perf_doc),
            runtime_root=Path("data/sessions"),
            model=args.model,
            ollama_url=args.ollama_url,
            embed_device=args.embed_device,
            retrieval_device=args.retrieval_device,
        )
        logger.info("Performans olcumu tamamlandi.")

    triad = benchmark["aggregate"]["triad_score"]
    status = "PASS" if triad >= float(args.pass_threshold) else "FAIL"

    report = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "pass_threshold": float(args.pass_threshold),
            "benchmark_file": args.benchmark_file,
            "adversarial_file": args.adversarial_file,
        },
        "summary": {
            "triad_score": triad,
            "status": status,
            "adversarial_pass_rate": adversarial["pass_rate"],
            "session_isolation_ok": session_iso["isolation_ok"],
        },
        "benchmark": benchmark,
        "adversarial": adversarial,
        "session_isolation": session_iso,
        "performance": perf,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Rapor yazildi: %s", out)
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
