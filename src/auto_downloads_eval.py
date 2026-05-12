from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from chunk_embedding_pipeline import run_pipeline
from document_processor import process_document_to_markdown
from generation_pipeline import ask_question
from retrieval_pipeline import DEFAULT_COLLECTION, build_vector_index


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _list_download_pdfs(downloads_dir: Path) -> List[Path]:
    return sorted(
        [p for p in downloads_dir.glob("*.pdf") if p.is_file()] + [p for p in downloads_dir.glob("*.PDF") if p.is_file()],
        key=lambda p: p.name.lower(),
    )


def _clear_dir(path: Path) -> None:
    if not path.exists():
        return
    for item in path.iterdir():
        if item.is_file():
            item.unlink(missing_ok=True)
        elif item.is_dir():
            _clear_dir(item)
            try:
                item.rmdir()
            except OSError:
                pass


def _list_data_pdf_names(data_dir: Path) -> set[str]:
    names: set[str] = set()
    for ext in ("*.pdf", "*.PDF"):
        for p in data_dir.rglob(ext):
            if p.is_file():
                names.add(_normalize_name(p.name))
    return names


def _generate_questions(doc_id: str, md_text: str) -> List[str]:
    lines = [ln.strip() for ln in md_text.splitlines() if ln.strip()]
    headings = [ln.lstrip("#").strip() for ln in lines if ln.startswith("#")]
    questions: List[str] = []

    questions.append(f"{doc_id} belgesinin ana konusu nedir?")
    if headings:
        for h in headings[:2]:
            questions.append(f"{doc_id} belgesinde '{h}' basliginda hangi konu anlatiliyor?")

    # Kisa sayisal/anahtar kelime sorulari
    text = " ".join(lines[:120])
    if any(k in text.lower() for k in ("tarih", "date", "deadline", "süre", "sure")):
        questions.append(f"{doc_id} belgesinde belirtilen tarih veya sure bilgisi nedir?")
    if any(k in text.lower() for k in ("teslim", "submission", "due")):
        questions.append(f"{doc_id} belgesinde teslim kosulu veya son tarih nedir?")

    return questions[:4]


def main() -> None:
    parser = argparse.ArgumentParser(description="Downloads PDF'lerinden otomatik soru uretip pipeline testi yapar")
    parser.add_argument("--downloads-dir", default=str(Path.home() / "Downloads"))
    parser.add_argument("--data-dir", default="src/data")
    parser.add_argument("--work-root", default="src/results/eval/downloads_auto_eval")
    parser.add_argument("--max-docs", type=int, default=6)
    parser.add_argument("--max-questions-per-doc", type=int, default=4)
    parser.add_argument("--include-name", default="", help="Belirli bir PDF adini (tam veya parcali) secmek icin filtre")
    parser.add_argument("--ollama-model", default="qwen3:8b")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate")
    parser.add_argument("--embed-device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--retrieval-device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--strict-guardrail", action="store_true", help="Strict guardrail ile calistir")
    parser.add_argument("--retrieval-min-overlap", type=float, default=0.08)
    args = parser.parse_args()

    downloads_dir = Path(args.downloads_dir)
    data_dir = Path(args.data_dir)
    work_root = Path(args.work_root)
    uploads_dir = work_root / "uploads"
    ocr_md_dir = work_root / "ocr_md"
    chunk_dir = work_root / "chunks"
    vector_dir = work_root / "vector"
    for p in (uploads_dir, ocr_md_dir, chunk_dir, vector_dir):
        p.mkdir(parents=True, exist_ok=True)
    for p in (uploads_dir, ocr_md_dir, chunk_dir, vector_dir):
        _clear_dir(p)

    download_pdfs = _list_download_pdfs(downloads_dir)
    existing_data_names = _list_data_pdf_names(data_dir)
    candidates = [p for p in download_pdfs if _normalize_name(p.name) not in existing_data_names]
    if args.include_name.strip():
        needle = _normalize_name(args.include_name)
        candidates = [p for p in candidates if needle in _normalize_name(p.name)]
    candidates = sorted(candidates, key=lambda p: p.stat().st_size)
    selected = candidates[: max(1, args.max_docs)]
    if not selected:
        raise RuntimeError("Downloads icinde src/data disinda PDF bulunamadi.")

    md_map: Dict[str, str] = {}
    prepared_docs: List[str] = []
    for pdf in selected:
        target = uploads_dir / pdf.name
        target.write_bytes(pdf.read_bytes())
        md = process_document_to_markdown(target, use_gpu=(args.embed_device == "cuda"))
        md_path = ocr_md_dir / f"{target.stem}.md"
        md_path.write_text(md, encoding="utf-8")
        md_map[target.stem] = md
        prepared_docs.append(target.name)

    run_pipeline(
        input_dir=ocr_md_dir,
        output_dir=chunk_dir,
        model_name="newmindai/Mursit-Large-TR-Retrieval",
        chunk_size=1200,
        chunk_overlap=200,
        min_chunk_size=120,
        device=args.embed_device,
        batch_size=32,
    )
    build_vector_index(
        artifacts_dir=chunk_dir,
        persist_dir=vector_dir,
        collection_name=DEFAULT_COLLECTION,
        batch_size=256,
    )

    all_questions: List[Dict[str, str]] = []
    for doc_id, md_text in md_map.items():
        for q in _generate_questions(doc_id, md_text)[: max(1, args.max_questions_per_doc)]:
            all_questions.append({"doc_id": doc_id, "question": q})

    results: List[Dict[str, Any]] = []
    for item in all_questions:
        q = item["question"]
        t0 = time.perf_counter()
        out = ask_question(
            question=q,
            persist_dir=vector_dir,
            collection_name=DEFAULT_COLLECTION,
            initial_k=12,
            final_k=4,
            device=args.retrieval_device,
            model_name=args.ollama_model,
            ollama_url=args.ollama_url,
            fast_mode=True,
            context_limit=4,
            auto_doc_filter=False,
            doc_id=item["doc_id"],
            retrieval_min_overlap=float(args.retrieval_min_overlap),
            strict_guardrail=bool(args.strict_guardrail),
        )
        latency = time.perf_counter() - t0
        results.append(
            {
                "doc_id": item["doc_id"],
                "question": q,
                "answer": out.get("answer", ""),
                "source_count": len(out.get("sources", []) or []),
                "latency_sec": latency,
                "verification": out.get("verification", {}),
            }
        )

    answered = [r for r in results if str(r["answer"]).strip()]
    avg_latency = sum(r["latency_sec"] for r in results) / max(1, len(results))
    with_sources = sum(1 for r in results if int(r["source_count"]) > 0)
    fallback_count = sum(1 for r in results if "baglamda" in str(r["answer"]).lower())

    report = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "downloads_dir": str(downloads_dir),
            "data_dir": str(data_dir),
            "selected_docs": prepared_docs,
            "question_count": len(results),
            "embed_device": args.embed_device,
            "retrieval_device": args.retrieval_device,
            "strict_guardrail": bool(args.strict_guardrail),
            "retrieval_min_overlap": float(args.retrieval_min_overlap),
        },
        "summary": {
            "answered_count": len(answered),
            "with_sources_count": with_sources,
            "fallback_count": fallback_count,
            "avg_latency_sec": avg_latency,
        },
        "results": results,
    }

    out_path = work_root / "downloads_auto_eval_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(out_path), "summary": report["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
