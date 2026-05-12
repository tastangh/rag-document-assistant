OCR TESTLERI

12-05-2026 - QUICK 3Q GPU TEST

ortam hazirligi:
- cd E:\Workspace\rag-document-assistant
- .\.venv\Scripts\Activate.ps1
- (gerekirse) Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

calistirilan komut:
- python src/auto_downloads_eval.py --downloads-dir "C:\Users\MT\Downloads" --data-dir src/data --max-docs 1 --max-questions-per-doc 3 --embed-device cuda --retrieval-device cuda --ollama-model qwen3:8b --ollama-url http://localhost:11434/api/generate

incelenen ciktilar:
- src/results/eval/downloads_auto_eval/quick_3q_report.json
- src/results/eval/downloads_auto_eval/quick_3q_eval.json

sonuclar (ozet):
- soru sayisi: 3
- fallback cevap: 3/3
- supported_ratio: 0.0 (tum sorular)
- reason: low_retrieval_overlap
- q1 latency: 377.51 sn
- q2 latency: 0.10 sn
- q3 latency: 0.09 sn
- total: 377.7 sn

yorum:
- ilk sorguda belirgin warmup maliyeti var.
- retrieval overlap dusuk kaldigi icin answer quality fallback'te kaliyor.



-----------------------------------------
tek tek döküman yükle soru sor gibi testler yapıyorum yorumlama zayıf kalıyor buna göre güncelleme yapıyorum.

Çok yavaş çalışıyor

Ui da belge yüklüyorum belgeyi hazırlatıyorum ve sorular soruyorum.
