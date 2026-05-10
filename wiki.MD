# RAG Proje Wiki

Bu dosya, projede yaptigimiz isleri temel seviyeden anlatmak icin tutulur.
Amac: gun sonunda "ne yaptik, neden yaptik, ne ogrendik" bilgisini sade sekilde kaydetmek.

## 1) Proje Nedir?

Bu proje bir RAG (Retrieval-Augmented Generation) sistemi.
Kisa mantik:
- Once belgelerden ilgili parcalari buluyoruz (retrieval).
- Sonra LLM bu parcalari kullanarak cevap olusturuyor (generation).

Boylece model uydurmak yerine belgeye dayali cevap veriyor.

## 2) Fazlarin Kisa Ozeti

### Faz 1 - OCR ve Markdown
- PDF/PNG/JPG belgelerden metin cikardik.
- Ciktiyi markdown formatinda kaydettik.
- Tablolari korumaya calistik.

### Faz 2 - Chunk + Embedding
- Uzun metinleri kucuk parcalara (chunk) ayirdik.
- Her parcayi embedding (vektor) haline cevirdik.
- Ciktilar:
  - `chunks.jsonl`
  - `embeddings.npy`
  - `manifest.json`

### Faz 3 - Retrieval Pipeline
- Chroma vector DB ile index kurduk.
- Soru sorunca en ilgili chunklari getiren akisi test ettik.
- Reranker acik/kapali karsilastirmasi yapmaya basladik.

## 3) Temel Terimler

- Chunk: Belgeden bolunmus kucuk metin parcasi.
- Embedding: Metnin sayisal vektor temsili.
- Vector DB (Chroma): Vektorleri saklayip benzerlik arayan veritabani.
- Index: Aramayi hizlandiran yapi.
- initial_k: Ilk asamada getirilen aday chunk sayisi.
- final_k: Son asamada LLM'e gidecek nihai chunk sayisi.
- Reranker: Adaylari ikinci kez daha iyi siralayan model.
- recall@k / mrr / ndcg@k: Retrieval kalitesini olcen metrikler.

## 4) Komut Sonrasi Aciklama Kurali

Komut verirken su format kullanilir:
- Komut
- Ne yapti
- Neden yaptik
- Yorum (sonucu nasil okumaliyiz)

## 5) GPU Kurali

Varsayilan yaklasim GPU-first:
- Mumkun olan tum adimlarda `cuda`/GPU kullanilir.
- CPU sadece zorunlu fallback olarak kullanilir.

## 6) Gun Sonu Not Formati

Her gun su basliklarla kisa ozet gecilir:

### Tarih: YYYY-MM-DD
- Bugun ne yaptik?
- Neden yaptik?
- Cikan sonuc ne?
- Sonraki adim ne?

## 7) Gun Sonu Ozet (2026-05-10)

- Faz 3 retrieval tarafi aktif olarak kullanildi.
- `build-index` ile Faz 2 artefaktlarindan Chroma index kurulu oldugu dogrulandi.
- `query` testleri ile ilgili chunklarin dondugu goruldu.
- `evaluate` adimina gecis icin tum OCR dokumanlarini kapsayan eval soru seti hazirlandi (`eval_docs.jsonl`).
- Kural seti netlestirildi:
  - Kisa/orta/detay anlatim
  - Komut + Ne yapti + Neden + Yorum
  - GPU-first varsayimi

## 8) Faz 3 Final Durumu (Kapanis)

Faz 3 tamamlandi.

Final karar:
- Varsayilan retrieval ayari kalici olarak `initial_k=16`, `final_k=5`, `rerank acik`, `device=cuda` secildi.

Kisa gerekce:
- 16/5 + rerank acik metrikleri en iyi sonucu verdi.
- Rerank kapandiginda metrikler belirgin dustugu icin reranker acik tutuldu.
