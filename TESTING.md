
# TEST CASE : 

------------------------------------------------------------
1 TEST CASE: tarama_eng.png (EN OCR + RAG davranisi)
------------------------------------------------------------

test kimligi:
- TC-EN-PNG-001

amac:
- ingilizce taranmis PNG dokumanda OCR + retrieval + cevaplama kalitesini olcmek.
- fakt sorulari, ozet sorusu ve anlamsal soru tiplerinde sistem davranisini gozlemek.

test ortami:
- oturum: `a1da23cc-b5af-42b6-8390-b34e84a2650b`
- run fingerprint:
  - `llm=qwen3:8b`
  - `emb=newmindai/Mursit-Large-TR-Retrieval`
  - `rerank=BAAI/bge-reranker-v2-m3`
  - `temp=0.3`
  - `strict=True`
  - `fast=True`
- belge: `tarama_eng.png` (116.6KB)
- durum: `ready`

test adimlari:
1) `tarama_eng.png` yuklendi.
2) belge hazirlama islemi tamamlandi ve aktif dokuman olarak goruldu.
3) asagidaki sorular sirasiyla soruldu:
   - `solution 5 ise dosage kaç belgede`
   - `belgede en anlatılmaktadır`
   - `belgeyi özetle`
   - `hangi hayvanlar`
4) her soru icin cevap + kaynaklar + debug/details kaydedildi.

beklenen sonuc:
- fakt sorularinda ilgili doz/hayvan bilgisi kaynakli donmeli.
- anlamsiz veya belirsiz soruda sistem netlestirme istemeli veya kontrollu fallback vermeli.
- ozet sorusunda dokumanin ana bulgulari kaynakli ve tutarli donmeli.

gerceklesen sonuc:
- S1 `solution 5 ise dosage kaç belgede`
  - cevap: `Solution 5 için dozaj 1800 belgede belirtilmiştir.`
  - sonuc: basarili (kaynakli, confidence high, fallback yok).
  - debug:
    - `latency_sec=45.798`
    - `sources_count=3`
    - `fallback_used=false`
    - `supported_ratio=1`
    - `confidence=high`
    - `query_mode=rag_fact`
- S2 `belgede en anlatılmaktadır`
  - cevap: `Bağlamda yeterli bilgi bulunamadı`
  - sonuc: fallback (soru yapisi belirsiz/anlamsal).
- S3 `belgeyi özetle`
  - cevap: toksisite, doz araligi, semptomlar, iyilesme suresi, yayin bilgisi ozetlendi.
  - sonuc: basarili (icerik olarak anlamli ozet).
- S4 `hangi hayvanlar`
  - cevap: `Bağlamda yeterli bilgi bulunamadı`
  - sonuc: kismi basarisiz.
  - not: kaynaklarda `ACUTE TOXICITY IN MICE` ifadesi gecmesine ragmen cevap fallback oldu.

analiz:
- sistem fakt sorusunda (S1) guclu performans verdi.
- ozet sorusu (S3) islevsel.
- kisa ve baglamsiz soru kaliplarinda (S2/S4) intent + evidence eslestirmesi zayif kalabiliyor.
- OCR ciktisinda gürültu/bozulma oldugunda (`MICE` gibi tek kritik anahtar) guardrail bazen gereksiz sertlesiyor.

aksiyon maddeleri:
1) kisa soru normalizasyonu eklenmeli (`hangi hayvanlar` -> `bu belgede deney hangi hayvanda yapilmis`).
2) source text keyword boosting eklenmeli (`mice`, `rat`, `rabbit`, vb. biyolojik anahtarlar).
3) fallback oncesi tek-cumle extractive deneme adimi (strict modda kontrollu) korunmali/guclendirilmeli.
4) OCR gurultusu yuksek belgeler icin `rag_fact` temperature ust siniri dusuk tutulmali (mevcut cap korunacak).
