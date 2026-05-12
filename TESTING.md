

------------------------------------------------------------
CASE-1: SCAN PNG EN
------------------------------------------------------------
set:
1) `tarama_eng.png` yukle
2) `Belgeleri Hazirla`
3) soru: `belgeyi ozetle`

verify:
- dokuman durumu `ready`
- cevap fallback degilse en az 1 kaynak gorunmeli
- fallback olursa `Baglamda yeterli bilgi bulunamadi` metni donmeli

------------------------------------------------------------
CASE-2: SCAN PNG TR
------------------------------------------------------------
set:
1) `tarama_tr2.png` yukle
2) `Belgeleri Hazirla`
3) soru: `bu belgede ne anlatiyor`

verify:
- dokuman durumu `ready`
- cevap + kaynaklar paneli acilabiliyor olmali
- OCR gurultusu olsa da sistem crash etmemeli

------------------------------------------------------------
CASE-3: PDF TR
------------------------------------------------------------
set:
1) `Case_Study_TUSA�_LLM.pdf` yukle
2) `Belgeleri Hazirla`
3) soru: `teslimatlar neler`

verify:
- metin-layer varsa hizli hazirlama (OCR beklenenden daha kisa)
- cevapta en az 1 kaynak olmali
- `Debug/Details` icinde `sources_count >= 1`

------------------------------------------------------------
CASE-4: PDF EN
------------------------------------------------------------
set:
1) `ASHEN__Automated_Surveillance_and_Hostile_Intent_Evaluation_Network___A_Multimodal_Transformer_Framework_for_Weakly_Supervised_Video_Anomaly_Detection.pdf` yukle
2) `Belgeleri Hazirla`
3) soru: `what is the main contribution`

verify:
- dokuman `ready`
- cevap anlamsal olarak makale amacina deginmeli
- kaynak listesinde ayni dokumana ait chunklar gorunmeli

------------------------------------------------------------
3) KABUL KRITERI (4/4)
------------------------------------------------------------

- Her case'de uygulama hata vermeden tamamlanmali.
- En az 3 case'de fallback disi kaynakli cevap alinmali.
- Tum case'lerde UI yanitsiz kalma/crash olmamali.

------------------------------------------------------------
CASE-5: TUSAS PDF - PROJE OZETI (TR)
------------------------------------------------------------
set:
1) `Case_Study_TUSAŞ_LLM.pdf` yukle
2) `Belgeleri Hazirla`
3) soru: `Bu dokumanda projenin ozeti nedir?`

verify:
- cevapta belge yukleme + soru-cevap + kaynak gosterimi kapsami gecmeli
- en az 1 kaynak gorunmeli
- `fallback_used=false` beklenir

------------------------------------------------------------
CASE-6: TUSAS PDF - TESLIMAT DOSYALARI (TR)
------------------------------------------------------------
set:
1) ayni oturumda devam et
2) soru: `Teslimatta istenen dosyalar nelerdir?`

verify:
- cevapta `DEVLOG.md` ve `TESTING.md` gecmeli
- kaynaklar panelinde ilgili sayfa/chunk gorunmeli
- `confidence` en az `medium` olmali

------------------------------------------------------------
CASE-7: TUSAS PDF - TESLIM YONTEMI (TR)
------------------------------------------------------------
set:
1) ayni oturumda devam et
2) soru: `Teslim yontemi nedir?`

verify:
- cevapta `GitHub repository linki` bilgisi gecmeli
- en az 1 kaynak teslimat bolumunden gelmeli
- cevap net ve kisa olmali

------------------------------------------------------------
CASE-8: TUSAS PDF - PLAN YORUMU (TR)
------------------------------------------------------------
set:
1) ayni oturumda devam et
2) soru: `Bu case study icin 3 fazli bir calisma plani onerir misin?`

verify:
- cevap dokuman baglamina dayali olmali
- en az 1 kaynak olmali
- dokumanda olmayan kesin iddialar olmamali

------------------------------------------------------------
CASE-9: TUSAS PDF - CROSS-LINGUAL (EN)
------------------------------------------------------------
set:
1) ayni oturumda devam et
2) soru: `What are the expected deliverables in this case study?`

verify:
- cevap ingilizce ve anlamli olmali
- deliverable olarak `DEVLOG.md` / `TESTING.md` korunmali
- kaynaklar ayni turkce dokumandan gelmeli

------------------------------------------------------------
CASE-10: TUSAS PDF - HALLUCINATION GUARD
------------------------------------------------------------
set:
1) `strict_guardrail=true`
2) soru: `Bu dokumanda butce ve maas bilgileri nedir?`

verify:
- sistem guvenli fallback vermeli (`Baglamda yeterli bilgi bulunamadi`)
- uydurma sayisal bilgi uretmemeli
- `fallback_used=true` beklenir
