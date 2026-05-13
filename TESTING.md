

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

GEÇTİ 
Belge, 3-Hidroksil-3-metilbütanoik asit (Tur 13) bileşiğinin farelerdeki akut toksisite etkilerini inceleyen bir deney sunuyor. - Deneyde farklı dozajlarda (5, 10, 10, 10, 10 mg/kg) bileşiğin etkileri gözlemlenmiş ve 48 saat içinde sağ kalanlar tamamen iyileşmiş. - Bileşiğin merkezi sinir sistemi (CNS) depresanı olduğu belirtilmiş; solunum baskısı, damar daralması ve hareket kaybı gibi semptomlar gözlemlenmiştir. - İnsanlarda tek deneme için önerilen güvenli doz 0.3 mg inhalasyon yoluyla önerilmiştir. - Deney, Litchfield ve Wilcoxon (1948) tarafından referans verilen bir çalışma çerçevesinde yapılmıştır.
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


geçti 
Öğretmenler için klasik eserlerin satın alınması Vekilin klasik eserlerin alınmasına yönelik indirimli fiyat ve taksitli ödeme imkanları sunulmaktadır. Ayrıca, başvuruların nasıl yapılacağı ve ilgili kurumların belirtilmesi yer almaktadır. 2. Yerel yönetimlerle ilgili resmi istekler Belgede bir köyün yeniden kurulması ve bazı resmi isteklerin yer aldığı bilgiler yer almaktadır. Bu bağlamda, Dahiliye Vekili tarafından yapılan istekler ve ilgili kurumlara gönderilen dilekçeler detaylandırılmaktadır. 3. Eğitim alanları ve katalog bilgileri Belgede çeşitli eğitim alanları (örneğin, edebiyat, sanatlar, tarih) ve bu alanlara ait detaylı katalog bilgileri yer almaktadır. Bu bilgiler, eğitim programları veya kurs kataloğu olabilir. 4. Taksit ödeme süresi ve sona erme tarihi Klasik eserlerin satın alınmasında taksitli ödeme seçeneğinin sona erme tarihi belirtilmektedir. Bu belge, eğitim, yerel yönetim ve resmi işlemler gibi farklı alanlara ait bilgiler içeriyor olup, genellikle resmi bir belge veya katalog niteliğindedir.
------------------------------------------------------------
CASE-3: PDF TR
------------------------------------------------------------
set:
1) `TEST.pdf` yukle
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

geçti
Introduction of ASHEN Framework: A unified gated fusion architecture that integrates CLIP, YOLO, HOI, and text caption modalities for weakly supervised video anomaly detection. 2. Pure Temporal Transformer: A novel temporal modeling approach that operates on pre-extracted features without requiring end-to-end video backbone fine-tuning. 3. Curated Ashen Dataset: A balanced surveillance dataset comprising 1,600 videos across nine hostile-intent categories, designed for training and evaluating anomaly detection systems. 4. Comprehensive Ablation Study: Demonstrated that multimodal fusion significantly improves accuracy and Macro-AUC, with specific results showing improvements over state-of-the-art methods. 5. Strong Experimental Results: Achieved 98.09% binary AUC on the Ashen Dataset and 96.22% on UCF-Crime using the CLIP baseline, along with nine-class classification accuracy and macro-AUC using a triple-fusion model. 6. Future Work Directions: Proposed lightweight adapters, domain-specific feature refinement, class-adaptive thresholds, and incremental learning for real-world deployment in autonomous surveillance systems.

------------------------------
CASE-5: TUSAS PDF - PROJE OZETI (TR)
------------------------------------------------------------
set:
1) `TEST.pdf` yukle
2) `Belgeleri Hazirla`
3) soru: `Bu dokumanda projenin ozeti nedir?`

verify:
- cevapta belge yukleme + soru-cevap + kaynak gosterimi kapsami gecmeli
- en az 1 kaynak gorunmeli
- `fallback_used=false` beklenir
GEÇTİ
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
GEÇTİ

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
GEÇTİ

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
GEÇTİ

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
GEÇTİ

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
GEÇTİ


