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

GECTI 
Belge, 3-Hidroksil-3-metilbutanoik asit (Tur 13) bilesiginin farelerdeki akut toksisite etkilerini inceleyen bir deney sunuyor. - Deneyde farkli dozajlarda (5, 10, 10, 10, 10 mg/kg) bilesigin etkileri gozlemlenmis ve 48 saat icinde sag kalanlar tamamen iyilesmis. - Bilesigin merkezi sinir sistemi (CNS) depresani oldugu belirtilmis; solunum baskisi, damar daralmasi ve hareket kaybi gibi semptomlar gozlemlenmistir. - Insanlarda tek deneme icin onerilen guvenli doz 0.3 mg inhalasyon yoluyla onerilmistir. - Deney, Litchfield ve Wilcoxon (1948) tarafindan referans verilen bir calisma cercevesinde yapilmistir.
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


gecti 
Ogretmenler icin klasik eserlerin satin alinmasi Vekilin klasik eserlerin alinmasina yonelik indirimli fiyat ve taksitli odeme imkanlari sunulmaktadir. Ayrica, basvurularin nasil yapilacagi ve ilgili kurumlarin belirtilmesi yer almaktadir. 2. Yerel yonetimlerle ilgili resmi istekler Belgede bir koyun yeniden kurulmasi ve bazi resmi isteklerin yer aldigi bilgiler yer almaktadir. Bu baglamda, Dahiliye Vekili tarafindan yapilan istekler ve ilgili kurumlara gonderilen dilekceler detaylandirilmaktadir. 3. Egitim alanlari ve katalog bilgileri Belgede cesitli egitim alanlari (ornegin, edebiyat, sanatlar, tarih) ve bu alanlara ait detayli katalog bilgileri yer almaktadir. Bu bilgiler, egitim programlari veya kurs katalogu olabilir. 4. Taksit odeme suresi ve sona erme tarihi Klasik eserlerin satin alinmasinda taksitli odeme seceneginin sona erme tarihi belirtilmektedir. Bu belge, egitim, yerel yonetim ve resmi islemler gibi farkli alanlara ait bilgiler iceriyor olup, genellikle resmi bir belge veya katalog niteligindedir.
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

gecti
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
GECTI
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
GECTI

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
GECTI

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
GECTI

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
GECTI

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
GECTI

------------------------------------------------------------
CASE-11: MERKEZBANKASI PDF (TR) - YORUM + BILGI
------------------------------------------------------------
set:
1) `merkezbankası.pdf` yukle
2) `Belgeleri Hazirla`
3) yorum sorusu: `Bu metnin ana mesaji ve olasi etkisi hakkinda kisa bir yorum yapar misin?`
4) bilgi sorusu: `Bu belgede gecen temel tarih, kurum veya karar bilgileri nelerdir?`

verify:
- dokuman durumu `ready`
- iki soruya da ayri ve anlamli cevap donmeli
- her cevapta en az 1 kaynak gorunmeli
YORUM
2026 yılında yıllık APİ portföy büyüklüğü nominal 450 milyar TL olarak belirlenmiş ve bu hedefe ulaşmak için 255 milyar TL tutarında kıymet alımı planlanmıştır .
Doğrudan alım ihaleleri, piyasa koşulları dikkate alınarak perşembe ve cuma günleri düzenlenmiştir ve 48,9 milyar TL doğrudan alım gerçekleştirilmiştir .
İhalelerin ikincil piyasa fiyatlarına etkisi analiz edilmiş ve doğrudan alımın piyasa etkisinin sınırlı olduğu sonucuna varılmıştır .
Parasal sıkılığın korunması ve enflasyon beklentilerinin iyileşmesi, 2027 yıl sonu enflasyon tahmin aralığının korunmasına yönelik stratejik önemi taşımaktadır .
Bilgi
24 Ocak 2026: Yurt dışından temin edilen 1 yıla kadar vadeli Türk lirası cinsi borçlara uygulanan zorunlu karşılık oranlarının 2 puan artırıldığı duyurulmuştur. - 2. 12 Aralık: Türk lirası cinsinden kredi kartı işlemlerinde uygulanan aylık azami faiz oranlarında 25 baz puan indirim yapılmıştır. - 3. 29 Ocak 2026: Bankacılık Düzenleme ve Denetleme Kurulu’nun kararı kapsamında kredi kartı borçlarının yeniden yapılandırılması için azami akdi faiz oranı referans oran ile sınırlandırılmıştır. - 4. 5 Aralık 2025: Sistemdeki fazla likidite 500 milyar TL etrafında dalgalı seyretmiştir. - 5. 9 Şubat 2026: Sistemdeki fazla likidite 899 milyar TL seviyesine ulaşmıştır. - 6. 150 ve 100 baz puan indirimleri: Aralık ve Ocak aylarında TCMB tarafından politika faizinde uygulanan indirimlerdir.
------------------------------------------------------------
CASE-12: MERKEZBANKASI PDF (EN) - YORUM + BILGI
------------------------------------------------------------
set:
1) `merkezbankası_eng.pdf` yukle
2) `Belgeleri Hazirla`
3) yorum sorusu: `Can you briefly comment on the document's tone and likely policy intent?`
The document's tone is analytical, cautious, and focused on controlled monetary policy adjustments to achieve disinflation. The policy intent appears to emphasize maintaining tight monetary policy until price stability is achieved, using macroprudential measures to support monetary transmission, and cautiously adjusting policy rates based on inflation trends and expectations. - The document highlights a balanced approach between tightening monetary conditions and supporting economic stability, with a focus on aligning policy actions with inflation forecasts and intermediate targets. - It underscores the importance of coordination among policy tools and prudential measures to sustain low inflation without compromising financial stability.
4) bilgi sorusu: `What are the key dates, institutions, or decisions explicitly mentioned in the document?`
The Central Bank of the Republic of Turkey (CBRT) reduced the policy rate by 150 basis points in December and 100 basis points in January 2026, delivering a total rate cut of 250 basis points [doc_id:merkezbankası_eng::p1::c12].
The CBRT raised reserve requirement ratios for TRY-denominated funding obtained from abroad with maturities up to one year by two percentage points on January 24, 2026 [doc_id:merkezbankası_eng::p1::c16].
The Banking Regulation and Supervision Agency (BRSA) decided on January 29, 2026, to limit the maximum contractual interest rate for restructuring personal credit card debts to the reference rate [doc_id:merkezbankası_eng::p1::c17].
The CBRT reduced the maximum monthly interest rates for credit card transactions in Turkish lira by 25 basis points, effective from January 1, 2026 [doc_id:merkezbankası_eng::p1::c16].
Excess liquidity in the system increased to TRY 899 billion as of February 9, 2026, following CBRT operations [doc_id:merkezbankası_eng::p1::c17].
The CBRT announced on December 2, 2025, that the zero-percent reserve requirement on long-term foreign currency liabilities would not be extended beyond year-end 2025 [doc_id:merkezbankası_eng::p1::c15].
verify:
- document status `ready`
- both questions receive distinct and grounded answers
- at least 1 source appears for each answer

------------------------------------------------------------
CASE-13: SCAN PNG EN (1) - YORUM + BILGI
------------------------------------------------------------
set:
1) `tarama_eng.png` yukle
2) `Belgeleri Hazirla`
3) yorum sorusu: `Bu belgeye gore calismanin guvenilirligi veya sinirlari hakkinda kisa bir yorum yapar misin?`
Baglamda yeterli bilgi yok.
4) bilgi sorusu: `Belgede belirtilen LD50 degeri ve gozlenen temel semptomlar nelerdir?`
Belgede belirtilen LD50 değeri 3.5 (3.1 ile 3.9) g/kg'dır . Gözlenen temel semptomlar:
Solunum depresyonu .
Kan damarlarının daralması .
Hareketsizlik .


verify:
- OCR gurultusune ragmen cevaplar anlasilir olmali
- bilgi sorusunda sayisal/veri odakli cevap donmeli
- en az 1 kaynak gorunmeli

------------------------------------------------------------
CASE-14: SCAN PNG EN (2) - YORUM + BILGI
------------------------------------------------------------
set:
1) `tarama_eng2.png` yukle
2) `Belgeleri Hazirla`
3) yorum sorusu: `Reviewer notuna gore bu duzeltmenin bilimsel anlatim kalitesi acisindan onemi nedir?`
"Mass spectroscopic" ifadesinin "Mass spectrometric" olarak düzeltilmesi, bilimsel terminolojinin doğru kullanilmasi ve anlasilabilirligi arttirmaktadir .
Duzeltmenin uygulanmasi, cikarilan verilerin bilimsel anlamda dogru ve tutarli sunulmasini saglar .
Duzeltmenin uygulanmasi, bilimsel raporlamada terimlerin karmasikligini azaltarak okuyucularin daha kolay anlasilmasini destekler .

VERIFY: cevabı belgeyle uyumlu ve makul yorum içeriyor.
4) bilgi sorusu: `Reviewer hangi satirda hangi terimin neyle degistirilmesini istemis?`

Reviewer, 4. sayfanın son satırında "Mass spectroscopic" teriminin "Mass spectrometric" ile değiştirilmesini istemiştir.

 cevabı doğru ve net: “Page 4 - Last line … Mass spectroscopic -> Mass spectrometric”.
5) Halüsinasyon sorusu :  `Belgede gecen cilt, sayi ve yayim tarihini yazar misin? `
Baglamda yeterli bilgi yok.

doğru davranış
verify:
- cevaplar review form icerigiyle tutarli olmali
- bilgi sorusu cevabinda satir ve terim degisikligi acikca gecmeli
- en az 1 kaynak gorunmeli

------------------------------------------------------------
CASE-15: SCAN PNG TR (1) - YORUM + BILGI
------------------------------------------------------------
set:
1) `tarama_tr.png` yukle
2) `Belgeleri Hazirla`
3) yorum sorusu: `Bu sayfaya gore donemin egitim/yayin politikasina dair kisa bir yorum yapar misin?`
Eğitimde yerel kültürün ve halk dilinin değerlerinin vurgulandığı "Halk ağzından söz derleme dergisi" gibi eserlerin okullara ve öğretmenlere yaygınlaştırılması hedeflenmiştir .
Yayin politikalarında fiyat indirimleri ve ücretsiz dağıtım (örneğin, ilgili makamlara paralı olmaksızın gönderme) gibi uygulamalar yer almakta olup, erişilebilirlik arttırmayı amaçlamaktadır .
Eğitim kurumlarında (örneğin, Güzel Sanatlar Akademisi) düzenli düzenlemeler ve kuralların güncellenmesi, eğitim standartlarının belirlenmesi açısından önemlidir .
Yayin ve eğitim politikalarının uygulanmasında hem kamu kurumları hem de bakanlıkların koordineli bir şekilde rol oynadığı görülmektedir .

verify : büyük ölçüde doğru yorum .
halk ağzından söz derleme dergisi ve kurum örnekleri de sayfadaki içerikle genel olarak örtüşüyor.
4) bilgi sorusu: `Belgede gecen cilt, sayi ve yayim tarihini yazar misin?`

uhığanda temin edilebilecek olan bu kitabin ilgililere tavsiyesini rica ederim. Milli Eğitim Bakanı y.K. S.Müstepar B. Kadirgan YAYIM MUDURLUGU 14/II/1949 Sayn: 82/1979ÖZET: 2434Halk ağzndan söz derle-me dergisi" h. Türk... - # ;: 28ŞUBAT 1949 vin 100 kuruş karphğnda okul kooperatiflerinde satılmasında bir sakınca görülmemiştir.Bu hosusun ilgililere duyurulmasım rica ederim. Milli Eğitim Bakama y.K. S. .Müstegar B. Kadırgan YAYIM MUDURLUGU 14... - ## Talim ve Terbiye Kuralu Kararı: Karar sayısi : 9ÖZET:Karar Tarihi 8/2/1949Güzel Sanatlar Akademisi Süsleme Sanstlar bölümü yönetmeliğinin 5. madde sinin degistrilmesi h. Güzel Sanatlar Akademisi Süsleme Sanatlarıbölüm... - tmen Okulu Müdürü,Edirne" adresinden 200 kurus karşhgunda temin edilebilecek olan bu esrin ilgillere tavysni rica ederin.Mili Ejtim Bakans y.Müstepar K. s B. Kadirgan

temiz ve güvenilir bir çıktı gelmedi 
 Cilt: 7, Sayı: 306, 11 gibi bir çıktı gelmesi lazımdı 

 5) bilgi sorusu: `fevzi selen'in görev yaptığı şehir neresidir?`
"Fevzi Selen. Öğretmen Okulu Müdürü, Edirne" adresinden 200 kuruş karşılığında temin edilebilecek olan eserin ilgili kişilerce tavsiye edilmesi ricası yapılmıştır .
"Fevzi Selen. Öğretmen Okulu Müdürü, Edirne" adresinden 200 kuruş karşılığında temin edilebilecek olan eserin ilgili kişilerce tavsiye edilmesi ricası yapılmıştır .
"Edirne" adresinden 200 kuruş karşılığında temin edilebilecek olan eserin ilgili kişilerce tavsiye edilmesi ricası yapılmıştır .

Cevap kısa net olmasa da doğru 

verify:
- osmanlica/tarihsel baski kaynakli OCR hatalari sistemi bozmamali
- bilgi sorusuna kisa ve dogrudan cevap donmeli
- en az 1 kaynak gorunmeli



