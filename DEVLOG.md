06-05-2026
Case Study'deki sistem için neleri yapmama gerek olduğunu bularak başladım.
LLM yardımı ile bu proje için gerekli ihtiyaçları çıkarttım.
Gerekli ihtiyaçları fazlara böldüm.

Çıkardığım fazlar;
# Faz 1: Görüntü İşleme ve Optik Karakter Tanıma (OCR) Mimarisi Kurma
Bir kere her şeyden önce belge ve görüntüleden metin çıkarımı yapacağız. görüntülerden metin çıkarmayı koşaylaştırma için OCR optik karakter tanıma mimarisini kullanmamız gerekiyor.
# Faz 2: Metin Bölütleme (Chunking) ve Vektör Gömme (Embedding)
YTÜ YL de edindiğim NLP dersi bilgilerim ve projelerime göre metinlerin ve kelimelerin ngram ve chunkları çok önemli. Ayrıca vektör uzayları da bizim için çok önemli. BÖlünen metinler LLM tarafından anlaşılabilmesi için belirli embedding modeller kullanılmalı . Ayrıca diğer bir öngörüm türkçe sondan eklemeli bir dil olduğu için bize sorun çıkaracaktır. 
# Faz 3 : Vektör veri tabanı ve geri getirme 
Daha önce bir projede de kullandığım gibi kosinüs benzerliği ile vektör arama yapmalıyız. bir vektör veritabanı seçip bizim vektörümüze en benzer yanı sorulan soruya en benzer vektörü bulmalıyız.  hazır bir veri tabanı da olabilir , belgeleri parçalayıp biz de bir şeyler yapabiliriz. 

Burda büyük ihtimalle direkt başarı elde edemiyeceğiz.  bir sıralama vb yapmalıyız ayrıca belki de blue score gibi şeyler de kullanabiliriz bilmiyorum.
# Faz 4: Yerel Büyük Dil Modeli (LLM) Entegrasyonu ve Ollama
yerel bir llm ile çalışmalıyız. LLM çalıştırmak için qwen3-coder için daha önce ollama kullanmıştım onla çalıştırırım diye düşündüm. Modelfile vb oluşturulabilir belki 
# Faz 5: Halüsinasyon Önleme ve Çıktı Doğrulama Stratejileri
yine google gemini deep research gibi kanıt isteme olabilir llm'den., modelin thinking özelliği adım adım çalışabilir , yanıtını modele doğrulattırırız. 
diğer bir çözüm olarak lettucedetect gibi halüsinasyon tespiti frameworkleri kullanabilriiz. 
# Faz 6: Kullanıcı Arayüzü Geliştirme 
Normalde react kullanabiliriz ama direkt bu iş için streamlit diye bir şey varmış önce onu deneyeceğim.
# Faz 7: Test ve Doğrulama Metodolojisi 
Test için birden fazla makale pdf , belge görüntüsü ile denemeliyiz.
context precision, recall vb değerlerini incelemeliyiz edge case ve normal test senaryoları geliştirmeliyiz .
# Faz 8: Dökümantasyon ve Video 
istenen md dosyaları son hale getirilir.

faz 1 e başlandı 

paddleocr problemleri var