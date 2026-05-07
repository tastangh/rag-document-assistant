## Sayfa 1

# TEKNİK DEĞERLENDİRME 

## Case Study 

Pozisyon: Yazılım Geliştirici (Al/ML Odaklı)

Teslim Süresi: 7 gün 

Tahmini Çalışma Süresi: 25-35 saat

## Sayfa 2

### 1. Proje Özeti 

Kullanıcıların çeşitli formatlardaki belgeleri yükleyerek bu belgeler üzerinden doğal dil ile soru sorabileceği, yapay zeka destekli bir Belge Analiz ve Soru-Cevap Sistemi geliştirmenizi bekliyoruz.

Sistem, yüklenen belgeleri işleyecek, içeriğini analiz edecek ve kullanıcı sorularına belgelerden elde ettiği bilgiye dayalı yanıtlar verecektir.



### 2. Fonksiyonel Gereksinimler 

<div style="text-align: center;">Sistemin aşağıdaki temel işlevleri yerine getirmesi bekleniyor:</div>



<div style="text-align: center;"><html><body>| İşlev | Beklenen Davranış |
| --- | --- |
| Belge Yükleme | Kullanıcı PDF ve resim formatlarında (JPG, PNG) belge yükleyebilmeli. |
| Metin Çıkarımı | Resim belgelerindeki metinler okunabilmeli. Türkçe ve Ingilizce desteklenmeli. |
| Soru-Cevap | Kullanıcı, yüklenen belgeler hakkında doğal dilde soru sorabilmeli ve tutarlı vanıtlar alabilmeli. |
| Doğruluk | Sistem, belgede olmayan bilgileri üretmemeli (hallucination). |
| Kullanılabilirlik | Kullanıcı sistemi bir arayüz üzerinden kullanabilmeli. |</body></html></div>


### 3. Teknik Yaklaşım 

Bu problemi nasıl çözeceğiniz tamamen size bırakılmıştır.

Hangi teknolojileri, kütüphaneleri, modelleri veya mimari yaklaşımları kullanacağınız sizin kararınızdır. Biz sizin bu kararları nasıl aldığınızı, hangi alternatifleri değerlendirdiğinizi ve neden bu yolu seçtiğinizi görmek istiyoruz.



Beklentimiz: Bu çalışmada işlevsel bir MvP (Minimum Viable Product) bekliyoruz. Ancak teslimatınızın başkaları tarafından kolayca çalıştırılabilir, test edilebilir ve üzerine geliştirilebilir formatta olmasıdeğerlendirmede avantaj sağlayacaktır.



### 4. Teslimatlar 

Aşağıdaki teslimatlar beklenmektedir:

## Sayfa 3

<div style="text-align: center;"><html><body>| # | Teslimat | Açıklama |
| --- | --- | --- |
| 1 | DEVLOG.md | Geliştirme sürecinizin kaydı. Detaylar aşağıda. |
| 2 | TESTING.md | Test senaryolarınız ve sonuçları. Detaylar aşağıda. |
| 3 | Demo Video | Sisteminizi müşteriye tanıttığınız kısa bir video (3-5 dk). Format ve içerik tercihi size aittir. |
| 4 | Kaynak Kod | Çalışan sistemin tüm kaynak kodları. |
| 5 | README.md | Projenin kısa tanıtımı ve sistemi çalıştırmak için gereken adımlar. |</body></html></div>


##### 4.1. DEVLOG.md — Geliştirme Süreci Kaydı

Projeniz boyunca izlediğiniz yolu, aldığınız kararları, deneyip vazgeçtiklerinizi, karşılaştığınız zorlukları ve bunları nasıl aştığınızı kronolojik olarak belgeleyin. Düşünce sürecinizi, problem çözme yaklaşımınızı ve öğrenme şeklinizi görmek istiyoruz.



Bu dosya bir "final rapor" değil, bir "yolculuk günlüğü" olmali, fikir vermesi açısından bu dosya ile aşağıdakilere benzer soruların cevaplanmasını istiyoruz:

•Problemi nasıl parçaladınız?

• Hangi yaklaşımları denediniz? Hangisi işe yaramadı ve neden?

• Kritik karar noktalarında hangi alternatifleri değerlendirdiniz?

• Nerede takıldınız? Nasıl çözdünüz?

• Zamaninizi nasıl harcadıniz?

• Şu an bildiğinizle baştan başlasanız neyi farklı yapardınız?

##### 4.2. TESTING.md — Test ve Doğrulama 

Sisteminizi nasıl test ettiğinizi ve sınırlarını belgeleyin.

Bu dosyada aşağıdakilere benzer sorulara cevap bekliyoruz:

•Sisteminizi hangi senaryolarla, nasıl test ettiniz?

• Farklı belge tiplerinde (Türkçe, İngilizce, taranmış, tablolu) nasıl performans gösterdi?

• Örnek sorular ve aldığınız yanıtlar 

Sistemin başarısız olduğu veya yetersiz kaldığı durumlar 

• Belgede olmayan bir bilgi sorulduğunda sistem nasıl davranıyor?

## Sayfa 4

### 5. Teslimat Bilgileri 

Teslim Yöntemi: GitHub repository linki 

### 6. Sonraki Adım 

Teslimatınız olumlu değerlendirildikten sonra bir teknik mülakat yapılacaktır. Bu mülakattta projenizi sunmanız, sisteminizi demo yapmanız ve teknik kararlarınızı tartışmanız beklenecektir.

## Notlar:

• LLM araçları kullanmak serbesttir. Projenin tamamını anlamanız ve savunabilmeniz beklenmektedir.



• Sistemle ilgili belirtilmeyen detaylar ve teknik kararlar için inisiyatif almanız beklenmektedir. Bu kararlar da değerlendirmenin bir parçasi olacaktir.



• Sorularınız olursa çekinmeden iletişime geçebilirsiniz.

Başarılar dileriz!