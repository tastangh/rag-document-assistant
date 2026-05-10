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

# faz 1 e başlandı 
Şimdi yapmamız gereken ilk aşama pdf ve resimleri OCR İle okuyup belirli bir formata çevirmek. Burdaki sıkıntı olabilecek şeyler tablolar ,türkçe karakterler ve pdf içindeki resimler vb olabilir diye öngörüyorum.
 PDF/JPG/PNG dosyalarını alıp RAG için tek bir Markdown çıktısı üretmeyi uygulamayı düşünüyorum.
 
  `requirements.txt` dosyasını oluşturdum ve temel bağımlılıkları ekledim (`paddleocr`, `paddlepaddle`, `pymupdf`, `opencv-python-headless`, `numpy`, `beautifulsoup4`). paddle ocr kurulumunda sorunlar yaşadım fakat çözdüm.

  document_processor.py diye dosya oluşturdum faz 1 de planlanan döküman resim okuma ve formatlama işlemini burada yapacağım.

py'a girdi kontrolü ekledimn . validate_input func ile artık tanımladığım girdiler haricinde format kabul edilmiyor. 

import fitz  # PyMuPDF diye bişi buldum llm ile konuşarak bunun pdf i sayfa sayfa görsele çevirdiğini öğrendim.    _pdf_to_images i fitz kullanarak yapmaya çalışıldı.

layout + OCR + table parsing + reading order'I PP-structure v3 ile yapmaya çalıştım.

abloları korumak için HTML tablo çıktısını Markdown grid tabloya (`| ... |`) dönüştüren fonksiyon ekledim.

Sayfa içeriklerini sıralı şekilde birleştirip tek Markdown metni döndüren akışı tamamladım.

 Script sonuna `if __name__ == "__main__"` bloğu ekleyip `input_file` ve `--output` ile test edilebilir hale getirdim.

  NumPy 2.x uyumsuzluğu yaşadım; `numpy==1.26.4` ile sabitleyerek çözdüm.
  
  PaddleOCR 3.5 API farkları nedeniyle fallback OCR parametrelerini sürüme uyumlu hale getirdim (`use_textline_orientation`, `device` vb.).

  CPU’da görülen PP-StructureV3 `onednn/pir` hatasını `enable_mkldnn=False` ile giderdim.

  Son durumda sistem PDF’den `.md` üretir hale geldi fakat birebir düzgün ocr etmediğini gözle doğruladım 

# 07-05-2026 
faz'1 de yaptıklarımı devlog a aktardım. 
Nasıl bir çözüm izleyeceğimi araştıramya başlayacağım llm ve internette bununla ilgili şeylere bakacağım.

kullandığım pdfde yazım hatalarının olduğunu da gördüm özellikle bu iş için bir pdf bulmata çalışacağım 

merkez bankası enflasyon raporu türkçe pdf olarak projeye indirdim aynı şekilde belli bir kısmının ingilizce halini buldum onu da koydum. taranmış görseller de eklemek istedim cord datasetinden 2 tane ingilizce taranmış belge görseli indirdim. fiş dataseti de buldum ama fiş belge görsel kapsamında mı emin olamadım şimdilik zamanı efektif kullanmak için atladım . türkçe için de meb sitesinden eski arşiv taramalarından 2 tanesini indirdim. Case study pdf inin de bir pdf olduğu için kullanmaya karar verdim.

hata düzeltme öncesi cpu yerine kendi bilgisayarımda laptop rtx 5070 ti kullanabilmek için gpu kullanmak için gerekli paketleri kurdum .
python document_processor.py "tarama_tr2.png" --gpu --output "tarama_tr2.md" gibi çalışmalarla eklediğim case pdf , merkez bankası pdf , merkezbankası eng pdf , taramalar eng /tr ocr ile md olarak oluşturuldu. 

pdfleri ve çıktıları karşılaştırdım gözle aynı zamanda llm'e yükleyerek bu yorumu aldım . " merkezbankası.md:1 ve merkezbankası_eng.md:1 en güçlü çıktılar. Sayfa yapısı, başlıklar ve içerik akışı korunmuş. RAG ve demo için en güvenilir örnekler bunlar. Buna rağmen Türkçe karakterlerde, boşluklarda ve bazı kelimelerde hatalar var. tarama_eng2.md:1 taranmış belgeler içinde en kullanılabilir olanı. Form yapısı büyük ölçüde korunmuş, okunabilirlik kabul edilebilir seviyede. Yine de bazı satırlarda karakter bozulmaları ve küçük tablo/hizalama sorunları var. tarama_eng.md:1 orta seviyede. İçerik genel olarak yakalanmış, ama tablo yoğun form nedeniyle alan ilişkileri tam temiz değil. Yine de anlamı büyük ölçüde taşıyor. tarama_tr2.md:1 zayıf. Türkçe karakterler, satır sırası ve tablo yapısı ciddi biçimde bozulmuş. Sadece kaba içerik fikri veriyor. tarama_tr.md:1 en zayıf çıktı. Karakter hataları ve kırık kelimeler çok fazla. Bilgi çıkarımı ve alıntı için güvenilir değil."

eski türkçe belgelerinde sayfa bölme  ,gürültü düzeltme ekledim ama iyileşme olsa da görüntü zor olduğu için çok iyi değiliz hala ama faz 2 ye geçeceğim artık baya uğraştırdı. 


# faz 2

chunk embedding pipeline olusturacagim.
requirements tarafinda faz 2 icin su kutuphaneleri ekledim:
sentence-transformers
langchain-text-splitters
torch

torch tarafini da paddle gibi duzenledim.
cpu satirini yoruma aldim gpu satirini aktif yaptim.
cu128 icin extra index ekledim.

faz 2 yi ayri bir pipeline olarak kurguladim ki faz 1 e dokunmadan md lerden devam edelim.
src/chunk_embedding_pipeline.py dosyasini olusturdum.

girdi klasoru src/results/ocrMdResults
cikti klasoru src/results/chunkEmbeddings

pipeline akisinda once md yi ## Sayfa N formatina gore sayfalara ayiriyorum.
sonra block parse var heading text table ayiriyorum.
tabloyu metinden ayri tutuyorum paragrafla birlestirmiyorum.

text blocklarda recursive splitter kullandim.
ayiraclari paragraf -> satir -> cumle -> bosluk seklinde kurdum.
defaultler:
chunk_size=1200
chunk_overlap=200
min_chunk_size=120

cok kisa ve gurultulu chunklari elemek icin alnum orani + min size kontrolu koydum.
table chunklari recursive split e sokmadim.

chunk id formatini sabitledim:
{doc_stem}::p{page}::c{index}

cikti artefaktlar:
chunks.jsonl
embeddings.npy
manifest.json

chunks.jsonl alanlari:
chunk_id, doc_id, page, section, chunk_type, text, char_len

embeddings float32 olarak npy ye yaziliyor.
manifest icine model adi embedding dim chunk count source docs device yazdim.

burda sorun yasadim:
ilk kurulumda torch cpu geldi (2.11.0+cpu) o yuzden pipeline cpu calisti.
sonra torch u kaldirip cu128 ile tekrar kurdum.
bge-m3 modelinde model.safetensors cok buyuk oldugu icin ilk indirme uzun surdu.

en sonda tekrar kostum ve cuda ile bitti.
sonuc:
doc_count: 7
chunk_count: 124
embedding_dim: 1024
model_name: BAAI/bge-m3
device: cuda

belge bazli chunk sayilari:
Case_Study_TUSAS_LLM.md -> 10
merkezbankasi.md -> 36
merkezbankasi_eng.md -> 34
tarama_eng.md -> 1
tarama_eng2.md -> 3
tarama_tr.md -> 21
tarama_tr2.md -> 19

manifest ile chunks.jsonl ve embeddings.npy tutarliligini da kontrol ettim.
124 chunk satiri var ve embedding shape (124, 1024) geldi.

faz 2 bu haliyle tamam gibi .


