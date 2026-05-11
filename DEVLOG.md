06-05-2026
Case Study'deki sistem iÃ§in neleri yapmama gerek olduÄŸunu bularak baÅŸladÄ±m.
LLM yardÄ±mÄ± ile bu proje iÃ§in gerekli ihtiyaÃ§larÄ± Ã§Ä±karttÄ±m.
Gerekli ihtiyaÃ§larÄ± fazlara bÃ¶ldÃ¼m.

Ã‡Ä±kardÄ±ÄŸÄ±m fazlar;
# Faz 1: GÃ¶rÃ¼ntÃ¼ Ä°ÅŸleme ve Optik Karakter TanÄ±ma (OCR) Mimarisi Kurma
Bir kere her ÅŸeyden Ã¶nce belge ve gÃ¶rÃ¼ntÃ¼leden metin Ã§Ä±karÄ±mÄ± yapacaÄŸÄ±z. gÃ¶rÃ¼ntÃ¼lerden metin Ã§Ä±karmayÄ± koÅŸaylaÅŸtÄ±rma iÃ§in OCR optik karakter tanÄ±ma mimarisini kullanmamÄ±z gerekiyor.
# Faz 2: Metin BÃ¶lÃ¼tleme (Chunking) ve VektÃ¶r GÃ¶mme (Embedding)
YTÃœ YL de edindiÄŸim NLP dersi bilgilerim ve projelerime gÃ¶re metinlerin ve kelimelerin ngram ve chunklarÄ± Ã§ok Ã¶nemli. AyrÄ±ca vektÃ¶r uzaylarÄ± da bizim iÃ§in Ã§ok Ã¶nemli. BÃ–lÃ¼nen metinler LLM tarafÄ±ndan anlaÅŸÄ±labilmesi iÃ§in belirli embedding modeller kullanÄ±lmalÄ± . AyrÄ±ca diÄŸer bir Ã¶ngÃ¶rÃ¼m tÃ¼rkÃ§e sondan eklemeli bir dil olduÄŸu iÃ§in bize sorun Ã§Ä±karacaktÄ±r. 
# Faz 3 : VektÃ¶r veri tabanÄ± ve geri getirme 
Daha Ã¶nce bir projede de kullandÄ±ÄŸÄ±m gibi kosinÃ¼s benzerliÄŸi ile vektÃ¶r arama yapmalÄ±yÄ±z. bir vektÃ¶r veritabanÄ± seÃ§ip bizim vektÃ¶rÃ¼mÃ¼ze en benzer yanÄ± sorulan soruya en benzer vektÃ¶rÃ¼ bulmalÄ±yÄ±z.  hazÄ±r bir veri tabanÄ± da olabilir , belgeleri parÃ§alayÄ±p biz de bir ÅŸeyler yapabiliriz. 

Burda bÃ¼yÃ¼k ihtimalle direkt baÅŸarÄ± elde edemiyeceÄŸiz.  bir sÄ±ralama vb yapmalÄ±yÄ±z ayrÄ±ca belki de blue score gibi ÅŸeyler de kullanabiliriz bilmiyorum.
# Faz 4: Yerel BÃ¼yÃ¼k Dil Modeli (LLM) Entegrasyonu ve Ollama
yerel bir llm ile Ã§alÄ±ÅŸmalÄ±yÄ±z. LLM Ã§alÄ±ÅŸtÄ±rmak iÃ§in qwen3-coder iÃ§in daha Ã¶nce ollama kullanmÄ±ÅŸtÄ±m onla Ã§alÄ±ÅŸtÄ±rÄ±rÄ±m diye dÃ¼ÅŸÃ¼ndÃ¼m. Modelfile vb oluÅŸturulabilir belki 
# Faz 5: HalÃ¼sinasyon Ã–nleme ve Ã‡Ä±ktÄ± DoÄŸrulama Stratejileri
yine google gemini deep research gibi kanÄ±t isteme olabilir llm'den., modelin thinking Ã¶zelliÄŸi adÄ±m adÄ±m Ã§alÄ±ÅŸabilir , yanÄ±tÄ±nÄ± modele doÄŸrulattÄ±rÄ±rÄ±z. 
diÄŸer bir Ã§Ã¶zÃ¼m olarak lettucedetect gibi halÃ¼sinasyon tespiti frameworkleri kullanabilriiz. 
# Faz 6: KullanÄ±cÄ± ArayÃ¼zÃ¼ GeliÅŸtirme 
Normalde react kullanabiliriz ama direkt bu iÅŸ iÃ§in streamlit diye bir ÅŸey varmÄ±ÅŸ Ã¶nce onu deneyeceÄŸim.
# Faz 7: Test ve DoÄŸrulama Metodolojisi 
Test iÃ§in birden fazla makale pdf , belge gÃ¶rÃ¼ntÃ¼sÃ¼ ile denemeliyiz.
context precision, recall vb deÄŸerlerini incelemeliyiz edge case ve normal test senaryolarÄ± geliÅŸtirmeliyiz .
# Faz 8: DÃ¶kÃ¼mantasyon ve Video 
istenen md dosyalarÄ± son hale getirilir.

# faz 1 e baÅŸlandÄ± 
Åimdi yapmamÄ±z gereken ilk aÅŸama pdf ve resimleri OCR Ä°le okuyup belirli bir formata Ã§evirmek. Burdaki sÄ±kÄ±ntÄ± olabilecek ÅŸeyler tablolar ,tÃ¼rkÃ§e karakterler ve pdf iÃ§indeki resimler vb olabilir diye Ã¶ngÃ¶rÃ¼yorum.
 PDF/JPG/PNG dosyalarÄ±nÄ± alÄ±p RAG iÃ§in tek bir Markdown Ã§Ä±ktÄ±sÄ± Ã¼retmeyi uygulamayÄ± dÃ¼ÅŸÃ¼nÃ¼yorum.
 
  `requirements.txt` dosyasÄ±nÄ± oluÅŸturdum ve temel baÄŸÄ±mlÄ±lÄ±klarÄ± ekledim (`paddleocr`, `paddlepaddle`, `pymupdf`, `opencv-python-headless`, `numpy`, `beautifulsoup4`). paddle ocr kurulumunda sorunlar yaÅŸadÄ±m fakat Ã§Ã¶zdÃ¼m.

  document_processor.py diye dosya oluÅŸturdum faz 1 de planlanan dÃ¶kÃ¼man resim okuma ve formatlama iÅŸlemini burada yapacaÄŸÄ±m.

py'a girdi kontrolÃ¼ ekledimn . validate_input func ile artÄ±k tanÄ±mladÄ±ÄŸÄ±m girdiler haricinde format kabul edilmiyor. 

import fitz  # PyMuPDF diye biÅŸi buldum llm ile konuÅŸarak bunun pdf i sayfa sayfa gÃ¶rsele Ã§evirdiÄŸini Ã¶ÄŸrendim.    _pdf_to_images i fitz kullanarak yapmaya Ã§alÄ±ÅŸÄ±ldÄ±.

layout + OCR + table parsing + reading order'I PP-structure v3 ile yapmaya Ã§alÄ±ÅŸtÄ±m.

ablolarÄ± korumak iÃ§in HTML tablo Ã§Ä±ktÄ±sÄ±nÄ± Markdown grid tabloya (`| ... |`) dÃ¶nÃ¼ÅŸtÃ¼ren fonksiyon ekledim.

Sayfa iÃ§eriklerini sÄ±ralÄ± ÅŸekilde birleÅŸtirip tek Markdown metni dÃ¶ndÃ¼ren akÄ±ÅŸÄ± tamamladÄ±m.

 Script sonuna `if __name__ == "__main__"` bloÄŸu ekleyip `input_file` ve `--output` ile test edilebilir hale getirdim.

  NumPy 2.x uyumsuzluÄŸu yaÅŸadÄ±m; `numpy==1.26.4` ile sabitleyerek Ã§Ã¶zdÃ¼m.
  
  PaddleOCR 3.5 API farklarÄ± nedeniyle fallback OCR parametrelerini sÃ¼rÃ¼me uyumlu hale getirdim (`use_textline_orientation`, `device` vb.).

  CPUâ€™da gÃ¶rÃ¼len PP-StructureV3 `onednn/pir` hatasÄ±nÄ± `enable_mkldnn=False` ile giderdim.

  Son durumda sistem PDFâ€™den `.md` Ã¼retir hale geldi fakat birebir dÃ¼zgÃ¼n ocr etmediÄŸini gÃ¶zle doÄŸruladÄ±m 

# 07-05-2026 
faz'1 de yaptÄ±klarÄ±mÄ± devlog a aktardÄ±m. 
NasÄ±l bir Ã§Ã¶zÃ¼m izleyeceÄŸimi araÅŸtÄ±ramya baÅŸlayacaÄŸÄ±m llm ve internette bununla ilgili ÅŸeylere bakacaÄŸÄ±m.

kullandÄ±ÄŸÄ±m pdfde yazÄ±m hatalarÄ±nÄ±n olduÄŸunu da gÃ¶rdÃ¼m Ã¶zellikle bu iÅŸ iÃ§in bir pdf bulmata Ã§alÄ±ÅŸacaÄŸÄ±m 

merkez bankasÄ± enflasyon raporu tÃ¼rkÃ§e pdf olarak projeye indirdim aynÄ± ÅŸekilde belli bir kÄ±smÄ±nÄ±n ingilizce halini buldum onu da koydum. taranmÄ±ÅŸ gÃ¶rseller de eklemek istedim cord datasetinden 2 tane ingilizce taranmÄ±ÅŸ belge gÃ¶rseli indirdim. fiÅŸ dataseti de buldum ama fiÅŸ belge gÃ¶rsel kapsamÄ±nda mÄ± emin olamadÄ±m ÅŸimdilik zamanÄ± efektif kullanmak iÃ§in atladÄ±m . tÃ¼rkÃ§e iÃ§in de meb sitesinden eski arÅŸiv taramalarÄ±ndan 2 tanesini indirdim. Case study pdf inin de bir pdf olduÄŸu iÃ§in kullanmaya karar verdim.

hata dÃ¼zeltme Ã¶ncesi cpu yerine kendi bilgisayarÄ±mda laptop rtx 5070 ti kullanabilmek iÃ§in gpu kullanmak iÃ§in gerekli paketleri kurdum .
python document_processor.py "tarama_tr2.png" --gpu --output "tarama_tr2.md" gibi Ã§alÄ±ÅŸmalarla eklediÄŸim case pdf , merkez bankasÄ± pdf , merkezbankasÄ± eng pdf , taramalar eng /tr ocr ile md olarak oluÅŸturuldu. 

pdfleri ve Ã§Ä±ktÄ±larÄ± karÅŸÄ±laÅŸtÄ±rdÄ±m gÃ¶zle aynÄ± zamanda llm'e yÃ¼kleyerek bu yorumu aldÄ±m . " merkezbankasÄ±.md:1 ve merkezbankasÄ±_eng.md:1 en gÃ¼Ã§lÃ¼ Ã§Ä±ktÄ±lar. Sayfa yapÄ±sÄ±, baÅŸlÄ±klar ve iÃ§erik akÄ±ÅŸÄ± korunmuÅŸ. RAG ve demo iÃ§in en gÃ¼venilir Ã¶rnekler bunlar. Buna raÄŸmen TÃ¼rkÃ§e karakterlerde, boÅŸluklarda ve bazÄ± kelimelerde hatalar var. tarama_eng2.md:1 taranmÄ±ÅŸ belgeler iÃ§inde en kullanÄ±labilir olanÄ±. Form yapÄ±sÄ± bÃ¼yÃ¼k Ã¶lÃ§Ã¼de korunmuÅŸ, okunabilirlik kabul edilebilir seviyede. Yine de bazÄ± satÄ±rlarda karakter bozulmalarÄ± ve kÃ¼Ã§Ã¼k tablo/hizalama sorunlarÄ± var. tarama_eng.md:1 orta seviyede. Ä°Ã§erik genel olarak yakalanmÄ±ÅŸ, ama tablo yoÄŸun form nedeniyle alan iliÅŸkileri tam temiz deÄŸil. Yine de anlamÄ± bÃ¼yÃ¼k Ã¶lÃ§Ã¼de taÅŸÄ±yor. tarama_tr2.md:1 zayÄ±f. TÃ¼rkÃ§e karakterler, satÄ±r sÄ±rasÄ± ve tablo yapÄ±sÄ± ciddi biÃ§imde bozulmuÅŸ. Sadece kaba iÃ§erik fikri veriyor. tarama_tr.md:1 en zayÄ±f Ã§Ä±ktÄ±. Karakter hatalarÄ± ve kÄ±rÄ±k kelimeler Ã§ok fazla. Bilgi Ã§Ä±karÄ±mÄ± ve alÄ±ntÄ± iÃ§in gÃ¼venilir deÄŸil."

eski tÃ¼rkÃ§e belgelerinde sayfa bÃ¶lme  ,gÃ¼rÃ¼ltÃ¼ dÃ¼zeltme ekledim ama iyileÅŸme olsa da gÃ¶rÃ¼ntÃ¼ zor olduÄŸu iÃ§in Ã§ok iyi deÄŸiliz hala ama faz 2 ye geÃ§eceÄŸim artÄ±k baya uÄŸraÅŸtÄ±rdÄ±. 


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

# 10-05-2026

faz 3 e basladim.
faz 2 de urettigim chunk + embedding kullanip chroma tarafinda index kurdum (build-index). bu adimla birlikte retrieval icin veri tabani hazir hale geldi.

ardindan query testleri yaptim.
query komutlariyla case study ve turkce ocr/chunking ile ilgili sorular sordum, sistemin en ilgili chunklari getirip getirmedigini kontrol ettim. genel olarak akisin calistigini gordum.

rerank tarafini da karsilastirdim.
ayni soruyu bir de --disable-rerank ile kostum, yani reranker acik/kapali sekilde sonuclarin siralamasina etkisini gozlemledim.

ozetle bugun faz 3 te:

index kuruldu
retrieval query testleri yapildi
rerank karsilastirmasi yapildi
sonraki adim olarak kucuk bir eval seti (jsonl) hazirlayip evaluate ile recall@k, mrr, ndcg@k metriklerine bakacagim.

eval.json dosyasÄ± oluÅŸturuldu


# 10-05-2026 (devam)

eval tarafini tum ocr md'leri kapsayacak sekilde genislettim.
src/results/eval/eval_docs.jsonl dosyasini olusturup her dokuman icin soru seti hazirladim (toplam 35 soru).

sonra evaluate komutlarini karsilastirmali kostum (gpu/cuda):

1) initial_k=12 final_k=4 rerank acik
recall@k: 0.1714
mrr: 0.0643
ndcg@k: 0.0912

2) initial_k=16 final_k=5 rerank acik
recall@k: 0.2571
mrr: 0.1000
ndcg@k: 0.1389

3) initial_k=12 final_k=4 rerank kapali
recall@k: 0.0571
mrr: 0.0286
ndcg@k: 0.0361

yorumlarsim:
- en iyi sonuc 16/5 + rerank acik konfigde geldi.
- reranker kapaninca metrikler ciddi dustu, yani reranker faydali.
- su an icin varsayilan retrieval ayarini 16/5 + rerank acik kullanmak daha mantikli.

ayrica bu sureci ogrenirken temel aciklamalari toplayalim diye repo kokune wiki.MD ekledim.
wiki icine rag temel kavramlari, faz ozeti, metrikler ve gun sonu not formati yazildi.

sonraki adim:
- eval setini daha da netlestirip (mumkunse chunk bazli etiketleyip) metrikleri tekrar test etmek.

faz 3 kapanis notu:
- faz 3 tamamlandi.
- retrieval default ayarini proje icinde kaliciya aldim:
  - initial_k=16
  - final_k=5
  - rerank acik
  - device=cuda
- boylece bundan sonra komutlarda elle 16/5 yazmadan da ayni varsayilanlarla devam edebilecegim.

# 10-05-2026 (faz 4)

faz 4'e gectim. hedefim faz 3 retrieval sonucunu yerel llm ile birlestirip ucdan uca soru-cevap almakti.

ilk is olarak src/generation_pipeline.py dosyasini olusturdum.
burda su akis var:
- soru al
- retrieval_pipeline.retrieve_contexts ile baglam cek
- baglami prompta yerlestir
- ollama api ile cevabi uret
- kaynaklari (doc_id, page, chunk_id) ile birlikte json olarak don

ayrica cli tarafina 2 komut ekledim:
- ask (tek soru)
- smoke-test (hazir soru setiyle toplu test)

faz 4 icin hazir soru seti de olusturdum:
src/results/eval/faz4_smoke_questions.jsonl

burda ilk buyuk sorunum ollama modeliydi.
ask komutunu ilk kostugumda su hatayi aldim:
HTTP 404 model 'qwen3:8b' not found

bu durumun sebebi koddan degil modelin localde olmamasiymis.
model adini/kurulumunu duzeltince generation akisina gecebildim.

ikinci sorun retrieval filtre tarafinda cikti.
--doc-id ve --chunk-type birlikte verilince chroma su hatayi verdi:
Expected where to have exactly one operator

sebebi where filtresini duz obje vermemdi.
retrieval_pipeline.py icindeki _build_where_filter fonksiyonunu degistirdim.
artik birden fazla filtre varsa $and ile gonderiyor.

ucuncu sorun prompt cikti formatinda oldu.
model bazen cevabin sonuna su satiri kopyaliyordu:
"3) Baglamda yeterli bilgi yok."

bunu cozmwk icin generation_pipeline.py tarafinda promptu sadeledim.
- cikti formati satirlarini daha net ve kosullu hale getirdim
- clean_answer adinda bir temizlik fonksiyonu ekledim
- sablon sizintisi yapan satirlari post-process ile siliyorum

sonra smoke-test tarafini guclendirdim.
ilk hali sadece soru listesi okuyordu.
ben bunu relevant_doc_ids okuyacak hale getirdim ve her soru icin ilgili doc_id ile filtreli retrieval yaptirdim.
ayrica otomatik kabul kontrolu ekledim:
- has_answer
- has_sources
- source_match
- no_template_leak
- passed

ve ozet metrikler:
- pass_threshold (default 0.70)
- pass_rate
- passed_count
- accepted (true/false)

burda bir kritik hata daha yasadim:
ilk smoke sonucunda case_study + merkezbankasi sorulari fail gorundu ve source bos geldi.
sebep doc_id mismatch idi.
jsonlde ascii doc_id kullaniyordum ama indexte turkce karakterli id vardi.

ornek:
- Case_Study_TUSAS_LLM yerine Case_Study_TUSAŞ_LLM
- merkezbankasi yerine merkezbankası
- merkezbankasi_eng yerine merkezbankası_eng

faz4_smoke_questions.jsonl dosyasinda bu idleri birebir indexteki adlarla duzelttim.

tekrar smoke-test kostum.
son durumda checks tarafinda tum sorular passed:true geldi.
source_match true, has_sources true, no_template_leak true oldu.

kisa ozet:
- faz 4 generation pipeline kuruldu
- ask + smoke-test komutlari calisiyor
- retrieval + llm + kaynakli cikti ucdan uca calisiyor
- prompt sizinti sorunu temizlendi
- filtreleme ve doc_id eslesme sorunlari cozuldu
- kabul kriteri otomatik olarak eklendi ve testten gecti

faz 4 final karari:
- faz 4 tamamlandi.
- varsayilan ayarlarla devam:
  - initial_k=16
  - final_k=5
  - rerank acik
  - device=cuda
  - model_name=qwen3:8b

# 11-05-2026 (faz 5)

faz 5'e gectim. hedefim halusinasyon onleme + cikti dogrulama katmanini eklemekti.

faz 4 sonunda cevap alabiliyordum ama cevaplarin kaynakla ne kadar uyumlu oldugunu otomatik olcecek bir katman yoktu.
o yuzden generation pipeline uzerine verification katmani ekledim.

src/generation_pipeline.py tarafinda yaptiklarim:
- promptu kaynak zorunlu hale getirdim.
- modelden iddialari kaynak etiketiyle yazmasini istedim.
- citation parse + claim parse + source eslestirme mantigi ekledim.
- verification cikti alanlari eklendi:
  - claim_count
  - supported_count
  - supported_ratio
  - citation_coverage
  - confidence (high/medium/low)
  - hallucination_risk
  - fallback_used

strict guardrail ekledim:
- strict_guardrail=true oldugunda dusuk guvenli cevaplari otomatik olarak
  "Baglamda yeterli bilgi yok." fallbackine cekiyor.

faz 5 icin test mimarisi kurdum:
- normal soru seti: src/results/eval/faz4_smoke_questions.jsonl
- adversarial soru seti: src/results/eval/faz5_adversarial_questions.jsonl

ayrica yeni komut eklendi:
- safety-eval

safety-eval iki ayri orani olcuyor:
- normal soru pass rate (beklenen: >= 0.80)
- adversarial soru pass rate (beklenen: >= 0.90)

burda onemli hatalarla karsilastim:

1) citation parse bugi:
modelden gelen format [doc_id:...:p1::c1] oldugu halde parser bazen yanlis parse edip
chunk_id'yi c:c1 seklinde uretiyordu.

cozum:
- regex ve normalize fonksiyonunu guncelledim.
- p1::c1 / p1:c1 / doc_id prefiksli formatlari destekledim.
- normalize tarafinda bastaki ':' karakterlerini temizleyip canonical chunk_id olusturdum.

2) confidence false-negative:
parse bugi varken citation_exists false oldugu icin supported_ratio 0 cikiyordu.
bugi duzelttikten sonra citation_exists true gelmeye basladi ve gercekci skorlar goruldu.

3) performans:
safety-eval cok uzun surdu cunku her soru icin retrieval + rerank + llm + verify calisiyor.
bu beklenen bir durum.
hizli tur icin limit dusurme / rerank kapali kosma notunu aldim.

test gozlemi:
- adversarial sorularda sistem fallback veriyor, guardrail calisiyor.
- normal sorularda clean dokumanlarda daha iyi, noisy ocr dokumanlarda supported_ratio daha dusuk.
- dusuk supported_ratio kalan kisimlarin buyuk bolumu pipeline bugi degil, ocr veri kalitesi ile ilgili.

faz 5 final karari:
- faz 5 tamamlandi.
- halusinasyon onleme ve cikti dogrulama katmani aktif.
- strict guardrail ile guvenlikli cikis aliniyor.

not:
- bir sonraki teknik iyilestirme alani faz 5 degil, noisy ocr belgelerde kalite artirma (preprocess/normalization).

# 11-05-2026 (faz 6 - ui stabilizasyon ve gercek kullanici akisi)

faz 6 tarafinda streamlit ui yi baya degistirdim cunku mevcut akista son kullanici index/chunk gibi teknik seyleri goruyordu ve deneyim iyi degildi.
hedefi netlestirdim:
- kullanici sadece belge yuklesin
- sistem arkada analiz etsin
- kullanici direkt soru sorsun

ilk asamada ui yi chatgpt benzeri sade akis yaptim:
- belge yukle
- sohbetten soru sor
- kaynaklari expandable alanda goster

sonra buyuk bir sorun cikti:
- yeni yuklenen belge baglama girmiyor gibi davrandi
- retrieval eski dokumanlardan sonuc getiriyordu

sebep:
- tum ocr md havuzunu indexliyordu, oturum izole degildi

cozum:
- ui runtime altinda session bazli klasor yapisi kurdum
- uploads / ocr_md / chunks / vector dizinleri session altina alindi
- ask tarafi sadece bu session vector store uzerinden calisiyor

ikinci buyuk sorun:
- paddle tarafinda pp-structure hatasi geldiginde md icine hata metni yaziliyordu
- bu hata metni bazen indexe girip anlamsiz retrieval uretiyordu

cozum:
- pdf icin once direct text-layer extraction (pymupdf) denedim
- text varsa ocr a dusmeden direct metni kullandim
- ocr hata metni patternlerini yakalayip bu dokumani failed sayma mantigi ekledim

ucuncu sorun:
- refresh atinca sohbet + aktif dokumanlar gidiyordu

cozum:
- disk kalici state modeli eklendi
- src/results/ui_runtime/<session_id>/state.json ile
  - docs
  - messages
  - index_signature
  - ready
  - model ayarlari
  - son hata bilgisi
  saklaniyor
- acilista state geri yukleniyor
- silinmis dosyalar state ten temizleniyor
- onemli her olaydan sonra atomik state write yapiliyor (tmp + replace)

dorduncu sorun:
- "islenemedi" gibi hata metni encoding bozulunca kaciyordu (islenemedi / iÅŸlenemedi vb)

cozum:
- hata tespitini unicode normalize + ascii sadelestirme + regex imzalarina cevirdim
- markerlar:
  - hata
  - islenemedi varyantlari
  - unimplemented
  - onednn_instruction
  - convertpirattribute2runtimeattribute

ui/ux iyilestirmeleri:
- belge hazirla butonu kaldirildi (tam otomatik akis)
- uploadta iki faz spinner:
  - yukleniyor...
  - analiz ve index guncelleniyor...
- sol panelde aktif dokumanlar:
  - checkbox ile aktif/pasif
  - X ile kaldirma
  - durum rozeti: analyzing / ready / failed
  - failed ise kisa hata metni

soru-cevap tarafinda:
- ready yoksa artik tek satir generic uyari yerine tanilayici mesaj var:
  - aktif ready dokuman yok
  - failed dokuman sayisi
  - en son hata

bu fazdaki kritik ogrenim:
- ana problem llm veya retrieval degildi, ingestion ve state yonetimiymis
- dokuman kalitesi + kalici session + izole index olmadan ui tarafi guven vermiyor

mevcut durum:
- faz 6 stabilizasyonun ana parcalari tamamlandi
- sistem artik session-izole calisiyor
- refresh sonrasi state geri geliyor
- failed dokumanlar indexe girmiyor

acik not:
- bazi image-only veya cok zor pdflerde paddle hala fail olabilir
- bu durumda dokuman failed gorunmesi beklenen davranis
- sonraki iyilestirme alani dokuman kalite metrikleri ve yeniden dene akisi olabilir

# 11-05-2026 (yeni yol haritasi faz 1 - konteynerizasyon ve kurulum izolasyonu)

bu fazda hedefim hardcoded yol bagimliliklarini azaltip calisma ortamini daha tasinabilir yapmakti.

yaptigim degisiklikler:
- src/config.py eklendi.
  - path ve servis ayarlari env degiskenlerinden okunuyor.
  - varsayilanlar relative path uzerinden src/results/... kullaniyor.
- src/ui_streamlit.py icinde sabit runtime/model degerleri config'e baglandi.
  - runtime root env ile degistirilebilir oldu.
  - embed model ve ollama model ayarlari merkezi hale geldi.
- src/retrieval_pipeline.py default artifacts/persist/collection/model sabitleri config'ten okunacak sekilde guncellendi.
- src/generation_pipeline.py default ollama url ve model config uzerinden okunacak sekilde guncellendi.
- src/chunk_embedding_pipeline.py default input/output dizinleri merkezi config'ten turetilir hale getirildi.
- README.md bastan yazildi.
  - hardcoded /home/... path temizlendi.
- pyproject.toml eklendi (poetry gecis baslangici).

yan etki notlari:
- poetry lock dosyasi bu asamada uretilmedi; ekipte bir kez poetry lock alinmasi gerekiyor.
- mevcut requirements.txt akisi korunuyor; gecis donemi icin iki yontem birlikte mevcut.

# 11-05-2026 (yeni yol haritasi faz 2 - gelismis ocr/chunking)

bu fazda odak metin bolutlemeyi karakter bazli yaklasimdan cikarip daha semantik bir akis kurmakti.

yaptigim degisiklikler:
- chunk_embedding_pipeline dosyasini faz 2 odagina gore sade ve temiz sekilde yeniden yazdim.
- recursivecharactertextsplitter bagimliligini aktif akisdan cikardim.
- yeni semantik split fonksiyonu eklendi:
  - once paragraf/normalizasyon
  - sonra cumle sinirlarina gore birlestirerek chunk olusturma
  - asiri uzun tek cumle varsa kontrollu sert bolme
- tablo bloklari markdown grid olarak tek parca korunuyor (table chunk), text chunk ile karistirilmiyor.
- chunk metadata guclendirildi:
  - doc_id
  - page
  - page_no
  - section
  - chunk_type
- run_pipeline imzasi korunarak ui ve diger modullerle uyumluluk bozulmadi.
- chunk_overlap parametresi geri uyumluluk icin tutuldu; semantik akisda aktif rol oynamadigi kodda notlandi.

beklenen etki:
- ozellikle turkce belgelerde cumle butunlugu daha iyi korunacak.
- gereksiz/parcalanmis OCR kirintilarinin retrievale girmesi azalacak.
- kaynak gosterimi tarafinda page/page_no metadata daha tutarli olacak.

risk/not:
- performans olarak cumle tabanli parcalama daha stabil ama cok uzun teknik cumlelerde chunk boyutu sinirina bagli sert bolme yapabilir.
- sonraki fazda model degisimi (TR odakli embedding) ile birlikte retrieval kalitesi tekrar olculmeli.

# 11-05-2026 (yeni yol haritasi faz 3 - turkce odakli embedding gecisi)

bu fazda amac bge-m3 varsayilanindan cikarak turkce agirlikli retrieval kalitesini artirmakti.

yaptigim degisiklikler:
- merkezi configte varsayilan embedding modeli `newmindai/Mursit-Large-TR-Retrieval` olarak guncellendi.
- chunk pipeline cli default modeli de ayni modele baglandi.
- embedding yukleme tarafina kontrollu fallback zinciri eklendi:
  1) istenen model
  2) newmindai/Mursit-Large-TR-Retrieval
  3) BAAI/bge-m3
  4) intfloat/multilingual-e5-base
- boylece model mirror/indirilebilirlik sorunu olsa bile pipeline tamamen kirilmiyor.
- README env default bilgisi yeni modelle guncellendi.

teknik not:
- retrieval kalitesini net gormek icin ayni eval setiyle faz 2'ye gore karsilastirmali metrik kosulmasi gerekiyor (nDCG/Recall/MRR).

# 11-05-2026 (yeni yol haritasi faz 4 - rrf tabanli hibrit retrieval)

bu fazda retrieval katmanini sadece dense (cosine) aramadan cikarip dense+sparse hibrit yapıya tasidim.

yaptigim degisiklikler:
- retrieval_pipeline icine turkce uyumlu basit tokenizasyon eklendi.
- collection dokumanlari uzerinde bm25-benzeri sparse skorlayan fonksiyon eklendi.
- dense aday listesi ile sparse aday listesi rrf (reciprocal rank fusion) ile birlestirildi.
- rrf ciktilari mevcut cross-encoder rerank asamasina gonderiliyor; final siralama burada netleniyor.
- retrieval cevabina yeni alanlar eklendi:
  - sparse_score
  - rrf_score
  - hybrid_applied
- cli tarafina --disable-hybrid parametresi eklendi (ab test ve geri donus icin).
- evaluate akisi hibrit acik/kapali kosulabilecek hale getirildi.

beklenen etki:
- tam eslesme gerektiren teknik kod/sayi/kisalma sorgularinda daha iyi retrieval precision.
- anlamsal benzerlik gucunu korurken sparse sinyal ile eksik kalan noktalarin tamamlanmasi.

risk/not:
- sparse asama collection.get ile tum filtrelenmis dokumanlari okuyup skorladigi icin buyuk koleksiyonlarda ek maliyet getirebilir.
- olasi performans darbogazinda disable-hybrid ile dense moda aninda donulebilir.

# 11-05-2026 (yeni yol haritasi faz 5 - semantik guardrail)

bu fazda generation tarafindaki lexical token-overlap tabanli dogrulama kaldirildi ve semantik guardrail katmani eklendi.

yaptigim degisiklikler:
- generation pipeline icine TurkLettuceGuardrail adinda bir denetleyici eklendi.
- guardrail, Turk-LettuceDetect uyumlu token-classification modelini kullanacak sekilde tasarlandi.
- varsayilan model env ile tanimlandi:
  - RAG_GUARDRAIL_MODEL=newmindai/ettin-encoder-150M-TR-HD
  - RAG_GUARDRAIL_THRESHOLD=0.55
- claim dogrulama artik citation + semantik destek kriteri ile yapiliyor.
- model yuklenemez veya inference calismazsa sistem guvenli tarafta kaliyor (supported=false).
- strict guardrail modunda dusuk guvenli cevaplar gizlenip "Veri bulunamadı." yaniti donuluyor.
- adversarial degerlendirme guvenli red cevabinda hem klasik fallback hem yeni guardrail fallback metnini kabul edecek sekilde guncellendi.

not:
- air-gapped ortamlarda guardrail modeli local mirror/cache ile onceden hazir olmali; aksi halde sistem fail-open degil fail-safe davranir.

# 11-05-2026 (yeni yol haritasi faz 6 - rag triad otomatik degerlendirme)

bu fazda ci/cd uyumlu otomatik degerlendirme scripti eklendi.

yaptigim degisiklikler:
- src/faz6_eval.py dosyasi eklendi.
- script su triad metriklerini hesapliyor:
  - context_relevance
  - faithfulness
  - answer_relevance
  - triad_score (ortalama)
- soru seti olarak mevcut faz4 smoke jsonl dosyasi ile calisacak sekilde tasarlandi.
- guardrail acik/kapali calisma destekleniyor (strict flag).
- pass_threshold ile pipeline kabul/reddet davranisi eklendi.
- rapor json olarak disari yaziliyor:
  - src/results/eval/faz6_triad_report.json
- ragas bagimliligi projeye eklendi (requirements + pyproject).
- air-gapped ortamlari bozmamak icin ragas runtime'da opsiyonel ele alindi;
  paket yoksa script fallback metriklerle rapor uretebiliyor.

kullanim notu:
- script accepted false donerse exit code 2 veriyor, boylece ci asamasinda fail condition net yakalaniyor.

# 11-05-2026 (yeni yol haritasi faz 7 - session izolasyonu ve state kaliciligi)

bu fazda streamlit ui tarafinda refresh sonrasi kayip ve oturumlarin birbirine karismasi sorununu ele aldim.

yaptigim degisiklikler:
- ui_streamlit.py icine session id mekanizmasi eklendi.
  - url query param `sid` yoksa uuid uretiliyor.
  - runtime path artik `src/results/.../sessions/<sid>/` altinda aciliyor.
- state kaliciligi eklendi:
  - her session icin `state.json` dosyasi kullaniliyor.
  - yuklenen alanlar: messages, ready, docs, last_error, model_name.
  - acilista state otomatik geri yukleniyor.
  - onemli olaylardan sonra atomik yazim yapiliyor (`tmp + replace`).
- windows dosya kilidi icin `_clear_dir` yumusatildi.
  - `PermissionError` durumunda kisa bekleme + tekrar deneme var.
  - kilit devam ederse tum akis kirilmadan sonraki dosyaya geciliyor.
- sohbet ve dokuman listesi state ile senkronlandi.
  - chat mesajlari her user/assistant adimindan sonra diske kaydediliyor.
  - belge hazirlama sonrasi docs/status bilgisi state'e yaziliyor.
- hazir degil durumda hata mesaji tanisal hale getirildi.
  - last_error varsa kullaniciya gosteriliyor.

beklenen etki:
- refresh sonrasinda sohbet ve belge durumu geri gelir.
- farkli session'larin vector/chunk/upload klasorleri fiziksel olarak ayrilir.
- dosya kilidi kaynakli temizleme hatalari tum ui akisinin patlamasini azaltir.

# 11-05-2026 (gpu zorlamasi)

kullanici talebine gore ui akisinda tum agir adimlar gpu zorlamali hale getirildi.
- OCR: process_document_to_markdown(..., use_gpu=OCR_USE_GPU)
- embedding/chunk: run_pipeline device=EMBED_DEVICE (varsayilan cuda)
- retrieval/rerank: ask_question device=RETRIEVAL_DEVICE (varsayilan cuda)
- confige yeni bayraklar eklendi:
  - RAG_OCR_USE_GPU=1
  - RAG_EMBED_DEVICE=cuda
  - RAG_RETRIEVAL_DEVICE=cuda

# 11-05-2026 (windows stabilizasyon - gpu/cpu uyumluluk)

ortam incelemesinde paddle gpu yerine cpu paketinin yuklu oldugu ve torch tarafinda windows dll uyumsuzlugu goruldu.
bu nedenle kurulum profili windows icin stabilize edildi.

yapilanlar:
- requirements guncellendi:
  - paddlepaddle-gpu kaldirildi, yerine paddlepaddle==3.3.0 alindi (windows stabil profil)
  - torchvision pin'i cu128 ile hizalandi (0.26.0+cu128)
  - transformers <5.0 pin'i eklendi
- ui_streamlit tarafina paddle cuda destek kontrolu eklendi.
  - RAG_OCR_USE_GPU=1 olsa bile paddle cuda derlemesi yoksa otomatik cpu kullaniliyor.
  - hazirlama sonrasi OCR cihazi (gpu/cpu) kullaniciya gosteriliyor.
- README'ye windows notu eklendi (tam paddle gpu icin Linux/WSL2 onerisi).
