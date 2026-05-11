# AGENTS.md

Bu dosya, bu repo icindeki Codex oturumlarinda iletisim ve token verimliligi kurallarini tanimlar.

## Varsayilan Calisma Modu
- Varsayilan cevap modu: `KISA`
- Hedef: en dusuk token ile en yuksek ilerleme

## Cevap Modlari
- `KISA`: 3-5 satir, direkt sonuc + sonraki adim
- `ORTA`: kisa sonuc + gerekli teknik detay
- `DETAY`: tam teknik aciklama
- `DEVLOG`: kullanicinin diline yakin, yapistirilabilir gunluk ozeti

Kullanici mesajinda bu etiketlerden biri gecerse o mod uygulanir.
Etiket yoksa `KISA` kullanilir.

## Token Verimliligi Kurallari
- Gereksiz giris cumleleri, tekrarlar ve uzun arka plan anlatimlari yapma.
- Uzun dosya icerigini oldugu gibi kopyalama; sadece ilgili kisimlari ozetle.
- Tek mesajda tek hedef: "simdi ne yaptik / siradaki en kucuk adim ne".
- Alternatifleri sadece karar gerekiyorsa ver; en fazla 2 secenek sun.
- Kullanici istemedikce uzun teori veya genis kapsamli listeleme yapma.

## Kod Degisikligi Rapor Formati
Kod degisikligi yapildiysa cikti su sirada verilir:
1. Ne degisti
2. Hangi dosyada degisti
3. Nasil dogrulanir

## Faz Bazli Calisma
- Kullanici faz faz ilerlemek istiyorsa yalniz aktif faza odaklan.
- Once aktif fazin mini ozeti, sonra tek sonraki adim.
- Faz disi onerileri "opsiyonel" olarak tek satirda belirt.

## Donanim Kurali
- Varsayilan: GPU-first calisma.
- OCR, embedding, query ve evaluate adimlarinda mumkun olan her durumda `cuda`/GPU kullan.
- GPU secenegi olan komutlarda varsayilan parametreyi GPU olacak sekilde ver (or. `--gpu`, `--device cuda`).
- Sadece teknik olarak zorunluysa CPU fallback oner.

## Dil ve Uslup
- Ana dil Turkce.
- Kisa, net, destekleyici.
- Teknik terimler gerekli oldugunda kullanilir, uzatilmaz.
