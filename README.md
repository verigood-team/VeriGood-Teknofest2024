# Acıkhack2024TDDİ

<!-- <br /> -->
<!-- <div align="center">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
</div> -->

## Hakkında

* Veri seti, çeşitli sektörlerden farklı firmalar veya kurumlar hakkında müşteri yorumlarını içermektedir.
* Proje, verilen yorumları öncelikle entity'e göre sıralar, sonrasında ise entity'lerin sunduğu hizmetler veya ürünlerle ilgili yorumlardaki duyguları (olumlu, olumsuz veya nötr) belirler.


## Gereklilikler
Projenin çalıştırılması için gerekli tüm bağlılıkların listesini [requirements.txt](requirements.txt) sayfasında bulabilirsiniz.  


## Derleme
Projenin çalıştırılması için izlenecek tüm adımlar:
#### Kaynak kodu üzerinden yükleme
```bash
git clone https://github.com/verigood-team/nlp-scenario.git
```
#### Test Kodu
```
triplet_extractor = ASTE.AspectSentimentTripletExtractor(r"checkpoints\dataset_45.31")

# example = "Vodafone hem ucuz hem de her yerde çekiyor herkese tavsiye ederim"
    
for i in range(1000):
    triplet_extractor.predict(input())

```

## Veri Seti Bağlantısı

https://drive.google.com/drive/folders/1YMFxBPbuQjaEGwL-OdXXR-QF7AMJogOY?usp=sharing

## Katkıda Bulunanlar
Bu süreç içerisinde projeye katkıda bulunanlar  
[Esra Ablak](https://github.com/eablak)  
[Fatih Ayaz](https://github.com/fatihayaz78)  
[Semanur Büdün](https://github.com/semanurbudun)  
[İsmail Furkan Atasoy](https://github.com/ifurkanatasoy)  


## Lisans

Bu proje Apache lisansı altında lisanslanmıştır. Daha fazla bilgi için [LİSANS](LICENSE) sayfasına bakın.  


## Referanslar

* [PyABSA](https://github.com/yangheng95/PyABSA)