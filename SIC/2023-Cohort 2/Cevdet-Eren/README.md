# Deprem sonrası, deprem arazisinin gerçek zamanlı analizi.

Bu proje, deprem afetinden sonra ortaya çıkan enkazları ayırt edebilen ve kullanılabilir yolları tanıyan bir model oluşturmayı amaçlamaktadır. Bu model, hem insan hayatını kurtarmak hem de hasar görmüş bölgelere yardım ulaştırmak için çok faydalı olabilir. Bu model, Türkiye'de 6 Şubat 2023 tarihinde yaşanan deprem sonrası haber kanallarının İHA görüntülerini kullanarak oluşturduğum bir veri seti üzerinde eğitilmiştir. Bu veri setinde, enkazlar için debris, kullanılabilir yollar için road adında iki sınıf tanımlanmıştır. Model, YOLOV8-L Instance Segmentation algoritmasını kullanarak görüntülerdeki enkazları ve yolları segmente etmektedir. Modelin maksimum mAP50 doğruluk değeri %59.2 olarak elde edilmiştir. Bu proje, dünya literatüründe daha önce çalışma yapılmamış bir konuyu ele almaktadır ve elde ettiği sonuçlar bu proje bazında en iyi sonuçlardır.

## Özellikler

Bu projenin özellikleri şunlardır:

- Deprem afetinden sonra ortaya çıkan enkazları ve kullanılabilir yolları segmente edebilen bir model sunar.
- Model, haber kanallarının İHA görüntülerini kullanarak oluşturulan bir veri seti üzerinde eğitilmiştir.
- Model, YOLOV8-L Instance Segmentation algoritmasını kullanmaktadır.
- Model, maksimum mAP50 doğruluk değeri %59.2 olarak elde etmiştir.
- Model, test için oluşturulan bir video üzerinde denenmiştir.
- Model, bazı durumlarda kullanılabilir yolları bulamama veya binaları enkaz sanma gibi hatalar yapabilmektedir.

Modelin test videosu üzerindeki performansı

[![youtube_logo](logo/icons8-youtube-logo.svg)](https://www.youtube.com/watch?v=E61xVKoWljs)

Modelin performansı, veri seti üzerinde maksimum mAP50 doğruluk değeri %59.2 olarak elde edilmiştir. Model, bazı durumlarda kullanılabilir yolları bulamama veya binaları enkaz sanma gibi hatalar yapabilmektedir. Bu hataların nedenleri, veri setinin küçük olması, görüntülerin kalitesiz olması, görüntülerdeki ışık ve renk değişimleri, görüntülerdeki perspektif ve açılardır.
