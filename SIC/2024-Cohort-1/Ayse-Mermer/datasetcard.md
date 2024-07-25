---
# Dataset Card
---

# Dataset Card for Garbage Classification

Bu veri seti, altı farklı atık türünü (karton, cam, metal, kağıt, plastik, çöp) içeren 2527 adet görüntüden oluşmaktadır. Atık sınıflandırma modelleri geliştirmek ve eğitmek için kullanılabilir.

## Dataset Details

### Dataset Description

Atık türlerinin (karton, cam, metal, kağıt, plastik, çöp) görüntülerini içeren bir sınıflandırma veri setidir.
- **Curated by:** asdasdasasdas adlı kullanıcı tarafından Kaggle'a yüklenmiştir.
- **License:** Data files © Original Authors

## Uses

<!-- Address questions around how the dataset is intended to be used. -->

### Direct Use
Atık sınıflandırma modellerini eğitmek ve doğrulamak için kullanılabilir.

## Dataset Structure

Veri seti, her biri bir atık türünü temsil eden altı farklı klasöre ayrılmıştır. Her klasör, o atık türüne ait görüntüleri içerir (JPEG formatında). Sınıfların dağılımı aşağıdaki gibidir:
| Sınıf | Görüntü Sayısı |
|---|---|
| cardboard (karton) | 393 |
| glass (cam) | 491 |
| metal (metal) | 400 |
| paper (kağıt) | 584 |
| plastic (plastik) | 482 |
| trash (çöp) | 187 |

## Dataset Creation

### Source Data

Veri seti, çeşitli kaynaklardan toplanmıştır. [Kaggle](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)

#### Data Collection and Processing

- Görüntüler, farklı ışıklandırma koşullarında ve açılardan çekilmiştir.
- Veri setindeki görüntülerin boyutu ve çözünürlüğü değişkenlik gösterebilir.
- Görüntüler, herhangi bir ön işleme tabi tutulmamıştır.
  
#### Features and the target

- Features (Özellikler): Görüntülerin piksel değerleri
- Target (Hedef): Görüntünün ait olduğu atık türü (karton, cam, metal, kağıt, plastik, çöp)


## Bias, Risks, and Limitations

- Bias: Veri setindeki bazı sınıflar (özellikle "trash" sınıfı) diğerlerinden daha az görüntü içermektedir. Bu durum, modelin bu sınıflara karşı önyargılı olmasına neden olabilir.
- Risks: Modelin yanlış sınıflandırmaları, geri dönüşüm süreçlerinin verimliliğini düşürebilir veya atıkların yanlış şekilde bertaraf edilmesine yol açabilir.
- Limitations: Veri seti sınırlı sayıda atık türünü içermektedir ve farklı ışıklandırma koşullarında veya açılardan çekilmiş görüntüleri içermemektedir. Bu durum, modelin genelleme yeteneğini sınırlayabilir.


