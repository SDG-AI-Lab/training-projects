---
# MODEL CARD

# Model Card for Garbage Classification CNN101

Bu model, atık türlerini (karton, cam, metal, kağıt, plastik, çöp) sınıflandırmak için evrişimli sinir ağlarını (CNN) kullanır. Keras ve TensorFlow ile geliştirilmiştir ve Google Colab ortamında (T4 GPU) eğitilmiştir.

## Model Details

### Model Description

Atık sınıflandırması için geliştirilmiş CNN modeli.

- **Developed by:** Ayşe Mermer
- **Model date:** 06/2024
- **Model type:** Evrişimli Sinir Ağı (CNN)
- **Language(s):** Python


## Uses
### Direct Use

Atık türlerini otomatik olarak sınıflandırmak için kullanılabilir.

### Downstream Use [optional]

Geri dönüşüm tesislerinde atık ayırma süreçlerini otomatikleştirmek, çevresel etkileri izlemek veya atık yönetimi politikalarını geliştirmek için kullanılabilir.

### Out-of-Scope Use

Tıbbi atıklar veya tehlikeli maddelerin sınıflandırılması gibi hassas alanlarda kullanılmamalıdır.

## Bias, Risks, and Limitations

- Bias: Eğitim verilerindeki sınıf dengesizlikleri ("trash" sınıfında daha fazla örnek) modelin performansını olumsuz etkileyebilir.
- Risks: Yanlış sınıflandırmalar, geri dönüşüm süreçlerini aksatabilir ve çevresel sorunlara yol açabilir.
- Limitations: Model, eğitim verilerinde yer almayan yeni atık türlerini tanımayabilir ve farklı ışıklandırma koşullarında veya açılarda çekilen görüntülerde zorlanabilir.
- 
### Recommendations

- Kullanıcılar, modelin sınırlılıkları ve potansiyel önyargıları konusunda bilgilendirilmelidir.
- Sınıf dengesizliklerini gidermek için veri artırma teknikleri (örneğin, rastgele döndürme, çevirme) veya sınıf ağırlıklandırma uygulanabilir.
- Modelin performansını düzenli olarak izlemek ve gerektiğinde güncel veriyle yeniden eğitmek önemlidir.

## How to Get Started with the Model

- Hugging Face Spaces demosunu ziyaret edin: [Hugging Face](https://huggingface.co/spaces/ayse0/Garbage_Classification)
- Sınıflandırmak istediğiniz atık görselini yükleyin veya örnek görsellerden birini seçin.
- "Submit" butonuna tıklayarak modelin tahminini alın.

## Training Details

### Training Data

- Training Data: [Kaggle](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)

### Training Procedure

- Veri ön işleme: Yeniden boyutlandırma (224x224 piksel), piksel değerlerinin 0-1 aralığına ölçeklenmesi, veri artırma (döndürme, çevirme, yakınlaştırma vb.)
- Model mimarisi: 5 evrişim katmanı, 5 pooling katmanı, 2 yoğun katman, dropout katmanı
- Optimizasyon: Adam optimizer (öğrenme hızı 0.0001)

#### Training Hyperparameters

- Batch size: 32
- Epochs: 100 (early stopping ile erken durdurulabilir)

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data
Eğitim verisinden rastgele seçilmiş %20'lik bir bölüm (506 görüntü)

#### Factors
Sınıf dengesizlikleri

#### Metrics
Doğruluk (accuracy), kayıp (loss)

### Results

Eğitim Verisi Üzerindeki Sonuçlar:
Kayıp (Loss): 0.4381
Doğruluk (Accuracy): 0.8776

Doğrulama Verisi Üzerindeki Sonuçlar:
Kayıp (Loss): 0.6690
Doğruluk (Accuracy): 0.8667

#### Summary
Garbage Classification CNN101 modeli, atık sınıflandırma görevinde iyi bir performans sergilemektedir. Eğitim ve doğrulama verileri üzerindeki yüksek doğruluk oranları, modelin atık türlerini etkili bir şekilde ayırt edebildiğini göstermektedir. Ancak, sınıflardaki dengesizlikler nedeniyle bazı atık türlerinde daha iyi performans gösterebilmektedir. 
