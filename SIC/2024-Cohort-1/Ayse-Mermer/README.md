# Garbage Classification with CNN

Bu proje, atık türlerini (karton, cam, metal, kağıt, plastik, çöp) sınıflandırmak için bir Evrişimli Sinir Ağı (CNN) modeli geliştirmeyi amaçlamaktadır.

## Veri Seti

Model eğitimi için Kaggle'da bulunan [Garbage Classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) veri seti kullanılmıştır. Veri seti, 2527 adet görüntüden oluşmaktadır ve her bir görüntü altı farklı atık türünden birine aittir.

## Model

Model, Keras ve TensorFlow kütüphaneleri kullanılarak geliştirilmiştir. Modelin mimarisi 5 evrişim katmanı, 5 pooling katmanı, 2 yoğun katman ve bir dropout katmanından oluşmaktadır. Model eğitimi Google Colab'de T4 GPU kullanılarak gerçekleştirilmiştir.

## Hugging Face Entegrasyonu

Eğitilen model, Hugging Face'e yüklenerek kolayca erişilebilir ve kullanılabilir hale getirilmiştir. Modelin demosunu şu adreste deneyebilirsiniz: [Garbage_Classification](https://huggingface.co/spaces/ayse0/Garbage_Classification)

Örnek bir atık sınıflandırma sonucu:

![image](https://github.com/user-attachments/assets/d95221cd-b9fa-4713-9d6a-e5f41c974751)


## Model Kartı

Model hakkında daha detaylı bilgi için lütfen [Model Card](modelcard.md) dosyasına bakınız.
