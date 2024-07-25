---
# MODEL CARD

# YOLOV8-L Instance Segmentation

<!-- Provide a quick summary of what the model is/does. -->
Bu model görüntüleri kullanarak segmentasyon yapar.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

Model, görüntü verilerini ve etiketlerin sınırlarını kullanarak, görüntüler üzerinde segmentasyon yapmaktadır. Model, YOLOV8-L adlı bir algoritma kullanmaktadır. YOLOV8-L, YOLO (You Only Look Once) ailesinin en yeni üyesidir. YOLOV8-L, önceki YOLO sürümlerine göre daha hızlı, daha doğru ve daha hafiftir. YOLOV8-L, görüntüleri ışık, renk, açı, perspektif gibi faktörlere karşı dayanıklı hale getirmek için veri artırma teknikleri kullanmaktadır. YOLOV8-L, görüntüleri ışık, renk, açı, perspektif gibi faktörlere karşı dayanıklı hale getirmek için veri artırma teknikleri kullanmaktadır.

- **Developed by:** Ultralytics 
- **Model date:** 10/02/2024
- **Model type:** Segmentation Model
- **Language(s):** Python, Ultralytics

### Model Sources

<!-- Provide the basic links for the model. -->

- **Repository:** [Github Repo](https://github.com/ultralytics/ultralytics)

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->
### Direct Use

Model, görüntüleri girdi olarak alır ve çıktı olarak görüntülerdeki nesnelerin sınıflarını ve segmentasyon maskelerini verir. Model, görüntü işleme, nesne tespiti, video analizi gibi alanlarda kullanılabilir. Model, aşağıdaki gibi doğrudan kullanılabilir:

- Görüntülerdeki nesneleri tanımak ve saymak
- Görüntülerdeki nesnelerin konumlarını ve boyutlarını belirlemek
- Görüntülerdeki nesnelerin arka planından ayırmak
- Görüntülerdeki nesnelerin ilişkilerini ve etkileşimlerini anlamak
- Görüntülerdeki nesnelerin özniteliklerini ve durumlarını tanımlamak
<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

### Downstream Use
Model, başka bir görev için ince ayar yapmak veya başka bir sistem veya uygulamaya entegre etmek için kullanılabilir. Model, aşağıdaki gibi dolaylı kullanılabilir:

- Görüntülerdeki nesneleri sınıflandırmak ve segmente etmek için bir ön işleme adımı olarak kullanmak
- Görüntülerdeki nesnelerin özniteliklerini ve durumlarını tanımlamak için bir ara işleme adımı olarak kullanmak
- Görüntülerdeki nesnelerin ilişkilerini ve etkileşimlerini anlamak için bir son işleme adımı olarak kullanmak
- Görüntülerdeki nesneleri tanımak ve saymak için bir görsel analiz aracı olarak kullanmak
- Görüntülerdeki nesnelerin konumlarını ve boyutlarını belirlemek için bir haritalama aracı olarak kullanmak
- Görüntülerdeki nesnelerin arka planından ayırmak için bir görüntü düzenleme aracı olarak kullanmak
<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

### Out-of-Scope Use

Model, aşağıdaki gibi uygun olmayan veya kötüye kullanılabilecek şekillerde kullanılmamalıdır:

- Görüntülerdeki insanların yüzlerini, kimliklerini, duygularını, cinsiyetlerini, ırklarını, yaşlarını, sağlık durumlarını, mahremiyetlerini, güvenliklerini veya onurlarını ihlal etmek için kullanmak
- Görüntülerdeki nesneleri yanlış, yanıltıcı, taraflı, ayrımcı, saldırgan, zararlı veya yasa dışı bir şekilde tanımlamak, sınıflandırmak veya segmente etmek için kullanmak
- Görüntülerdeki nesneleri değiştirmek, sahteleştirmek, manipüle etmek veya kötü amaçlı bir şekilde kullanmak için kullanmak
- Görüntülerdeki nesneleri tanımak, saymak, konumlandırmak, boyutlandırmak, ayırmak veya anlamak için yeterli veri, bilgi, deneyim veya yetkinliğe sahip olmadan kullanmak
<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

## Bias, Risks, and Limitations
Model, hem teknik hem de sosyoteknik sınırlamalar içermektedir. Bu sınırlamalar, aşağıdaki gibidir:

- Model, proje amacına yönelik daha önce yayınlanmamış bir veri seti ile eğitilmiştir. Model oluşturulmuş veri seti içerisinde bulunmayan sınıfları tanıyamaz. Ve modelin doğruluk oranına göre bazı hatalar yapabilir.

- Model, YOLOV8-L adlı bir algoritma kullanmaktadır. YOLOV8-L, önceki YOLO sürümlerine göre daha hızlı, daha doğru ve daha hafiftir. Ancak, YOLOV8-L, hala bazı zorluklarla karşılaşmaktadır. YOLOV8-L, küçük, yoğun, üst üste binen veya arka planda kalan nesneleri tespit etmekte zorlanabilir. YOLOV8-L, görüntülerdeki ışık, renk, açı, perspektif gibi faktörlere karşı hassas olabilir. YOLOV8-L, görüntülerdeki nesnelerin özniteliklerini ve durumlarını tanımlamak için yeterli değildir.

- Model, görüntü işleme, nesne tespiti, video analizi gibi alanlarda kullanılabilir. Model, bu alanlarda kullanılmadan önce, uygun veri setleri ve hiperparametreler ile ince ayar yapılması gerekmektedir.

- Model, görüntülerdeki nesneleri sınıflandırmak ve segmente etmek için olumlu bir amaç için kullanılmalıdır. Model, insanların yüzlerini, kimliklerini, duygularını, cinsiyetlerini, ırklarını, yaşlarını, sağlık durumlarını, mahremiyetlerini, güvenliklerini veya onurlarını ihlal etmemelidir. Model, görüntülerdeki nesneleri yanlış, yanıltıcı, taraflı, ayrımcı, saldırgan, zararlı veya yasa dışı bir şekilde tanımlamak, sınıflandırmak veya segmente etmek için kullanılmamalıdır. Model, görüntülerdeki nesneleri değiştirmek, sahteleştirmek, manipüle etmek veya kötü amaçlı bir şekilde kullanmak için kullanılmamalıdır.
<!-- This section is meant to convey both technical and sociotechnical limitations. -->

### Recommendations

Modelin önyargı, risk ve teknik sınırlamalarını önlemek veya azaltmak için, aşağıdaki tavsiyelere uyulması gerekmektedir:

- YOLOV8-L, önceki YOLO sürümlerine göre daha hızlı, daha doğru ve daha hafiftir. Ancak, YOLOV8-L, hala bazı zorluklarla karşılaşmaktadır. YOLOV8-L, küçük, yoğun, üst üste binen veya arka planda kalan nesneleri tespit etmekte zorlanabilir. YOLOV8-L, görüntülerdeki ışık, renk, açı, perspektif gibi faktörlere karşı hassas olabilir. YOLOV8-L, görüntülerdeki nesnelerin özniteliklerini ve durumlarını tanımlamak için yeterli değildir.

- Model, görüntülerdeki nesneleri sınıflandırmak ve segmente etmek için olumlu bir amaç için kullanılmalıdır. Model, insanların yüzlerini, kimliklerini, duygularını, cinsiyetlerini, ırklarını, yaşlarını, sağlık durumlarını, mahremiyetlerini, güvenliklerini veya onurlarını ihlal etmemelidir. 
<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

## How to Get Started with the Model

Modelin pretrained halini kullanmak için aşağıdaki kodu yazarak başlayabilirsiniz.

```python

import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = torch.hub.load('/file-path/model_best_output.pt')

```

Eğitilmiş modele [buradan](output_model.pt) ulaşabilirsiniz.

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

Model proje için üretilen veri seti üzerinde eğitilmiştir ve 2 farklı sınıfı tanıyabilmektedir. Veri setinin orjinalinde 280 adet görsel etiketlenmiş olup, daha sonrasında data augementation yöntemleri ve CLAHE algoritmasını kullanarak bu görsel sayısı 837'ye çıkarılmıştır.


### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing

Model, görüntüleri girdi olarak almadan önce, aşağıdaki ön işleme adımlarını uygulamaktadır:

- Görüntüleri RGB formatına dönüştürmek
- Görüntüleri 640x640 piksel boyutuna yeniden boyutlandırmak
- Görüntüleri tensor formatına dönüştürmek
- Görüntüleri normalize etmek


#### Training Hyperparameters

- **Batch size:** 16
- **Learning rate:** 0.01
- **imgsz:** 640
- **plots:** True
- **Momentum:** 0.937
- **Weight decay:** 0.0005
- **Epochs:** 30
- **Optimizer:** SGD
- **Loss function:** Cross entropy + IoU + Dice

#### Times 

Modeli yukarıdaki gibi ince ayarlarsanız modelin eğitimi 0.284 saat sürmektedir.

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

Model, test için oluşturulan veri setinin bir parçası olan 70 görüntüyü kullanmaktadır. Bu görüntüler, modelin eğitiminde kullanılmamıştır. Bu görüntüler, 2 farklı sınıfı tanımlayan örnekler içermektedir. Bu görüntüler, görüntülerdeki nesnelerin sınırlarını ve segmentasyon maskelerini de sağlamaktadır.
<!-- This should link to a Dataset Card if possible. -->

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. Decision tresholds, model performance measures -->

Model, performansını aşağıdaki metrikler ile ölçmektedir:

- **mAP50:** Bu metrik, modelin nesneleri doğru sınıflandırma ve segmente etme yeteneğini ölçmektedir. Bu metrik, modelin ürettiği nesne sınırları ve segmentasyon maskeleri ile gerçek nesne sınırları ve segmentasyon maskeleri arasındaki benzerliği IoU (Intersection over Union) ve Dice katsayısı ile hesaplamaktadır. Bu metrik, IoU ve Dice katsayısı 0.5’ten büyük olan tahminleri doğru kabul etmektedir. Bu metrik, tüm sınıflar ve boyutlar için ortalama bir değer vermektedir.
- **mAP75:** Bu metrik, mAP50 metriğine benzerdir, ancak daha katı bir kriter kullanmaktadır. Bu metrik, IoU ve Dice katsayısı 0.75’ten büyük olan tahminleri doğru kabul etmektedir. Bu metrik, modelin nesneleri daha hassas bir şekilde tanıma ve segmente etme yeteneğini ölçmektedir.
- **mAP:** Bu metrik, mAP50 ve mAP75 metriklerinin genelleştirilmiş bir versiyonudur. Bu metrik, IoU ve Dice katsayısı 0.5’ten 0.95’e kadar 0.05’lik aralıklarla değişen farklı eşikler için mAP değerlerini hesaplamakta ve bunların ortalamasını almaktadır. Bu metrik, modelin nesneleri farklı hassasiyet seviyelerinde tanıma ve segmente etme yeteneğini ölçmektedir.

### Results

Model, test veri seti üzerinde aşağıdaki performans değerlerine ulaşmıştır:

- **mAP50:** %59.2
- **mAP75:** %40.3
- **mAP:** %42.7

#### Summary

Bu sonuçlar, modelin görüntülerdeki nesneleri sınıflandırma ve segmente etme konusunda oldukça başarılı olduğunu göstermektedir. Fakat bu başarı oranlarının veri setinin büyütülmesi ile daha da artacağını tahmin ediyorum.







