---
# MODEL CARD

# Model Card for Climate Impact Prediction on Agricultural Production in Turkey

## Model Summary
This model was developed to estimate the impacts on agricultural production using temperature and precipitation data recorded between 1960 and 2022 in Turkey. The most efficient results were obtained with the LSTM (Long Short-Term Memory) model.

## Model Details

### Model Description

This model uses the LSTM model to predict agricultural production data using temperature and precipitation data.

Developed by: Hochreiter and Schmidhuber

Model date: 1997

Model type: Long Short-Term Memory (LSTM)

Language: Python


## Uses
### Direct use
The model can be used directly to analyze climate factors affecting agricultural production and predict future production levels. For example:

- Assessing the Potential Impacts of Climate Changes on Agricultural Production: The model can analyze the effects of climate changes on agricultural production using temperature and precipitation data. This is useful for predicting possible future production losses or increases.

- Making Forecasts for Agricultural Production Planning and Management: Farmers and agricultural managers can forecast production levels for a particular year or season and optimize planting, harvesting and irrigation strategies based on these forecasts.

- Making Projections of Annual Production Levels in a Specific Region: The model provides information for long-term planning and resource management by making projections of annual production levels in a specific geographic region. This can help in planning water resources management, fertilizer use and agricultural support programs.

- Early Warning Systems: The model can detect possible agricultural production losses early and thus contribute to the creation of early warning systems. In this way, farmers can take precautions against possible adverse climatic conditions.

### Downstream Use
The outputs of the model can be used in various fields. For example:

- Agricultural Policies: It can be used in the process of determining and regulating agricultural policies. Policy makers can use this model to assess the impacts of climate change on agriculture and take appropriate measures. This can help develop sustainable agricultural practices.

- Farmer Planning: Farmers can make production plans according to future climate conditions. This can help determine planting timing, crop selection and irrigation strategies. Farmers can develop more resilient agricultural methods against possible climate changes.

- Climate Change Impact: It can be used to develop sustainable agricultural practices by assessing the long-term effects of climate change. This can contribute to minimizing environmental impacts and making agricultural production systems more resilient.

- Research and Education: The model can be used in academic research and educational purposes, especially to understand the relationships between climate and agriculture. Researchers can conduct further studies using the model's results, and students can gain practical knowledge about agricultural and climate sciences.

- Insurance Industry: Agricultural insurance companies can make risk assessments and determine premium rates using the model's predictions. This could help provide farmers with more affordable and fair insurance policies.

 Finance and Investment: Agricultural investors and financial institutions can make more informed investment decisions with the outputs of the model. This enables better management of risks in financing agricultural projects.

### Out-of-Scope Use
Situations outside the model's scope of use include:

* Forecasts in Other Countries: Since the model is trained only with data specific to Turkey, it is not suitable for agricultural production forecasts in other countries. Climatic and agricultural conditions in other regions may differ and the model does not account for these differences.

* Unverified Data: The model should not be used to make unverified predictions using different climate data or different data sources. The reliability of the model depends on the quality and accuracy of the data used.

* Economic and Political Variables: The model does not take into account other factors (economic, political, technological) affecting agricultural production and is not suitable for this type of analysis. Agricultural production is affected by economic and political decisions, technological developments and social factors, as well as climate factors.

* Long-Term Climate Changes: Designed to model seasonal and annual fluctuations and may not fully reflect the effects of long-term climate changes. Larger and different data sets may be required to assess the effects of long-term climate changes.

* Natural Disasters and Extraordinary Situations: The model is not suitable for predicting unpredictable events such as natural disasters (such as floods, droughts, storms) or extraordinary situations (such as war, epidemics). Such events can have sudden and major impacts on agricultural production.

## Bias, Risks, and Limitations
* The LSTM model is capable of learning long-term dependencies in time series data, but this may ignore the impact of short-term events.
* The performance of the model depends on the accuracy and quality of the dataset; Incorrect or incomplete data may lead to misleading results.
* The model's predictive capacity is limited by the data used during training, and the model may not accurately predict unexpected or extreme events.
* LSTM models typically require large amounts of data and long training times, which can be a challenge when computational resources are limited.
* The model is only valid for a specific geographic region and time period; therefore, its generalizability to different regions or time periods may be limited.
* Changes in agricultural production may depend not only on climatic factors, but also on economic, political and technological factors.
  
### Recommendations
Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model.

## How to Get Started with the Model
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Yıl, sıcaklık, yağış ve Scaled_Value sütunlarını kullanarak veri kümesini oluştur
data = grouped[['Year', 'Temperature', 'Precipitation', 'Scaled_Value']].values
```
The necessary libraries were loaded and the columns to be examined in the data set were grouped.


## Training Details

### Training Data

** For dataset card : https://github.com/iremgulcin/tugba-melisa-gungor/blob/main/datasetcard.md
There is an instance :
In order to analyze animal products from past to present, first the rows containing animal products were included in a separate data set. Then, the production values ​​of these products were averaged for each year and the data was scaled. The same procedures were carried out for the other 3 product groups.

```
grouped = df.groupby(['Year'])['Value'].mean().reset_index()
grouped['Scaled_Value'] = grouped['Value'].apply(lambda x: float('{:.0f}'.format(x)))/200000
```
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/0e6dd716-be9b-4088-929f-8769a6bf7afd)



### Training Procedure
#### Preprocessing 

```
# Veriyi ölçeklendirme
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# LSTM modeli için veri hazırlama fonksiyonu
def prepare_data(values, n_steps):
    X, y = [], []
    for i in range(len(values) - n_steps):
        X.append(values[i:i + n_steps, :3])  # First 3 columns : Year , Temperature, Precipitation
        y.append(values[i + n_steps, 3])     # Target: Scaled_Value
    return np.array(X), np.array(y)
X, y = prepare_data(data_scaled, n_steps)
```

#### Training 
```
# LSTM modelini oluşturma
def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model
 n_steps = 3

 
# Veriyi LSTM giriş formatına uygun hale getirme
X_train = X.reshape((X.shape[0], X.shape[1], X.shape[2]))
y_train = y

# Modeli oluşturma ve eğitme
input_shape = (X_train.shape[1], X_train.shape[2])
model = create_model(input_shape)
model.fit(X_train, y_train, epochs=200, verbose=0)
```
## Evaluation
```
# Gelecek yıllar için tahmin yapma
def forecast(model, input_data, n_steps, future_steps):
    predictions = []
    input_data = input_data[-n_steps:, :3].reshape((1, n_steps, 3))  # İlk 3 sütunu kullan
    for _ in range(future_steps):
        yhat = model.predict(input_data, verbose=0)
        predictions.append(yhat[0, 0])
        new_input = np.array([[0, 0, yhat[0, 0]]])  # Yeni tahmini ekle
        input_data = np.append(input_data[:, 1:, :], new_input.reshape(1, 1, 3), axis=1)
    return predictions
```

### Testing Data, Factors & Metrics
```
# Gelecek yıllar için tahminler (2023-2025)
future_steps = 3
future_years = np.arange(2023, 2023 + future_steps)
scaled_predictions = forecast(model, data_scaled, n_steps, future_steps)

# Ölçeklendirilmiş tahminleri orijinal ölçeğe dönüştürme
predictions = scaler.inverse_transform(np.column_stack((np.zeros((len(scaled_predictions), 3)), scaled_predictions)))[:, 3]
```
### Visualization of Results
```
# Grafik oluşturma
plt.figure(figsize=(10, 6))
plt.plot(grouped['Year'], grouped['Scaled_Value'], marker='o', linestyle='-', label='Gerçek Scaled_Value', color='b')
plt.plot(future_years, predictions, marker='x', linestyle='--', label='Tahminler (2023-2025)', color='r')
plt.xlabel('Yıl')
plt.ylabel('Scaled_Value')
plt.title('2022 Sonrası 3 Yıl için Tahminler')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/14375086-21ee-4612-be48-7b5b55a534f2)


#### Hardware

* GPU: NVIDIA ( To speed up model training)


#### Software
* Python
* TensorFlow
* Pandas
* NumPy




