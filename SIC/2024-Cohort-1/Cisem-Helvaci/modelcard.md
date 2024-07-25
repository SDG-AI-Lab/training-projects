---
# MODEL CARD

# Model Card for Solar Energy Consumption Forecasting Model

This model forecasts the solar energy production of various countries using time series analysis, specifically ARIMA.

## Model Details

### Model Description

This model examines historical solar energy consumption data of countries to predict future solar power. The analysis includes preprocessing steps, exploratory data analysis, feature engineering, model training with ARIMA and evaluation. Cross-validation and advanced error analysis are performed. The model's forecasts are visualized to increase understanding of future solar energy trends by country.

- **Developed by:** Çisem Helvacı
- **Model date:** 2024-07-11
- **Model type:** Time Series Forecasting (ARIMA)
- **Language(s):** Python


### Direct Use

The model can be used directly to forecast solar energy production for countries included in the dataset. It would contribute for energy analysts, policymakers and researchers interested in renewable energy trends.

### Out-of-Scope Use

The model is not suitable for short-term forecasting or for countries not represented in the training dataset. It should not be used for real-time decision-making without further validation.

## Bias, Risks, and Limitations

The model's accuracy depends on the quality and reliability of the historical dataset. It may not account for sudden changes in solar energy policies or technological advancements. The model is also limited by the assumption that past trends will continue into the future.

### Recommendations

Users should be made aware of the risks, biases, and limitations of the model. It is recommended to regularly update the model with new data and validate its predictions.

## How to Get Started with the Model

The code:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from pmdarima import auto_arima
import joblib

#Load dataset
df = pd.read_csv("solar-energy-consumption.csv")
df['Year'] = pd.to_datetime(df['Year'], format='%Y')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

#Normalize the data
scaler = StandardScaler()
df['Electricity from solar - TWh'] = scaler.fit_transform(df[['Electricity from solar - TWh']])

#ARIMA Model example for one country
country = 'United States'
country_data = df[df['Entity'] == country].set_index('Year')['Electricity from solar - TWh']

model = ARIMA(country_data, order=(1, 1, 1))
model_fit = model.fit()

#Prophet Model example
def fit_prophet_model(data):
    data_prophet = data.reset_index().rename(columns={'Year': 'ds', 'Electricity from solar - TWh': 'y'})
    model = Prophet()
    model.fit(data_prophet)
    future = model.make_future_dataframe(periods=10, freq='Y')
    forecast = model.predict(future)
    return model, forecast

model, forecast = fit_prophet_model(country_data.reset_index())

#Save the model
joblib.dump(model_fit, 'arima_model.pkl')

## Training Details

### Training Data

The training data includes historical solar energy consumption records for various countries. The data has been preprocessed to handle missing values, duplicates and to normalize the target variable.

### Training Procedure

The training involves fitting ARIMA models to the normalized time series data. Cross-validation is also performed. Detailed error analysis is conducted to evaluate model performance.

#### Preprocessing [optional]

The preprocessing steps include handling missing values, removing duplicates, and normalizing the target variable.

#### Training Hyperparameters

- **Training regime:** fp32

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

The testing data is a subset of the historical solar energy consumption data, reserved for model evaluation.

#### Factors

The evaluation is broken down by country to account for regional differences in solar energy trends.

#### Metrics

Evaluation metrics has Mean Absolute Error (MAE) and Mean Squared Error (MSE) to assess the accuracy of the forecasts.

### Results

The model demonstrates varying levels of accuracy across different countries, reflecting regional differences in solar energy adoption and usage patterns.

#### Summary

The model effectively captures long-term trends in solar energy consumption, with certain limitations in handling short-term fluctuations and sudden changes.




