#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from pmdarima import auto_arima
import joblib


# In[3]:


df = pd.read_csv("solar-energy-consumption.csv")


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.info()


# In[7]:


df.drop(columns=['Code'], inplace=True)


# In[8]:


df.info()


# In[9]:


df['Year'] = pd.to_datetime(df['Year'], format='%Y')


# In[10]:


df.info()


# In[28]:


df.describe()


# In[11]:


df.drop_duplicates(inplace=True)


# In[12]:


df.isnull().sum()


# In[26]:


df.info()


# In[13]:


# Plot the solar energy consumption over the years for each country
plt.figure(figsize=(12, 6))
for entity in df['Entity'].unique():
    entity_data = df[df['Entity'] == entity]
    plt.plot(entity_data['Year'], entity_data['Electricity from solar - TWh'], label=entity)
plt.xlabel('Year')
plt.ylabel('Electricity from solar - TWh')
plt.title('Solar Energy Consumption Over the Years by Country')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.show()


# In[14]:


# Decompose the time series for a specific country
country = 'United States'
country_data = df[df['Entity'] == country].set_index('Year')
result = seasonal_decompose(country_data['Electricity from solar - TWh'], model='multiplicative')
result.plot()
plt.show()


# In[15]:


# Creating lagged features
for lag in range(1, 4):
    df[f'solar_lag_{lag}'] = df['Electricity from solar - TWh'].shift(lag)
df.dropna(inplace=True)


# In[16]:


# Normalize the 'Electricity from solar - TWh' column
scaler = StandardScaler()
df['Electricity from solar - TWh'] = scaler.fit_transform(df[['Electricity from solar - TWh']])


# In[17]:


countries = df['Entity'].unique()


# In[18]:


# Dictionaries to store results
model_summaries = {}
forecast_results = {}


# In[19]:


# Fit ARIMA models and forecast for each country
for country in countries:
    print(f"Processing country: {country}")
    country_data = df[df['Entity'] == country].reset_index(drop=True)

    if len(country_data) < 30:
        print(f"Not enough data points for {country}")
        continue
    
    result = adfuller(country_data['Electricity from solar - TWh'])
    print(f'{country} - ADF Statistic: {result[0]}')
    print(f'{country} - p-value: {result[1]}')
    
    if result[1] > 0.05:
        country_data_diff = country_data['Electricity from solar - TWh'].diff().dropna()
    else:
        country_data_diff = country_data['Electricity from solar - TWh']
    
    try:
        model = ARIMA(country_data_diff, order=(1, 1, 1))
        model_fit = model.fit()
        print(model_fit.summary())
        model_summaries[country] = model_fit.summary()
        
        pred = model_fit.predict(start=0, end=len(country_data_diff)-1, typ='levels')
        mae = mean_absolute_error(country_data_diff, pred)
        mse = mean_squared_error(country_data_diff, pred)
        print(f"{country} - Mean Absolute Error: {mae}")
        print(f"{country} - Mean Squared Error: {mse}")
        
        forecast = model_fit.predict(start=len(country_data_diff), end=len(country_data_diff)+10, typ='levels')
        forecast_results[country] = forecast
        
        plt.figure(figsize=(12, 6))
        plt.plot(country_data_diff.index, country_data_diff, label='Actual')
        plt.plot(range(len(country_data_diff), len(country_data_diff) + len(forecast)), forecast, label='Forecast')
        plt.xlabel('Year')
        plt.ylabel('Electricity from solar - TWh')
        plt.title(f'Forecast of Solar Energy Consumption for {country}')
        plt.legend()
        plt.show()
        
    except Exception as e:
        print(f"Could not fit ARIMA model for {country} due to error: {e}")


# In[21]:


# Combine all forecasts into a DataFrame
forecasts_df = pd.DataFrame()

for country, forecast in forecast_results.items():
    forecast_df = pd.DataFrame({
        'Year': range(df[df['Entity'] == country]['Year'].iloc[-1].year + 1, df[df['Entity'] == country]['Year'].iloc[-1].year + 1 + len(forecast)),
        'Forecast': forecast,
        'Country': country
    })
    forecasts_df = pd.concat([forecasts_df, forecast_df], ignore_index=True)

# Display aggregated forecasts
print(forecasts_df)


# In[22]:


# Plot comparisons
plt.figure(figsize=(14, 8))
sns.lineplot(data=forecasts_df, x='Year', y='Forecast', hue='Country')
plt.title('Forecasted Solar Energy Consumption for All Countries')
plt.xlabel('Year')
plt.ylabel('Electricity from solar - TWh')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[23]:


# Evaluate ARIMA model errors on the original scale
for country in model_summaries.keys():
    mae = mean_absolute_error(country_data_diff, pred)
    mse = mean_squared_error(country_data_diff, pred)
    
    original_scale_mae = mae * scaler.scale_[0] + scaler.mean_[0]
    original_scale_mse = mse * (scaler.scale_[0] ** 2)
    
    print(f'{country} - MAE on original scale: {original_scale_mae}')
    print(f'{country} - MSE on original scale: {original_scale_mse}')


# In[24]:


# Save the model
joblib.dump(model_fit, 'arima_model.pkl')

# Load the model for future use
loaded_model = joblib.load('arima_model.pkl')


# In[25]:


# Auto ARIMA example
for country in countries:
    country_data = df[df['Entity'] == country]['Electricity from solar - TWh'].dropna()
    if len(country_data) < 30:
        continue

    stepwise_model = auto_arima(country_data, start_p=1, start_q=1, max_p=3, max_q=3, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)
    print(stepwise_model.summary())

    n_periods = 10
    forecast, conf_int = stepwise_model.predict(n_periods=n_periods, return_conf_int=True)

    plt.figure(figsize=(12, 6))
    plt.plot(country_data.index, country_data, label='Actual')
    plt.plot(pd.date_range(start=country_data.index[-1], periods=n_periods, freq='Y'), forecast, label='Forecast')
    plt.fill_between(pd.date_range(start=country_data.index[-1], periods=n_periods, freq='Y'), conf_int[:, 0], conf_int[:, 1], color='pink', alpha=0.3)
    plt.xlabel('Year')
    plt.ylabel('Electricity from solar - TWh')
    plt.title(f'Auto ARIMA Forecast for {country}')
    plt.legend()
    plt.show()


# In[29]:


from sklearn.model_selection import TimeSeriesSplit

# Time Series Cross-Validation
def cross_validate_arima(data, order, splits=3):
    tscv = TimeSeriesSplit(n_splits=splits)
    errors = []
    
    for train_index, test_index in tscv.split(data):
        train, test = data[train_index], data[test_index]
        model = ARIMA(train, order=order)
        model_fit = model.fit()
        predictions = model_fit.forecast(steps=len(test))
        error = mean_squared_error(test, predictions)
        errors.append(error)
    
    return np.mean(errors), np.std(errors)

# Example for one country
country = 'United States'
country_data = df[df['Entity'] == country]['Electricity from solar - TWh'].dropna().values

mean_error, std_error = cross_validate_arima(country_data, order=(1, 1, 1))
print(f'Cross-Validation Mean Squared Error: {mean_error}')
print(f'Cross-Validation Standard Deviation of Error: {std_error}')


# In[30]:


# Advanced Error Analysis
def plot_residuals(data, model_fit):
    residuals = pd.DataFrame(model_fit.resid)
    residuals.plot(title="Residuals")
    residuals.plot(kind='kde', title='Density')
    plt.show()
    print(residuals.describe())

# Example for one country
country = 'United States'
country_data = df[df['Entity'] == country]['Electricity from solar - TWh'].dropna()

model = ARIMA(country_data, order=(1, 1, 1))
model_fit = model.fit()
plot_residuals(country_data, model_fit)


# In[32]:


get_ipython().system('pip install prophet')


# In[33]:


from prophet import Prophet

# Example using Prophet
def fit_prophet_model(data):
    data_prophet = data.reset_index().rename(columns={'Year': 'ds', 'Electricity from solar - TWh': 'y'})
    model = Prophet()
    model.fit(data_prophet)
    future = model.make_future_dataframe(periods=10, freq='Y')
    forecast = model.predict(future)
    return model, forecast

# Example for one country
country = 'United States'
country_data = df[df['Entity'] == country][['Year', 'Electricity from solar - TWh']].dropna().set_index('Year')

model, forecast = fit_prophet_model(country_data)

fig = model.plot(forecast)
plt.title(f'Prophet Forecast for {country}')
plt.xlabel('Year')
plt.ylabel('Electricity from solar - TWh')
plt.show()


# In[ ]:




