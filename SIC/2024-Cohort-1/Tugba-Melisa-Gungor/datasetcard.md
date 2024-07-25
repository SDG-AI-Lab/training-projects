---
# Dataset Card
---

# Dataset Card for Climate and Agricultural Production Data in Turkey

This data set was created to examine the effects of temperature increases and precipitation fluctuations on agricultural products in Turkey. Two main data sources were used: FAOSTAT website were used for the product data set, and Meteoblue website were used for precipitation and temperature data.


## Dataset Details

### Dataset Description
* The product data dataset taken from FAOSTAT includes :
  
The names, quantities, quantity units, harvest areas of various agricultural products produced in Turkey between 1960-2022.

* The data set taken from Meteoblue :
  
Contains 24 temperature measurement data for each month between 1960-2022 and basal precipitation data for each year.

### Dataset Sources 

- **For Product Data** : https://www.fao.org/faostat/en/#data/QCL/visualize
- **For Climate Data** : https://www.meteoblue.com/tr/hava/historyclimate/weatherarchive


## Uses

The fact that the data set consists of various components allowed me to deepen the research.
Agriculture actually depends on many external factors such as wind, humidity, temperature, fertilizer, time.. It might not even be reasonable to expect a connection or a pattern related only to temperature, so precipitation data was also added and it was investigated whether there was a pattern with both.

### Direct Use
This dataset is used to analyze how changes in agricultural product production are related to climate changes and to predict future trends. In addition, agricultural production forecasts for the period until 2025,2030 and 2040 are made using artificial intelligence and machine learning techniques.

## Dataset Structure
This data set consists of two main parts:

1-) Product Data (FAOSTAT):
Columns : Domain Code,	Domain	, Area Code (M49)	, Area,	Element Code	,Element,	Item Code (CPC),	Item	, Year Code	, Year,	Unit	, Value,	Flag	,Flag Description	, Note

2-) Climate Data (Meteoblue):
Columns : timestamp	,[2 m elevation corrected]	,Basel Precipitation Total	Year

## Dataset Creation

### Source Data

The source data set consists of the following columns: Item,  Year, Value,  avg_temp, avg_precipitation, production_metric, water_metric, Product Group, Scaled_Value

#### Data Collection and Processing

- First of all, care was taken to choose reliable platforms when selecting data.
- In the selection of agricultural product data, the most comprehensive and up-to-date data possible was selected.
- When choosing temperature and precipitation data, attention was paid to whether there was time information and whether the annual average of this information would be consistent.
- Temperature data contained 2 measurement data for each month, averaged and converted into avg_temp column. Then, the part of this column between 1960 and 2022 was taken to be compatible with the agricultural product data set.
-In agricultural product data, they were divided into 4 groups using product names (Item column): animal products, seed and legume products, fruits and vegetables, products used as industrial raw materials (such as cotton, silk, beeswax, etc.).
_ Then, when calculating the production amount of these products for each group, the equivalents of the Product values ​​in the Element column and the Value column were added and the average was taken for each year.
- Both precipitation data and production values; Scaling was done so that the numbers were relatively close to each other and the relationship between them could be easily examined.
- It was checked whether there were values ​​that were Null or NaN, and whether the columns were consistent.
- The project was developed in python using jupyter notebook
- Numpy, pandas, sklearn libraries were used for data pre-processing and cleaning, matplotlib for data visualization, and kreas, sklearn libraries for analysis and future predictions.
#### Features and the target

The project examined the relationship between annual production amounts of 4 different agricultural product groups between 1960 and 2022 and temperature, precipitation, and then temperature and precipitation, and predictions were made for the next years 2025, 2030 and 2040.

The project also includes sections where temperature forecasts for future years, precipitation forecasts and the relationship between temperature and precipitation are examined.


## Bias, Risks, and Limitations

This data set is limited only to a specific geographical region (Turkey) and a specific time period (1960-2022).

More contextual data may be required to fully understand the relationship between climate data and agricultural production data.

The accuracy and reliability of data depends on the quality of the data provided by the sources.

Meteorological data can show seasonal and annual fluctuations, which can create uncertainty in modeling and forecasts.

Changes in agricultural data may depend not only on climatic factors, but also on economic, political and technological factors.

