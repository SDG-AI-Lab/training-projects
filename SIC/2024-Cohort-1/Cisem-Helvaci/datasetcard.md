---
# Dataset Card
---

# Dataset Card for Solar Energy Consumption Dataset

Historical data of solar energy consumption for various countries.

## Dataset Details

### Dataset Description

This dataset includes historical records of solar energy consumption in terawatt-hours (TWh) for various countries. The data is collected and curated by Our World in Data.

- **Curated by:** Our World in Data
- **License:** {{ license | default("[More Information Needed]", true)}}

### Dataset Sources [optional]

<!-- Provide the basic links for the dataset. -->

- **Repository:** {{ repo | default("[More Information Needed]", true)}}
- **Paper [optional]:** {{ paper | default("[More Information Needed]", true)}}
- **Demo [optional]:** {{ demo | default("[More Information Needed]", true)}}

## Uses

<!-- Address questions around how the dataset is intended to be used. -->

### Direct Use

<!-- This section describes suitable use cases for the dataset. -->

The dataset contributes for analysis and forecasting of solar energy consumption trends. It is suitable for use in time series forecasting models, energy policy analysis and research in renewable energy trends.

## Dataset Structure

The dataset has the following fields:

Entity: The name of the country.
Code: The code for countries.
Year: The year of the record.
Electricity from solar - TWh: The amount of electricity generated from solar energy in terawatt-hours.

## Dataset Creation

### Source Data

The source data is collected from Our World in Data, which compiles energy statistics from various reliable sources.

#### Data Collection and Processing

The data was processed to handle missing values, remove duplicates, and normalize the Electricity from solar - TWh field using StandardScaler.

#### Features and the target

Features:

Year (datetime)
Lagged features of solar energy consumption (created during preprocessing).
Target:

Electricity from solar (normalized value in TWh).


## Bias, Risks, and Limitations

The dataset may have biases due to variations in data reporting standards and availability across different countries. In addition to this, the dataset does not account for future changes in technology, policy, or economic factors that could affect solar energy consumption trends.

