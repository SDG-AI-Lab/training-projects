---
# RESULTS
## ANALYSIS
### Temperature and Precipitation 
*Yearly Variation of Average Temperature*

 ![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/17764e31-8465-49b0-a626-04e4b71363c0)
 **temperature unit : °C**

*Yearly Variation of Average Precipitation*

![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/bf1fcffd-cffe-45e6-a9c0-333bc9002c5e)
**average basel precipitation unit : mm**
**How are the precipitation values scaled**
```
water_values=merged_df.groupby('Year')['avg_precipitation'].mean()
precipitation_scaledc=water_values*100
```
*Yearly Variation of Average Temperature and Precipitation*

![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/b3e26537-44ca-4e18-9d5c-a53bd9de6133)

### Animal Product

*Yearly Variation of Average Animal Product Production*

![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/6367c016-7c8b-4d5b-8de2-ebd6545ce85d)

**How are the values ​​scaled?**
```
grouped = df.groupby(['Year'])['Value'].mean().reset_index()
grouped['Scaled_Value'] = grouped['Value'].apply(lambda x: float('{:.0f}'.format(x)))/200000
```

*Yearly Variation of Average Temperature and Yearly Variation of Average Animal Product Production*

![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/a90c74a6-9b49-413c-81f1-216e358616b1)

*Yearly Variation of Precipitation and Average Animal Product Production*

![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/9ad4def6-ae54-423c-935b-8e2568558f00)

*Yearly Variation of Temperature, Precipitation, and Value for Animal products*

![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/0d8a4903-5a5d-4f10-b1a3-4ee2217f0fa5)

### Fruit and Vegetable

*Yearly Variation of Average Fruit and Vegetalbe Production*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/e8d0a6c7-6811-40cd-801f-b64413574d0b)

*Yearly Variation of Average Temperature and Fruits and Vegetables*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/87b78280-b85a-4d15-b99b-28d5eb579de8)

*Yearly Variation of Precipitation and Fruits and Vegetables'*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/b52db66d-1783-4638-9203-2a8432c5dba3)

*Yearly Variation of Temperature, Precipitation, Fruits and Vegetables*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/f1908b80-ca7d-442e-9ab7-989ac8c00991)

### Legume and Cereal

*Yearly Variation of Average Legume and Cereal Production*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/ace900ff-8102-42e4-8a2d-292b4ed02b31)

*Yearly Variation of Temperature and Legumes and Cereals*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/55b36e76-982f-4c8b-a7f8-e4c826af1137)

*Yearly Variation of Precipitation and Legumes and Cereals*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/20ee1f03-a994-4fe4-a64d-e72906b268d6)

*Yearly Variation of Temperature, Precipitation, legumes and cereals*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/24eb8784-8a02-4d39-b5de-6c1301b762cf)

### Industrial Raw Material

*Yearly Variation of Industrial Raw Material*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/acdadbbf-e1c1-40c6-8cf0-94c5e23c795f)

*Yearly Variation of Temperature and Industrial Raw Material*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/7c27e90f-af7e-4842-86e1-c235b4ba307a)

*Yearly Variation of Precipitation and Industrial Raw Material*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/d7530950-e95e-48fc-8b6f-fb43ab20d872)

*Yearly Variation of Temperature, Precipitation, and Industrial Raw Material*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/e2eccaae-defb-4549-b038-4b7dc6838a8d)


## FORECASTS

### Temperature Forecasts
*Yearly Average Temperature and Future Forecasts Until 2025*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/55320b67-daf5-4ff2-90e3-9b3a2ec6317f)

*Yearly Average Temperature and Future Forecasts Until 2030*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/58af5ec0-5061-4fbe-b0b3-bbf568fe9ec0)

*Yearly Average Temperature and Future Forecasts Until 2040*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/3bfc7d43-a24b-49dc-887f-2ccd96a6bc4b)

### Precipitation Forecasts

*Yearly Precipitation (mm) and Future Forecasts Until 2025*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/5b32ef5f-d16d-4b8b-ad8e-fcb2f295a1d8)

*Yearly Precipitation (mm) and Future Forecasts Until 2030*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/80d1aca9-605d-44b4-9199-56f7d779c41c)

*Yearly Precipitation (mm) and Future Forecasts Until 2040*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/d58166d6-2e3c-4af4-88d9-68094d638d6f)

### Animal Product Producing
*Average animal product production estimates for 3 years after 2022, using temperature and precipitation characteristics*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/222c5709-5c65-48f7-b620-757576bb3228)

*Average animal product production estimates for the years 2023-2030 , using temperature and precipitation characteristics*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/c2a76a9e-b196-46bf-bf9a-19f8dfadc05f)

*Animal product production forecasts for the years 2023 - 2030, created with animal product production statistics from past to present.*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/078e10f6-f8be-4dcc-86e7-f6fd60509406)

*Animal product production forecasts for the years 2023 - 2040, created with animal product production statistics from past to present.*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/102a8af2-057b-412a-bba1-137d99656ecc)


### Fruits and Vegetables
*Average fruit and vegetable production estimates for 3 years after 2022, using temperature and precipitation characteristics*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/41755838-a70a-40a0-bc8e-bb67cc4cdac4)

*Fruit and vegetable production forecasts for the years 2023 - 2030, created with  production statistics from past to present.*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/31797adf-1f42-48ce-b882-f25fd1decbed)

*Fruit and vegetable production forecasts for the years 2023 - 2040, created with  production statistics from past to present.*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/7e8046da-f1af-4027-9fa0-112149826cf3)

### Legume and Cereals
*Average legume and cereal production estimates for 3 years after 2022, using temperature and precipitation characteristics*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/eece6014-1081-4b7b-9bd0-7ce247ea78fd)

*Legume and Cereal production forecasts for the years 2023 - 2025, created with  production statistics from past to present.*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/4d684ef1-06e3-4c16-8555-ab4d47217b2d)

*Legume and cereal production forecasts for the years 2023 - 2030, created with  production statistics from past to present*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/a33d50be-67cf-4590-a8d1-a8996319958b)

*Average legume and cereal production estimates for the years 2023-2030 , using temperature and precipitation characteristics*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/fef49bfe-f662-4a79-bf8a-75480d290cf0)

*Average legume and cereal production estimates for the years 2023-2040 , using temperature and precipitation characteristics*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/3c7f897d-05f6-4f5e-8688-ba4779de2df0)

*Legume and cereal production forecasts for the years 2023 - 2040, created with  production statistics from past to present*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/ce97a5d1-8d02-45d5-be06-e74f64d2cca8)

### INDUSTRIAL RAW MATERIALS

*Industrial raw material production estimates for 3 years after 2022, using temperature and precipitation characteristics*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/f186d3f7-7d60-4a9f-9dd1-4ede3dbebabf)

*Industrial raw material production estimates for the years 2023-2030 , using temperature and precipitation characteristics*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/dd92ce4c-4a39-4c11-affa-aba1091c5a0d)

*Industrial raw material production forecasts for the years 2023 - 2030, created with  production statistics from past to present*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/ab21af2c-bd1b-4628-b33b-88c57411389a)

*Industrial raw material production forecasts for the years 2023 - 2040, created with  production statistics from past to present*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/1618496e-fdce-49ab-8811-fe47ccd07789)

*Industrial raw material production estimates for the years 2023-2040 , using temperature and precipitation characteristics*
![resim](https://github.com/iremgulcin/tugba-melisa-gungor/assets/105967050/a2660562-03b5-46aa-b1e1-383edcdb2e71)















