---
# Dataset Card
---

# Dataset Card for Impact of Dune Plants on Loggerhead Sea Turtles (Caretta Caretta)
This dataset includes information on nest locations, dates, and types of vegetation surrounding the nests. It is intended to study the impact of environmental factors on the nesting success of loggerhead sea turtles.


## Dataset Details

### Dataset Description

This dataset contains comprehensive information on the nesting habits of loggerhead sea turtles (Caretta caretta) and the surrounding vegetation at various nesting sites. The data was collected to understand the environmental factors influencing nesting success and habitat preferences.


- **Curated by:** Mihriban Özdemir, Türkan Rişvan, Ceren Kılıç

### Dataset Sources 

https://doi.org/10.5061/dryad.zw3r228dk

- **Repository:** Dryad Digital Repository

## Uses

### Direct Use
This dataset can be directly used for training and evaluating machine learning models for analyzing the impact of vegetation types on nesting success of loggerhead sea turtles.


## Dataset Structure
This dataset includes various fields related to the nesting and environmental conditions of loggerhead sea turtles. The dataset contains 837 records.


## Dataset Creation

### Source Data

<!-- This section describes the source data (e.g. news text and headlines, social media posts, translated sentences, ...). -->

#### Data Collection and Processing

The data was collected from field surveys and monitoring efforts along the coast of Florida. Raw case study data was collected by excavating loggerhead sea turtle nests and identifying surrounding plants in 2022.

#### Features and the target
Features variables for the dataset:
- Lat - Latitude of nest
- Long - Longitude of nest
- VegPresence - Presence/absence (1/0) of vegetation around nest
- VegType - Species of vegetation around nest
- RootPresence - Presence/absence (1/0) of roots around nest
- PlantRoot - Species of plant roots belonged to
- DistBarrier - Distance of nest to the barrier (m)
- DistHighWater - Distance of nest to the high water mark (m)
- TotalDist - Total width of beach (m)
- LocOnBeach - Location of nest on the beach
- Division - Which division nest was located on beach; beach was divided into thirds
- SurfaceDepth - Depth from surface to first egg (cm)
- BottomDepth - Depth from surface to bottom of nest chamber (cm)
- InternalDepth - Internal nest chamber depth (cm)
- CavityWidth - Width of the nest cavity, from wall to wall (cm)
- Hatched - Number of eggs hatched
- Unhatched - Number of eggs unhatched
- Developed_UH - Number of unhatched eggs with developed hatchling
- LivePip - Number of live pipped hatchlings
- DeadPip - Number of dead pipped hatchlings
- Yolkless - Number of yolkless eggs
- EncasedTotal - Number of total root-encased eggs
- DevEnc_UH - Number of root-encased unhatched eggs with developed hatchling
- H_Encased - Number of root-encased hatched eggs
- UH_Encased - Number of root-encased unhatched eggs
- InvadedTotal - Number of total root-invaded eggs
- H_Invaded - Number of root-invaded hatched eggs
- UH_Invaded - Number of root-invaded unhatched eggs
- Live - Number of live hatchlings
- Dead - Number of dead hatchlings
- Depredated - Depredation of nest (yes/no; 1/0)
- RootDamageProp - Proportion of root damaged eggs
- ES - Emergence success.
- TotalEggs - Total eggs within the nest

The target variable for the dataset is:
HS: Hatching success rate.



## Bias, Risks, and Limitations

This dataset may have biases due to incomplete data collection or varying data quality based on different surveyors. The dataset also contains missing values in some columns.


## Citation 

Redding, Olivia; Castorani, Max; Lasala, Jake (2024). Case study data 2022: The effects of dune plant roots on loggerhead turtle (Caretta caretta) nest success [Dataset]. Dryad. https://doi.org/10.5061/dryad.zw3r228dk

