---
# Dataset Card
---

# Dataset Card for ADHD-200 Dataset

The ADHD-200 dataset is a collection of neuroimaging data used for the prediction and study of Attention Deficit Hyperactivity Disorder (ADHD). It includes structural and functional MRI data from individuals diagnosed with ADHD and control subjects.

## Dataset Details

### Dataset Description

The ADHD-200 dataset comprises MRI data collected from multiple sites, including resting-state functional MRI (rs-fMRI) and structural MRI (sMRI). The dataset is designed to facilitate the development of predictive models for ADHD diagnosis and to understand the neural correlates of ADHD. It includes data from 973 individuals, of which 491 are diagnosed with ADHD, and 582 are controls. The data is preprocessed and made available in a standardized format.

- **Curated by:** {{ curators | default("[More Information Needed]", true)}}
- **License:** {{ license | default("[More Information Needed]", true)}}

### Dataset Sources 

<!-- Provide the basic links for the dataset. -->

- **Repository:** https://fcon_1000.projects.nitrc.org/indi/adhd200/
- **Paper [optional]:** {{ paper | default("[More Information Needed]", true)}}
- **Demo [optional]:** {{ demo | default("[More Information Needed]", true)}}

## Uses

### Direct Use

The dataset is primarily used for developing and evaluating machine learning models for ADHD diagnosis. It is also used in academic research to understand the neural mechanisms underlying ADHD and to explore neuroimaging biomarkers associated with the disorder.


## Dataset Structure

The dataset is organized by site and includes both raw and preprocessed neuroimaging data. Each subject has associated metadata, including age, gender, diagnostic status, and medication status. The sites involved are:

New York University Child Study Center (NYU)
Peking University (Peking)
Kennedy Krieger Institute (KKI)
NeuroIMAGE Sample (NeuroIMAGE)
Oregon Health & Science University (OHSU)
University of Pittsburgh (Pitt)
Washington University in St. Louis (WashU)
Brown University (Brown)
University of Michigan (UM)

## Dataset Creation

### Source Data

The data were collected from multiple international sites as part of the ADHD-200 Global Competition. Each site followed a standardized protocol for data acquisition.


#### Data Collection and Processing

<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, tools and libraries used, etc. -->

{{ data_collection_and_processing_section | default("[More Information Needed]", true)}}

#### Features and the target

<!-- This section describes the features of the dataset and the target of the project -->

### Annotations [optional]

<!-- If the dataset contains annotations which are not part of the initial data collection, use this section to describe them. -->

#### Annotation process

<!-- This section describes the annotation process such as annotation tools used in the process, the amount of data annotated, annotation guidelines provided to the annotators, interannotator statistics, annotation validation, etc. -->

{{ annotation_process_section | default("[More Information Needed]", true)}}

#### Who are the annotators?

<!-- This section describes the people or systems who created the annotations. -->

{{ who_are_annotators_section | default("[More Information Needed]", true)}}


## Bias, Risks, and Limitations
Bias: The dataset may have site-specific biases due to differences in MRI scanners and acquisition protocols. <br />
Risks: The use of neuroimaging data for diagnostic purposes should be done cautiously, considering the potential for misdiagnosis.<br />
Limitations: The dataset includes a limited age range (7-21 years) and may not generalize to older populations. Additionally, the variability in diagnostic criteria across sites may affect the consistency of the data.<br />


## Citation [optional]

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

