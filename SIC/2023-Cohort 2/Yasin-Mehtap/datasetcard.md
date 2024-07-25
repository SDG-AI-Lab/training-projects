---
# Dataset Card
---

# Dataset Card for Anemia Detection with Palm Images

<!-- Provide a quick summary of the dataset. -->

Dataset Summary: This dataset consists of palm images. The dataset includes a total of 3867 images, each labeled as either anemic or non-anemic, with 2169 anemic and 1698 non-anemic instances. 

## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->

{{ This dataset contains palm images collected from hospitals in the city of Accra, Ghana, for the purpose of anemia detection in children under the age of 5. The dataset consists of a total of 3867 images, each labeled as either anemic or non-anemic, with 2169 anemic and 1698 non-anemic instances. Region of Interest (ROI) has been applied to the images. }}

- **Curated by:** {{ curators | default("[More Information Needed]", true)}}
- **License:** {{ license | default("[More Information Needed]", true)}}

### Dataset Sources [optional]

<!-- Provide the basic links for the dataset. -->

- **Repository:** {{ repo | default("[More Information Needed]", true)}}
- **Paper [optional]:** {{ paper | default("[More Information Needed]", true)}}
- **Demo [optional]:** {{ demo | default("[More Information Needed]", true)}}

## Uses

<!-- Address questions around how the dataset is intended to be used. -->

### Direct Use

This dataset can be directly used for training and evaluating machine learning models for anemia detection using palm images. It provides a suitable resource for researchers and practitioners in the field of healthcare diagnostics and machine learning.


## Dataset Structure

The dataset consists of palm images collected for anemia detection in children under the age of 5. Each data point in the dataset includes the following fields:

Image: Palm image representing the sample.
Label: Anemia status of the individual in the image, categorized as either "anemic" or "non-anemic".
The dataset has been split into training, validation, and test subsets. The training dataset contains 80% of the original data, while the validation dataset comprises the remaining 20%. The validation dataset has been further divided into 50% for validation and 50% for testing.

The images are represented in the RGB color model and have been resized to 224x224 dimensions. Pixel values of the images have been rescaled to the range [0,1]. The batch size used for processing is 32.

## Dataset Creation

### Source Data

Data collected from hospitals in Africa, Ghana.

#### Data Collection and Processing

Palm images were collected from pediatric patients under the age of 5. Each image was labeled as anemic or non-anemic based on clinical diagnosis. Region of Interest (ROI) was applied to focus on relevant areas in the images.

The dataset was divided into training, validation, and test subsets. The training dataset contains 80% of the original dataset, while the validation dataset contains the remaining 20%. The validation dataset was further split into 50% validation and 50% test subsets. The RGB color model was used for the images, and the images were resized to 224x224 dimensions. The rescale parameter was set to 1./255 to scale the pixel values of the images to the range [0,1]. A batch size of 32 was used.

#### Features and the target

Features: Palm images with Region of Interest (ROI) applied
Target: Anemia status (Anemic or Non-Anemic)

### Annotations [optional]

<!-- If the dataset contains annotations which are not part of the initial data collection, use this section to describe them. -->

#### Annotation process

<!-- This section describes the annotation process such as annotation tools used in the process, the amount of data annotated, annotation guidelines provided to the annotators, interannotator statistics, annotation validation, etc. -->

{{ annotation_process_section | default("[More Information Needed]", true)}}

#### Who are the annotators?

<!-- This section describes the people or systems who created the annotations. -->

{{ who_are_annotators_section | default("[More Information Needed]", true)}}


## Bias, Risks, and Limitations

The dataset may have inherent biases due to its source from a specific geographic location and demographic.
Annotator bias may also be present depending on the criteria used for labeling images.
Limitations include potential variations in image quality and lighting conditions, which could affect model performance.


## Citation [optional]

Asare, J. W. (2022). Anemia Detection using Palpable Palm Image Datasets from Ghana. Mendeley Data. https://doi.org/10.17632/ccr8cm22vz.1

