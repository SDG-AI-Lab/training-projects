---
# Dataset Card
---

# Dataset Card for Brain Tumor MRI Dataset

<!-- Provide a quick summary of the dataset. -->
Dataset Summary:
A dataset consisting of MRI images of brain tumors, including meningioma, glioma, pituitary tumors, and no tumor cases.


## Dataset Details

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->
This dataset contains 7,000 MRI images of brain tumors, categorized into four types: meningioma, glioma, pituitary, and no tumor. The images are used for training and evaluating machine learning models for tumor detection and classification.


- **Curated by:** Berkant ARTAN / Emre Can Şen
- **License:** https://figshare.com/articles/dataset/brain_tumor_dataset/1512427

### Dataset Sources [optional]

<!-- Provide the basic links for the dataset. -->
Source 1 : https://figshare.com/articles/dataset/brain_tumor_dataset/1512427


Source 2: https://www.kaggle.com/datasets/davidbroberts/brain-tumor-object-detection-datasets/data

## Uses

<!-- Address questions around how the dataset is intended to be used. -->

### Direct Use

<!-- This section describes suitable use cases for the dataset. -->
This dataset is directly used for training and evaluating machine learning models, specifically ResNet50 for classification and YOLOv5 for localization of brain tumors in MRI images.


## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->
The dataset is structured into four folders, each corresponding to a tumor type. Each folder contains MRI images in DICOM or JPEG format.

## Dataset Creation

### Source Data

<!-- This section describes the source data (e.g. news text and headlines, social media posts, translated sentences, ...). -->

#### Data Collection and Processing

The images were preprocessed to standardize their size and format. Data augmentation techniques were applied to increase the dataset's diversity.

#### Features and the target

<!-- This section describes the features of the dataset and the target of the project -->
The features are the pixel values of the MRI images, and the target is the tumor type + tumor location.


## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

The dataset may contain biases due to the diversity of imaging equipment and patient demographics. There are risks of misclassification, which could impact medical diagnosis. Limitations include the need for further validation in clinical settings.


