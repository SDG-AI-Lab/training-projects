---
# Dataset Card
---

# Dataset Card for MIT-BIH Arrhythmia Dataset

The MIT-BIH Arrhythmia Dataset is a publicly available dataset widely used for the detection of arrhythmias in ECG recordings. It contains annotations of various types of heartbeats, which can be used to train and evaluate models for automated arrhythmia detection.

## Dataset Details

### Dataset Description

The MIT-BIH Arrhythmia Dataset consists of ECG recordings obtained from 47 subjects studied by the BIH Arrhythmia Laboratory between 1975 and 1979. The dataset includes over 109,000 ECG segments, each labeled with one of five heartbeat classes. The recordings have been digitized at 360 samples per second, and each record is 30 minutes long. The dataset is crucial for the development and benchmarking of algorithms for arrhythmia detection and classification.

Curated by: MIT Laboratory for Computational Physiology
License: Open Database License (ODbL)

## Uses

### Direct Use

The dataset is primarily used for training and evaluating machine learning models aimed at detecting and classifying arrhythmias from ECG recordings. It can be utilized in healthcare research, clinical studies, and the development of real-time monitoring systems for patients with heart conditions.

## Dataset Structure

The dataset is structured as follows:

ECG signals: The raw ECG recordings are provided in binary files, with each file containing 30 minutes of data recorded at 360 Hz.
Annotations: Each ECG recording is accompanied by annotation files that label the heartbeats into five categories:
0: Normal beat
1: Supraventricular premature beat
2: Premature ventricular contraction
3: Fusion of ventricular and normal beat
4: Unclassifiable beat

## Dataset Creation

### Source Data

#### Data Collection and Processing

The ECG recordings were collected from subjects undergoing long-term ECG monitoring. The data was then digitized and annotated by expert cardiologists. The annotations include both the type of heartbeat and the time at which each heartbeat occurs.

#### Features and the target

Features: The primary feature in the dataset is the ECG signal, represented as a time series of voltage measurements.
Target: The target is the classification of heartbeats into one of the five categories mentioned above.

## Bias, Risks, and Limitations

The dataset may have some biases due to the demographic characteristics of the subjects, who were primarily male and from a limited geographical area. Additionally, the manual annotation process, despite rigorous validation, may introduce some subjectivity. The dataset's recordings are also limited to 30-minute segments, which might not capture all types of arrhythmias that occur less frequently.

