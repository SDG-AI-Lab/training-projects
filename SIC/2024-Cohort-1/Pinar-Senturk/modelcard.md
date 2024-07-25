---
# MODEL CARD

# Model Card for MIT-BIH Arrhythmia Detection Model

The MIT-BIH Arrhythmia Detection Model is designed to classify ECG signals into five distinct categories of heartbeats. This model leverages a custom neural network architecture based on Network in Network (NIN) blocks to accurately detect and classify arrhythmias in ECG data.

## Model Details

### Model Description

The model employs a series of Network in Network (NIN) blocks to extract intricate features from 1D ECG signals. It uses convolutional layers, batch normalization, and GELU activation functions to process the input data. The model is designed to handle the complexity of ECG signals and provide robust classification across five heartbeat categories.

Developed by: Pınar Şentürk
Model date: [2024-07-23]
Model type: Neural Network (Network in Network architecture)
Language(s): Python
Finetuned from model [optional]: Not applicable

## Uses

### Direct Use

The model is intended for the automated detection and classification of arrhythmias from ECG recordings. It can be used in healthcare applications for real-time monitoring and diagnosis of heart conditions.

### Downstream Use 

The model can be further fine-tuned for specific types of arrhythmias or integrated into larger healthcare systems for comprehensive patient monitoring.

### Out-of-Scope Use

The model should not be used as a sole diagnostic tool without expert verification. It is not intended for use in life-critical applications without further validation and testing.

## Bias, Risks, and Limitations

The model may exhibit biases related to the demographic characteristics of the training dataset, such as age, gender, and ethnicity of the subjects. It may also struggle with generalization to data collected from different sources or under varying conditions.

### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases, and limitations of the model. More information is needed for further recommendations.

## How to Get Started with the Model

Use the code below to get started with the model.

import torch
from model import Encoder

# Load model
model = Encoder()
model.load_state_dict(torch.load('path_to_model.pth'))
model.eval()

# Load data
# Add your data loading and preprocessing code here

# Make predictions
inputs = torch.tensor(your_data).float()
outputs = model(inputs)

## Training Details

### Training Data

The training data consists of ECG recordings from the MIT-BIH Arrhythmia Dataset. Each recording is labeled with one of five heartbeat categories.

### Training Procedure

The training procedure includes data preprocessing steps such as normalization, reshaping, and splitting into training and validation sets. The model is trained using a WeightedRandomSampler to handle class imbalance.

#### Training Hyperparameters

Batch size: 1024
Learning rate: 1e-3
Learning rate decay: 0.3
Epochs: 50

## Evaluation

The testing data includes ECG recordings from the MIT-BIH Arrhythmia Dataset, separate from the training data.

### Testing Data, Factors & Metrics

#### Testing Data

The testing data includes ECG recordings from the MIT-BIH Arrhythmia Dataset, separate from the training data.

#### Factors

Key factors include class imbalance, signal noise, and variability in ECG patterns.

{{ testing_factors | default("[More Information Needed]", true)}}

#### Metrics

Accuracy: Overall classification accuracy
Precision: Precision for each class
Recall: Recall for each class
F1-Score: F1-Score for each class

The model achieves an overall accuracy of 98.85% on the test set, with detailed metrics for each class provided below.

#### Summary

The model demonstrates robust performance in detecting and classifying arrhythmias from ECG signals, with strong metrics across all heartbeat categories.

### Compute Infrastructure

The model was trained on NVIDIA GeForce RTX 3050 Laptop GPU.

#### Hardware

GPU with at least 8GB VRAM for efficient training and inference.

#### Software

Python 3.7+, PyTorch 1.7+, and necessary data processing libraries.





