---
# MODEL CARD

# Model Card for {{ model_id | Emotion deduction by AI }}

These codes utilize the `MobileNetV2` model from TensorFlow's Keras API for transfer learning. The model is pretrained and then customized for facial expression recognition using the "fer2013" dataset obtained from Kaggle. The dataset comprises seven emotion classes: anger, disgust, fear, happiness, sadness, surprise, and neutral. The MobileNetV2 model is adapted by extracting some of its layers to create a new model specifically tailored for recognizing facial expressions. Additionally, OpenCV's Haar Cascade classifier with the `haarcascade_frontalface_default.xml` file is employed for face detection within the images from the "fer2013" dataset.

{{ model_summary | default("", true) }}

## Model Details

### Model Description

MobileNetV2 Model:

Architecture: MobileNetV2 is a lightweight neural network architecture designed for mobile and edge devices, known for its efficiency and low computational cost.
Purpose in Code: The MobileNetV2 model is employed for transfer learning, allowing the utilization of pre-trained weights to boost performance in facial expression recognition tasks.
Transfer Learning: The code initializes the MobileNetV2 model, extracts certain layers, and then creates a new model for transfer learning by connecting a custom output layer for facial expression recognition.
Haar Cascade Classifier:

Purpose: The Haar Cascade classifier, part of the OpenCV library, is used for face detection within images from the "fer2013" dataset.
File Used: The haarcascade_frontalface_default.xml file is employed as a pre-trained classifier specifically designed for detecting frontal faces.
Functionality: This classifier works by applying a series of progressively more complex classifiers to sub-regions of the image until a face is identified or rejected.

{{ model_description | default("", true) }}

- **Developed by:** {{ developers | Selma Kınacıoğlu)}}
- **Model date:** {{ model_date | 19-02-2024)}}
- **Model type:** {{ model_type | Transform Learning}}
- **Language(s):** {{ language | English)}}
- **Finetuned from model [optional]:** {{ base_model | The Haar Cascade classifie)}}

### Model Sources [optional]

--

- **Repository:** {{ repo | default("[More Information Needed]", true)}}
- **Paper [optional]:** {{ paper | default("[More Information Needed]", true)}}
- **Demo [optional]:** {{ demo | default("[More Information Needed]", true)}}

## Uses

You need to use the link for The Haar Cascade classifie that have been added to github.

### Direct Use

Go to my model.


### Downstream Use [optional]

-

### Out-of-Scope Use

The model will not wok well for low GPU and google colab.


## Bias, Risks, and Limitations



### Recommendations



## How to Get Started with the Model

Go to file my model and run the codes.


## Training Details

### Training Data

Training data will be import it from the files, it is recommended to change the name of the folders to 1-7 from A to Z.


### Training Procedure

The training procedure can be 1-15 then can be increaed to 25

#### Preprocessing [optional]



#### Training Hyperparameters

the most important parametere is learning batch and activation fuunction which is 15 and ReLu function.

#### Speeds, Sizes, Times [optional]

Training the data may take about 1 to 2 hour according to the GPU.


## Evaluation

Open the online stream video and watch your face and test it.

### Testing Data, Factors & Metrics

#### Testing Data

it can be tesed it accoring to test data.

{{ testing_data | default("[More Information Needed]", true)}}

#### Factors


{{ testing_factors | default("[More Information Needed]", true)}}

#### Metrics

Be attention for the photos quality.


### Results

Model gave 0.5 accuracy for this data. it can be increased.



## Model Examination [optional]


{{ model_examination | default("[More Information Needed]", true)}}


## Technical Specifications [optional]

### Model Architecture and Objective

{{ model_specs | The Haar Cascade classifie}}

### Compute Infrastructure

{{ compute_infrastructure | about 1 hour }}

#### Hardware

{{ hardware_requirements | C and D }}

#### Software

{{ software | Google colab will not work, vs code}}





