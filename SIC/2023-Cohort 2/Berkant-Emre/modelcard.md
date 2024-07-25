---
# MODEL CARD

# Model Card for  Brain Tumor Detection Model

<!-- Provide a quick summary of what the model is/does. -->
A deep learning model trained to detect and classify brain tumors in MRI images.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->
This model is designed to identify and classify brain tumors in MRI images. It utilizes a combination of ResNet50 for classification and YOLOv5 for localization to detect the presence of meningioma, glioma, pituitary tumors, and no tumor cases.

- **Developed by:** Berkant Artan & Emre Can Şen
- **Model date:** 19.02.2024
- **Model type:** Convolutional Neural Network (CNN) with ResNet50 and YOLOv5
- **Language(s):** Python, TensorFlow, PyTorch
- **Finetuned from model [optional]:** Pretrained ResNet50 and YOLOv5 models

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** {{ repo | default("[More Information Needed]", true)}}
- **Paper [optional]:** {{ paper | default("[More Information Needed]", true)}}
- **Demo [optional]:** {{ demo | default("[More Information Needed]", true)}}

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use
The model is directly used for detecting and classifying brain tumors in MRI images for medical diagnosis and research purposes.

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->
The model's outputs can be integrated into medical imaging software, used in clinical trials, or applied in research studies focused on brain tumor treatments and patient outcomes.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->
The model is not intended for use in non-medical applications or as a standalone diagnostic tool without expert medical supervision.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->
There may be biases due to the diversity of imaging equipment and patient demographics. Risks include potential misclassification of tumors, which could impact medical diagnosis. The model requires further validation in clinical settings and may have limitations in generalizing to unseen data or different tumor types.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->
Users, both direct and downstream, should be made aware of the risks, biases, and limitations of the model. It is recommended to use the model in conjunction with expert medical evaluation and to continually update the model with diverse and representative data.

## How to Get Started with the Model

Use the provided .ipynb files to implement the respective models for tumor detection and classification.  

## Training Details

### Training Data

7000 MRI images with 4 different brain tumor type were used. Training dataset was obtained by splitting data randomly.
The whole dataset : https://figshare.com/articles/dataset/brain_tumor_dataset/1512427

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->


#### Preprocessing [optional]
Images were resized, normalized, and augmented with techniques such as rotation, flipping, and zooming to increase the dataset's diversity.


#### Training Hyperparameters

- **Training regime:**  <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->
The model was trained using a batch size of 32, a learning rate of 0.001, and the Adam optimizer for 50 epochs.

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

For ResNet50 model: 45 minutes and for YOLOv5 model: 1.5 hours using Google Colab environment

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics
The model was evaluated on a separate test set of 1,000 MRI images.


#### Testing Data

<!-- This should link to a Dataset Card if possible. -->
7000 MRI images with 4 different brain tumor type were used. Training dataset was obtained by splitting data randomly.
The whole dataset was given above.

#### Factors
Factors considered in the evaluation include the diversity of tumor types, image quality, and the presence of artifacts.


#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. Decision tresholds, model performance measures -->
Accuracy: The percentage of correctly classified images.
Precision, Recall, F1 Score: Metrics to evaluate the model's performance in classifying each tumor type.
Intersection over Union (IoU): A metric to evaluate the localization accuracy of YOLOv5.

### Results
The model achieved an accuracy of 95% on the test set, with precision and recall values above 90% for each tumor type. The IoU for tumor localization was 0.85.

#### Summary

The brain tumor detection model demonstrates high accuracy in classifying and localizing tumors in MRI images. It shows promise for assisting in medical diagnosis and research.



## Technical Specifications [optional]

### Model Architecture and Objective

The model combines ResNet50 for classification and YOLOv5 for localization, utilizing transfer learning and fine-tuning techniques.

#### Software

The model requires Python 3.x, TensorFlow, PyTorch, and other dependencies listed in the repository's requirements file.




