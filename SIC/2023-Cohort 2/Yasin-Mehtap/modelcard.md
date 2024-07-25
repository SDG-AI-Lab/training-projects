---
# MODEL CARD

# Model Card for Anemia Detection Model

This model detects anemia in children under the age of 5 using palm images.

## Model Details

### Model Description

This model utilizes multiple pre-trained convolutional neural network (CNN) architectures, including CNN, MobileNetV2, VGG16, and InceptionV3, for the task of anemia detection using palm images. Each architecture brings its own strengths and characteristics to the task, allowing for a comprehensive exploration of different approaches to image classification.

- **Developed by:** Mehtap Saatçı, Yasin Aksu
- **Model date:** 23.01.2024
- **Model type:** Convolutional Neural Network (CNN), Transfer Learning
- **Language(s):** Python
- **Finetuned from model [optional]:** [Pre-trained models include CNN, MobileNetV2, VGG16, and InceptionV3]

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** Asare, J. W. (2022). Anemia Detection using Palpable Palm Image Datasets from Ghana. Mendeley Data. https://doi.org/10.17632/ccr8cm22vz.1
- **Paper [optional]:** Asare, J. W., Appiahene, P., Donkoh, E. T., & Dimauro, G. (2023). Iron deficiency anemia detection using machine learning models: A comparative study of fingernails, palm and conjunctiva of the eye images. Engineering Reports, 5(11). https://doi.org/10.1002/eng2.12667
- **Demo [optional]:** {{ demo | default("[More Information Needed]", true)}}

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

This model is intended to be used directly for detecting anemia in children under the age of 5 using palm images. It can be deployed in healthcare settings or mobile applications to provide rapid and non-invasive screening for anemia.

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

{{ downstream_use | default("[More Information Needed]", true)}}

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

{{ out_of_scope_use | default("[More Information Needed]", true)}}

## Bias, Risks, and Limitations

### Technical Limitations:

1. Dataset Bias: The model's performance may be affected by biases present in the dataset used for training. For example, if the dataset predominantly consists of palm images from a specific demographic group, the model may not generalize well to other populations.

2. Generalization: While the model has been trained on a diverse dataset, its performance may vary when applied to palm images from populations with different skin tones, ages, or environmental conditions. Care should be taken to ensure the model's generalization across diverse populations.

3. Sensitivity to Image Quality: The model's performance may degrade when presented with low-quality palm images, such as those with poor lighting, motion blur, or artifacts. Robust preprocessing techniques and quality control measures are necessary to mitigate this limitation.

### Sociotechnical Limitations:

1. Ethical Considerations: Deployment of the model in real-world settings raises ethical concerns regarding data privacy, informed consent, and potential biases in decision-making. Careful consideration of these ethical implications is essential to ensure responsible use of the model.

2. Accessibility: While the model offers a non-invasive and potentially cost-effective approach to anemia detection, its accessibility may be limited in resource-constrained settings where access to healthcare infrastructure, internet connectivity, and technical expertise is lacking.

3. Interpretability: Deep learning models, such as CNNs, are often considered black-box models, making it challenging to interpret their decisions. Clinicians and end-users may require explanations or justifications for the model's predictions, which may pose challenges in real-world deployment.

4. Algorithmic Bias: There is a risk of algorithmic bias in the model's predictions, leading to disparities in diagnosis or treatment across different demographic groups. Ongoing monitoring and evaluation of the model's performance across diverse populations are necessary to identify and address any biases.

5. Human-in-the-loop Considerations: The model should be used as a decision support tool rather than a replacement for clinical judgment. Clinicians should exercise caution and critically evaluate the model's predictions in the context of other clinical information and patient history.

6. Legal and Regulatory Compliance: Deployment of the model may be subject to regulatory requirements and legal considerations, particularly regarding medical device regulations, data protection laws, and liability issues. Compliance with relevant regulations and standards is essential to ensure legal and ethical use of the model.

### Recommendations

1. Bias Mitigation Strategies:
   - Ensure diversity and representativeness in the training dataset by actively collecting data from diverse demographic groups.
   - Implement data augmentation techniques to artificially increase the diversity of the dataset and reduce bias.
   - Regularly monitor and evaluate the model's performance across different subpopulations to detect and mitigate biases.

2. Transparency and Interpretability:
   - Develop tools and techniques for explaining the model's predictions, such as feature importance analysis and attention mechanisms, to enhance transparency and interpretability.
   - Provide clinicians and end-users with access to model outputs, confidence scores, and uncertainty estimates to facilitate informed decision-making.

3. Ethical Considerations:
   - Establish clear guidelines and protocols for data collection, storage, and sharing to ensure compliance with ethical standards and data protection regulations.
   - Obtain informed consent from individuals participating in data collection and ensure their privacy and confidentiality are protected throughout the process.
   - Conduct regular audits and reviews of the model's performance and decision-making processes to identify and address potential ethical issues or biases.

4. Human-in-the-loop Integration:
   - Integrate the model into clinical workflows as a decision support tool, allowing clinicians to validate and corroborate the model's predictions with their own expertise and judgment.
   - Provide mechanisms for feedback and correction, enabling clinicians to flag erroneous predictions and provide additional context or information to improve the model's performance.

5. Continuous Monitoring and Evaluation:
   - Establish a robust monitoring and evaluation framework to track the model's performance, including accuracy, fairness, and safety metrics, over time.
   - Conduct regular audits and assessments to identify and mitigate emerging risks, biases, or technical challenges associated with the model.

6. Collaboration and Stakeholder Engagement:
   - Foster collaboration and engagement with diverse stakeholders, including clinicians, researchers, policymakers, and community representatives, to ensure the model's development and deployment align with stakeholders' needs and values.
   - Solicit feedback and input from end-users and affected communities throughout the model's lifecycle to promote transparency, accountability, and trust.

## How to Get Started with the Model

Use the code below to get started with the model.

{{ get_started_code | default("[More Information Needed]", true)}}

## Training Details

### Training Data

80% of the dataset

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

{{ preprocessing | default("[More Information Needed]", true)}}


#### Training Hyperparameters

- **Training regime:** train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='File_Path',
    y_col='Target',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    color_mode='rgb',
    class_mode='categorical'
)

    Epoch Number
    Batch Size
    Activation Function for Dense Layer
    Model Layers for CNN

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

{{ speeds_sizes_times | default("[More Information Needed]", true)}}

## Evaluation

# Accuracy
The model's accuracy was evaluated using a held-out test dataset, where each image was labeled as either anemic or non-anemic. The accuracy metric measures the proportion of correctly classified images out of the total number of images in the test set.

# Processing Time
The processing time of the model refers to the time taken for the model to process each image and make predictions. This includes the time for data preprocessing, feature extraction, model inference, and post-processing steps.

### Testing Data, Factors & Metrics

#### Testing Data

20% of the dataset (10% for validation - 10% for test)

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

{{ testing_factors | default("[More Information Needed]", true)}}

#### Metrics

Accuracy, AUC, Precision, Recall, Processing Time

### Results

MobileNetv2 gives the best accuracy in less processing time. 

#### Summary

CNN: Achieved an accuracy score of 0.865.
MobileNetV2: Achieved the highest accuracy score of 0.937.
VGG16: Achieved an accuracy score of 0.832.
InceptionV3: Achieved an accuracy score of 0.932.
Overall, MobileNetV2 performed the best in terms of accuracy, followed by InceptionV3, CNN, and VGG16, respectively.

CNN: Has a processing time of 48 minutes.
MobileNetV2: Demonstrates the fastest processing time, taking only 18 minutes.
VGG16: Requires the longest processing time among the models, with 240 minutes.
InceptionV3: Shows a processing time of 32 minutes.
In summary, MobileNetV2 has the shortest processing time, making it the most efficient model, while VGG16 exhibits the longest processing time among the models evaluated.

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

{{ model_examination | default("[More Information Needed]", true)}}


## Technical Specifications [optional]

### Model Architecture and Objective

{{ model_specs | default("[More Information Needed]", true)}}

### Compute Infrastructure

{{ compute_infrastructure | default("[More Information Needed]", true)}}

#### Hardware

{{ hardware_requirements | default("[More Information Needed]", true)}}

#### Software

{{ software | default("[More Information Needed]", true)}}

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->


## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

{{ glossary | default("[More Information Needed]", true)}}

## More Information [optional]

{{ more_information | default("[More Information Needed]", true)}}




