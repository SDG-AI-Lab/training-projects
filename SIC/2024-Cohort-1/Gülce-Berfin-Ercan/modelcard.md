---
# MODEL CARD

# Model Card for PopulasyonTahmini

This model predicts the number of cats and dogs on the streets based on various factors such as the number of people living on the street, the area of the street, the number of volunteers, the percentage of animal lovers, and the availability of parks/gardens.

## Model Details

### Model Description

The model uses various regression algorithms (Linear Regression, Decision Tree, Random Forest, SVR) to predict the number of cats and dogs on the streets.



- **Developed by:** Gulce Berfin Ercan
- **Model date:** 2024
- **Model type:** Regression
- **Language(s):** Python


## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

Predicting the number of street animals for planning and management purposes.

### Out-of-Scope Use
The model should not be used for purposes other than population prediction of street animals.

## Bias, Risks, and Limitations

Bias: The model may be biased based on the data it was trained on.
Risks: Incorrect predictions could lead to improper resource allocation.
Limitations: The model's accuracy is limited to the quality and representativeness of the input data.

### Recommendations

Users (both direct and downstream) should be made aware of the risks, biases, and limitations of the model

## How to Get Started with the Model

Use the code below to get started with the model.

{{ get_started_code | default("[More Information Needed]", true)}}

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

{{ training_data | default("[More Information Needed]", true)}}

### Training Procedure

#### Training Hyperparameters

Default hyperparameters for the respective models.

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

 Same as the training data, split into training and testing sets.

#### Factors

Factors include the number of people, area, number of volunteers, percentage of animal lovers, and availability of parks.
#### Metrics
Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared.

### Results
The model's performance varies based on the regression technique used, with specific metrics provided for each model.

#### Summary

{{ results_summary | default("", true) }}

### Model Architecture and Objective

{{ model_specs | default("[More Information Needed]", true)}}

### Compute Infrastructure

{{ compute_infrastructure | default("[More Information Needed]", true)}}

#### Hardware

{{ hardware_requirements | default("[More Information Needed]", true)}}

#### Software

{{ software | default("[More Information Needed]", true)}}






