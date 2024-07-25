# Food Safety Innovation Risk Management Poultry with AI   


## Contents:

## Problem Statement

In recent times, significant shifts have been observed within the agricultural sector, largely driven by heightened competition concerning both quality and yield. Consequently, the advent of smart farming presents itself as an advantageous avenue for enhancing farm quality. Leveraging technology stands to bolster farm productivity and operational efficiency by integrating cutting-edge advancements such as biotechnology, drones, big data analytics, artificial intelligence, and robotics. These disruptive technologies have spurred the emergence of novel businesses and innovative business models tailored to assist smallholders, who constitute the majority of Thai farmers, in reducing costs and boosting crop output, thereby fostering economic growth.

Furthermore, the agricultural landscape contends with prevalent poultry diseases like Salmonella, Newcastle, and Coccidiosis, which indiscriminately impact farming systems of all scales. The ramifications of such widespread poultry ailments include elevated mortality rates and hindered competitiveness in both export and domestic consumption markets vis-à-vis other high-producing nations. Notably, these challenges disproportionately affect small to medium-scale farms, often managed by younger individuals who may lack the requisite expertise to effectively address such adversities, particularly during disease outbreaks. Hence, early detection emerges as a pivotal strategy in containing the proliferation of these diseases.

In response to these imperatives, efforts are directed towards the development of a Streamlit application aimed at facilitating the classification of chicken diseases based on images of chicken feces utilizing deep learning neural networks.

The primary objective of this initiative centered on creating a Streamlit application designed to aid in the identification and classification of chicken diseases. The application utilizes image analysis techniques to assess the health status of chickens based on samples of their fecal matter. This process involves the implementation of sophisticated deep learning neural networks to accurately discern various disease indicators present in the images.

## Executive Summary
We get the datasets from the [Machine Learning Dataset 
1]
[Machine Learning Dataset 
2]. This is a series of information about poultry disease diagnostics was annotated using Polymerase Chain Reaction (PCR) and the farm-labeled fecal image.The poultry fecal images were taken  between September 2020 and February 2021.




The   **Training subdirectory** encompasses data essential for neural network learning, aiding in the determination of optimal weights and architectural design. The **Validation subdirectory** is dedicated to ensuring that the neural network model avoids overfitting by cross-verifying its performance with the training datasets. Meanwhile, the Testing subdirectory serves as the ultimate evaluation set to gauge the model's overall proficiency.

Recognizing an imbalance within the 'ncd' class in the **Training subdirectory**, we have addressed this issue by implementing image augmentation techniques. This involves diversifying the dataset within this class through operations such as flipping, shifting, rotating, and introducing variations in noise and contrast.

Ultimately, we have meticulously prepared the data to suit the requirements of **training deep learning models**, leveraging the capabilities of TensorFlow and Keras.

In our project, the foundational model is based on **Convolutional Neural Network (CNN)**. To enhance its performance, we incorporate advanced techniques such as transfer learning and fine-tuning. For this purpose, we have selected the VGG16 and MobileNetV2 models, chosen for their ease of integration with the TensorFlow API and their demonstrated high accuracy in addressing the challenges posed by image classification in our project
