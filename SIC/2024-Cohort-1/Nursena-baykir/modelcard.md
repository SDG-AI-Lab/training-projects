---
# MODEL CARD

# Model Card for Lstm model and yolov7

## Model Details

Lstm use for wide range od field like  time series anlaysis,video analysis or financial anlaysis.yolov7 used for pose estimetion with pretraiend coco dataset.

### Model Description
Lstm Model used for classified behaviour such as normal or violent.

- **Developed by:** Nursena Baykır
- **Model date:** july 3 2024
- **Model type:** predicting behaviours,pose estimation
- **Language(s):** Python
- **Finetuned from model** yolov7

### Model Sources 


- **Repository:** (https://github.com/WongKinYiu/yolov7?tab=readme-ov-file)
- **Paper :** [A_Review_on_State-of-the-Art_Violence_Detection_Te.pdf](https://github.com/user-attachments/files/16100407/A_Review_on_State-of-the-Art_Violence_Detection_Te.pdf)


## Uses

lstm model use for classifed behaviour but yolov7 model used for pose estimation and detecting human and movements.

### Direct Use

Model can use with loading model first dowload model from here:https://github.com/iremgulcin/nursena-baykir/blob/main/model_rnn.h5
You can save model like this:




```model_save_path = "/kaggle/working/model_rnn.h5"
model_rnn.save(model_save_path)

model = load_model("model_rnn.h5")

```


### Downstream Use 

If this model use for app like streamlit we can analysis with real world camera safety guards could use it.

### Out-of-Scope Use

- Already trained with 2000 videos with 3 hours 45 minutes but still model may not be able to classified violent behaviour very well.

## Bias, Risks, and Limitations

Model may classified actual violent cases normal instead of violent.

### Recommendations

- Don't use this model for malicious or really important cases model may not be useful that much.
- You can use this model on subways cameras or real time camera contain people.

## How to Get Started with the Model

Use the code below to get started with the model.



```
model = load_model("model_rnn.h5")
#if you have a funciton like this you can use this model for video classification.you can find this function from modeling.ipynb

process_video_with_classification(video_path, output_path, model_rnn)


```


## Training Details

### Training Data

We have keypoints form 2000 videos for every frame and these are important for the predict new given videos.
You can find this data from the running moeling.ipynb.

### Training Procedure

#### Preprocessing 

Splitted datas 0.2 test size and Scaled all datas for the better model resahaped all train and test datas for the lstm input.

#### Training Hyperparameters

- **Training regime:** "Adam" and categorical_crossentopy
  
#### Speeds, Sizes, Times 

Epoch 1/10
8429/8429 ━━━━━━━━━━━━━━━━━━━━ 150s 18ms/step - accuracy: 0.8973 - loss: 0.3805 - val_accuracy: 0.9393 - val_loss: 0.2170


Epoch 2/10
8429/8429 ━━━━━━━━━━━━━━━━━━━━ 150s 18ms/step - accuracy: 0.9409 - loss: 0.2125 - val_accuracy: 0.9419 - val_loss: 0.2039


Epoch 3/10
8429/8429 ━━━━━━━━━━━━━━━━━━━━ 201s 18ms/step - accuracy: 0.9429 - loss: 0.2228 - val_accuracy: 0.9435 - val_loss: 0.1994


Epoch 4/10
8429/8429 ━━━━━━━━━━━━━━━━━━━━ 151s 18ms/step - accuracy: 0.9440 - loss: 0.1964 - val_accuracy: 0.9439 - val_loss: 0.1955


Epoch 5/10
8429/8429 ━━━━━━━━━━━━━━━━━━━━ 152s 18ms/step - accuracy: 0.9451 - loss: 0.1893 - val_accuracy: 0.9442 - val_loss: 0.1865


Epoch 6/10
8429/8429 ━━━━━━━━━━━━━━━━━━━━ 148s 18ms/step - accuracy: 0.9458 - loss: 0.1810 - val_accuracy: 0.9456 - val_loss: 0.1793


Epoch 7/10
8429/8429 ━━━━━━━━━━━━━━━━━━━━ 147s 17ms/step - accuracy: 0.9451 - loss: 0.1776 - val_accuracy: 0.9455 - val_loss: 0.1795


Epoch 8/10
8429/8429 ━━━━━━━━━━━━━━━━━━━━ 152s 18ms/step - accuracy: 0.9451 - loss: 0.1727 - val_accuracy: 0.9448 - val_loss: 0.1705
Epoch 9/10

8429/8429 ━━━━━━━━━━━━━━━━━━━━ 151s 18ms/step - accuracy: 0.9461 - loss: 0.1683 - val_accuracy: 0.9456 - val_loss: 0.1703


Epoch 10/10
8429/8429 ━━━━━━━━━━━━━━━━━━━━ 151s 18ms/step - accuracy: 0.9459 - loss: 0.1649 - val_accuracy: 0.9455 - val_loss: 0.1677



## Evaluation

2634/2634 ━━━━━━━━━━━━━━━━━━━━ 18s 7ms/step
LSTM Model Performance on Test Set
Accuracy: 0.9469
Precision: 0.9400
Recall: 0.9814
F1 Score: 0.9602

### Testing Data, Factors & Metrics

#### Testing Data

Datas provided from modeling.ipynb

#### Factors

Depends on the ivdeo qualty model may not be able to classifed behaviou correctly .

#### Metrics

Presicion and recaall and accuracy used for the better idea about model and our model classified succesful but depends on the video model may not be classified right way.

### Results

Accuracy: 0.9469
Precision: 0.9400
Recall: 0.9814
F1 Score: 0.9602

#### Summary

Model with lstm achieved higher performane.

## Model Examination 



## Technical Specifications 

### Model Architecture and Objective

Model used yolov7 to collecting keypoints and trained lstm for the behaviour  cllassified.


### Compute Infrastructure

Model trained and evaulate on kaggle notebook with GPU P100 acceleretor.

#### Hardware

- GPU: NVIDIA Tesla P100
- CPU: Intel i5
- OS:Macos Bigsur
- RAM: 8 GB 
- Disk: 20 GB

#### Software

OpenCv
Torch
Scikit-learn
keras
Yolov7 pose estimation
Matplotlib
Pandas

## Citation 
Kevin B.



## Glossary 

Better understanding using data and creating model like lstm is using annotations.json for every info.

## More Information
I tried this model on my computer without using any accelaretor but that one very slow like 6 hour training time and 3 hours collecting keypoint from every video.
if you don't wnat to run like 12 hours use cloud products.




