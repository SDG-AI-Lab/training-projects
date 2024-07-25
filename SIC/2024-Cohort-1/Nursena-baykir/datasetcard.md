---
# Dataset Card
---

# Dataset Card for Violent and Normal Behavior Videos Dataset

This dataset contain 35000 labeled videos with normal and violent behaviours.There is a json file for the annotate videos and giving information about each frames of the videos.

## Dataset Details

### Dataset Description

Kevin B.
normal videos were obtained using a surveillance camera feed and were automatically segmented using an object detector since portions of the original videos are not useful if there are no pedestrians. As for the violent videos, they were obtained from YouTube. Kickboxing tutorial videos were found and in this case, manually segmented when the point of view changed. The purpose of this dataset is to be able to detect each pedestrian, track them, extract their pose estimation, and classify their behavior. In order to track pedestrians, the same camera viewpoint has to be kept. Due to privacy matters, pedestrian's faces were blurred.

- **Curated by:** Kevin B.


### Dataset Sources 

Dataset link:(https://www.kaggle.com/datasets/kevinbkwanloo/kranoknv)

- **Repository:** https://www.kaggle.com/code/nursenabaykr/predicting-with-pose-estimetion-yolov7?scriptVersionId=186809341
- **Paper :** (https://ieeexplore.ieee.org/document/9857832)
## Uses
We can predict pedestrians behaviour with pose estimetion and
Detecting pedestrians.

### Direct Use

Predicting human behaviour for safer and better place in the world .

## Dataset Structure

data has 2 sub field these are annotations.json and videos.
- json has bbox,frame,pedestrains,video id.
- videos has 35000 labeled normal and violent mp4 files.

## Dataset Creation

### Source Data

Videos were taking form youtube.

#### Data Collection and Processing

1000 normal 1000 violent videos randomly selected.We have to plot skeleton each video for taking keypoint data from human body.This datas used for the classified human behaviour.
OpenCv,Torch,yolov7 used for the plotting skeleton and creating model.

#### Features and the target

Main target is train with keypoints and predicting the normal or violent behaviours.


#### Who are the annotators?

This dataset annotadet by Kevin B. His github account is https://github.com/kranok-dev

## Bias, Risks, and Limitations

 Subjective Annotation ; The labels are based on human judgment, which can introduce subjectivity and inconsistency,
 Bias ;These dataset contains real world and not real world videos which may not seen real world scenerious.

