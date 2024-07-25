---
# Dataset Card
---

# Dataset Card for {{ pretty_name | FER 2017 }}

<!-- Provide a quick summary of the dataset. -->

{{ dataset_summary |Dataset for emotion deduction  }}

## Dataset Details

### Dataset Description

The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image.

The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). The training set consists of 28,709 examples and the public test set consists of 3,589 examples.


- **Curated by:** {{ curators | }}
- **License:** {{ license | Database Contents License (DbCL) v1.0}}

### Dataset Sources [optional]

<!-- Provide the basic links for the dataset. -->

- **Repository:** {{ repo | 2 main file for train and tes}}
- **Paper [optional]:** {{ paper | no paper}}
- **Demo [optional]:** {{ demo | no demo}}

## Uses

<!-- Address questions around how the dataset is intended to be used. -->

### Direct Use

<!-- This section describes suitable use cases for the dataset. -->

{{ direct_use | default("[More Information Needed]", true)}}


## Dataset Structure

The task is to categorize each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). The training set consists of 28,709 examples and the public test set consists of 3,589 examples.

## Dataset Creation

### Source Data

Social Media.

#### Data Collection and Processing

--


#### Features and the target

The feature based on the face deduction and emotion deduction according to the categories.
### Annotations [optional]

The data was annotated by pre_trained model.

#### Annotation process

The data was annotated by pre_trained model.


{{ annotation_process_section | default("[More Information Needed]", true)}}

#### Who are the annotators?

The data was annotated by pre_trained model.

{{ who_are_annotators_section | default("[More Information Needed]", true)}}


## Bias, Risks, and Limitations

Bias may be evaluated according to the pictures quality.
The accurasy will be increased if we can augmante the dataset.
GPU risk may be considered.

## Citation [optional]

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

