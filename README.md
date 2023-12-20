# Two Stream S3D Architecture for Word Level Sign Language Recognition


## Introduce

This repository accompanies the paper [Two Stream S3D Architecture for Word Level Sign Language Recognition](). The article addresses sign language recognition at the word level based on the Separable 3D CNN (S3D) model. We propose a low-cost model because we recognize the potential for future use of identification systems on handheld devices. We have conducted experiments on many different data sets and achieved the expected results.

## Two Stream S3D Architecture

![Architecture](images/architecture.png)

## Dataset

We tested on three different datasets including: [Large-Scale Multimodal Turkish Signs (AUTSL)](https://ieeexplore.ieee.org/abstract/document/9210578), [Large-Scale Dataset for Word-Level American Sign Language (WLASL)](https://github.com/dxli94/WLASL), and [A Dataset for Argentinian Sign Language (LSA64)](https://facundoq.github.io/datasets/lsa64/). You can explore and download the data on our link provided in this repository.

## Data Folder

To make it easier for you to use your custom data, we describe in detail the structure of the folder containing the data as follows:

```
Root Folder
├── Videos
│   ├── Gloss_1
|   |   |──video_1.1.mp4
|   |   |──video_1.2.mp4
│   ├── Gloss_2
|   |   |──video_2.1.mp4
|   |   |──video_2.2.mp4
├── Preprocessing
│   ├── test
|   |   ├──frames
|   |   |  ├──Gloss_1
|   |   |  |  ├──video_1.1
|   |   |  |  |  ├──frame_1.1.0.jpg
|   |   |  |  |  ├──frame_1.1.1.jpg
|   |   |  ├──Gloss_2
|   |   |  |  ├──video_2.1
|   |   |  |  |  ├──frame_2.1.0.jpg
|   |   |  |  |  ├──frame_2.1.1.jpg
|   |   ├──poses
|   |   |  ├──Gloss_1
|   |   |  |  ├──video_1.1
|   |   |  |  |  ├──pose_1.1.0.jpg
|   |   |  |  |  ├──pose_1.1.1.jpg
|   |   |  ├──Gloss_2
|   |   |  |  ├──video_2.1
|   |   |  |  |  ├──pose_2.1.0.jpg
|   |   |  |  |  ├──pose_2.1.1.jpg
│   ├── train
|   |   ├──frames
|   |   |  ├──Gloss_1
|   |   |  |  ├──video_1.1
|   |   |  |  |  ├──frame_1.1.0.jpg
|   |   |  |  |  ├──frame_1.1.1.jpg
|   |   |  ├──Gloss_2
|   |   |  |  ├──video_2.1
|   |   |  |  |  ├──frame_2.1.0.jpg
|   |   |  |  |  ├──frame_2.1.1.jpg
|   |   ├──poses
|   |   |  ├──Gloss_1
|   |   |  |  ├──video_1.1
|   |   |  |  |  ├──pose_1.1.0.jpg
|   |   |  |  |  ├──pose_1.1.1.jpg
|   |   |  ├──Gloss_2
|   |   |  |  ├──video_2.1
|   |   |  |  |  ├──pose_2.1.0.jpg
|   |   |  |  |  ├──pose_2.1.1.jpg
├── folder2label_int.txt
```
We will need a root folder containing all the necessary things. The videos folder will be the folder containing the video data you want to use. Let's say your dataset or one of the datasets we introduced like LSA64, AUTSL and WLASL. The preprocesisng folder will be automatically created by our code. The file folder2label_int.txt contains labels of glosses numbered from 0 to N-1 and N is the number of glosses you have in the data set.

The process of creating the preprocessing directory will take a considerable amount of time, so here we provide you with the full Root directory of the following datasets:
[LSA64]()<br>
[AUTSL]()<br>
[WLASL100]()<br>
[WLASL300]()<br>




