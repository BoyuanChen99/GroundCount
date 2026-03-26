# GroundCount
This repo stores the major code for implementing experiments in our paper [GroundCount: Grounding Vision-Language Models with Object Detection for Mitigating Counting Hallucinations](https://arxiv.org/abs/2603.10978). 


## Setting Up Environment
First create a virtual environment. Then, read requirements.txt. Install `ultralytics` by source code and THEN `pip install -r requirements.txt`. The object detection model (ODM) YOLOv13x will be automatically ddownloaded during evaluation. 


## Preparing Datasets

### MSCOCO
Download MSCOCO 2014 and 2017 - all train, val and annotations. No need to download other years or test sets. This is the image source for most of our experiments. Please copy and save the path where you put it.

To preprocess the COCO dataset, cd to `scripts/preprocess`, and run the following files in sequence. Note that you must change the paths in these directories first. 
- coco_sort.py
- coco_categories.py
- change_path.py


### [PhD](https://github.com/jiazhen-code/PhD) (CVPR'25)
PhD is by Sep 2025, the biggest benchmark dataset on VLM hallucination, including 15,398 distinct images 102,564 VQA triplets. Most images are from MSCOCO, while a small portion are customly generated. Download their "data.json".

Now, go to `/scripts/preprocess`, and run the following scripts in sequence: 
- `phd.py`
- Then run `coco_count.py` to generate COCO-Count dataset for training the fuser model. This could take a while, as coco/train2017 has 118k images.


## Reimplementation - Data Preparation, Training and Evaluation

### Prepare dataset: COCO-Count and PhD
First, go to COCO official website to download COCO2017 train and val images (https://cocodataset.org/#download), as well as annotations. If using HPC, you should create a singularity. 

Then run `/scripts/preprocess/coco_count.py` to generate COCO-Count dataset for training the fuser model. This could take a while, as coco/train2017 has 118k images, and 343k vqa tuples. 

For our evaluation benchmark - PhD, download their "data.json" from [their github repo](https://github.com/jiazhen-code/PhD). 


### Train fusion model using COCO-Count
Go to `scripts/train/train_fuser.py`. Set the data path to your COCO-Count dataset path. Adjust other hyperparameters if needed. Then run the training script. 

Note that the current evaluation is based on a previous round of baseline result. We set the default val size to 0, so that the evaluation is skipped during training. You may manually add it by setting the paths correctly. 


### Evaluate on PhD benchmark
Go to `scripts/benchmark`. Pick the script you want to test on, and adjust the paths accordingly, including the fusion model path, COCO dataset path, and PhD dataset path. Then run the evaluation script.

To get accuracy, run `scripts/analysis/judge_phd.py`. Adjust the paths accordingly, including the PhD dataset path and the evaluation result path. Then run the script to get accuracy.

### Visualize
We also provide a vanilla GUI for visualizing PhD results. To launch the GUI, run `scripts/analysis/gui.py` and follow the on-screen instructions.



## COCO Full Annotations Anatomy
- **captions**:
    - info: 
        - description: "COCO 2014 Dataset"
        - url: http://cocodataset.org
        - version: 1.0
        - year: 2014
        - contributor: COCO Consortium
        - date_created: 2017/09/01
    - licenses: 
        - url: http://creativecommons.org/licenses/by-nc-sa/2.0/
        - id: 1
        - name: Attribution-NonCommercial-ShareAlike License
    - images: 
        - license: 4
        - file_name: 000000397133.jpg
        - coco_url: http://images.cocodataset.org/val2017/000000397133.jpg
        - height: 427
        - width: 640
        - date_captured: '2013-11-14 17:02:52'
        - flickr_url: http://farm7.staticflickr.com/6116/6255196340_da26cf2c9e_z.jpg
        - id: 397133
    - annotations: 
        - image_id: 203564
        - id: 37
        - caption: A bicycle replica with a clock as the front wheel.


- **image_info** (only for test sets): 
    - info: (same as captions)
    - licenses: (same as captions)
    - images: (same as captions)
    - categories: (a library of all objects. test2014 has 90 categories.)
        - supercategory: person
        - id: 1
        - name: person




- **instances** (intentionally unsorted, and each image could have multiple instances)
    - info: (same as captions)
    - licenses: (same as captions)
    - images: (same as captions)
    - categories: (same as image_info)
    - annotations: 
        - segmentation: [polygon]
        - area: 702.0
        - iscrowd: 0
        - image_id: 244654
        - bbox: [x,y,width,height]
        - category_id: 18
        - id: 1768




- **panoptic** (2017 only)
    - info: (same as captions)
    - licenses: (same as captions)
    - images: (same as captions)
    - categories: (same as image_info)
    - annotations:
        - segments_info:
            - id: 3226956 (a unique dataset-wise integer identifier of this particular segment within the image’s panoptic mask.)
            - category_id: 1
            - iscrowd: 0
            - bbox: [413, 158, 53, 138]
            - area: 2840
            - ...
        - file_name: 000000000139.png
        - image_id: 139




- **person_keypoints**
    - info: (same as captions)
    - licenses: (same as captions)
    - images: (same as captions)
    - categories: (same as image_info)
    - annotations:
        - segmentation: [polygon]
        - num_keypoints: (0-17)
        - area: 28292.08625
        - iscrowd: 0/1
        - keypoints: Al length of 3x17=51, in the order of [x1, y1, v1, x2, y2, v2,...], where vi is the visibility flag of each keypoint. 
            - v=0: not labeled (in which case x=y=0)
            - v=1: labeled but not visible
            - v=2: labeled and visible
        - image_id: 244654
        - bbox: [x,y,width,height]
        - category_id: 1 (person)
        - id: 1768 (a unique dataset-wise integer identifier of this particular segment within the image’s panoptic mask.)
    
    COCO defines 17 keypoints for each person, corresponding to:

    0: nose
    1: left_eye
    2: right_eye
    3: left_ear
    4: right_ear
    5: left_shoulder
    6: right_shoulder
    7: left_elbow
    8: right_elbow
    9: left_wrist
    10: right_wrist
    11: left_hip
    12: right_hip
    13: left_knee
    14: right_knee
    15: left_ankle
    16: right_ankle


- **stuff** (2017 only)
    - info: (same as captions)
    - licenses: (same as captions)
    - images: (same as captions)
    - categories: (same as image_info)
    - annotations:
        - segmentation: [polygon]
        - area: (same as above)
        - iscrowd: 0/1
        - image_id: (same as above)
        - bbox: (same as above)
        - category_id: (save as above)
        - id: (same as above)