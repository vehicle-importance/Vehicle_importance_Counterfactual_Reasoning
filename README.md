# Object Importance Estimation Using Counterfactual Reasoning


**News**: \
**1.10.2023:** We released the code to generate the counterfactual trajectories. \
**22.09.2023:** We released the code to reproduce the results in the paper.


## [Project Page](http://vehicle-importance.github.io) | [Paper (Coming Soon)]() 

This repository provides code for the following paper:

- [Pranay Gupta](https://pranaygupta36.github.io), [Abhijat Biswas](https://www.cs.cmu.edu/~abhijatb/), [Henny Admoni](https://hennyadmoni.com/), [David Held](https://davheld.github.io/), Vehicle Importance Estimation using Counterfactual Reasoning, *Under Submission*  

<!-- ![demo]() -->

# Content
* [Setup](#setup)
* [Download HOIST Dataset](#Download-HOIST-Dataset)
* [Evaluation](#evaluation)
* [Generating True and Counterfactual Trajectories](#Generating-True-and-Counterfactual-Trajectories)
* [Citation *(Coming Soon)*]()



## Setup
First, you have to install the conda environment.

``` bash
git clone https://github.com/vehicle-importance/oiecr.git
cd oiecr
conda env create -f environment.yml
conda activate annotation
```


## Download HOIST Dataset 
You can download the HOIST dataset by executing:
``` bash
chmod +x download_hoist.sh
./download_hoist.sh
```
If this doesn't work, HOIST can be manually downloaded from [here](https://cmu.box.com/s/4j0g9hz9rimyctl1ftins0eq0cbyqgoq).
```bash
cp <path_to_HOIST.zip> ./
unzip HOIST.zip
```

You should see a directory structure like this.
```
.
├── annotations
│   ├── gt_annotations_xx.csv
│   ├── gt_annotations_xx1.csv
│   .
|   .
|   .
│   └── removed_pids.txt
├── data
│   ├── 14
│   │   ├── data
│   │   │   ├── xx.npy
|   |   |   .
│   │   │   ├── plant_data_xx.npy
|   |   |   .
│   │   └── recording_data
│   │       └── data
│   │           ├── xx.npy
|   |           |   .
│   ├── 19
│   ├── 2
│   ├── 24
│   ├── 31
│   └── 9
├── download_hoist.sh
├── environment.yml
├── get_consensus_score_pedestrians.py
├── get_consensus_score.py
├── HOIST.zip
├── images
│   ├── image_xx.png
│   ├── image_xx1.png
│   .
├── README.md
├── recordings
│   ├── RouteScenario_13_rep0.log
│   ├── RouteScenario_18_rep0.log
│   ├── RouteScenario_1_rep0.log
│   ├── RouteScenario_23_rep0.log
│   ├── RouteScenario_30_rep0.log
│   └── RouteScenario_8_rep0.log
├── videos
│   ├── video_xx0.webm
│   ├── video_xx1.webm
|   .
└── train_images.npy
```

1. The `annotations` directory contains csv files with the collected groud-truth annotations. The relevant headers in each csv file are : annotations (bounding boxes), image_name (name of the image), pid (annotator id), dl (whether they have driving license or not), exp (how many years have they been driving), habit (how many hours in a week do they drive), vehicle_ids (the CARLA ids of the annotated vehicles)

2. The `videos` directory contains the 409 RGB birds_eye_videos. Each video is named `video_<hand-assigned_id_for_interesting_cases>_<route_id>_<frame_number>.webm`

3. The `images` directory contains the last frame of the 409 RGB birds_eye_videos. Each image is named `image_<hand-assigned_id_for_interesting_cases>_<route_id>_<frame_number>.png`

4. The `recordings` directory contains the CARLA simulation recording of the 6 routes.

5. The `data` directory contains the true and the counterfactual trajectories of the vehicles for the removal score and the true and perturbed trajectories for the velocity perturbation score. It also contains the PlanT relevance scores.

6. `train_images.npy` contains the a small subset of the driving scenario used to develop the approach. The results in the paper are calculated using the remaining scenarios.

## Evaluation
1. Quantitative results comparing our method with the baseline methods for the task of object importance estimation on the **entire HOIST dataset** (Table I in the paper) and for **vehicles only** (Table III in the paper). We report Avg. Precision Score, Optimal Threshold F1 Score and Optimal Threshold Accuracy values.

    ```
    python get_consensus_score.py 
    ```

2. Quantitative results comparing our method with the baseline methods for **pedestrian** importance estimation (Table IV in the paper). 
    
    ```
    python get_consensus_score_pedestrians.py 
    ```

## Generating True and Counterfactual Trajectories
The code for generating true and coutefactual trajectories is [here](https://github.com/vehicle-importance/generating_counterfactual_trajectories/).

## Citation (Coming Soon)
