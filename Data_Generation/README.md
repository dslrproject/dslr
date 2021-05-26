## Dataset
---
- Download the dataset using the following script.
```
bash ./Data_Generation/download_dataset.sh ati_dataset
```
- We create ARD-16 (Ati Realworld Dataset), a
first of its kind real-world paired correspondence dataset, by
applying our dataset generation method on 16-beam VLP-16
Puck LiDAR scans on a slow-moving Unmanned Ground
Vehicle. We obtain ground truth poses by using fine reso-
lution brute force scan matching, similar to Cartographer.
(Hess et al. 2016).

# Steps to convert Carla ply to npy to be fed to our lidar generation code:

Copy all the ipynb files from this folder.

Let's say you would like to generate "npys" for this path "/home/sabyasachi/Projects/ati/data/data/datasets/Carla/64beam-Data" which constains folders "static" and "dynamic" which themselves contain sub-folders numbered as "0", "1", ... and these sub folders contains all ply/pcd in "\_out" folder, corresponding non-player agent info in "\_nonplayers" and a groundtruth file called "groundTruth.csv".

Do the following:

1. Generate annotations for dynamic frames: In 1_Dynamic_Frame_Annotation.ipynb file, set BASE_PATH="/home/sabyasachi/Projects/ati/data/data/datasets/Carla/64beam-Data", check other params in the same cell and then run all the cells in the notebook. This will create a file called "annotation.csv" inside every subfolder which lists all pcds which are dynamic.

2. Get corresponding static dynamic pcd files: In 2_Static_Dynamic_PCD_Pairs_with_pose.ipynb file, set BASE_PATH again, check other params in the same cell and then run all the cells in the notebook. This will create a file called "pair_with_pose.csv" inside every subfolder which lists dynamic pcds and corresponding static pcd with paths and relative pose difference. It uses the annotations created in previous step.

3. Create a "pair" folder: In 3_Pair2Folder.ipynb filem set BASE_PATH again, check other params in the same cell and then run all the cells in the notebook. This will create a "pair" folder containing a "static" folder and "dynamic folder" which contains dynamic pcds in "dynamic" folder numbered by "1.pcd", "2.pcd", ... and corresponding static pcds with same corresponding numbering. It use pair file from previous step.

4. Create nps: In 4_ply2processednpy.ipynb file, generate npys for static and dynamic in the following two steps separately for "static" and "dynamic" folders:
    a. set BASE_PATH again and set PCD_FOLDER="static" and set appropriate BATCH_SIZE (more on this below), check other params in the same cell and then run all cells in the notebook. This will create static npys.
    b. set BASE_PATH again and set PCD_FOLDER="dynamic" and set appropriate BATCH_SIZE (more on this below), check other params in the same cell and then run all cells in the notebook. This will create dynamic npys.

BATCH_SIZE is the size of npy chunks i.e. no of pcds per npy. If you want one training npy and one validation npy, then set BATCH_SIZE=#pcds in training npy. If you can accomodate such large npy in your CPU ram then you can set a smaller value like BATCH_SIZE=2048.


Your static npys will be in the path : "/home/sabyasachi/Projects/ati/data/data/datasets/Carla/64beam-Data/pair/static_out_npy"
Your dynamic npys will be in the path : "/home/sabyasachi/Projects/ati/data/data/datasets/Carla/64beam-Data/pair/dynamic_out_npy"