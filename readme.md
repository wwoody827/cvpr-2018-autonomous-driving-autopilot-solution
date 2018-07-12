Hello!

Below you can find a outline of how to reproduce my solution for the cvpr-2018-autonomous-driving competition.
I tested on my local setup. But due to time limits I did not re-train all models and re-generating all predictions I made. If you run into any trouble with the setup/code or have any questions please contact me at wwoody827@gmail.com

# ARCHIVE CONTENTS
pass

# HARDWARE: (The following specs were used to create the original solution)
You need a gpu to run this code, since some functions are cuda only.

* Ubuntu 16.04 LTS (2TB SSD boot disk)
* 1 x NVIDIA 1080Ti 
* 1 x NVIDIA 1070 (powerful enough)
* 16GB RAM


# SOFTWARE (python packages are detailed separately in `requirements.txt`):
* Python 3.6
* CUDA 8.0
* cuddn 7.0.5



# DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)
## Below are the shell commands used in each step, as run from the top level directory
```shell
mkdir data/
kaggle competitions download -c cvpr-2018-autonomous-driving -p data
```

then unzip all data file to data/ dir with your own zip program


# DATA PROCESSING

**The train/predict code will NOT call this script if it has not already been run on the relevant data.**

This script will split data into training set and validation set. To make training faster, I only kept training set with at least 20 instances.
```shell
cd source/preprocess/
python keep_instance_larger_than_threshod.py
```

This will copy training images to TRAIN_DATA_CLEAN_PATH/train_val and split them into train and val

This is no preprocessing for test set.

# MODEL BUILD: 

There are two sub models to produce the solution, all based on Mask-RCNN. One is a keras implements and the other one is pytorch implements. The final solution is the ensemble of multiple solutions.

## Mask RCNN based on Keras and tensorflow

### Training:
Mack RCNN code is modified from https://github.com/matterport/Mask_RCNN. 
Before you run the training script, make sure you have pretrained model 'mask_rcnn_coco.h5' in MODEL_CHECKPOINT_DIR. The init weights pretrained on COCO is also from https://github.com/matterport/Mask_RCNN.


```shell
cd source/mask_rcnn
python train_mrcnn.py
```

Training a model will take about 3 or 4 days on a GTX 1070 gpu. When training on all layers, I used a P100 instance on GCP. Model files and checkpoints will be saved to LOGS_DIR dir. To make predictions, you should place trained checkpoints to MODEL_CHECKPOINT_DIR.

### Prediction
#### generating predictions
To make predictions, one should use the following command:
```shell
cd source/mask_rcnn
python prediction.py
```

Running this will read a trained weights ('mask_rcnn_adriving_aug_1024_1024_1e-5_4p_0428.h5') from MODEL_CHECKPOINT_DIR and test images from ('TEST_DATA_CLEAN_PATH'), and save predictions results to SUBMISSION_DIR. It will create at results dir under SUBMISSION_DIR and a prediction file. The folder contains raw predictions for each image and prediction file contained run-length-encoded results for each image.

Prediction will take 4 to 5 hours on my machine with GTX 1080Ti and i7-4770K. Most of the time was taken in run-lenght-encoding.


#### post processing
To get final prediction, I removed some small objects and low confidence objects from predictions. They are filtered by setting a threshold for number of pixels and threshold for scores for each instance. This is done in Microsoft Excel.



## Mask RCNN based on pytorch

### Training:
Pytorch implements is based on https://github.com/multimodallearning/pytorch-mask-rcnn, with heavliy modifications. The initial weights are converted from 'mask_rcnn_adriving_aug_1024_1024_1e-5_4p_0428.h5'.

Before running the code, you should compile some functions:
(You can refere to https://github.com/multimodallearning/pytorch-mask-rcnn for more details.)

(Assume you are in source/pytorch-mask-rcnn)

```shell
 cd nms/src/cuda/
 nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch=[arch]
 cd ../../
 python build.py
 cd ../

 cd roialign/roi_align/src/cuda/
 nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch=[arch]
 cd ../../
 python build.py
 cd ../../
 ```
 
After that you can go to source/pytorch-mask-rcnn and run the following command for training, assume you've downloaded coresponding weigiths and put them in MODEL_CHECKPOINT_DIR.

```shell
python train_resnet.py
python train_resnet_v2.py
python train_seresnext.py
python train_resnext101.py
```

### Prediction:
Since my implementation of RLE is very slow even with multiprocessing, I seperate rle with model prediction. Running the following command can generate a result dir containing compressed predictions for each test image (if their predictions are not empty, in which case the empty prediction will be skipped). The parameters used for my final submission have been hardcoded into the python file. I did not use seresenext and resnext101 for submission.

```shell
python predict_resnet.py
python predict_resnet_v2.py
```

Then, to get run-length-encoded results you should run prediction_rle_mp.py.

## Ensembling

My final submission are the ensemble of 4 different results, from 'mask_rcnn_adriving_aug_1024_1024_1e-5_4p_0428.h5', 'mask_rcnn_adriving_0521_1e-5_bz_32_0.82.pth'(x2, with horizontal flip as TTA), 'mask_rcnn_adriving_resnetv2_0094.pth'. You should have move all results dirs to SUBMISSION_DIR/ensemble and run prediction_ensemble.py under source/ensemble. ~~Note result dirs needed for ensembling are hardcoded into the python script, so you may need to change that if you have different results folder names.~~ It will perform postprocessing and ensembling and out put filtered results to SUBMISSION_DIR. It will take more than 20h even with multiprocessing on a laptop with i7-6700HQ and 16G RAM.



