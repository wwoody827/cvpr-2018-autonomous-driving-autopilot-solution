import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from pathlib import Path
import skimage.io
import tensorflow as tf
import torch
os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Root directory of the project
with open('../../settings.json') as f:
    setting = json.load(f)
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)
DEFAULT_LOGS_DIR = os.path.join('../../', setting['LOGS_DIR'])

# Import Mask RCNN
sys.path.append(ROOT_DIR)

from config import Config
import utils
import model_resnet_v2 as modellib

from imgaug import augmenters as iaa

# Path to trained weights file


############################################################
#  Configurations
############################################################
IMGSIZE = (1024, 2048)


class AdrivingConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Adriving"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 32
    PRINT_ITER = 400
    
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 50
    VALIDATION_STEPS = 50
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 7  # Background + baloon
    MEAN_PIXEL = np.array([88.59672608, 95.91837699, 98.90089033])

    RPN_NMS_THRESHOLD = 0.7
    TRAIN_ROIS_PER_IMAGE = 320
    RPN_TRAIN_ANCHORS_PER_IMAGE = 500
    
    MAX_GT_INSTANCES = 100
    
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 2000
    
    DETECTION_NMS_THRESHOLD = 0.3
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_MAX_INSTANCES = 100
    
    IMAGE_MIN_DIM = IMGSIZE[0]
    IMAGE_MAX_DIM = IMGSIZE[1]
    IMAGE_RESIZE_MODE = "none"
    
    ROI_POSITIVE_RATIO = 0.5
    
    MASK_SHAPE = [56, 56]
    
    OPTIMIZER = 'Adam'
    LEARNING_RATE = 1e-5
    EPSILON = 1e-8
    GRADIENT_CLIP_NORM = 5
    ACCUM_ITERS = 1

############################################################
#  Dataset
############################################################
from adriving_util import *

def train(model):
    """Train the model."""
    # Training dataset.
    
    augmentation = iaa.SomeOf((0, 2), [
        iaa.Fliplr(0.5),
        iaa.Multiply((0.6, 1.1)),
        iaa.GaussianBlur(sigma=(0.0, 0.5))
    ])
    
    dataset_train = AdrivingDatasetNoResize()
    dataset_train.load_adriving(data_dir, "train", size = IMGSIZE)
    # dataset_train.load_adriving(data_dir, "train")
    dataset_train.prepare()
    print(len(dataset_train.image_ids))

    # Validation dataset
    dataset_val = AdrivingDatasetNoResize()
    dataset_val.load_adriving(data_dir, "val", size = IMGSIZE)
    #dataset_val.load_adriving(data_dir, "val")
    dataset_val.prepare()
    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
#     model.train_model(dataset_train, dataset_val,
#                 learning_rate=config.LEARNING_RATE/10,
#                 epochs=10,
#                 layers='heads')
    
    model.train_model(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE, augmentation= augmentation,
                epochs=200,layers='heads')
    model.train_model(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE/10, augmentation= augmentation,
                epochs=200,layers='4+')
# --------------------------------------------------------------------------

############################################################
#  Training
############################################################

if __name__ == '__main__':
    config = AdrivingConfig()
    config.display()
    DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0
    # with tf.device(DEVICE):
    model = modellib.MaskRCNN(config=config,
                              model_dir=os.path.join(DEFAULT_LOGS_DIR, 'pytorch_resnet_v2'))
    
    INIT_WEIGHTS_PATH = os.path.join('../../', 
                                     setting['MODEL_CHECKPOINT_DIR'],
                                     'mask_rcnn_adriving_resnetv2_0094.pth')

    weights_path = INIT_WEIGHTS_PATH
    print(weights_path)
    model.load_weights(weights_path, strict = True)
    model = model.cuda()

    data_dir = Path(os.path.join('../../', setting['TEST_DATA_CLEAN_PATH'], 'train_val'))
    train(model)

