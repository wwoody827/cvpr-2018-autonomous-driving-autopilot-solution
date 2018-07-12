import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from pathlib import Path
import skimage.io
import tensorflow as tf
# os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

with open('../../settings.json') as f:
    setting = json.load(f)


# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)

DEFAULT_LOGS_DIR = os.path.join('../../', setting['LOGS_DIR'])

from mrcnn.config import Config
from mrcnn import utils as utils
from mrcnn import model as modellib

from imgaug import augmenters as iaa



############################################################
#  Configurations
############################################################
IMGSIZE = (1024, 1024)


class AdrivingConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Adriving"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 10
    VALIDATION_STEPS = 1
    
    # Number of classes (including background)
    NUM_CLASSES = 1 + 7  # Background + baloon
    MEAN_PIXEL = np.array([88.59672608, 95.91837699, 98.90089033])
    

    RPN_NMS_THRESHOLD = 0.6
    TRAIN_ROIS_PER_IMAGE = 600

    RPN_TRAIN_ANCHORS_PER_IMAGE = 320
    MAX_GT_INSTANCES = 80
    
    POST_NMS_ROIS_TRAINING = 4000
    POST_NMS_ROIS_INFERENCE = 2000
    
    DETECTION_NMS_THRESHOLD = 0.3
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_MAX_INSTANCES = 100
    
    IMAGE_MIN_DIM = IMGSIZE[0]
    IMAGE_MAX_DIM = IMGSIZE[1]
    IMAGE_RESIZE_MODE = "none"
    
    MASK_SHAPE = [28, 28]
    
    OPTIMIZER = 'SGD'
    LEARNING_RATE = 1e-6
    EPSILON = 1e-6
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
        iaa.Multiply((0.9, 1.1))
        # iaa.GaussianBlur(sigma=(0.0, 1.0))
    ])
    
    dataset_train = AdrivingDatasetNoResize()
    dataset_train.load_adriving(data_dir, "train", size = IMGSIZE)
    dataset_train.prepare()
    print(len(dataset_train.image_ids))

    # Validation dataset
    dataset_val = AdrivingDatasetNoResize()
    dataset_val.load_adriving(data_dir, "val", size = IMGSIZE)
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                augmentation=augmentation,
                epochs=40,
                layers='heads')
    
    model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            augmentation=augmentation,
            epochs=50,
            layers='4+')
    
    model.train(dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE/10,
        augmentation=augmentation,
        epochs=100,
        layers='3+')
    
    model.train(dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE/10,
        augmentation=augmentation,
        epochs=100,
        layers='all')
    
    
# --------------------------------------------------------------------------

############################################################
#  Training
############################################################

if __name__ == '__main__':
    config = AdrivingConfig()
    config.display()
    DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=os.path.join(DEFAULT_LOGS_DIR, 'mask_rcnn'))
    
    COCO_WEIGHTS_PATH = os.path.join('../../', setting['MODEL_CHECKPOINT_DIR'], 'mask_rcnn_coco.h5')

    
    weights_path = COCO_WEIGHTS_PATH
    model.load_weights(weights_path, by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])
    
    # weights_path = model.find_last()[1]
    # weights_path = 'weights/run1/mask_rcnn_adriving_aug_1024_1024_1e-4_0507_ep0052.h5'
    # model.load_weights(COCO_WEIGHTS_PATH, by_name=True)
    
    data_dir = Path(os.path.join('../../', setting['TEST_DATA_CLEAN_PATH'], 'train_val'))
    train(model)

