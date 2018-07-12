import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import PureWindowsPath as Path
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
from mrcnn import model as modellib
from mrcnn.model import log
import skimage.io
import skimage.transform
from mrcnn.config import Config
import pickle
from tqdm import tqdm

from adriving_util import *

os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
image_size = (2048, 3584)

with open('../../settings.json') as f:
    setting = json.load(f)

class AdrivingConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "Adriving"

    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 7  # Background + baloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000

    RPN_NMS_THRESHOLD = 0.7
    TRAIN_ROIS_PER_IMAGE = 500
    RPN_TRAIN_ANCHORS_PER_IMAGE = 320
    POST_NMS_ROIS_TRAINING = 4000
    POST_NMS_ROIS_INFERENCE = 2000

    IMAGE_MIN_DIM = image_size[0]
    IMAGE_MAX_DIM = image_size[1]
    IMAGE_RESIZE_MODE = "none"
    MEAN_PIXEL = np.array([88.59672608, 95.91837699, 98.90089033])
    DETECTION_MIN_CONFIDENCE = 0.3


config = AdrivingConfig()
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
config = InferenceConfig()
config.display()
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

import mrcnn.utils as utils
import skimage.io
import skimage.transform
from skimage import data, img_as_float
from skimage import exposure

def crop_and_resize_test(image, contrast = False):
    img_crop = np.zeros([image_size[0], image_size[1], 3], dtype = np.float)
    img_roi = image[-image_size[0]:, :, :]
    if contrast:
        img_adapteq = exposure.equalize_adapthist(img_roi, clip_limit=0.01)
    else:
        img_adapteq = img_roi / 255.0
    img_adapteq = img_adapteq * 255.0
    img_crop[:, 72:(72+3384), :] = img_adapteq
    return img_crop


def load_test_image(image_filename, test_dir):
    if os.path.islink(str(test_dir/image_filename)):
        image_path = os.readlink(test_dir/image_filename)
    else:
        image_path = str(test_dir/image_filename)

    image = skimage.io.imread(image_path)
    image = crop_and_resize_test(image)
    return image

from scipy import sparse
def prediction_to_sparse(prediction):
    prediction_sparse = dict()
    prediction_sparse['rois'] = prediction['rois']
    prediction_sparse['class_ids'] = prediction['class_ids']
    prediction_sparse['scores'] = prediction['scores']

    prediction_sparse['masks'] = []
    for i in range(len(prediction['scores'])):
        prediction_sparse['masks'].append(sparse.bsr_matrix(prediction['masks'][:, :, i]))
    return prediction_sparse


def predict(model, test_image, test_dir, results_folder, write_rle = True):
    file_name = results_folder + '.txt'
    if write_rle:
        with open(file_name, 'w+') as prediction_file:
            prediction_file.write('ImageId,LabelId,Confidence,PixelCount,EncodedPixels\n')

    for image_filename in tqdm(test_image, ncols = 50):
        image = load_test_image(image_filename, test_dir)
        image_id = image_filename[:-4]
        prediction = model.detect([image], verbose=0)[0]
        if len(prediction['class_ids']) == 0:
            # prediction_file.write(image_id + ',' + '33, 1, 100,1 100|\n')
            continue
        prediction_sparse = prediction_to_sparse(prediction)
        with open(results_folder + '/' + image_id + '.p', 'wb') as f:
            pickle.dump(prediction_sparse, f)
        if write_rle:
            with open(file_name, 'a+') as prediction_file:
                mask_pred = np.zeros([2710, 3384, len(prediction['scores'])], dtype = bool)
                mask_pred[-image_size[0]:, :, :] = prediction['masks'][:, 72:(72+3384), :]
                mask, instance_score = instance_to_mask(mask_pred, prediction['class_ids'],
                                                          prediction['scores'])
                rle_string_list =  write_mask(image_id, mask, score = instance_score)
                for rle_str in rle_string_list:
                    prediction_file.write(rle_str)
                    prediction_file.write('\n')



def predict_rle(model, test_image, test_dir, file_name):
    with open(file_name, 'w') as prediction_file:
        prediction_file.write('ImageId,LabelId,Confidence,PixelCount,EncodedPixels\n')

    with open(file_name, 'a') as prediction_file:
        for image_filename in tqdm(test_image, ncols = 40):
            image = load_test_image(image_filename, test_dir)
            image_id = image_filename[:-4]
            prediction = model.detect([image], verbose=0)[0]
            if len(prediction['class_ids']) == 0:
                # prediction_file.write(image_id + ',' + '33, 1, 100,1 100|\n')
                continue
            mask, score = instance_to_mask(prediction['masks'], prediction['class_ids'], score = prediction['scores'])
            mask_original = np.zeros([2710, 3384], dtype = np.int)
            mask_original[-image_size[0]:, :] = mask[:, 72:(72+3384)]
            rle_string_list =  write_mask(image_id, mask_original, score = score)
            for rle_str in rle_string_list:
                prediction_file.write(rle_str)
                prediction_file.write('\n')

if __name__ == '__main__':
    mode = 'test'

    if mode == 'test':
        test_dir = Path(os.path.join('../../',setting['TEST_DATA_CLEAN_PATH'], "test"))
    else:
        test_dir = Path('../../data/train_full/val/image')

    test_image = os.listdir(str(test_dir))
    test_image = [x for x in test_image if x[0] != '.']
    test_image.sort()
    if mode == 'val':
        test_image = test_image[:100]

    MODEL_DIR = 'log'
    # with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

    weights_path = os.path.join('../../', 
                                setting['MODEL_CHECKPOINT_DIR'], 
                                'mask_rcnn_adriving_aug_1024_1024_1e-5_4p_0428.h5')

    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    results_folder = os.path.join('../../', 
                                  setting['SUBMISSION_DIR'], 
                                  'mask_rcnn/test_20180506_00')
    os.makedirs(results_folder)
    predict(model, test_image, test_dir, results_folder)
