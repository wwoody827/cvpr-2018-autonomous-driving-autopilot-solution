import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import PureWindowsPath as Path
import utils
import visualize
from visualize import display_images
import model_resnet_v2 as modellib
from model_resnet_v2 import log
import skimage.io
import skimage.transform
from config import Config
import pickle
from tqdm import tqdm
import torch

from adriving_util import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
with open('../../settings.json') as f:
    setting = json.load(f)

image_size_crop = (2048, 3384)
image_size = (2048, 3584)
SCALE = 1
FLIP = False
if SCALE == 2:
    image_size = (int(image_size[0]/2), int(image_size[1]/2))
else:
    image_size = (image_size[0], image_size[1])
    
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

    RPN_NMS_THRESHOLD = 0.8
    TRAIN_ROIS_PER_IMAGE = 1500
    RPN_TRAIN_ANCHORS_PER_IMAGE = 1000
    POST_NMS_ROIS_TRAINING = 4000
    POST_NMS_ROIS_INFERENCE = 2000

    IMAGE_MIN_DIM = image_size[0]
    IMAGE_MAX_DIM = image_size[1]
    
    IMAGE_RESIZE_MODE = "none"
    MEAN_PIXEL = np.array([88.59672608, 95.91837699, 98.90089033])
    DETECTION_MIN_CONFIDENCE = 0.1
    DETECTION_NMS_THRESHOLD = 0.3
    
    MASK_THRESHOLD = 0.5
    DETECTION_MAX_INSTANCES = 150
    
config = AdrivingConfig()

config.display()
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

import utils
import skimage.io
import skimage.transform
from skimage import data, img_as_float
from skimage import exposure

def crop_and_resize_test(image, scale = SCALE, flip = FLIP):
    img_crop = np.zeros([image_size[0], image_size[1], 3], dtype = np.float)
    img_roi = image[-image_size_crop[0]:, :, :]
    if scale == 2:
        img_resize = skimage.transform.resize(img_roi, (image_size_crop[0]/2, image_size_crop[1]/2),
                                                order=1, mode="constant", 
                                                preserve_range=True)
    else:
        img_resize = img_roi
    start_y = int((img_crop.shape[1] - img_resize.shape[1])/2)
        # print(start_y)
    img_crop[:, start_y:(start_y+img_resize.shape[1]), :] = img_resize
    if flip:
        img_crop = np.fliplr(img_crop)
    return img_crop


def load_test_image(image_filename, test_dir):
    if os.path.islink(str(test_dir/image_filename)):
        image_path = os.readlink(test_dir/image_filename)
    else:
        image_path = str(test_dir/image_filename)

    image = skimage.io.imread(image_path)
    image = crop_and_resize_test(image, scale = SCALE)
    return image

from scipy import sparse

def prediction_to_sparse(prediction, flip = FLIP):
    prediction_sparse = dict()
    prediction_sparse['rois'] = prediction['rois']
    prediction_sparse['class_ids'] = prediction['class_ids']
    prediction_sparse['scores'] = prediction['scores']

    prediction_sparse['masks'] = []
    for i in range(len(prediction['scores'])):
        if flip:
            mask = np.fliplr(prediction['masks'][:, :, i])
        else:
            mask = prediction['masks'][:, :, i]
            
        prediction_sparse['masks'].append(sparse.bsr_matrix(mask))
    return prediction_sparse
    

def predict(model, test_image, test_dir, results_folder, write_rle = False):
    file_name = results_folder + '.txt'
    if write_rle:
        with open(file_name, 'w+') as prediction_file:
            prediction_file.write('ImageId,LabelId,Confidence,PixelCount,EncodedPixels\n')

    for image_filename in tqdm(test_image, ncols = 50):
        image = load_test_image(image_filename, test_dir)
        image_id = image_filename[:-4]
        prediction = model.detect([image])[0]
        
        if prediction is None:
            continue
            
        if len(prediction['class_ids']) == 0:
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

if __name__ == '__main__':
    test_dir = Path(os.path.join('../../', setting['TEST_DATA_CLEAN_PATH'], 'test'))

    test_image = os.listdir(str(test_dir))
    test_image = [x for x in test_image if x[0] != '.']
    test_image.sort()

    MODEL_DIR = 'log'
    # with tf.device(DEVICE):
    model = modellib.MaskRCNN(model_dir=MODEL_DIR,
                              config=config)

    weights_path = os.path.join('../../', 
                                setting['MODEL_CHECKPOINT_DIR'], 
                                'mask_rcnn_adriving_resnetv2_0094.pth')

    print("Loading weights ", weights_path)
    model.load_weights(weights_path, strict=True)
    model = model.cuda()

    results_folder = 'pytorch_resnet_v2/test_resnet_v2_201806010_00_mask_th_' \
                        + str(config.MASK_THRESHOLD) \
                        + '_nms_th_' + str(config.RPN_NMS_THRESHOLD) \
                        + '_Scale_' + str(SCALE) + '_Flip_' + str(FLIP) \
                        + '_DT_NMS_TH_' + str(config.DETECTION_NMS_THRESHOLD)
    
    results_folder = os.path.join('../../', 
                              setting['SUBMISSION_DIR'], 
                              results_folder)
    print(results_folder)
    if not os.path.isdir(results_folder):
        os.makedirs(results_folder)
    predict(model, test_image, test_dir, results_folder, write_rle = False)
