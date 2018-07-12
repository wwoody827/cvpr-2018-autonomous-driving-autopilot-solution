import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing
from functools import partial

from pathlib import Path
import skimage.io
import scipy.ndimage as ndimage
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from adriving_util import *
from visualize import *
import skimage.morphology
import skimage.transform
from tqdm import tqdm
import pickle
import time

SC_TH = 0.3
PX_TH = 30
prediction_folder = 'test_20180521_00_Scale_1'
output_folder = prediction_folder + '_px_{}sc_{}'.format(PX_TH, SC_TH)

import skimage.morphology
def fill_and_remove(mask_instance):

    mask_instance_filled = mask_instance.copy()
    for n in range(mask_instance.shape[2]):
        # print(n)
        n_pixels = np.sum(mask_instance[:, :, n])
        if n_pixels > 5000:
            selem = skimage.morphology.square(5)
        else:
            selem = skimage.morphology.square(3)
        mask_instance_filled[:, :, n] = skimage.morphology.binary_opening(mask_instance[:, :, n], selem = selem)
        # mask_instance_filled[:, :, n] = skimage.morphology.binary_closing(mask_instance_filled[:, :, n], selem = selem)

    return mask_instance_filled


def read_and_rle(image_id, prediction_folder=prediction_folder,
                output_folder=output_folder, sc_th = 0.5, px_th = 30):
    with open(prediction_folder + '/'+ image_id + '.p', 'rb') as f:
        prediction = pickle.load(f)
    class_ids_pred = prediction['class_ids']
    scores_pred = prediction['scores']
    mask_pred = np.zeros([2710, 3384, len(scores_pred)], dtype = bool)

    for n in range(len(class_ids_pred)):
        h, w = prediction['masks'][0].toarray().shape
        if h == 2048:
            mask_pred[-2048:, :, n] = prediction['masks'][n].toarray()[:, 100:(100+3384)]
        else:
            mask_recover = prediction['masks'][n].toarray()
            mask_recover =  skimage.transform.resize(
                    mask_recover, (2*h, 2*w),
                    order=0, mode="constant", preserve_range=True)
            mask_pred[-2048:, :, n] = mask_recover[:, 100:(100+3384)]

    # pipeline:
    n_px_per_instance = np.sum(mask_pred, axis = (0, 1))
    instance_keep = np.where((n_px_per_instance > px_th) * (scores_pred > sc_th))[0]
    if len(instance_keep) == 0:
        return 0
    instance_reorder = instance_keep[np.argsort(scores_pred[instance_keep])]
    score_reorder = scores_pred[instance_reorder]
    class_ids_reorder = class_ids_pred[instance_reorder]
    mask_reorder = mask_pred[:, :, instance_reorder]
    # mask_reorder = fill_and_remove(mask_reorder)
    mask, instance_score = instance_to_mask(mask_reorder, class_ids_reorder,
                                                  score_reorder,
                                                  order_by_score = False)

    # pipeline_end
    rle_string_list = write_mask(image_id, mask, score = instance_score)
    fileoutput_name = os.path.join(output_folder, image_id + '.csv')
    with open(fileoutput_name, 'w+') as prediction_file:
        for rle_str in rle_string_list:
            prediction_file.write(rle_str)
            prediction_file.write('\n')
    return 0



if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on MS COCO.')

    parser.add_argument('--prediction', required=True,
                        metavar="test_20180507_00",
                        help='Directory of output dir')
    parser.add_argument('--px_th', required=False,
                        default=20,
                        metavar="<px>",
                        help='px')
    parser.add_argument('--sc_th', required=False,
                        default=0.5,
                        metavar="<sc>",
                        help='px')

    args = parser.parse_args()
    prediction_folder = args.prediction
    PX_TH = int(args.px_th)
    SC_TH = float(args.sc_th)

    output_folder = prediction_folder + '_px_{}sc_{}'.format(PX_TH, SC_TH)
    image_list = os.listdir(prediction_folder)
    image_list = [x[:-2] for x in image_list if x[0] != '.']
    image_list = sorted(image_list)

    print(len(image_list))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    n_core = min(multiprocessing.cpu_count(), 4)

    with multiprocessing.Pool(n_core) as p:
        r = list(tqdm(p.imap(partial(read_and_rle,
                            prediction_folder=prediction_folder,
                            output_folder=output_folder,
                            sc_th = SC_TH, px_th = PX_TH),
                            image_list),
                            total=len(image_list)))

    output_file = output_folder + '.csv'

    filenames = os.listdir(output_folder)
    filenames = [x for x in filenames if x[0] != '.']
    filenames = sorted(filenames)
    filenames = [os.path.join(output_folder, x) for x in filenames]
    outfile_list = []

    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                line = line[:-1]
                outfile_list.append(line.split(','))

    out_file = pd.DataFrame(outfile_list, columns = ['ImageId','LabelId','Confidence','PixelCount','EncodedPixels'])
    out_file['PixelCount'] = (out_file['PixelCount']).astype(int)
    out_file['LabelId'] = (out_file['LabelId']).astype(int)
    out_file['Confidence'] = (out_file['Confidence']).astype(float)
    out_file = out_file.loc[out_file.PixelCount > PX_TH]
    out_file = out_file.sort_values(['ImageId', 'Confidence'], ascending=[True, False])
    out_file.to_csv(output_file, index = False)
