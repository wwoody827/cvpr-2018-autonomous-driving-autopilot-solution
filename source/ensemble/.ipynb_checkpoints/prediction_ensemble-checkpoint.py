import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing
from functools import partial

from pathlib import Path
import skimage.io
import scipy.ndimage as ndimage
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '../pytorch-mask-rcnn'))

from adriving_util import *
from visualize import *
import skimage.morphology
import skimage.transform
from tqdm import tqdm
import pickle
import time

from unionfind import UnionFind

import json
with open('../../settings.json') as f:
    setting = json.load(f)

SC_TH = 0.5
PX_TH = 50
prediction_folder_list = []
output_folder = 'dummy'

def compute_iou_masks_partial_bck(masks1, mask2):
    '''Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width]
    '''
    # flatten masks
    masks1 = np.reshape(masks1, (-1, masks1.shape[-1])).astype(bool)
    mask2 = np.reshape(mask2, -1).astype(bool)

    area1 = np.sum(masks1, axis = 0) + 1e-3
    area2 = np.repeat(np.sum(mask2), masks1.shape[-1])
    # return area1, area2

    # intersections and union
    intersections = np.dot(masks1.T, mask2)
    union = area1 + area2 - intersections
    overlaps = intersections / area1

    return overlaps

def compute_iou_masks_partial(masks1, mask2):
    '''Computes IoU overlaps between two sets of masks.
    masks1, masks2: [Height, Width]
    '''
    # flatten masks
    # mask2 = np.repeat(mask2[:, :, np.newaxis], masks1.shape[2], axis=2)
    area1 = np.sum(masks1, axis = (0,1)) + 1e-3
    area2 = np.array([np.sum(mask2)] * masks1.shape[2])
    # return area1, area2

    # intersections and union
    intersections = []
    for i in range(masks1.shape[2]):
        intersections.append(np.sum(masks1[:, :, i] * mask2))
    intersections = np.array(intersections)
    # print(intersections)
    union = area1 + area2 - intersections
    overlaps = intersections / area1

    return overlaps

def compute_iou_masksets_partial(masks1, masks2):
    results = np.zeros([masks1.shape[-1], masks2.shape[-1]])
    for n in range(masks2.shape[-1]):
        # print(n)
        results[:, n] = compute_iou_masks_partial(masks1, masks2[:, :, n])
    return results

import scipy.ndimage.measurements
def remove_disconnected_instance(mask_this):
    mask_label, num_features = scipy.ndimage.measurements.label(mask_this)
    n_pixels = []
    if num_features <= 1:
        return mask_this

    for n in range(np.max(mask_label)):
        n_pixels.append(np.sum(mask_label==n))
    n_pixels = np.array(n_pixels)

    index = np.argmax(n_pixels)

    return (mask_label == n)

def remove_disconnected(mask_instance):
    mask_remove = np.zeros(mask_instance.shape, dtype = bool)
    for n in range(mask_instance.shape[2]):
        mask_remove[:, :, n] = remove_disconnected_instance(mask_instance[:, :, n])
    return mask_remove

def remove_duplicates_instance_to_mask(mask, class_ids, score, PX_TH = 20, SC_TH = 0.3):
    
    mask_resize = mask[::5, ::5, :]
    iou_matrix = compute_iou_masksets_partial(mask_resize, mask_resize)

    uf = UnionFind(list(range(mask.shape[2])))
    overlap = list()
    for i in range(mask.shape[2]):
        for j in range(mask.shape[2]):
            if i == j:
                continue
            else:
                if iou_matrix[i, j] > 0.8 and class_ids[i] == class_ids[j]:
                    uf.union(i, j)
                    overlap.append(i)
                    overlap.append(j)

    overlap = np.unique(overlap)

    keep = []
    for n in range(iou_matrix.shape[0]):
        if n not in overlap:
            keep.append(n)
    # print('keep', keep)

    mask_instance_new = mask[:, :, keep]
    class_ids_new = list(class_ids[keep])
    score_new = list(score[keep])

    merged_sets = []
    for n, pair in enumerate(uf.components()):
        if len(pair) >= 2:
            merged_sets.append(pair)


    mask_instance_merged = np.zeros([mask.shape[0],
                                     mask.shape[1], len(merged_sets)],
                                    dtype = bool)

    for n, pair in enumerate(merged_sets):
        mask_instance_merged[:, :, n] = np.zeros([mask.shape[0],
                                     mask.shape[1]],
                                    dtype = bool)
        scores_this_set = []
        index_this_set = []
        class_id_this_set = []
        px_num_this_set = []
        for p in pair:
            scores_this_set.append(score[p])
            index_this_set.append(p)
            class_id_this_set.append(class_ids[p])
            px_num_this_set.append(np.sum(mask[:, :, p], axis = (0, 1)))
            
        index = np.argmax(np.array(scores_this_set))
        mask_instance_merged[:, :, n] = mask[:, :, index_this_set[index]]

        class_ids_new.append(class_id_this_set[index])
        score_new.append(scores_this_set[index])

    # print('before', mask_instance_new.shape)
    mask_instance_new = np.dstack((mask_instance_new, mask_instance_merged))
    # mask_instance_new = mask_instance_merged
    class_ids_pred = np.array(class_ids_new)
    scores_pred = np.array(score_new)

    # print('after', mask_instance_new.shape)

    n_px_per_instance = np.sum(mask_instance_new, axis = (0, 1))
    instance_keep = np.where(np.logical_and((n_px_per_instance > PX_TH) , (scores_pred > SC_TH)))[0]
    if len(instance_keep) == 0:
        return None, None
    # print(instance_keep)
    instance_reorder = instance_keep[np.argsort(scores_pred[instance_keep])]
    # print(instance_reorder)
    score_reorder = scores_pred[instance_reorder]
    class_ids_reorder = class_ids_pred[instance_reorder]
    mask_reorder = mask_instance_new[:, :, instance_reorder]
    mask_reorder = remove_disconnected(mask_reorder)
    # print(mask_reorder.shape)
    # mask_reorder = fill_and_remove(mask_reorder)
    mask, instance_score = instance_to_mask(mask_reorder, class_ids_reorder,
                                                  score_reorder,
                                                  order_by_score = False)

    return mask, instance_score


import skimage.morphology

def fill_and_remove(mask_instance):
    selem = skimage.morphology.disk(3)
    mask_instance_filled = mask_instance.copy()
    for n in range(mask_instance.shape[2]):
        n_pixels = np.sum(mask_instance[:, :, n])
        if n_pixels > 5000:
            selem = skimage.morphology.disk(3)
        else:
            selem = skimage.morphology.disk(1)

    return mask_instance_filled

def get_prediction_from_csv(prediction, image_id):
    prediction_this_image = prediction.loc[prediction['ImageId'] == image_id]
    num_instances = len(prediction_this_image)
    mask_pred_list = list()
    # mask_pred = np.zeros([2710, 3384, num_instances], dtype = bool)
    class_ids_pred = np.zeros(num_instances, dtype = int)
    score_ids_pred = np.zeros(num_instances, dtype = float)
    for n in range(num_instances):
        rle = prediction_this_image.iloc[n]['EncodedPixels']
        mask_pred_list.append(rle_decode(rle, (2710, 3384)).astype(bool))
        # assert(np.sum(mask_pred[:,:, n]) == int(prediction_this_image.iloc[n]['PixelCount']))
        class_ids_pred[n] = label_to_class[int(prediction_this_image.iloc[n]['LabelId'])]
        score_ids_pred[n] = float(prediction_this_image.iloc[n]['Confidence'])
    return mask_pred_list, class_ids_pred, score_ids_pred

def get_prediction_and_csv(prediction_folder_list, image_id):
    class_ids_pred_list = []
    scores_pred_list = []
    mask_pred_list = []
    for prediction_folder in prediction_folder_list:
        if os.path.isdir(prediction_folder):
            if not os.path.isfile(prediction_folder + '/'+ image_id + '.p'):
                continue
            with open(prediction_folder + '/'+ image_id + '.p', 'rb') as f:

                prediction = pickle.load(f)
                #print(prediction)
                class_ids_pred = prediction['class_ids']
                class_ids_pred_list += class_ids_pred.tolist()
                scores_pred = prediction['scores']
                scores_pred_list += scores_pred.tolist()

                for n in range(len(class_ids_pred)):
                    mask_pred = np.zeros([2710, 3384], dtype = bool)
                    h, w = prediction['masks'][0].toarray().shape
                    if h == 2048:
                        mask_pred[-2048:, :] = prediction['masks'][n].toarray()[:, 100:(100+3384)]
                    else:
                        mask_recover = prediction['masks'][n].toarray()
                        mask_recover =  skimage.transform.resize(
                                mask_recover, (2*h, 2*w),
                                order=0, mode="constant", preserve_range=True)
                        mask_pred[-2048:, :] = mask_recover[:, 100:(100+3384)]
                    mask_pred_list.append(mask_pred.astype(bool))
        else:
            prediction_file = pd.read_csv(prediction_folder)
            image_list = list(np.unique(prediction_file.ImageId))
            if image_id not in (image_list):
                continue
            mask_pred, class_ids_pred, scores_pred = get_prediction_from_csv(prediction_file, image_id)
            class_ids_pred_list += class_ids_pred.tolist()
            scores_pred_list += scores_pred.tolist()
            mask_pred_list += mask_pred

    return np.stack(mask_pred_list, axis = -1), np.array(class_ids_pred_list), np.array(scores_pred_list)


def read_and_rle(image_id, prediction_folder_list=prediction_folder_list,
                output_folder=output_folder, SC_TH = 0.3, PX_TH = 20):
    mask_pred, class_ids_pred, scores_pred = get_prediction_and_csv(prediction_folder_list, image_id)
    # pipeline:
    n_px_per_instance = np.sum(mask_pred, axis = (0, 1))
    instance_keep = np.where(np.logical_and((n_px_per_instance > PX_TH) , (scores_pred > SC_TH)))[0]
    # print(instance_keep)
    if len(instance_keep) == 0:
        return 0

    instance_reorder = instance_keep[np.argsort(scores_pred[instance_keep])]

    score_reorder = scores_pred[instance_reorder]
    class_ids_reorder = class_ids_pred[instance_reorder]
    mask_reorder = mask_pred[:, :, instance_reorder]

    # mask_reorder = fill_and_remove(mask_reorder)
    # print(class_ids_reorder)
    mask, instance_score = remove_duplicates_instance_to_mask(mask_reorder, class_ids_reorder,
                                                                score_reorder,
                                                                SC_TH = SC_TH, PX_TH = PX_TH)

    # pipeline_end
    if mask is not None:
        rle_string_list = write_mask(image_id, mask, score = instance_score)
        fileoutput_name = output_folder + '/'+ image_id + '.csv'
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

    parser.add_argument('--px_th', required=False,
                        default=50,
                        metavar="<px>",
                        help='px')
    parser.add_argument('--sc_th', required=False,
                        default=0.1,
                        metavar="<sc>",
                        help='px')

    args = parser.parse_args()

    PX_TH = int(args.px_th)
    SC_TH = float(args.sc_th)

    # prediction_folder_list = ['test_resnet_v2_201806010_00_mask_th_0.5_nms_th_0.8_Scale_1_Flip_False_DT_NMS_TH_0.3', 
    #                           'test_20180521_00_mask_th0.35_nms_th_0.6_Scale_1Flip_False',
    #                           'test_20180521_00_mask_th0.35_nms_th_0.6_Scale_1Flip_True',
    #                           'test_20180506_01_px_30sc_0.5.csvpx20_th0.5.csv' ]
    prediction_folder_list = os.listdir(os.path.join('../../', setting['SUBMISSION_DIR'], 'ensemble'))
    prediction_folder_list = [x for x in prediction_folder_list if not x.startswith('.')]
    prediction_folder_list = [os.path.join('../../', setting['SUBMISSION_DIR'], 'ensemble', x) for x in prediction_folder_list]
    print(prediction_folder_list)
    # output_folder = ''
    # for prediction_folder in prediction_folder_list:
    #     output_folder += prediction_folder + '_'
    # output_folder = output_folder + '_px_{}sc_{}_ensemble_keep'.format(PX_TH, SC_TH)
    output_folder = 'final_prediction' + '_px_{}sc_{}'.format(PX_TH, SC_TH)
    output_folder = os.path.join('../../', setting['SUBMISSION_DIR'], output_folder)

    os.makedirs(output_folder)

    # output_folder = prediction_folder + '_px_{}sc_{}'.format(PX_TH, SC_TH)
    image_list = []
    for prediction_folder in prediction_folder_list:
        if os.path.isdir(prediction_folder):

            image_list_this = os.listdir(prediction_folder)
            image_list_this = [x[:-2] for x in image_list_this if x[0] != '.']
        else:
            prediction_file = pd.read_csv(prediction_folder)
            image_list_this = list(np.unique(prediction_file.ImageId))
        image_list += image_list_this
    image_list = list(set(image_list))
    image_list = sorted(image_list)
    print(len(image_list))
    # image_list = image_list[195:]

    n_core = min(multiprocessing.cpu_count(), 4)

    with multiprocessing.Pool(n_core) as p:
        r = list(tqdm(p.imap(partial(read_and_rle,
                            prediction_folder_list=prediction_folder_list,
                            output_folder=output_folder,
                            SC_TH = SC_TH, PX_TH = PX_TH),
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
