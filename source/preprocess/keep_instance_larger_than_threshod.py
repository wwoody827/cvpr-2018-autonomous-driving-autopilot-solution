import os
import json
import numpy as np
from matplotlib import pyplot as plt
threshod = 20
from shutil import copyfile
from tqdm import tqdm
from pathlib import Path

with open('../../settings.json') as f:
    setting = json.load(f)

train_imageid_ninstance = dict()
with open("train_imageid_ninstance.txt") as f:
    for line in f:
        (key, val) = line.split(',')
        train_imageid_ninstance[key] = int(val)

val_imageid_ninstance = dict()
with open("val_imageid_ninstance.txt") as f:
    for line in f:
        (key, val) = line.split(',')
        val_imageid_ninstance[key] = int(val)
        
        
train_color_dir = Path(os.path.join('../../' , setting['RAW_DATA_DIR'] , 'train_color'))
train_label_dir = Path(os.path.join('../../' , setting['RAW_DATA_DIR'] , 'train_label'))

output_dir =  Path(os.path.join('../../', setting['TRAIN_DATA_CLEAN_PATH'], 'train_val'))

for mode in ['train', 'val']:
    if mode == 'train':
        image_dict = train_imageid_ninstance
    else:
        image_dict = val_imageid_ninstance
    output_dir_image = output_dir / mode / 'image'
    output_dir_label = output_dir / mode / 'label'
    
    output_dir_image.mkdir(parents=True, exist_ok=True)
    output_dir_label.mkdir(parents=True, exist_ok=True)
    
    for image_id, value in tqdm(image_dict.items()):
        if value >= threshod:
            copyfile(train_color_dir/(image_id + '.jpg'), output_dir_image/(image_id + '.jpg'))
            copyfile(train_label_dir/(image_id + '_instanceIds.png'), output_dir_label/(image_id + '_instanceIds.png'))
