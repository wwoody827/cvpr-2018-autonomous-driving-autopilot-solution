from mrcnn import utils
from pathlib import Path
import numpy as np
import math
import skimage.io
import os

label_to_name = {33: 'car', 
                 34: 'motorbicycle', 35: 'bicycle', 36: 'person', 
                 38: 'truck', 39: 'bus', 40: 'tricycle'} 

# label_to_class = {33:1, 34:2, 35:3, 36:4, 38:5, 39:6, 40:7}
label_to_class = {33:1, 34:2, 35:3, 36:4, 38:5, 39:6, 40:7,
                 161:1, 162:2, 163:3, 164:4, 166:5, 167:6, 168:7}
class_to_label = {1:33, 2:34, 3:35, 4:36, 5:38, 6:39, 7:40}

def get_label_name(image_name):
    label_name = image_name[:-4]
    label_name += '_instanceIds.png'
    return label_name

def contain_instance(mask):
    mask_int = np.floor(mask/1000).astype(int)
    instances = np.unique(mask_int)
    instance_in = []
    for instance in instances:
        if instance in label_to_name.keys():
            instance_in.append(instance)
        
    if len(instance_in) > 0:
        return True
    else:
        return False

def filelink(link):
    if os.path.islink(link): 
        return(os.readlink(link))
    else:
        return(link)

def mask_to_instance(mask):
        instances_all = np.unique(mask)
        instances = []
        for i in range(len(instances_all)):
            instance = instances_all[i]
            if np.floor(instance/1000).astype(int) in label_to_name.keys():
                instances.append(instance) 
        instances = np.array(instances)
        num_instance = len(instances)
        mask_out = np.zeros([mask.shape[0], mask.shape[1], num_instance], dtype=bool)
        class_ids = np.zeros(num_instance)
        for i in range(num_instance):
            instance = instances[i] 
            class_ids[i] = label_to_class[np.floor(instance/1000).astype(int)]
            mask_out[:, :, i] = (mask == instance)
        return mask_out, class_ids.astype(np.int32)
    
class AdrivingDataset(utils.Dataset):
    def mask_to_instance(self, mask):
        instances_all = np.unique(mask)
        instances = []
        for i in range(len(instances_all)):
            instance = instances_all[i]
            if np.floor(instance/1000).astype(int) in label_to_name.keys():
                instances.append(instance) 
        instances = np.array(instances)
        num_instance = len(instances)
        mask_out = np.zeros([mask.shape[0], mask.shape[1], num_instance], dtype=bool)
        class_ids = np.zeros(num_instance)
        for i in range(num_instance):
            instance = instances[i] 
            class_ids[i] = label_to_class[np.floor(instance/1000).astype(int)]
            mask_out[:, :, i] = (mask == instance)
        return mask_out, class_ids.astype(np.int32)
    
    def load_adriving(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        dataset_dir = Path(dataset_dir)
        # Add classes. We have only one class to add.
        for key, name in label_to_name.items():
            if key != 0:
                class_id = label_to_class[key]
                self.add_class("adriving", class_id, name)

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = dataset_dir / subset
        self.dataset_dir = dataset_dir
        
        train_image_dir = dataset_dir/ 'image'
        
        color_images = os.listdir(train_image_dir)
        for img in color_images:
            if img[0] == '.':
                continue
            self.add_image(
                "adriving",
                image_id=img,  # use file name as a unique image id
                path = train_image_dir / img,
            )

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        img_path = self.image_info[image_id]['path']
        mask_path_filename = utils.get_label_name(os.path.basename(img_path))
        mask_path = self.dataset_dir / 'label' / mask_path_filename
        # [height, width, instance_count]
        mask_raw = np.array(skimage.io.imread(mask_path))
        mask, class_ids = self.mask_to_instance(mask_raw)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask, class_ids
    
class AdrivingDatasetNoResize(utils.Dataset):
    
    def mask_to_instance(self, mask):
        instances_all = np.unique(mask)
        instances = []
        for i in range(len(instances_all)):
            instance = instances_all[i]
            if np.floor(instance/1000).astype(int) in label_to_class.keys():
                instances.append(instance) 
        instances = np.array(instances)
        num_instance = len(instances)
        mask_out = np.zeros([mask.shape[0], mask.shape[1], num_instance], dtype=bool)
        class_ids = np.zeros(num_instance)
        for i in range(num_instance):
            instance = instances[i] 
            class_ids[i] = label_to_class[np.floor(instance/1000).astype(int)]
            mask_out[:, :, i] = (mask == instance)
        return mask_out, class_ids.astype(np.int32)
    
    def load_adriving(self, dataset_dir, subset, size = (768, 768)):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        dataset_dir = Path(dataset_dir)
        # Add classes. We have only one class to add.
        for key, name in label_to_name.items():
            if key != 0:
                class_id = label_to_class[key]
                self.add_class("adriving", class_id, name)

        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = dataset_dir / subset
        self.dataset_dir = dataset_dir
        self.size = size
        
        train_image_dir = dataset_dir / 'image'
        
        color_images = os.listdir(train_image_dir)
        for img in color_images:
            if img[0] == '.':
                continue
            self.add_image(
                "adriving",
                image_id=img,  # use file name as a unique image id
                path = filelink(train_image_dir / img),
            )
            
    def peek_mask(self, image_id):
        # If not a balloon dataset image, delegate to parent class.
        img_path = self.image_info[image_id]['path']
        mask_path_filename = utils.get_label_name(os.path.basename(img_path))
        if os.path.islink(self.dataset_dir / 'label' / mask_path_filename): 
            mask_path = os.readlink(self.dataset_dir / 'label' / mask_path_filename)
        else:
            mask_path = self.dataset_dir / 'label' / mask_path_filename
            
        # [height, width, instance_count]
        mask_raw = np.array(skimage.io.imread(mask_path))
        return mask_raw
    
    def load_image(self, image_id):
        # Load image
        image = skimage.io.imread(self.image_info[image_id]['path'])
        mask_raw = self.peek_mask(image_id)
        original_size = (image.shape[0], image.shape[1])
        crop_size = self.size
        crop_max = (original_size[0] - crop_size[0], 
                    original_size[1] - crop_size[1])
        
        for i in range(100):
            crop_start = (np.random.randint(800, crop_max[0]), 
                          np.random.randint(crop_max[1]))
            mask_crop = mask_raw[crop_start[0]:crop_start[0]+self.size[0], 
                           crop_start[1]:crop_start[1]+self.size[1]]
            if contain_instance(mask_crop):
                break
        
        image_crop = image[crop_start[0]:crop_start[0]+self.size[0], 
                           crop_start[1]:crop_start[1]+self.size[1],
                           :]
        self.image_info[image_id]['crop_start'] = crop_start
        return image_crop
    
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a balloon dataset image, delegate to parent class.
        img_path = self.image_info[image_id]['path']
        mask_path_filename = utils.get_label_name(os.path.basename(img_path))
        
        if os.path.islink(self.dataset_dir / 'label' / mask_path_filename): 
            mask_path = os.readlink(self.dataset_dir / 'label' / mask_path_filename)
        else:
            mask_path = self.dataset_dir / 'label' / mask_path_filename
            
        # [height, width, instance_count]
        mask_raw = np.array(skimage.io.imread(mask_path))
        crop_start = self.image_info[image_id]['crop_start']
        mask_crop = mask_raw[crop_start[0]:crop_start[0]+self.size[0], 
                           crop_start[1]:crop_start[1]+self.size[1]]
        mask, class_ids = self.mask_to_instance(mask_crop)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask, class_ids
    
    
def instance_to_mask(instance_mask, class_ids, score = None):
    mask = np.zeros([instance_mask.shape[0], instance_mask.shape[1]], dtype = np.int)
    n_instances = instance_mask.shape[2]
    class_added = np.zeros(7)
    instance_score = dict()
    for n in range(n_instances):
        class_this = class_ids[n]
        label_this = class_to_label[class_this]
        mask_instance = int(label_this * 1000 + class_added[class_this - 1])
        class_added[class_this - 1] += 1
        loc = np.where(instance_mask[:, :, n])
        mask[loc] = mask_instance
        if score is not None:
            instance_score[mask_instance] = score[n]
    if score is not None:
        return mask, instance_score
    else:
        return mask
    
def instance_to_mask_score(instance_mask, class_ids, score, score_th = 0.0, px_th = 0):
    mask = np.zeros([instance_mask.shape[0], instance_mask.shape[1]], dtype = np.int)
    n_instances = instance_mask.shape[2]
    class_added = np.zeros(7)
    instance_score = dict()
    order = np.argsort(score)
    # print(order)
    for n in order:
        if score[n] < score_th or np.sum(instance_mask[:, :, n]) < px_th:
            continue
        class_this = class_ids[n]
        label_this = class_to_label[class_this]
        mask_instance = int(label_this * 1000 + class_added[class_this - 1])
        class_added[class_this - 1] += 1
        loc = np.where(instance_mask[:, :, n])
        mask[loc] = mask_instance
        if score is not None:
            instance_score[mask_instance] = score[n]
    if score is not None:
        return mask, instance_score
    else:
        return mask
    

def recover_cropped(mask, original_size = (2710, 3384)):
    mask_crop = mask[:, 50:1742]
    mask_original = np.zeros(original_size, dtype = np.int)
    mask_resize = skimage.transform.resize(
                    mask_crop, (1024, 3384),
                    order=0, mode="constant", preserve_range=True)
    mask_original[-1324:-300, :] = mask_resize
    return mask_original

def rle_encode(instance_map):
    check = 0
    begin = 0
    length = 0
    idmap1d = np.reshape(instance_map,(-1))
    InstanceIds = np.unique(instance_map)
    Totalcount = np.sum(idmap1d)
    rle_string = "{},".format(Totalcount)
    
    find = False
    for index in range(idmap1d.shape[0]):
        if find:
            if idmap1d[index]:
                length = length+1
            else:
                rle_string += "{} {}|".format(begin,length)
                check = check + length
                length = 0
                find = False
        else:
            if idmap1d[index]:
                begin = index
                length = 1
                find = True
        if index == idmap1d.shape[0]-1 and find:
            rle_string +=  "{} {}|".format(begin,length)
            check = check + length
            length = 0
            find = False
    return rle_string

def rle_encode_fast(instance_map, score = 1.0):
    """Encodes a mask in Run Length Encoding (RLE).
    Returns a string of space-separated values.
    """
    assert instance_map.ndim == 2, "Mask must be of shape [Height, Width]"
    # Flatten it column wise
    m = np.reshape(instance_map,(-1))
    Totalcount = np.sum(m)
    # Compute gradient. Equals 1 or -1 at transition points
    g = np.diff(np.concatenate([[0], m, [0]]), n=1)
    # 1-based indicies of transition points (where gradient != 0)
    rle = np.where(g != 0)[0].reshape([-1, 2])
    # Convert second index in each pair to lenth
    rle[:, 1] = rle[:, 1] - rle[:, 0]
    rle_string = "{:4.3f},{},".format(score, Totalcount)
    check = 0
    for n in range(len(rle)):
        rle_string += "{} {}|".format(rle[n, 0],rle[n, 1])
        check += rle[n, 1]
    return rle_string

def rle_decode(rle, shape):
    """Decodes an RLE encoded list of space separated
    numbers and returns a binary mask."""
    rle = rle.replace('|', ' ').split(' ')
    rle = [x for x in rle if x != '']
    # print(rle)
    rle = list(map(int, rle))
    rle = np.array(rle, dtype=np.int32).reshape([-1, 2])
    rle[:, 1] += rle[:, 0]
    mask = np.zeros([shape[0] * shape[1]], np.bool)
    for s, e in rle:
        assert 0 <= s < mask.shape[0]
        assert 1 <= e <= mask.shape[0], "shape: {}  s {}  e {}".format(shape, s, e)
        mask[s:e] = 1
    # Reshape and transpose
    mask = mask.reshape([shape[0], shape[1]])
    return mask

def write_mask(imageid, mask, score = None):
    mask_string = []
    instances = np.unique(mask)
    index = np.argwhere(instances==0)
    instances = np.delete(instances, index)
    for n, instance in enumerate(instances):
        if score is not None:
            confidence = score[instance]
        else:
            confidence = 1.0
        label_id = int(math.floor(instance/1000))
        mask_string_this = imageid + ',' + str(label_id) + ',' 
        rle_string_this = rle_encode_fast(mask == instance, confidence)
        mask_string_this += rle_string_this
        mask_string.append(mask_string_this)
    return mask_string

def write_mask_mp(imageid, mask, score = None):
    mask_string = []
    instances = np.unique(mask)
    index = np.argwhere(instances==0)
    instances = np.delete(instances, index)
    for n, instance in enumerate(instances):
        if score is not None:
            confidence = score[instance]
        else:
            confidence = 1.0
        label_id = int(math.floor(instance/1000))
        mask_string_this = imageid + ',' + str(label_id) + ',' 
        rle_string_this = rle_encode_fast(mask == instance, confidence)
        mask_string_this += rle_string_this
        mask_string.append(mask_string_this)
    return mask_string

from mrcnn.visualize import *
def display_instances_mask(image, masks, class_ids,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True,
                      colors=None, captions=None):
    """
    """
    # Number of instances
    N = masks.shape[2]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    if auto_show:
        plt.show()