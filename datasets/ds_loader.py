import cv2
import numpy as np
import torch
from scipy import ndimage
import scipy.misc
import skimage
import os,sys
import itertools
from skimage import transform as stf
from PIL import Image
from torch.utils.data import Dataset
import imageio
from math import floor, ceil
import pickle
import gin
from .label_aug import aug_labels

# @gin.configurable
class IC15(Dataset):

    def __init__(self, file_list, label_path, image_path, ralph_path, num):

        super(IC15, self).__init__()

        self.file_label_list = get_files(file_list, label_path)
        self.file_image_list = get_image_files(file_list, image_path)
        self.file_lables = get_labels(self.file_label_list, num)
        alph = load_alph(ralph_path)
        self.ralph = dict (zip(alph.values(),alph.keys()))

    def __len__(self):

        return len(self.file_label_list)

    def __getitem__(self, index):

        images = get_images(self.file_image_list[index])
        images = images.transpose((2, 0, 1))

        return (images, self.file_lables[index])

def get_files(nfile, dpath):

    fnames = open(nfile, 'r').readlines()
    fnames = [ dpath + x.strip() for x in fnames ]
    return fnames

def get_image_files(nfile, dpath):

    fnames = open(nfile, 'r').readlines()
    fnames = ["_".join(x.strip().split("_")[1:]) for x in fnames]
    fnames = [ dpath + x.strip() for x in fnames ]
    return fnames


def get_labels(fnames, num):

    labels = []
    for id,image_file in enumerate(fnames):
        fn  = os.path.splitext(image_file)[0] + '.txt'
        lbl = open(fn, 'r').read()
        # lbl_new = []
        lbl_new = ""
        for lbl_item in lbl.split():
            txt = lbl_item.split(",")[-1]
            if txt != "###":
                lbl_new = lbl_new + ";:" + txt
            lbl_new = lbl_new[2:]
        # for i in range(num - len(lbl_new)):
        #     lbl_new.append("000")
        #lbl = ' '.join(lbl_new)

        labels.append(lbl_new)

    return labels

def load_alph(file_name):

    return np.load(file_name, allow_pickle="TRUE").item()

def rescale_img(img, size):
    image = np.zeros((size, size,3),dtype = np.uint8)
    # image = np.zeros((size, size),dtype = np.uint8)
    h, w = img.shape[:2]
    length = max(h, w)
    scale = 768 / length           ###768 is the train image size
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    image[:img.shape[0], :img.shape[1]] = img

    return image


@gin.configurable
def get_images(f_name, size, extend, n_dim):

    try:
        # image_data = np.array(Image.open(f_name + extend).convert('L'))
        image_data = np.array(Image.open(f_name + extend))
        image_data = rescale_img(image_data, size)
        image_data = skimage.img_as_float32(image_data)

        h, w = np.shape(image_data)[:2]

        if image_data.ndim < 3:
            image_data = np.expand_dims(image_data, axis=-1)

        if n_dim==3 and image_data.shape[2]!=3:
            image_data = np.tile(image_data,3)

    except IOError as e:
        print("could not read:", f_name, ":", e)

    return image_data

def init_gin():
    config_file = "./config/ic15_dataset.gin"
    gin.parse_config_file(config_file)



def build(file_list,
          label_path,
          image_path,
          ralph_path,
          num):

    return IC15(file_list, label_path, image_path, ralph_path, num)


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = torch.cat([torch.from_numpy(t).unsqueeze(0) for t in batch[0]])
    batch[1] = aug_labels(batch[1])
    return tuple(batch)


if __name__ == "__main__":

    from torch.utils.data import DataLoader, DistributedSampler
    import torch.distributed as dist

    init_gin()

    dataset_train = IC15()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '11111'
    dist.init_process_group("nccl", rank=0, world_size=1)
    sampler_train = DistributedSampler(dataset_train)
    #sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, 3, drop_last=True)

    data_loader_train = DataLoader(dataset_train,
                                   batch_sampler=batch_sampler_train,
                                   collate_fn=collate_fn,
                                   num_workers=1)

    # sampler_train = torch.utils.data.RandomSampler(dataset_train)
    # batch_sampler_train = torch.utils.data.BatchSampler(
    #     sampler_train,
    #     10,
    #     drop_last=True)
    # train_loader = torch.utils.data.DataLoader(
    #     dataset_train,
    #     batch_sampler=batch_sampler_train,
    #     collate_fn = collate_fn,
    #     num_workers = 1,
    # )

    # it = iter(train_loader)
    # first = next(it)
    #
    # print(len(first[1]))
    # print(first[1])
    # print(first[0][0].shape)

    for sample, target in data_loader_train:

        print(sample.shape)
        print(f"target[0]: {target[0]}")
        print(f"target[1]: {target[1]}")
        print(len(target[0]))
        break
