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
    h, w = img.shape[:2]
    length = max(h, w)
    scale = 768 / length           ###768 is the train image size
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    image[:img.shape[0], :img.shape[1]] = img

    return image


@gin.configurable
def get_images(f_name, size, extend, n_dim):

    try:
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

if __name__ == "__main__":

    init_gin()

    dataloader = IC15()
    train_loader = torch.utils.data.DataLoader(
        dataloader,
        batch_size = 4,
        shuffle = True,
        num_workers = 0,
        drop_last = True,
        pin_memory = True
    )

    it = iter(train_loader)
    first = next(it)
    print(first[0].shape)
    print(len(first[1]))
    print(first[1])
