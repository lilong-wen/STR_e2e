import imgproc
import cv2
from mep import mep
import numpy as np
import os
import torch
import torch.utils.data as data
import scipy.io as scio
import re
import itertools
import random
from PIL import Image
import torchvision.transforms as transforms
import time


def ratio_area(h, w, box):
    area = h * w
    ratio = 0
    for i in range(len(box)):
        poly = plg.Polygon(box[i])
        box_area = poly.area()
        tem = box_area / area
        if tem > ratio:
            ratio = tem
    return ratio, area

def rescale_img(img, box, h, w):
    image = np.zeros((768,768,3),dtype = np.uint8)
    length = max(h, w)
    scale = 768 / length           ###768 is the train image size
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    image[:img.shape[0], :img.shape[1]] = img
    box *= scale
    return image

def random_scale(img, bboxes, min_size):
    h, w = img.shape[0:2]
    # ratio, _ = ratio_area(h, w, bboxes)
    # if ratio > 0.5:
    #     image = rescale_img(img.copy(), bboxes, h, w)
    #     return image
    scale = 1.0
    if max(h, w) > 1280:
        scale = 1280.0 / max(h, w)
    random_scale = np.array([0.5, 1.0, 1.5, 2.0])
    scale1 = np.random.choice(random_scale)
    if min(h, w) * scale * scale1 <= min_size:
        scale = (min_size + 10) * 1.0 / min(h, w)
    else:
        scale = scale * scale1
    bboxes *= scale
    img = cv2.resize(img, dsize=None, fx=scale, fy=scale)
    return img

def padding_image(image,imgsize):
    length = max(image.shape[0:2])
    if len(image.shape) == 3:
        img = np.zeros((imgsize, imgsize, len(image.shape)), dtype = np.uint8)
    else:
        img = np.zeros((imgsize, imgsize), dtype = np.uint8)
    scale = imgsize / length
    image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    if len(image.shape) == 3:
        img[:image.shape[0], :image.shape[1], :] = image
    else:
        img[:image.shape[0], :image.shape[1]] = image
    return img

def random_crop(imgs, img_size, character_bboxes):
    h, w = imgs[0].shape[0:2]
    th, tw = img_size
    crop_h, crop_w = img_size
    if w == tw and h == th:
        return imgs

    word_bboxes = []
    if len(character_bboxes) > 0:
        for bboxes in character_bboxes:
            word_bboxes.append(
                [[bboxes[:, :, 0].min(), bboxes[:, :, 1].min()], [bboxes[:, :, 0].max(), bboxes[:, :, 1].max()]])
    word_bboxes = np.array(word_bboxes, np.int32)

    #### IC15 for 0.6, MLT for 0.35 #####
    if random.random() > 0.6 and len(word_bboxes) > 0:
        sample_bboxes = word_bboxes[random.randint(0, len(word_bboxes) - 1)]
        left = max(sample_bboxes[1, 0] - img_size[0], 0)
        top = max(sample_bboxes[1, 1] - img_size[0], 0)

        if min(sample_bboxes[0, 1], h - th) < top or min(sample_bboxes[0, 0], w - tw) < left:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
        else:
            i = random.randint(top, min(sample_bboxes[0, 1], h - th))
            j = random.randint(left, min(sample_bboxes[0, 0], w - tw))

        crop_h = sample_bboxes[1, 1] if th < sample_bboxes[1, 1] - i else th
        crop_w = sample_bboxes[1, 0] if tw < sample_bboxes[1, 0] - j else tw
    else:
        ### train for IC15 dataset####
        # i = random.randint(0, h - th)
        # j = random.randint(0, w - tw)

        #### train for MLT dataset ###
        i, j = 0, 0
        crop_h, crop_w = h + 1, w + 1  # make the crop_h, crop_w > tw, th

    for idx in range(len(imgs)):
        # crop_h = sample_bboxes[1, 1] if th < sample_bboxes[1, 1] else th
        # crop_w = sample_bboxes[1, 0] if tw < sample_bboxes[1, 0] else tw

        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i:i + crop_h, j:j + crop_w, :]
        else:
            imgs[idx] = imgs[idx][i:i + crop_h, j:j + crop_w]

        if crop_w > tw or crop_h > th:
            imgs[idx] = padding_image(imgs[idx], tw)

    return imgs


def random_horizontal_flip(imgs):
    if random.random() < 0.5:
        for i in range(len(imgs)):
            imgs[i] = np.flip(imgs[i], axis=1).copy()
    return imgs


def random_rotate(imgs):
    max_angle = 10
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        imgs[i] = img_rotation
    return imgs


class ICDAR2015(data.Dataset):
    def __init__(self, icdar2015_folder, target_size=768, viz=False, debug=False):
        super(ICDAR2015, self).__init__()
        self.target_size = target_size
        self.img_folder = os.path.join(icdar2015_folder, 'train')
        self.gt_folder = os.path.join(icdar2015_folder, 'train_gt')
        imagenames = os.listdir(self.img_folder)
        self.images_path = []
        for imagename in imagenames:
            self.images_path.append(imagename)

    def __getitem__(self, index):
        return self.pull_item(index)

    def __len__(self):
        return len(self.images_path)

    def get_imagename(self, index):
        return self.images_path[index]

    def load_image_gt(self, index):
        '''
        根据索引加载ground truth
        :param index:索引
        :return:bboxes 字符的框，
        '''
        imagename = self.images_path[index]
        gt_path = os.path.join(self.gt_folder, "gt_%s.txt" % os.path.splitext(imagename)[0])
        word_bboxes, words = self.load_gt(gt_path)
        word_bboxes = np.float32(word_bboxes)

        image_path = os.path.join(self.img_folder, imagename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = random_scale(image, word_bboxes, self.target_size)

        return image, word_bboxes, words

    def load_gt(self, gt_path):
        lines = open(gt_path, encoding='utf-8').readlines()
        bboxes = []
        words = []
        for line in lines:
            ori_box = line.strip().encode('utf-8').decode('utf-8-sig').split(',')
            box = [int(ori_box[j]) for j in range(8)]
            word = ori_box[8:]
            word = ','.join(word)
            box = np.array(box, np.int32).reshape(4, 2)
            if word == '###':
                words.append('###')
                bboxes.append(box)
                continue
            area, p0, p3, p2, p1, _, _ = mep(box)

            bbox = np.array([p0, p1, p2, p3])
            distance = 10000000
            index = 0
            for i in range(4):
                d = np.linalg.norm(box[0] - bbox[i])
                if distance > d:
                    index = i
                    distance = d
            new_box = []
            for i in range(index, index + 4):
                new_box.append(bbox[i % 4])
            new_box = np.array(new_box)
            bboxes.append(np.array(new_box))
            words.append(word)
        return bboxes, words

    def pull_item(self, index, num=100):
        # if self.get_imagename(index) == 'img_59.jpg':
        #     pass
        # else:
        #     return [], [], [], [], np.array([0])
        image, character_bboxes, words = self.load_image_gt(index)
        # print(image.shape)
        image = rescale_img(image, character_bboxes, image.shape[0], image.shape[1])

        image = Image.fromarray(image)
        image = image.convert('RGB')
        image = transforms.ColorJitter(brightness=32.0 / 255, saturation=0.5)(image)

        image = imgproc.normalizeMeanVariance(np.array(image), mean=(0.485, 0.456, 0.406),
                                              variance=(0.229, 0.224, 0.225))

        image = torch.from_numpy(image).float().permute(2, 0, 1)

        # return image, character_bboxes, words
        # for i in range(num - len(words)):
        #     words.append('000')
        words = np.asarray(words, dtype=np.str)
        words = np.pad(words, (0, num - len(words)), 'reflect')
        words = list(words)

        return image, words
        # return {"image": image, "text": words}


if __name__ == '__main__':

    dataloader = ICDAR2015('/home/zju/w4/datasets/ic15', target_size=768, viz=True)
    train_loader = torch.utils.data.DataLoader(
        dataloader,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True)
    '''
    total = 0
    total_sum = 0
    for index, (opimage, bboxes, words) in enumerate(train_loader):
        total += 1
        img = opimage[0].numpy().transpose(1, 2, 0)
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR) * 255.
        # print(img)
        print(img.shape)
        print(words.shape)
        bboxes = bboxes.cpu().numpy()[0]
        # print(bboxes.shape)
        for box in bboxes:
            # print(box.shape)
            box = np.array(box, dtype=int).reshape(-1, 1, 2)
            # print(box)
            cv2.polylines(img, [box], True, (0, 0, 255))
        # img = np.transpose(img, (1, 2, 0))
        cv2.imwrite("test.jpg", img)
        # print(img.shape)
        # print(bboxes)
        # print(index, words)
        break
    '''
    it = iter(train_loader)
    first = next(it)
    # print(first["image"].shape)
    # print(first["text"])
    print(first[1])
