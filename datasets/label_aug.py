import numpy as np
import torch
import random

#@gin.configurable
def aug_labels(label, num=50):

    label_list = [l.split(";:") for l in label]
    label_all = [j for i in label_list for j in i]
    mask = []
    label_final = []

    for item in label_list:

        item_mask = []
        own = list(set(item))
        if len(own) < num / 4:
            #own = own * int(((num / 4) - len(own)) / len(own))
            own = own * int(num / 8)
        others = list(set(label_all) - set(item))
        len_own = len(own)
        len_others = len(others)
        if len_own + len_others < num:
            for i in range(num - len_own - len_others):
                others.append(others[i%len(others)])
        else:
            others = list(set(label_all) - set(item))[:num-len(own)]
        item_mask = [1 for t in own]
        item_mask +=[0 for t in others]
        own += others

        tmp = list(zip(own, item_mask))
        random.shuffle(tmp)
        own, item_mask = zip(*tmp)

        label_final.append(own)
        # to torch
        item_mask = torch.from_numpy(np.array(item_mask))
        mask.append(item_mask.unsqueeze(0))

    mask_torch = torch.cat(mask)
    return label_final, mask_torch


if __name__ == "__main__":

    label = [';:meets;:Baby-G;:Baby-G', '', 'mpressor;:PrimeFresh;:299', ':EXPERIENCE;:THE;:DREAM;:TRAIN;:RIDE@', 'n;:boshi;:lehiban', 'pment;:Office', 'RAMEN;:CHAMPION;:NANTSUTTE;:SINGAPORE', ':you;:Note?', '', 'tion;:MONTHLY;:CALENDAR;:POSTER;:COLLECT', 'cup;:handmade', 'B);:Joho;:Iskadar;:Malaysia;:With;:;:unseated;:market;:hit;:sales;:RMS;:billion;:57;:billion);:the;:day;:launch.;:more;:than;:000;:from;:all;:over;:the;:world;:proudly;:Garden;:Bay;:home.', '', ';:N;:INGAPORE;:URAL;:ORCHARD;:SALES;:REBAJAS;:SOLDES;:ZNIZANJE;:UTA', 'OLOGICAL;:FORMULAS;:UNIQUE;:SKINCARE;:FOR;:ioma;:CREME;:SALE;:SELLERS', 'rm', ':NETS;:delight;:VISA;:ART;:COMMON;:personal;:choque', 'S;:EXIT', '', 'CCHA;:HOUSE']

    label_f, mask = aug(label)

    print(label_f)
    print(mask)
    print(len(mask[0]))
