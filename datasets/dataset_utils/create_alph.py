import numpy as np
import itertools
import os

def get_files(file_name, file_path):
    fnames = open(file_name, 'r').readlines()
    fnames = [ file_path + x.strip() for x in fnames ]
    return fnames


def get_labels(fnames):

    labels = []
    for id,image_file in enumerate(fnames):
        fn  = os.path.splitext(image_file)[0] + '.txt'
        lbl = open(fn, 'r').read()
        lbl_new = []
        for lbl_item in lbl.split():
            lbl_new.append(lbl_item.split(",")[-1])

        lbl = ' '.join(lbl_new) #remove linebreaks if present

        labels.append(lbl)

    return labels

def get_alphabet(labels):

    coll = ''.join(labels)
    unq  = sorted(list(set(coll)))
    unq  = [''.join(i) for i in itertools.product(unq, repeat = 1)]
    alph = dict( zip( unq,range(len(unq)) ) )

    return alph

def save_alph(alph, file_name):

    np.save(file_name, alph)

def load_alph(file_name):

    return np.load(file_name, allow_pickle="TRUE").item()


def ic15_dataset_utils():

    file_name = "/home/zju/w4/STR_e2e/ic15/all.ln"
    file_path = "/home/zju/w4/datasets/ic15/train_gt/"

    names = get_files(file_name, file_path)

    labels = get_labels(names)

    alph = get_alphabet(labels)

    print(f"alph: {alph}")

    save_alph(alph, '/home/zju/w4/STR_e2e/ic15/alph.npy')

    # print(load_alph('/home/zju/w4/STR_e2e/ic15/alph.npy'))

if __name__ == "__main__":

    ic15_dataset_utils()
