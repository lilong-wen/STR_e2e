import os
from sklearn.model_selection import train_test_split


def split_and_write(name_path, store_path, split_rate):
    '''
    name_path: path where have file names
    store_path: where to store train and test file name list
    '''

    file_name_all = os.listdir(name_path)

    train_list, test_list = train_test_split(file_name_all, test_size=split_rate)

    with open(store_path + 'all.ln', 'w') as all_f:
        for item in file_name_all:
            item = item.split(".")[0]
            all_f.write(item + "\n")

    with open(store_path + 'train.ln', 'w') as train_f:
        for item in train_list:
            item = item.split(".")[0]
            train_f.write(item + "\n")

    with open(store_path + 'test.ln', 'w') as test_f:
        for item in train_list:
            item = item.split(".")[0]
            test_f.write(item + "\n")



if __name__ == "__main__":

    name_path = "/home/zju/w4/datasets/ic15/train_gt"
    store_path = "/home/zju/w4/STR_e2e/ic15/"

    split_and_write(name_path, store_path, 0.4)
