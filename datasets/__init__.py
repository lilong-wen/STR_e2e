import gin
from .ds_loader import build

@gin.configurable
def build_dataset(train_file_list,
                  train_label_path,
                  train_image_path,
                  test_file_list,
                  test_label_path,
                  test_image_path,
                  ralph_path,
                  num):

    train_dataset = build(train_file_list,
                          train_label_path,
                          train_image_path,
                          ralph_path,
                          num)
    test_dataset = build(test_file_list,
                         test_label_path,
                         test_image_path,
                         ralph_path,
                         num)

    return train_dataset, test_dataset
