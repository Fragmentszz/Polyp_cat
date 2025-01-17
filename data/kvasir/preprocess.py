import json
import os
import shutil
from os.path import curdir, join, exists, isdir
import sys
sys.path.append('/applications/graduate_design/cat-sam')
import cat_sam
from cat_sam.datasets.misc import get_json_dict_from_dir

if __name__ == '__main__':
    curdir = '/applications/graduate_design/dataset/polyp'
    test_data_dir = join(curdir, 'test')
    train_data_dir = join(curdir, 'train')
    assert exists(train_data_dir) and exists(test_data_dir), \
        "TrainDataset and TestDataset donot exist! Please download them from Google Driver!"

    train_image_dir = join(train_data_dir, 'images')
    train_mask_dir = join(train_data_dir, 'gts')
    for file_name in os.listdir(train_image_dir):
        sample_name = file_name.split('.')[0]
        if not sample_name.isdigit():
            continue

        # image_file_path = join(train_image_dir, file_name)
        # while exists(image_file_path):
        #     os.remove(image_file_path)

        # mask_file_path = join(train_mask_dir, file_name)
        # while exists(mask_file_path):
        #     os.remove(mask_file_path)

    train_json = get_json_dict_from_dir(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir
        # mask_dir=join(curdir, 'Kvasir-SEG', 'masks'), mask_ext='jpg'
    )
    train_json_path = join(curdir, 'train.json')
    with open(train_json_path, 'w') as f:
        json.dump(train_json, f, indent=4)


    test_image_dir = join(test_data_dir,  'images')
    test_mask_dir = join(test_data_dir,  'gts')
    # for folder_name in os.listdir(test_data_dir):
    #     folder_path = join(test_data_dir, folder_name)
    #     if not isdir(folder_path) or folder_name == 'Kvasir':
    #         continue
    #     while exists(folder_path):
    #         shutil.rmtree(folder_path)

    test_json = get_json_dict_from_dir(
        image_dir=test_image_dir,
        mask_dir=test_mask_dir
        # mask_dir=join(curdir, 'Kvasir-SEG', 'masks'), mask_ext='jpg'
    )
    test_json_path = join(curdir, 'test.json')
    with open(test_json_path, 'w') as f:
        json.dump(test_json, f, indent=4)

