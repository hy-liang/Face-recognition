import os
import shutil
import random
import cv2


def divide_2_train_test():
    no_dir = 1
    dir_name_list = os.listdir(r'train')[1:]


    for old_dir_name in dir_name_list:
        new_dir_name = str(no_dir)
        print (no_dir)
        os.rename('train/' + old_dir_name, 'train/' + new_dir_name)
        os.mkdir('test/' + new_dir_name)

        img_name_list = os.listdir('train/' + new_dir_name)

        img_list_len = len(img_name_list)
        if img_list_len == 0:
            no_dir += 1
            continue
        num_test = int(img_list_len * 0.2)
        if num_test < 1:
            num_test = 1
        test_index = random.sample(range(0, img_list_len), num_test)

        for img_index in test_index:
            img_name = img_name_list[img_index]
            shutil.move('train/' + new_dir_name + '/' + img_name, 'test/' + new_dir_name + '/' + img_name)

        no_dir += 1

divide_2_train_test()