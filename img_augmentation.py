import os
import numpy as np

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


datagen = ImageDataGenerator(
        zca_whitening=True,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

def cal_multi(l):
    if l<5:
        return 16
    elif l<10:
        return 8
    elif l<20:
        return 4
    elif l<40:
        return 2

def img_augmentation(img_path):
    dir_name_list = os.listdir(img_path)[1:]

    for dir_name in dir_name_list:
        img_name_list = os.listdir(img_path+'/' + dir_name)
        img_list_len = len(img_name_list)
        if img_list_len == 0:
            continue
        img_batch = []
        for img_name in img_name_list:
            img = load_img(img_path+'/' + dir_name + '/' + img_name)
            x = img_to_array(img)
            img_batch.append(x)
        img_batch = np.array(img_batch)
        i = 0
        img_multi = cal_multi(img_list_len)

        for batch in datagen.flow(img_batch, batch_size=img_list_len, save_to_dir=img_path + '/' + dir_name
                , save_prefix=str(dir_name), save_format='jpeg'):
            i += 1
            if i > img_multi:
                break

img_augmentation('train')
img_augmentation('test')
