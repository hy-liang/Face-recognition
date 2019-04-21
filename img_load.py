import os
import numpy as np
from PIL import Image

def img_load(img_path):
    dir_name_list = os.listdir(img_path)
    dir_name_list.remove('.DS_Store')

    len_dir = len(dir_name_list)
    print len_dir

    img_list = []
    img_label_list_2d = []
    #img_label_list_1d = []

    for dir_name in dir_name_list:
        img_name_list = os.listdir(img_path+'/' + dir_name)

        for img_name in img_name_list:
            img_label = np.zeros([len_dir])
            img_label[int(dir_name) - 1] = 1
            img_label_list_2d.append(img_label)
            #img_label_list_1d.append(int(dir_name)-1)


            img = Image.open(img_path+'/' + dir_name +'/'+ img_name)
            img_array = np.asarray(img, 'float32')/255
            #img_array = np.asarray(img)
            img_list.append(img_array)

    img_list = np.array(img_list)
    img_label_list_2d = np.array(img_label_list_2d)
    #img_label_list_1d = np.array(img_label_list_1d)

    #np.save(img_path+'_y.npy', img_label_list_2d)
    #np.save(img_path+'_y_1d.npy', img_label_list_1d)
    #np.save(img_path+'_x.npy', img_list)
    print img_label_list_2d.shape
    print img_list.shape
    return img_list, img_label_list_2d

