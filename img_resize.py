import os
import cv2


def resize_img(img, height=64, width=64):
    top, bottom, left, right = 0, 0, 0, 0
    h, w, _ = img.shape
    max_edge = max(h, w)
    if h < max_edge:
        dh = max_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < max_edge:
        dw = max_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    BLACK = [0, 0, 0]

    constant = cv2.copyMakeBorder(img, top, bottom, left, right,
                                  cv2.BORDER_CONSTANT, value=BLACK)
    return cv2.resize(constant, (height, width))

def img_resize(img_path):
    dir_name_list = os.listdir(img_path)[1:]

    for dir_name in dir_name_list:
        img_name_list = os.listdir(img_path+'/' + dir_name)
        img_list_len = len(img_name_list)
        print (dir_name)

        for img_name in img_name_list:
            img = cv2.imread(img_path+'/' + dir_name +'/'+ img_name)
            resized_img = resize_img(img, 64, 64)
            cv2.imwrite(img_path+'/' + dir_name +'/'+ img_name, resized_img)



img_resize('55')

