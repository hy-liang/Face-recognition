import os,shutil
def rm(img_path):
    dir_name_list = os.listdir(img_path)[1:]

    for dir_name in dir_name_list:
        img_name_list = os.listdir(img_path+'/' + dir_name)
        img_list_len = len(img_name_list)
        if img_list_len<20:
            shutil.rmtree(img_path + '/' + dir_name)
    dir_name_list = os.listdir(img_path)[1:]
    print len(dir_name_list)
rm('55')