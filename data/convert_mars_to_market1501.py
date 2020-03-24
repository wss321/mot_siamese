from shutil import copyfile
import os

MARS_path = r"E:\PyProjects\datasets\MARS"
# ---------------------------------------
# train_all
train_path = MARS_path + '/bbox_train'
train_save_path = MARS_path + '/pytorch/train_all'
if not os.path.isdir(train_save_path):
    os.makedirs(train_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    print(root)
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name[:4]
        src_path = train_path + "/{}".format(ID) + '/' + name
        dst_path = train_save_path + '/' + ID
        if not os.path.isdir(dst_path):
            os.makedirs(dst_path)
        copyfile(src_path, dst_path + '/' + name)

# ---------------------------------------
# train_val
train_path = MARS_path + '/bbox_train'
train_save_path = MARS_path + '/pytorch/train'
val_save_path = MARS_path + '/pytorch/val'
if not os.path.isdir(train_save_path):
    os.makedirs(train_save_path)
    os.makedirs(val_save_path)

for root, dirs, files in os.walk(train_path, topdown=True):
    print(root)
    for name in files:
        if not name[-3:] == 'jpg':
            continue
        ID = name[:4]
        src_path = train_path + "/{}".format(ID) + '/' + name
        dst_path = train_save_path + '/' + ID
        if not os.path.isdir(dst_path):
            os.makedirs(dst_path)
            dst_path = val_save_path + '/' + ID  # first image is used as val image
            os.makedirs(dst_path)
        copyfile(src_path, dst_path + '/' + name)
