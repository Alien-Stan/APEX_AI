import os
import random
import shutil

data_dir = r'G:\yolov5\apex_data'
valid_ratio = 0.1
n = 200
print(os.listdir(os.path.join(data_dir, 'img'))[0])
def copyfile(filename, target_dir):
    """将文件复制到目标目录"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.copy(filename, target_dir)

def movefile(filename, target_dir):
    """将文件复制到目标目录"""
    os.makedirs(target_dir, exist_ok=True)
    shutil.move(filename, target_dir)


def reorg_train_valid(data_dir,  valid_ratio):

    label_file = os.path.join(data_dir, 'data', 'labels')
    img_file = os.path.join(data_dir, 'img')

    path = os.listdir(label_file)
    number = len(path)
    sample = random.sample(path , int(number * (1 - valid_ratio)))
    for name in sample:
        movefile(os.path.join(label_file, name),
                os.path.join(data_dir, 'train_data', 'labels', 'train'))

    for file in os.listdir(label_file):
        fname = os.path.join(label_file, file)
        movefile(fname, os.path.join(data_dir, 'train_data', 'labels', 'val'))

    for file in os.listdir(os.path.join(data_dir, 'train_data', 'labels', 'val')):
        i = file.split('_')[-1].split('.')[0]
        movefile(os.path.join(img_file, ('img_'+f'{i}'+'.jpg')),
                 os.path.join(data_dir, 'train_data', 'images', 'val'))

    for file in os.listdir(img_file):
        movefile(os.path.join(img_file, file ),
                 os.path.join(data_dir, 'train_data', 'images', 'train'))

    # for train_file in os.listdir(os.path.join(data_dir, 'img')):
    #     # 训练集图片
    #     if int(train_file.split('_')[-1].split('.')[0]) <= int((1-valid_ratio) * n):
    #         fname = os.path.join(data_dir, 'img', train_file)
    #         copyfile(fname, os.path.join(data_dir, 'train_data','images', 'train'))
    #     # 验证集图片
    #     elif int((1-valid_ratio) * n) < int(train_file.split('_')[-1].split('.')[0]) <= n :
    #         fname = os.path.join(data_dir, 'img', train_file)
    #         copyfile(fname, os.path.join(data_dir, 'train_data', 'images', 'val'))
    #
    # for train_label in os.listdir(os.path.join(data_dir, 'data', 'labels')):
    #     # 训练集图片
    #     if int(train_label.split('_')[-1].split('.')[0]) <= int((1-valid_ratio) * n):
    #         fname = os.path.join(data_dir, 'data', 'labels', train_label)
    #         copyfile(fname, os.path.join(data_dir, 'train_data','labels', 'train'))
    #     # 验证集图片
    #     elif int((1-valid_ratio) * n) < int(train_label.split('_')[-1].split('.')[0]) <= n :
    #         fname = os.path.join(data_dir, 'data', 'labels', train_label)
    #         copyfile(fname, os.path.join(data_dir, 'train_data', 'labels', 'val'))

reorg_train_valid(data_dir, valid_ratio)