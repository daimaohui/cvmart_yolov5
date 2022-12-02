# _*_ coding:utf-8 _*_
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import cv2
import logging
from glob import glob
from glob import glob
from random import randint
import time
import shutil
'''
YOLO v5 
xml -> txt
'''

# class2id = {"table": 0, "chair": 1, "table_chair": 2, "umbrella": 3, "uncertain": 4}
class2id = {"person": 0, "short_sleeve_red": 1, "short_sleeve_black": 2, "short_sleeve_white": 3, "short_sleeve_grey": 4,
            "short_sleeve_green": 5, "short_sleeve_blue": 6, "short_sleeve_dark_blue": 7, "long_sleeve_red": 8,
            "long_sleeve_black": 9, "long_sleeve_white": 10, "long_sleeve_grey": 11, "non_uniform": 12,
            "other_uniform": 13, "chef_hat_red": 14, "chef_hat_black": 15, "chef_hat_white": 16, "peaked_cap_red": 17,
            "peaked_cap_black": 18, "peaked_cap_white": 19, "peaked_cap_blue": 20, "peaked_cap_beige": 21,
            "disposable_cap_white": 22, "disposable_cap_blue": 23, "head": 24, "other_hat": 25, "apron_red": 26,
            "apron_black": 27, "apron_white": 28, "apron_grey": 29, "other_apron": 30}


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    x = min(x, 1.0)
    y = min(y, 1.0)
    w = min(w, 1.0)
    h = min(h, 1.0)

    return (x, y, w, h)


def convert_annotation(image_path):
    in_file = open(image_path.replace('.xml', '.xml'), encoding="utf-8")
    out_file = open(image_path.replace('.xml', '.txt'), 'w')
    # print(in_file)
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    # img = cv2.imread("./datasets/VOCdevkit/VOC2007/JPEGImages/"+image_id+".jpg")
    # sp = img.shape

    for obj in root.iter('object'):
        name = obj.find('name').text
        cls_id = class2id[name]
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b, )
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
def image_txt_copy(files,scr_path,dst_img_path,dst_txt_path):
    """
    :param files: 图片名字组成的list
    :param scr_path: 图片的路径
    :param dst_img_path: 图片复制到的路径
    :param dst_txt_path: 图片对应的txt复制到的路径
    :return:
    """
    for file in files:
        img_path=scr_path+file
        shutil.copy(img_path, dst_img_path+file)
        scr_txt_path=scr_path+file.split('.')[0]+'.txt'
        shutil.copy(scr_txt_path, dst_txt_path + file.split('.')[0]+'.txt')
if __name__ == '__main__':
    files = glob('/project/train/src_repo/trainval/*.xml')
    print(files)
    for file in files:
        print(file)
        convert_annotation(file)

    files = glob("/project/train/src_repo/trainval/*.txt")
    # print(files)
    train = []
    val = []

    for i in files:
        num = randint(1, 10)
        name = i.replace('.txt', '.jpg').split('/')[-1]
        if num < 8:
            train.append(name)
        else:
            val.append(name)
    file_List = ["train", "val"]
    for file in file_List:
        if not os.path.exists('/project/train/src_repo/VOC/images/%s' % file):
            os.makedirs('/project/train/src_repo/VOC/images/%s' % file)
        if not os.path.exists('/project/train/src_repo/VOC/labels/%s' % file):
            os.makedirs('/project/train/src_repo/VOC/labels/%s' % file)
    image_txt_copy(train,"/project/train/src_repo/trainval/",'/project/train/src_repo/VOC/images/train/','/project/train/src_repo/VOC/labels/train/')
    image_txt_copy(val, "/project/train/src_repo/trainval/", '/project/train/src_repo/VOC/images/val/', '/project/train/src_repo/VOC/labels/val/')

