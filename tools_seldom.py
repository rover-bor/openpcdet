import os
import numpy as np
import cv2
import shutil


def get_imageset():
    src_dir = "/media/chenliangjin/T7 Shield/datasets/waymo_open_dataset/"
    txt_path = src_dir + "ImageSets/train.txt"
    tfc_dir = src_dir + "raw_data/"
    with open(txt_path, 'w') as f:
        for tfc_name in os.listdir(tfc_dir):
            f.write(tfc_name+"\n")


if __name__ == "__main__":
    get_imageset()