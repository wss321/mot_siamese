# vim: expandtab:ts=4:sw=4
import os
import urllib.request
import hashlib
from functools import partial
import tarfile
import glob
import shutil
from xml.dom import minidom
import subprocess

import numpy as np
import cv2


def read_cvml_detections(
        filename, save_to, roi_scale_w=0.75, roi_scale_h=1.0):
    def fattr(node, name):
        return float(node.attributes[name].value)

    def rescale_roi(old_roi):
        x, y, w, h = old_roi
        new_w, new_h = roi_scale_w * w, roi_scale_h * h
        dw, dh = w - new_w, h - new_h
        x += dw / 2
        y += dh / 2
        return x, y, new_w, new_h

    xmldoc = minidom.parse(filename)
    with open(save_to, 'w') as f:
        for frame in xmldoc.getElementsByTagName("frame"):
            frame_idx = int(frame.attributes["number"].value)
            for obj in frame.getElementsByTagName("object"):
                box = obj.getElementsByTagName("box")[0]
                xc, yc = fattr(box, "xc"), fattr(box, "yc")
                w, h = fattr(box, "w"), fattr(box, "h")
                roi = xc - w / 2., yc - h / 2., w, h
                x, y, w, h = rescale_roi(roi)
                confidence = fattr(obj, "confidence")
                f.write(
                    '{},{},{:.2f},{:.2f},{:.2f},{:.2f},{}\n'.format(frame_idx + 1, -1, x, y, w, h, confidence))

    return None


def read_cvml_groundtruth(filename, save_to):
    def fattr(node, name):
        return float(node.attributes[name].value)

    xmldoc = minidom.parse(filename)
    with open(save_to, 'w') as f:
        for frame in xmldoc.getElementsByTagName("frame"):
            frame_idx = int(frame.attributes["number"].value)
            for obj in frame.getElementsByTagName("object"):
                box = obj.getElementsByTagName("box")[0]
                xc, yc = fattr(box, "xc"), fattr(box, "yc")
                w, h = fattr(box, "w"), fattr(box, "h")
                x, y, w, h = xc - w / 2., yc - h / 2., w, h

                track_id = int(obj.attributes["id"].value)
                f.write(
                    '{},{},{:.2f},{:.2f},{:.2f},{:.2f},{},{},{},{}\n'.format(frame_idx + 1, track_id, x, y, w,
                                                                             h, 1, -1, -1, -1))
    return None


def move(data_path=r"E:\PyProjects\datasets\Crowd_PETS09\S2\L1\Time_12-34"):
    img = cv2.imread(data_path + "/View_001/frame_0000.jpg")
    h, w, c = img.shape
    with open(os.path.join(data_path, "seqinfo.ini"), "w") as f:
        f.write("[Sequence]\n")
        f.write("name=PETS2009_S2_L1_V001\n")
        f.write("imDir=img1\n")
        f.write("frameRate=30\n")
        f.write("seqLength={}\n".format(len(os.listdir(data_path + "/View_001"))))
        f.write("imWidth={}\n".format(w))
        f.write("imHeight={}\n".format(h))
        f.write("imExt=.jpg")
    for fname in os.listdir(data_path + "/View_001"):
        frame_id = int(fname.strip("frame_").strip(".jpg"))
        print("{}".format(frame_id + 1).zfill(6) + ".jpg")
        if not os.path.exists(data_path + "/img1/"):
            os.makedirs(data_path + "/img1/")
            shutil.copy(data_path + "/View_001/" + fname,
                        data_path + "/img1/" + "{}".format(frame_id + 1).zfill(6) + ".jpg")

        # shutil.copy()


if __name__ == '__main__':
    read_cvml_detections('E:/PyProjects/datasets/data-tud/det/PETS2009/PETS2009-S2L1-c1-det.xml',
                         r'E:\PyProjects\datasets\Crowd_PETS09\S2\L1\Time_12-34\det/det.txt')
    read_cvml_groundtruth('E:/PyProjects/datasets/data-tud/gt/PETS2009/PETS2009-S2L1.xml',
                          r'E:\PyProjects\datasets\Crowd_PETS09\S2\L1\Time_12-34/gt/gt.txt')
    move()
