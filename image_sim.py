# -*- coding: utf-8 -*-

from __future__ import print_function, division

from random import shuffle
import argparse
import torch
from torch.autograd import Variable
from torchvision import transforms
import cv2
import os

import yaml
from siamese.model import Siamese, TripletSiamese
import numpy as np

# --------

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str,
                    help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch', default='last',
                    type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='../Market/pytorch',
                    type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50',
                    type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--PCB', action='store_true', help='use PCB')
parser.add_argument('--multi', action='store_true', help='use multiple query')
parser.add_argument('--fp16', action='store_true', help='use fp16.')

opt = parser.parse_args()

# load the training config
config_path = os.path.join(r'E:\PyProjects\MOT\siamese', 'opts.yaml')
with open(config_path, 'r') as stream:
    config = yaml.load(stream)

str_ids = opt.gpu_ids.split(',')
# which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

use_gpu = torch.cuda.is_available()


def load_network(network, save_path=r"E:\PyProjects\MOT\siamese\net_last.pth"):
    network.load_state_dict(torch.load(save_path))
    return network


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def verify(model, img1, img2):
    with torch.autograd.no_grad():
        img1 = torch.Tensor(data_transforms(img1)).cuda().unsqueeze_(0)
        img2 = torch.Tensor(data_transforms(img2)).cuda().unsqueeze_(0)
        f1 = model.get_feature(img1)
        f2 = model.get_feature(img2)
        # score = torch.mm(f1, f2.view(-1, 1))
        # score = score.squeeze(1).cpu()
        # score = score.numpy()
        sim = model.verify(f1, f2)
        sim = sim.cpu().data.numpy()[0][1]
    return sim


if __name__ == '__main__':
    from sklearn.utils.linear_assignment_ import linear_assignment

    model = TripletSiamese(751)
    model = load_network(model, save_path=r"E:\PyProjects\MOT\siamese\tripletnet_last.pth")
    # Change to test mode
    model = model.eval()
    if use_gpu:
        model = model.cuda()
    #
    # # img1 = cv2.imread(r"E:\PyProjects\datasets\Market\pytorch\gallery\0061\0061_c2s1_008001_01.jpg")
    # # img1 = cv2.imread(r"E:\PyProjects\datasets\Market\pytorch\gallery\0025\0025_c2s1_001751_01.jpg")
    # # img1 = cv2.imread(r"E:\PyProjects\MOT\track_img\S2_L1\1_243.jpg")
    # # img1 = cv2.imread(r"E:\PyProjects\MOT\track_img\S2_L1\17_185.jpg")
    # img1 = cv2.imread(r"E:\PyProjects\MOT\track_img\S2_L1\10_488.jpg")
    # img1 = cv2.resize(img1, (128, 256))
    # # img2 = cv2.imread(r"E:\PyProjects\datasets\Market\pytorch\gallery\0061\0061_c4s1_009401_02.jpg")
    # # img2 = cv2.imread(r"E:\PyProjects\MOT\track_img\S2_L1\1_372.jpg")
    # # img2 = cv2.imread(r"E:\PyProjects\MOT\track_img\S2_L1\9_2.jpg")
    # # img2 = cv2.imread(r"E:\PyProjects\datasets\Market\pytorch\gallery\0038\0038_c4s1_004226_02.jpg")
    # img2 = cv2.imread(r"E:\PyProjects\MOT\track_img\S2_L1\11_26.jpg")
    # img2 = cv2.resize(img2, (128, 256))
    # s = verify(model, img1, img2)
    # match = np.hstack((img1, img2))
    # white = np.ones(
    #     shape=(80, match.shape[1], match.shape[2])) * 255
    # match = np.vstack([match, white])
    # cv2.putText(match, "{:.2f}%".format(s * 100), (80, 40 + 256), 0,
    #             5e-3 * 200, (0, 0, 0), 2)
    # cv2.imwrite("./image_sim_test/triple_sim2.jpg", match)
    # print("done.")

    # 比较背景与人的相似度
    from object_matching import draw_grid
    from PIL import Image
    from tqdm import tqdm

    seq = 4
    bg_list = os.listdir(f"./output/crop/bg{seq}")
    person_list = os.listdir(f"output/crop/person{seq}")
    sim_matrix = np.zeros((len(bg_list), len(person_list)))
    img_bgs = []
    print(len(bg_list), len(person_list))

    bg_list = tqdm(bg_list)
    for i, bg in enumerate(bg_list):
        bg = cv2.imread(f"output/crop/bg{seq}/{bg}")
        img_bgs.append(bg)
        img_persons = []
        for j, per in enumerate(person_list):
            person = cv2.imread(f"output/crop/person{seq}/{per}")
            img_persons.append(person)
            s = verify(model, bg, person)
            sim_matrix[i, j] = s
    num_img_bg = len(bg_list)
    num_img_per = len(person_list)
    h_query = np.hstack(img_bgs)
    v_gallery = np.vstack(np.asarray(img_persons))
    bg = np.ones(
        shape=(256 * (num_img_per + 1), 128 * (num_img_bg + 1), 3), dtype='uint8') * 255
    for i, sim_i in enumerate(sim_matrix):
        for j, sim_j in enumerate(sim_i):
            loc_x = (i + 1) * 128 + 20
            loc_y = 120 + 256 * (j + 1)
            cv2.putText(bg, "{:.2f}".format(100 * sim_j), (loc_x, loc_y), 0,
                        5e-3 * 200, (0, 0, 0), 2)
        cv2.putText(bg, "{:.2f}".format(100 * sim_i.mean()), (loc_x, loc_y + 40), 0,
                    5e-3 * 200, (0, 0, 0), 2)
    draw_grid(bg)
    h_query = Image.fromarray(cv2.cvtColor(h_query, cv2.COLOR_BGR2RGB))
    v_gallery = Image.fromarray(cv2.cvtColor(v_gallery, cv2.COLOR_BGR2RGB))
    bg = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
    bg.paste(h_query, (128, 0, 128 * (num_img_bg + 1), 256))
    bg.paste(v_gallery, (0, 256, 128, 256 * (num_img_per + 1)))
    bg.save(f"./image_sim_test/bg_person{seq}.jpg")
    print("done.")
