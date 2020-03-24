# -*- coding: utf-8 -*-

from __future__ import print_function, division

import argparse
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
import cv2
import os

import yaml
from tracker.siamese.model import ft_net
from tracker.track_utils import load_network
import numpy as np

np.random.seed(0)
# --------

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='../Market/pytorch', type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=256, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--PCB', action='store_true', help='use PCB')
parser.add_argument('--multi', action='store_true', help='use multiple query')
parser.add_argument('--fp16', action='store_true', help='use fp16.')

opt = parser.parse_args()

# load the training config
config_path = os.path.join(r'E:\PyProjects\MOT\tracker\siamese', 'opts.yaml')
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


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def verify(model, img1, img2):
    img1 = torch.Tensor(data_transforms(img1)).cuda().unsqueeze_(0)
    img2 = torch.Tensor(data_transforms(img2)).cuda().unsqueeze_(0)
    f1 = model.get_feature(img1)
    f2 = model.get_feature(img2)
    sim = model.verify(f1, f2)
    sim = sim.cpu().data.numpy()[0][1]
    return sim


if __name__ == '__main__':
    crop_path = "./track_img/S2_L1"
    import os
    from tqdm import tqdm

    # Load Collected data Trained model
    model_structure = ft_net(751)
    model = load_network(model_structure)
    # Change to test mode
    model = model.eval()
    if use_gpu:
        model = model.cuda()

    num_image_pair = 200

    imgs_list = os.listdir(crop_path)
    maxlen = len(imgs_list)
    image_pairs = np.random.randint(1, maxlen, size=(num_image_pair, 2))
    match = np.zeros(num_image_pair, dtype=int)
    sim = np.zeros(num_image_pair)
    gt = np.zeros(num_image_pair, dtype=int)
    image_pairs = tqdm(image_pairs)

    for i, pair_idx in enumerate(image_pairs):
        img1 = cv2.imread(os.path.join(crop_path, imgs_list[pair_idx[0]]))
        img2 = cv2.imread(os.path.join(crop_path, imgs_list[pair_idx[1]]))
        c1 = imgs_list[pair_idx[0]].split("_")[0]
        c2 = imgs_list[pair_idx[1]].split("_")[0]
        if c1 == c2:
            gt[i] = 1
        s = verify(model, img1, img2)
        sim[i] = s
    for th in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995]:
        for i, s in enumerate(sim):
            if s > th:
                match[i] = 1
            else:
                match[i] = 0
        correct = 0
        for i in range(num_image_pair):
            if gt[i] == match[i]:
                correct += 1
        acc = correct / num_image_pair
        print("acc@{:2f}:\t{:4f}%".format(th, acc * 100))
