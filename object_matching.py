# -*- coding: utf-8 -*-

from __future__ import print_function, division
import matplotlib.pyplot as plt
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
from PIL import Image

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str,
                    help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch', default='last',
                    type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='../Market/pytorch/',
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
    network.load_state_dict(torch.load(save_path), strict=False)
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


def get_query(data_path, num_image=10, seed=0):
    imgs_list = os.listdir(data_path)
    imgs_label = [int(imgs_list[i].split("_")[0])
                  for i in range(len(imgs_list))]
    label_set = list(set(imgs_label))[:num_image]
    # shuffle(label_set)
    imgs_label = np.asarray(imgs_label, dtype=int)
    imgs = []
    for s in label_set:
        np.random.seed(s + seed)
        index = np.where(imgs_label == s)
        rnd = np.random.randint(0, len(index[0]))
        index = index[0][rnd]
        img = cv2.imread(os.path.join(crop_path, imgs_list[index]))
        imgs.append(img)
    img = np.hstack(imgs)
    # cv2.imwrite("./query.jpg", img)
    return imgs, label_set


def get_gallery(data_path, num_image=10, seed=0):
    imgs_list = os.listdir(data_path)
    imgs_label = [int(imgs_list[i].split("_")[0])
                  for i in range(len(imgs_list))]
    label_set = list(set(imgs_label))[:num_image]
    imgs_label = np.asarray(imgs_label, dtype=int)
    imgs = []
    for s in label_set:
        np.random.seed(s + seed)
        index = np.where(imgs_label == s)
        rnd = np.random.randint(0, len(index[0]))
        index = index[0][rnd]
        img = cv2.imread(os.path.join(crop_path, imgs_list[index]))
        imgs.append(img)
    img = np.hstack(imgs)
    # cv2.imwrite("./gallery.jpg", img)
    return imgs, label_set


def draw_grid(img, line_color=(0, 0, 0), thickness=1, type_=cv2.LINE_AA, pxstep_h=128, pxstep_v=256):
    '''(ndarray, 3-tuple, int, int) -> void
    draw gridlines on img
    line_color:
        BGR representation of colour
    thickness:
        line thickness
    type:
        8, 4 or cv2.LINE_AA
    pxstep:
        grid line frequency in pixels
    '''
    x = pxstep_h
    y = pxstep_v
    while x < img.shape[1]:
        cv2.line(
            img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep_h

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y),
                 color=line_color, lineType=type_, thickness=thickness)
        y += pxstep_v


if __name__ == '__main__':
    from sklearn.utils.linear_assignment_ import linear_assignment

    crop_path = "./track_img/S2_L1"
    # crop_path = r"E:\PyProjects\datasets\Market\pytorch\gallery"
    num_img = 8
    query, q_l = get_query(crop_path, num_img, seed=2108)
    gallery, g_l = get_gallery(crop_path, num_img, seed=230)
    sim_matrix = np.zeros((len(q_l), len(g_l)))
    model_structure = TripletSiamese(751)
    model = load_network(
        model_structure, save_path=r"E:\PyProjects\MOT\siamese\tripletnet_last.pth")
    # Change to test mode
    model = model.eval()
    m = []
    if use_gpu:
        model = model.cuda()
    for i in range(len(q_l)):
        img1 = query[i]
        for j in range(len(g_l)):
            if i > j:
                continue
            img2 = gallery[j]
            s = verify(model, img1, img2)
            if s > 0.99:
                m.append((i + 1, j + 1))
            sim_matrix[i, j] = 1 - s
            sim_matrix[j, i] = 1 - s
    # print(m)
    # print(1-sim_matrix)
    indices = linear_assignment(sim_matrix)
    orig_q = query
    orig_g = gallery
    query = np.asarray(query)[indices[:, 0]]
    # print(query.shape)
    white = np.ones(
        shape=(query.shape[0], 80, query.shape[2], query.shape[3])) * 255
    gallery = np.asarray(gallery)[indices[:, 1]]
    sim = 1 - sim_matrix
    txt = []
    for i in indices:
        txt.append("{:.2f}".format(100 * sim[i[0], i[1]]))
    h_query = np.hstack(query)
    white = np.hstack(white)
    h_gallery = np.hstack(gallery)
    match = np.vstack([h_query, h_gallery, white])
    for i, t in enumerate(txt):
        cv2.putText(match, t, (i * 128 + 20, 40 + 256 * 2), 0,
                    5e-3 * 200, (0, 0, 0), 2)
    cv2.imwrite("./image_sim_test/triplet_match.jpg", match)
    num_img = len(orig_g)
    h_query = np.hstack(orig_q)
    v_gallery = np.vstack(orig_g)
    bg = np.ones(
        shape=(256 * (num_img + 1), 128 * (num_img + 1), 3), dtype='uint8') * 255
    for i, sim_i in enumerate(sim):
        for j, sim_j in enumerate(sim_i):
            loc_x = (i + 1) * 128 + 20
            loc_y = 120 + 256 * (j + 1)
            cv2.putText(bg, "{:.2f}".format(100 * sim_j), (loc_x, loc_y), 0,
                        5e-3 * 200, (0, 0, 0), 2)
    draw_grid(bg)
    # cv2.imwrite("./image_sim_test/triplet_bg.jpg", bg)
    # print(h_query.shape)
    # print(h_query.dtype)
    h_query = Image.fromarray(cv2.cvtColor(h_query, cv2.COLOR_BGR2RGB))
    v_gallery = Image.fromarray(cv2.cvtColor(v_gallery, cv2.COLOR_BGR2RGB))
    bg = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
    bg.paste(h_query, (128, 0, 128 * (num_img + 1), 256))
    bg.paste(v_gallery, (0, 256, 128, 256 * (num_img + 1)))
    bg.save("./image_sim_test/triplet_matrix.jpg")
    print("done.")
    # print(sim_matrix)
    # print(indices)
    # print(sim_matrix[indices])
    # import os
    # from tqdm import tqdm
    #
    # # Load Collected data Trained model
    # model_structure = Siamese(751)
    # model = load_network(model_structure)
    # # Change to test mode
    # model = model.eval()
    # if use_gpu:
    #     model = model.cuda()
    #
    # num_image_pair = 200
    #
    # imgs_list = os.listdir(crop_path)
    # maxlen = len(imgs_list)
    # image_pairs = np.random.randint(1, maxlen, size=(num_image_pair, 2))
    # match = np.zeros(num_image_pair, dtype=int)
    # sim = np.zeros(num_image_pair)
    # gt = np.zeros(num_image_pair, dtype=int)
    # image_pairs = tqdm(image_pairs)
    #
    # for i, pair_idx in enumerate(image_pairs):
    #     img1 = cv2.imread(os.path.join(crop_path, imgs_list[pair_idx[0]]))
    #     img2 = cv2.imread(os.path.join(crop_path, imgs_list[pair_idx[1]]))
    #     c1 = imgs_list[pair_idx[0]].split("_")[0]
    #     c2 = imgs_list[pair_idx[1]].split("_")[0]
    #     if c1 == c2:
    #         gt[i] = 1
    #     s = verify(model, img1, img2)
    #     sim[i] = s
    # for th in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995]:
    #     for i, s in enumerate(sim):
    #         if s > th:
    #             match[i] = 1
    #         else:
    #             match[i] = 0
    #     correct = 0
    #     for i in range(num_image_pair):
    #         if gt[i] == match[i]:
    #             correct += 1
    #     acc = correct / num_image_pair
    #     print("acc@{:2f}:\t{:4f}%".format(th, acc * 100))
