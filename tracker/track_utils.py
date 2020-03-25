# -*- coding: utf-8 -*-
import argparse
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import transforms
import os
import cv2
from .loader import extract_image_patch, load_detections, detection_from_frame_id, show_bboxes
from scipy.optimize import linear_sum_assignment
from sklearn.utils.linear_assignment_ import linear_assignment

from PIL import Image

# --------

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--which_epoch', default='last', type=str, help='0,1,2,3...or last')
parser.add_argument('--test_dir', default='../Market/pytorch', type=str, help='./test_data')
parser.add_argument('--name', default='ft_ResNet50', type=str, help='save model path')
parser.add_argument('--batchsize', default=32, type=int, help='batchsize')
parser.add_argument('--use_dense', action='store_true', help='use densenet121')
parser.add_argument('--PCB', action='store_true', help='use PCB')
parser.add_argument('--multi', action='store_true', help='use multiple query')
parser.add_argument('--fp16', action='store_true', help='use fp16.')

opt = parser.parse_args()
###load config###
# load the training config
# config_path = os.path.join('./model', opt.name, 'opts.yaml')
# with open("../siamese/opts.yaml", 'r') as stream:
#     config = yaml.load(stream)

str_ids = opt.gpu_ids.split(',')
# which_epoch = opt.which_epoch
name = opt.name
test_dir = opt.test_dir

gpu_ids = []
for str_id in str_ids:
    id = int(str_id)
    if id >= 0:
        gpu_ids.append(id)

# set gpu ids
if len(gpu_ids) > 0:
    torch.cuda.set_device(gpu_ids[0])
    cudnn.benchmark = True

data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_network(network):
    save_path = r"E:\PyProjects\MOT\tracker\siamese\net_last.pth"
    # os.path.join('./siamese', 'net_last.pth')
    network.load_state_dict(torch.load(save_path), strict=False)
    return network


######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#
def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def get_id(img_path):
    camera_id = []
    labels = []
    for path, v in img_path:
        # filename = path.split('/')[-1]
        filename = os.path.basename(path)
        label = filename[0:4]
        camera = filename.split('c')[1]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


def compute_similarty(model, img1, img2):
    _, sim = model(img1, img2)
    return sim


def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox = np.asarray(bbox)
    candidates = np.asarray(candidates)
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def get_around_bboxes(src, targets, th=0):
    """

    :param src: src bbox
    :param targets: targets bbox
    :param th:
    :return:
    """
    IOU = iou(src, targets)
    output = []
    indices = []
    not_indices = []
    for idx, bbox in enumerate(targets):
        if IOU[idx] > th:
            output.append(bbox)
            indices.append(idx)
        else:
            not_indices.append(idx)
    return output, indices, not_indices


def cost_matrix(model, srcs, targets):
    """

    :param model:
    :param srcs: img
    :param targets: img
    :return:
    """
    cost = np.zeros((len(srcs), len(targets)))
    for i, src in enumerate(srcs):
        img1 = Variable(torch.Tensor(data_transforms(src)).cuda()).unsqueeze_(0)
        for j, tar in enumerate(targets):
            img2 = Variable(torch.Tensor(data_transforms(tar)).cuda()).unsqueeze_(0)
            _, sim = model(img1, img2)
            cost[i, j] = sim[0][0]
    return cost


def match(model, frame1, frame2, bbox1, bbox2):
    with torch.no_grad():
        srcs = []
        targets = []
        not_around_bboxes = []
        for bbox in bbox1:
            srcs.append(extract_image_patch(frame1, bbox))
            not_around_bboxes.append(get_around_bboxes(bbox, bbox2)[2])
        for bbox in bbox2:
            targets.append(extract_image_patch(frame2, bbox))
        cost_m = cost_matrix(model, srcs, targets)
        for i, nabbs in enumerate(not_around_bboxes):
            for j in nabbs:
                cost_m[i, j] = 1.0
        # print(cost_m)
        indices = linear_assignment(cost_m)
        detection_indices = np.arange(len(targets))
        track_indices = np.arange(len(srcs))
        max_distance = 0.5
        matches, unmatched_tracks, unmatched_detections = [], [], []
        for col, detection_idx in enumerate(detection_indices):
            if col not in indices[:, 1]:
                unmatched_detections.append(detection_idx)
        for row, track_idx in enumerate(track_indices):
            if row not in indices[:, 0]:
                unmatched_tracks.append(track_idx)
        for row, col in indices:
            track_idx = track_indices[row]
            detection_idx = detection_indices[col]
            if cost_m[row, col] > max_distance:
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(detection_idx)
            else:
                matches.append((track_idx, detection_idx))
        return matches, unmatched_tracks, unmatched_detections


# b1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# b2 = [0, 1, 4, 2, 3, 12, 8, 5, 10, 7, 11, 9]
# for i in range(len(b1)):
#     show_bboxes(frame1, [bbox1[b1[i]]], txt=i)
#     show_bboxes(frame2, [bbox2[b2[i]]], txt=i)
# img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
# show_image(img2)
# show_image(np.hstack([img1, img2]))
if __name__ == '__main__':
    path = "/home/wss/PycharmProjects/MOT/data/MOT16/train/MOT16-02/img1/000001.jpg"
    frame1 = cv2.imread(path)
    path = "/home/wss/PycharmProjects/MOT/data/MOT16/train/MOT16-02/img1/000002.jpg"
    frame2 = cv2.imread(path)
    data = load_detections("/home/wss/PycharmProjects/MOT/data/MOT16/train/MOT16-02/det/det.txt")
    bbox1, score1 = detection_from_frame_id(data, 1)
    bbox2, score2 = detection_from_frame_id(data, 2)
    # candidates, indices = get_around_bboxes(bbox1[0], bbox2)
    show_bboxes(frame1, bbox1)
    #
    # model_structure = ft_net(751)
    # model = load_network(model_structure)
    #
    # # Remove the final fc layer and classifier layer
    #
    # # Change to test mode
    # model = model.eval()
    # model = model.cuda()
    # matches, unmatched_tracks, unmatched_detections = match(model, frame1, frame2, bbox1, bbox2)
    # print(matches, unmatched_tracks, unmatched_detections)
