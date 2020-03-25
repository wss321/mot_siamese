import cv2
from tqdm import tqdm
import colorsys
import os
import numpy as np
from deep_sort import extractor as gdet

"""将S2L1裁剪作为目标匹配"""
rgb = lambda x: colorsys.hsv_to_rgb((x * 0.41) % 1, 1., 1. - (int(x * 0.41) % 4) / 5.)
colors = lambda x: (int(rgb(x)[0] * 255), int(rgb(x)[1] * 255), int(rgb(x)[2] * 255))

tracking_file = r"E:\PyProjects\datasets\Crowd_PETS09\S2\L1\Time_12-34\gt\gt.txt"
# tracking_file = r"E:\PyProjects\MOT\output\Time_12-34.txt"
imgs_path = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L1\Time_12-34\img1'
save_path = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L1\Time_12-34\tracklets_assiociate'
seqinfo = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L1\Time_12-34\seqinfo.ini'

# tracking_file = "./output/L2001.txt"
# imgs_path = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L2\L2001\img1'
# save_path = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L2\L2001\tracklets'
# seqinfo = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L2\L2001\seqinfo.ini'

# tracking_file = "./output/L3001.txt"
# imgs_path = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L3\L3001\img1'
# save_path = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L3\L3001\tracklets'
# seqinfo = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L3\L3001\seqinfo.ini'
if not os.path.exists(save_path):
    os.makedirs(save_path)


def get_foot_point(bbox):
    y = bbox[1] + bbox[3]
    x = bbox[0] + 0.5 * bbox[2]
    return int(x), int(y)


seqs = os.listdir(imgs_path)
tracks = np.genfromtxt(tracking_file, delimiter=',', dtype=np.int32)

with open(seqinfo, 'r') as info:
    lines = info.readlines()
    lines = [line.split('=') for line in lines]
    lines = {line[0]: line[1].strip("\n") for line in lines if len(line) > 1}
w, h, ext, img_dir = int(lines['imWidth']), int(lines['imHeight']), lines['imExt'], lines["imDir"]


# fourcc = cv2.VideoWriter_fourcc(*'MJPG')

def crop(seq_name):
    if not os.path.exists(seq_name):
        os.makedirs(seq_name)
    extractor = gdet.extractor()
    seqs = os.listdir(imgs_path)
    seqs = tqdm(range(len(seqs)))
    seqs.set_description('croping')
    for i, seq in enumerate(seqs):
        frame_id = i + 1
        file = os.path.join(imgs_path, "{}".format(frame_id).zfill(6)) + ".jpg"
        frame = cv2.imread(file)
        tracks_i = tracks[tracks[:, 0] == frame_id]
        bboxes, ids = tracks_i[:, 2:6], tracks_i[:, 1]
        if len(bboxes) == 0:
            continue
        bboxes = [bboxes[j] for j in range(bboxes.shape[0])]
        rois = extractor(frame, bboxes)
        for j, roi in enumerate(rois):
            tracklet_path = os.path.join(seq_name, str(ids[j]))
            if not os.path.exists(tracklet_path):
                os.makedirs(tracklet_path)
            cv2.imwrite(tracklet_path + "/frame_{}.jpg".format(frame_id), roi)


def crop_for_matching(save_to):
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    extractor = gdet.extractor()
    seqs = os.listdir(imgs_path)
    seqs = tqdm(range(len(seqs)))
    seqs.set_description('croping')
    for i, seq in enumerate(seqs):
        frame_id = i + 1
        file = os.path.join(imgs_path, "{}".format(frame_id).zfill(6)) + ".jpg"
        frame = cv2.imread(file)
        tracks_i = tracks[tracks[:, 0] == frame_id]
        bboxes, ids = tracks_i[:, 2:6], tracks_i[:, 1]
        if len(bboxes) == 0:
            continue
        bboxes = [bboxes[j] for j in range(bboxes.shape[0])]
        rois = extractor(frame, bboxes)
        for j, roi in enumerate(rois):
            cv2.imwrite(save_to + "/{}_{}.jpg".format(ids[j], frame_id), roi)


np.random.seed(0)
if __name__ == '__main__':
    # crop_path = save_path
    crop_path = "./track_img/S2_L1"
    crop_for_matching(crop_path)
    print("done.")
