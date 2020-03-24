import cv2
from tqdm import tqdm
import colorsys
import os
import numpy as np
from deep_sort import generate_detections as gdet

rgb = lambda x: colorsys.hsv_to_rgb((x * 0.41) % 1, 1., 1. - (int(x * 0.41) % 4) / 5.)
colors = lambda x: (int(rgb(x)[0] * 255), int(rgb(x)[1] * 255), int(rgb(x)[2] * 255))

tracking_file = r"E:\PyProjects\datasets\Crowd_PETS09\S2\L1\Time_12-34\gt\gt.txt"
imgs_path = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L1\Time_12-34\img1'
save_path = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L1\Time_12-34\tracklets'
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
            cv2.imwrite(seq_name + "/{}_{}.jpg".format(ids[j], frame_id), roi)


np.random.seed(0)
if __name__ == '__main__':
    crop_path = "./track_img/S2_L1"
    import os

    num_image_pair = 10

    imgs_list = os.listdir(crop_path)
    maxlen = len(imgs_list)
    # crop("./track_img/S2_L1")
    # print("done.")
    image_pairs = np.random.randint(1, maxlen, size=(num_image_pair, 2))
    similaritys = np.zeros(num_image_pair)
    print(similaritys)
