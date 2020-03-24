import cv2
from tqdm import tqdm
import colorsys
import os
import numpy as np

rgb = lambda x: colorsys.hsv_to_rgb((x * 0.41) % 1, 1., 1. - (int(x * 0.41) % 4) / 5.)
colors = lambda x: (int(rgb(x)[0] * 255), int(rgb(x)[1] * 255), int(rgb(x)[2] * 255))

# tracking_file = "./output/Time_12-34.txt"
# imgs_path = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L1\Time_12-34\img1'
# save_path = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L1\Time_12-34\tracklets'
# seqinfo = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L1\Time_12-34\seqinfo.ini'

# tracking_file = "./output/L2001.txt"
# imgs_path = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L2\L2001\img1'
# save_path = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L2\L2001\tracklets'
# seqinfo = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L2\L2001\seqinfo.ini'

tracking_file = "./output/L3001.txt"
imgs_path = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L3\L3001\img1'
save_path = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L3\L3001\tracklets'
seqinfo = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L3\L3001\seqinfo.ini'
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

def draw_tracklets(seq_name, start=1, end=100):
    seqs = os.listdir(imgs_path)
    seqs = tqdm(range(len(seqs)))
    seqs.set_description('')
    foot_set = {}
    track_ids = set()
    for i, seq in enumerate(seqs):
        # print('{}-{}'.format(mot.sqm[index], seq + 1))
        frame_id = i + 1
        if frame_id < start or frame_id > end:
            continue
        file = os.path.join(imgs_path, "{}".format(frame_id).zfill(6)) + ".jpg"
        frame = cv2.imread(file)
        tracks_i = tracks[tracks[:, 0] == frame_id]
        bboxes, ids = tracks_i[:, 2:6], tracks_i[:, 1]
        if len(bboxes) == 0:
            continue
        bboxes = [bboxes[j] for j in range(bboxes.shape[0])]

        for i, bbox in enumerate(bboxes):
            t = ids[i]
            track_ids = track_ids | set([t])
            foot = get_foot_point(bbox)
            foot_set.setdefault(t, []).append(foot)
            bbox[2:] += bbox[:2]
            top_left = (int(bbox[0]), int(bbox[1]))
            bottom_right = (int(bbox[2]), int(bbox[3]))
            cv2.rectangle(frame, top_left, bottom_right,
                          colors(t),
                          2)
            cv2.putText(frame, "{}".format(t), top_left, 0,
                        5e-3 * 200, colors(t), 2)
            for j in track_ids:
                foots = foot_set[j]
                for k, foot in enumerate(foots):
                    if k + 2 > len(foots):
                        break
                    distance = ((foot[0] - foots[k + 1][0]) ** 2 + (foot[1] - foots[k + 1][1]) ** 2) ** 0.5
                    if distance > 40:
                        # foot_set[j] = []
                        continue
                    cv2.line(frame, foot, foots[k + 1], colors(j), 2)

        cv2.imshow('', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.imwrite(os.path.join(save_path, "{}_{}-{}.jpg".format(seq_name, start, end)), frame)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    for i in range(4):
        draw_tracklets("S2_L3", 100 * i, 100 * (i + 1))
