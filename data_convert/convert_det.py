import numpy as np
import os
import cv2
import shutil

det_file = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L1\Time_12-34\PETS2009_S2L1_View1.txt'
save_path = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L1\Time_12-34\det'
save_img = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L1\Time_12-34\img1'
data_path = r"E:\PyProjects\datasets\Crowd_PETS09\S2\L1\Time_12-34"

# det_file = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L2\L2001\PETS2009_S2L2_View1.txt'
# save_path = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L2\L2001\det'
# save_img = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L2\L2001\img1'
# data_path = r"E:\PyProjects\datasets\Crowd_PETS09\S2\L2\L2001"

# det_file = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L3\L3001\PETS2009_S2L3_View1.txt'
# save_path = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L3\L3001\det'
# save_img = r'E:\PyProjects\datasets\Crowd_PETS09\S2\L3\L3001\img1'
# data_path = r"E:\PyProjects\datasets\Crowd_PETS09\S2\L3\L3001"
if not os.path.exists(save_path):
    os.makedirs(save_path)
if not os.path.exists(save_img):
    os.makedirs(save_img)
dets = np.genfromtxt(det_file, delimiter=',', dtype=np.float32)
with open(save_path + "/det.txt", 'w') as f:
    for det in dets:
        frame_idx = det[1]
        x = det[2]
        y = det[3]
        w = det[6]
        h = det[7]
        confidence = det[8]
        f.write(
            '{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n'.format(int(frame_idx) + 1, -1, x, y, w, h, confidence))


def move(data_path=r"E:\PyProjects\datasets\Crowd_PETS09\S2\L2"):
    img = cv2.imread(data_path + "/View_001/frame_0000.jpg")
    h, w, c = img.shape
    with open(os.path.join(data_path, "seqinfo.ini"), "w") as f:
        f.write("[Sequence]\n")
        f.write("name=PETS2009_S2_L2_V001\n")
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

# move(data_path)
