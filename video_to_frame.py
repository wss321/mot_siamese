import cv2
import os

# video_path = r"E:\PyProjects\MOT\videos\Time_12-34.avi"
# save_path = r"E:\PyProjects\MOT\videos\L1001_iouth0.99"

# video_path = r"E:\PyProjects\MOT\videos\L2001.avi"
# save_path = r"E:\PyProjects\MOT\videos\L2001"

# video_path = r"E:\PyProjects\MOT\videos\L3001.avi"
# save_path = r"E:\PyProjects\MOT\videos\L3001"
video_path = r"E:\PyProjects\MOT\output\videos\MOT16-02.avi"
save_path = r"E:\PyProjects\MOT\output\videos\MOT16-02"

if not os.path.exists(save_path):
    os.makedirs(save_path)
video = cv2.VideoCapture(video_path)
i = 0
while True:
    i += 1
    ret, frame = video.read()  # frame shape 640*480*3
    if ret is not True or frame is None:
        break
    cv2.imwrite(os.path.join(save_path, "{}.jpg".format(i)), frame)
