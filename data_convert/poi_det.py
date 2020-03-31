import numpy as np
import os

poi_path = r"E:\PyProjects\datasets\MOT16\POI_det"
dets = os.listdir(poi_path)
for det_file in dets:
    try:
        save_to = r"E:\PyProjects\datasets\MOT16\test\{}\det\poi.txt".format(det_file.strip(".npy"))
        det_file = os.path.join(poi_path, det_file)
        det = np.load(det_file)
        np.savetxt(save_to, det[:, :10], delimiter=",", fmt="%d,%d,%d,%d,%d,%d,%.4f,%d,%d,%d")
    except:
        pass
print("Done.")
