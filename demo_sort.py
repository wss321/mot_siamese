import os
import numpy as np
from PIL import Image
from deep_sort.tracker import Tracker
from deep_sort import nn_matching, extractor as gdet
from deep_sort.detection import Detection
import cv2
from tqdm import tqdm

class Mot(object):
    def __init__(self, main_dir, eval_type='train'):
        self.main_dir = main_dir
        self.type = eval_type
        if self.type not in ['train', 'test']:
            raise ValueError('eval_type must be one of train or test')

        self.map_dir, self.sqm = self.__get_seqmaps()

    def load_detections(self, detections):
        """
        Loads detections stored in a mot-challenge like formatted CSV or numpy array (fieldNames = ['frame', 'id', 'x', 'y',
        'w', 'h', 'score']).
        Args:
            detections：det.txt file path
        Returns:
            list: list containing the detections for each frame.
        """

        data = []
        if type(detections) is str:
            raw = np.genfromtxt(detections, delimiter=',', dtype=np.float32)
        else:
            # assume it is an array
            assert isinstance(detections, np.ndarray), "only numpy arrays or *.csv paths are supported as detections."
            raw = detections.astype(np.float32)

        end_frame = int(np.max(raw[:, 0]))
        for i in range(1, end_frame + 1):
            idx = raw[:, 0] == i
            bbox = raw[idx, 2:6]
            # bbox[:, 2:4] += bbox[:, 0:2]  # x1, y1, w, h -> x1, y1, x2, y2

            scores = raw[idx, 6]
            dets = []
            for bb, s in zip(bbox, scores):
                dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': s})
            data.append(dets)

        return data

    def __get_seqmaps(self):
        seqmaps = os.listdir(os.path.join(self.main_dir, self.type))
        seqmaps_dir = [os.path.join(self.main_dir, self.type) + '/{}'.format(i) for i in
                       seqmaps]

        return seqmaps_dir, seqmaps

    def detection_from_frame_id(self, detections, frame_id):
        return [dets['bbox'] for dets in detections[frame_id - 1]], [dets['score'] for dets in detections[frame_id - 1]]


def eval(data_dir, output_dir, e_type='train', show=False, save_video=False):
    # Definition of the parameters
    max_cosine_distance = 0.5
    nn_budget = None

    # deep_sort
    model_filename = 'models/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    mot = Mot(data_dir, e_type)
    for index, det_dir in enumerate(mot.map_dir):
        seqs = os.listdir(det_dir + '/img1')
        dets = mot.load_detections(det_dir + '/det/det.txt')
        seqinfo = det_dir + '/seqinfo.ini'
        with open(seqinfo, 'r') as info:
            lines = info.readlines()
            lines = [line.split('=') for line in lines]
            lines = {line[0]: line[1] for line in lines if len(line) > 1}
        w, h, ext, frame_rate = int(lines['imWidth']), int(lines['imHeight']), lines['imExt'], int(lines['frameRate'])
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        if save_video:
            out = cv2.VideoWriter(mot.sqm[index] + '.avi', fourcc, 7, (960, 540))
        tracker = Tracker(metric, img_shape=(w, h), max_eu_dis=0.1 * np.sqrt((w ** 2 + h ** 2)),
                          center_assingment=False, mean=True)
        with open(output_dir + mot.sqm[index] + '.txt', 'w') as f:
            for seq in tqdm(range(len(seqs))):
                # print('track {}'.format(seq + 1))
                frame_id = seq + 1
                file = os.path.join(det_dir, "img1", "{}".format(frame_id).zfill(6)) + ".jpg"
                frame = cv2.imread(file)  # np.asarray(Image.open(det_dir + '/img1/' + seq))
                bboxes, scores = mot.detection_from_frame_id(dets, frame_id)
                features = encoder(frame, bboxes)
                detections = [Detection(bbox_and_feature[0], scores[idx], bbox_and_feature[1]) for idx, bbox_and_feature
                              in
                              enumerate(zip(bboxes, features))]
                tracker.predict()
                tracks_dets = tracker.update(detections, frame)
                # frame = frame
                for td in tracks_dets:
                    t = td[0]
                    d = detections[td[1]].tlwh
                    f.write(
                        '{},{},{:.4f},{:.4f},{:.4f},{:.4f},{},{},{},{}\n'.format(frame_id, t, d[0], d[1], d[2], d[3], -1, -1, -1, -1))
                    if show or save_video:
                        bbox = detections[td[1]].to_tlbr()
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                      (255, 255, 255),
                                      2)
                        cv2.putText(frame, "{}".format(t), (int(bbox[0]), int(bbox[1])), 0,
                                    5e-3 * 200, (0, 255, 0), 2)
                if show:
                    frame = cv2.resize(frame, (960, 540))
                    cv2.imshow('', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if save_video:
                    out.write(frame)
        if save_video:
            out.release()


if __name__ == '__main__':
    eval(r'./data/MOT16',
         r"./output/", show=True, save_video=False)
