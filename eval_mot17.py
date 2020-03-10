import os
import numpy as np
from PIL import Image
from deep_sort.tracker import Tracker
from deep_sort import nn_matching, generate_detections as gdet
from deep_sort.detection import Detection
import cv2
import os
import cv2
import logging
import motmetrics as mm
# from utils import visualization as vis
from tools.log import logger
# from utils.timer import Timer
from tools.evaluation import Evaluator
from tqdm import tqdm
import colorsys

rgb = lambda x: colorsys.hsv_to_rgb((x * 0.41) % 1, 1., 1. - (int(x * 0.41) % 4) / 5.)
colors = lambda x: (int(rgb(x)[0] * 255), int(rgb(x)[1] * 255), int(rgb(x)[2] * 255))


class MotLoader(object):
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

    def detFromFrameID(self, detections, frame_id):
        return [dets['bbox'] for dets in detections[frame_id - 1]], [dets['score'] for dets in detections[frame_id - 1]]


def eval(data_dir, output_dir, e_type='train', show=False, save_video=False, budget=32, iou_th=0.3, area_rate_th=2):
    # Definition of the parameters

    mot = MotLoader(data_dir, e_type)
    for index, det_dir in enumerate(mot.map_dir):
        seqs = os.listdir(det_dir + '/img1')
        # if not "FRCNN" in det_dir:
        #     continue
        dets = mot.load_detections(det_dir + '/det/det.txt')
        # dets = mot.load_detections(det_dir + '/gt/gt.txt')
        seqinfo = det_dir + '/seqinfo.ini'
        with open(seqinfo, 'r') as info:
            lines = info.readlines()
            lines = [line.split('=') for line in lines]
            lines = {line[0]: line[1] for line in lines if len(line) > 1}
        w, h, ext, frame_rate = int(lines['imWidth']), int(lines['imHeight']), lines['imExt'], int(lines['frameRate'])
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        if show:
            cv2.namedWindow('{}'.format(mot.sqm[index]), cv2.WINDOW_NORMAL)
        if save_video:
            out = cv2.VideoWriter("./videos/" + mot.sqm[index] + '.avi', fourcc, 7, (w, h))
        distance_th = 0.4
        encoder = gdet.extractor()
        metric = nn_matching.SimilarityDistanceMetric(distance_th, budget, iou_th, area_rate_th)
        tracker = Tracker(metric, img_shape=(w, h), max_eu_dis=0.1 * np.sqrt((w ** 2 + h ** 2)),
                          center_assingment=False, mean=True)
        with open(output_dir + mot.sqm[index] + '.txt', 'w') as f:
            seqs = tqdm(range(len(seqs)))
            seqs.set_description(mot.sqm[index])
            for seq in seqs:
                # print('{}-{}'.format(mot.sqm[index], seq + 1))
                frame_id = seq + 1
                file = os.path.join(det_dir, "img1", "{}".format(frame_id).zfill(6)) + ".jpg"
                frame = cv2.imread(file)  # np.asarray(Image.open(det_dir + '/img1/' + seq))
                bboxes, scores = mot.detFromFrameID(dets, frame_id)

                bboxes = [bboxes[i] for i, s in enumerate(scores) if s > 0]
                scores = [s for i, s in enumerate(scores) if s > 0]

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
                        '{},{},{:.2f},{:.2f},{:.2f},{:.2f},{},{},{},{}\n'.format(frame_id, t, d[0], d[1], d[2], d[3], 1,
                                                                                 -1, -1, -1))
                    bbox = detections[td[1]].to_tlbr()
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                  colors(t),
                                  2)
                    cv2.putText(frame, "{}".format(t), (int(bbox[0]), int(bbox[1])), 0,
                                5e-3 * 200, colors(t), 2)
                # frame = cv2.resize(frame, (960, 540))
                if show:
                    cv2.imshow('{}'.format(mot.sqm[index]), frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if save_video:
                    out.write(frame)
        if save_video:
            out.release()
        cv2.destroyAllWindows()


def mkdirs(path):
    if os.path.exists(path):
        return
    os.makedirs(path)


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    elif data_type == 'kitti':
        save_format = '{frame} {id} pedestrian 0 0 -10 {x1} {y1} {x2} {y2} -10 -10 -10 -1000 -1000 -1000 -10\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            if data_type == 'kitti':
                frame_id -= 1
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def main(data_root='/data/MOT16/train', det_root=None,
         seqs=('MOT16-05',), exp_name='demo', save_image=False, show_image=True):
    logger.setLevel(logging.INFO)
    result_root = "./output"
    # mkdirs(result_root)
    data_type = 'mot'

    # run tracking
    accs = []
    for seq in seqs:
        output_dir = os.path.join(data_root, 'outputs', seq) if save_image else None

        logger.info('start seq: {}'.format(seq))
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))

    # get summary
    # metrics = ['mota', 'num_switches', 'idp', 'idr', 'idf1', 'precision', 'recall']
    metrics = mm.metrics.motchallenge_metrics
    # metrics = None
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    with open(os.path.join(result_root, "mot17_result"), 'a') as f:
        f.write(strsummary + "\n")
    print(strsummary)


if __name__ == '__main__':
    seqs = os.listdir(r'E:\PyProjects\datasets/MOT17/train')
    print(seqs)
    eval(r'E:\PyProjects\datasets/MOT17',
         r"./output/", show=True, save_video=True, budget=1, iou_th=0.0, area_rate_th=2)
    main(data_root=r'E:\PyProjects\datasets/MOT17/train',
         seqs=seqs,
         exp_name='mot17_val',
         show_image=False)
