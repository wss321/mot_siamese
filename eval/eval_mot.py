import numpy as np
from deep_sort.tracker import Tracker
from tools.extractor import extract
from deep_sort import nn_matching
from deep_sort.detection import Detection
import os
import torch
import cv2
import logging
import motmetrics as mm
from tools.log import logger
from tools.evaluation import Evaluator
from tqdm import tqdm
import colorsys
from tracker.siamese.model import Siamese
import copy

rgb = lambda x: colorsys.hsv_to_rgb((x * 0.41) % 1, 1., 1. - (int(x * 0.41) % 4) / 5.)
colors = lambda x: (int(rgb(x)[0] * 255), int(rgb(x)[1] * 255), int(rgb(x)[2] * 255))


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


class MotLoader(object):
    def __init__(self, main_dir):
        self.main_dir = main_dir
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
            feature = raw[idx, 7:]
            # bbox[:, 2:4] += bbox[:, 0:2]  # x1, y1, w, h -> x1, y1, x2, y2
            scores = raw[idx, 6]
            dets = []
            j = 0
            for bb, s in zip(bbox, scores):
                dets.append({'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': s, "feature": feature[j]})
                j += 1
            data.append(dets)

        return data

    def __get_seqmaps(self):
        seqmaps = os.listdir(self.main_dir)
        seqmaps_dir = [self.main_dir + '/{}'.format(i) for i in
                       seqmaps]

        return seqmaps_dir, seqmaps

    def detFromFrameID(self, detections, frame_id):
        return [dets['bbox'] for dets in detections[frame_id - 1]], \
               [dets['score'] for dets in detections[frame_id - 1]], \
               [dets['feature'] for dets in detections[frame_id - 1]]


def load_network(network, model_path):
    network.load_state_dict(torch.load(model_path), strict=False)
    return network


def tracking(args):
    # Definition of the parameters
    logger.setLevel(logging.INFO)
    logger.info("Loading Model...")
    mot = MotLoader(args.data_dir)
    model = Siamese(args.num_classes)
    model = load_network(model, args.model_path)
    model = model.eval()
    mkdirs(args.output_dir)
    for index, seq_dir in enumerate(mot.map_dir):
        imgs = os.listdir(seq_dir + '/img1')
        if args.use_feature:
            dets = np.load(os.path.join(args.data_dir, mot.sqm[index], "det",
                                        mot.sqm[index] + "_{}.npy".format(args.det_file.split(".")[0])))
            dets = mot.load_detections(dets)
        else:
            dets = mot.load_detections(seq_dir + '/det/{}'.format(args.det_file))
        seqinfo = seq_dir + '/seqinfo.ini'
        with open(seqinfo, 'r') as info:
            lines = info.readlines()
            lines = [line.split('=') for line in lines]
            lines = {line[0]: line[1].strip("\n") for line in lines if len(line) > 1}
        w, h, ext, img_dir = int(lines['imWidth']), int(lines['imHeight']), lines['imExt'], lines["imDir"]
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        if args.show:
            cv2.namedWindow('Tracking-{}'.format(mot.sqm[index]), cv2.WINDOW_NORMAL)
        if args.show_kalman:
            cv2.namedWindow('Kalman-{}'.format(mot.sqm[index]), cv2.WINDOW_NORMAL)
        if args.save_video:
            video_dir = args.output_dir + "/videos/"
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            out = cv2.VideoWriter(video_dir + mot.sqm[index] + '.avi', fourcc, 7, (w, h))
        metric = nn_matching.SimilarityMetric(-np.log(args.sim_th), copy.deepcopy(model), args.budget)

        tracker = Tracker(metric, img_shape=(w, h), max_age=args.age, max_iou_distance=1 - args.iou_th, mean=True)
        logger.info("Tracking...")
        with open(args.output_dir + mot.sqm[index] + '.txt', 'w') as f:
            img_len = tqdm(range(1, len(imgs) + 1))
            img_len.set_description(mot.sqm[index])
            for frame_id in img_len:
                file = os.path.join(seq_dir, img_dir, "{}".format(frame_id).zfill(6)) + ".jpg"
                frame = cv2.imread(file)
                bboxes, scores, features = mot.detFromFrameID(dets, frame_id)

                if "RCNN" in mot.sqm[index] or "SDP" in mot.sqm[index] or "poi" in args.det_file:
                    bboxes = [bboxes[i] for i, s in enumerate(scores) if s > 0.2]
                    features = [features[i] for i, s in enumerate(scores) if s > 0.2]
                    scores = [s for i, s in enumerate(scores) if s > 0.2]
                else:
                    bboxes = [bboxes[i] for i, s in enumerate(scores) if sigmoid(s) > args.score_th]
                    features = [features[i] for i, s in enumerate(scores) if sigmoid(s) > args.score_th]
                    scores = [sigmoid(s) for i, s in enumerate(scores) if sigmoid(s) > args.score_th]
                if args.use_feature and features[0].shape[0] > 0:
                    rois = features
                else:
                    rois = extract(frame, bboxes)
                    # with torch.no_grad():
                    #     from deep_sort.nn_matching import data_transforms
                    #     rois = np.asarray([data_transforms(src).numpy() for src in rois])
                    #     rois = torch.Tensor(rois).cuda()
                    #     f_rois = model.get_feature(rois).cpu().data.numpy()
                    #     print("rois", f_rois)
                detections = [Detection(bbox_with_roi[0], scores[idx], bbox_with_roi[1]) for idx, bbox_with_roi
                              in enumerate(zip(bboxes, rois))]
                tracker.predict()
                tracks_dets = tracker.update(detections)
                frame_copy = copy.copy(frame)

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
                if args.show_kalman:
                    for track in tracker.tracks:
                        t = track.track_id
                        bbox = track.to_tlwh()  # # bbox predicted by kalman
                        bbox[2:] += bbox[:2]
                        cv2.rectangle(frame_copy, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                                      colors(t),
                                      2)
                        cv2.putText(frame_copy, "{}".format(t), (int(bbox[0]), int(bbox[1])), 0,
                                    5e-3 * 200, colors(t), 2)
                    cv2.imshow('Kalman-{}'.format(mot.sqm[index]), frame_copy)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if args.show:
                    cv2.imshow('Tracking-{}'.format(mot.sqm[index]), frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                if args.save_video:
                    out.write(frame)
        if args.save_video:
            out.release()
        cv2.destroyAllWindows()
        logger.info("Track Done.")
        eval(args, [mot.sqm[index]])


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


def eval(args, seqs=None):
    logger.setLevel(logging.INFO)
    result_root = args.output_dir
    mkdirs(result_root)
    data_type = 'mot'
    if seqs is None:
        seqs = os.listdir(args.data_dir)
    accs = []
    for seq in seqs:
        logger.info('start seq: {}'.format(seq))
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        # eval
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(args.data_dir, seq, data_type)
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
    with open(os.path.join(result_root, "eval_result.txt"), 'a') as f:
        f.write(strsummary + "\n")
    print(strsummary)
