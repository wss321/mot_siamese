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
from siamese.model import Siamese, TripletSiamese
import copy
from deep_sort.preprocessing import non_max_suppression, remove_overlay


def rgb(x): return colorsys.hsv_to_rgb((x * 0.41) %
                                       1, 1., 1. - (int(x * 0.41) % 4) / 5.)


def colors(x): return (int(rgb(x)[0] * 255),
                       int(rgb(x)[1] * 255), int(rgb(x)[2] * 255))


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


class MotLoader(object):
    def __init__(self, main_dir):
        self.main_dir = main_dir
        self.map_dir, self.sqm = self.__get_seqmaps()

    def load_detections(self, det_file):
        """
        Loads detections stored in a mot-challenge like formatted CSV or numpy array (fieldNames = ['frame', 'id', 'x', 'y',
        'w', 'h', 'score']).
        Args:
            detections：det.txt file path
        Returns:
            list: list containing the detections for each frame.
        """

        data = []
        if type(det_file) is str:
            raw = np.genfromtxt(det_file, delimiter=',', dtype=np.float32)
        else:
            # assume it is an array
            assert isinstance(
                det_file, np.ndarray), "only numpy arrays or *.csv paths are supported as detections."
            raw = det_file.astype(np.float32)

        end_frame = int(np.max(raw[:, 0]))
        for i in range(1, end_frame + 1):
            idx = raw[:, 0] == i
            bbox = raw[idx, 2:6]
            if raw.shape[1] == 2058:
                feature = raw[idx, 10:]
            else:
                feature = raw[idx, 7:]
            # bbox[:, 2:4] += bbox[:, 0:2]  # x1, y1, w, h -> x1, y1, x2, y2
            scores = raw[idx, 6]
            dets = []
            j = 0
            for bb, s in zip(bbox, scores):
                dets.append(
                    {'bbox': (bb[0], bb[1], bb[2], bb[3]), 'score': s, "feature": feature[j]})
                j += 1
            data.append(dets)

        return data

    def __get_seqmaps(self):
        seqmaps = os.listdir(self.main_dir)
        seqmaps_dir = [self.main_dir + '/{}'.format(i) for i in
                       seqmaps]

        return seqmaps_dir, seqmaps

    def detFromFrameID(self, detections, frame_id):
        try:
            return [dets['bbox'] for dets in detections[frame_id - 1]], \
                   [dets['score'] for dets in detections[frame_id - 1]], \
                   [dets['feature'] for dets in detections[frame_id - 1]]
        except:
            return None, None, None


def load_network(network, model_path):
    network.load_state_dict(torch.load(model_path))
    return network


def tracking(args):
    # Definition of the parameters
    logger.setLevel(logging.INFO)
    logger.info("Loading Model...")
    mot = MotLoader(args.data_dir)
    if args.triplet_model:
        model = TripletSiamese(args.num_classes)
    else:
        model = Siamese(args.num_classes)
    model = load_network(model, args.model_path)
    model = model.eval()
    mkdirs(args.output_dir)
    for index, seq_dir in enumerate(mot.map_dir):
        # if "MOT16-13" not in mot.sqm[index]:
        #     continue
        imgs = os.listdir(seq_dir + '/img1')
        if args.use_feature:
            dets = np.load(os.path.join(args.data_dir, mot.sqm[index], "det",
                                        mot.sqm[index] + "_{}.npy".format(args.det_file.split(".")[0])))
            dets = mot.load_detections(dets)
        else:
            dets = mot.load_detections(
                seq_dir + '/det/{}'.format(args.det_file))
        seqinfo = seq_dir + '/seqinfo.ini'
        with open(seqinfo, 'r') as info:
            lines = info.readlines()
            lines = [line.split('=') for line in lines]
            lines = {line[0]: line[1].strip("\n")
                     for line in lines if len(line) > 1}
        w, h, ext, img_dir = int(lines['imWidth']), int(
            lines['imHeight']), lines['imExt'], lines["imDir"]
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        if args.show:
            cv2.namedWindow(
                'Tracking-{}'.format(mot.sqm[index]), cv2.WINDOW_NORMAL)
        if args.show_kalman:
            cv2.namedWindow(
                'Kalman-{}'.format(mot.sqm[index]), cv2.WINDOW_NORMAL)
        if args.save_video:
            video_dir = args.output_dir + "/videos/"
            save_frame_dir = args.output_dir + f"/images/{mot.sqm[index]}"
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)
            if not os.path.exists(save_frame_dir):
                os.makedirs(save_frame_dir)
            out = cv2.VideoWriter(
                video_dir + mot.sqm[index] + '.avi', fourcc, 7, (w, h))
        metric = nn_matching.SimilarityMetric(
            -np.log(args.sim_th), copy.deepcopy(model), args.budget)

        tracker = Tracker(metric, img_shape=(w, h), max_age=args.age,
                          max_iou_distance=1 - args.iou_th, mean=True)
        logger.info("Tracking...")
        with open(args.output_dir + mot.sqm[index] + '.txt', 'w') as f:
            img_len = tqdm(range(1, len(imgs) + 1))
            img_len.set_description(mot.sqm[index])
            for frame_id in img_len:
                file = os.path.join(seq_dir, img_dir, "{}".format(
                    frame_id).zfill(6)) + ".jpg"
                frame = cv2.imread(file)
                bboxes, scores, features = mot.detFromFrameID(dets, frame_id)
                if bboxes is None:
                    continue
                keep0 = non_max_suppression(np.asarray(bboxes), max_bbox_overlap=args.overlap_th,
                                            scores=np.asarray(scores))
                # _, keep = soft(np.asarray(bboxes), np.asarray(scores))
                keep1 = remove_overlay(np.asarray(bboxes), thresh=0.7)
                keep = list(set(keep0) & set(keep1))
                # keep = soft_nms(np.asarray(bboxes), scores=np.asarray(scores), iou_th=args.overlap_th)
                if "RCNN" in mot.sqm[index] or "SDP" in mot.sqm[index] or "poi" in args.det_file:
                    bboxes = [bboxes[i] for i, s in enumerate(
                        scores) if s > args.score_th and i in keep]
                    features = [features[i] for i, s in enumerate(
                        scores) if s > args.score_th and i in keep]
                    scores = [s for i, s in enumerate(
                        scores) if s > args.score_th and i in keep]
                else:
                    bboxes = [bboxes[i] for i, s in enumerate(
                        scores) if sigmoid(s) > args.score_th and i in keep]
                    features = [features[i] for i, s in enumerate(
                        scores) if sigmoid(s) > args.score_th and i in keep]
                    scores = [sigmoid(s) for i, s in enumerate(
                        scores) if sigmoid(s) > args.score_th and i in keep]

                if args.use_feature and len(features) > 0:
                    rois = features
                else:
                    rois = extract(frame, bboxes)
                    for i, roi in enumerate(rois):
                        crop_path = f"{args.output_dir}/crop/{mot.sqm[index]}"
                        if not os.path.exists(crop_path):
                            os.makedirs(crop_path)
                        cv2.imwrite(f"{crop_path}/{frame_id}_{i}.jpg", roi)

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
                    cv2.imwrite(f"{save_frame_dir}/{frame_id}.jpg", frame)
                    out.write(frame)
        if args.save_video:
            out.release()
        cv2.destroyAllWindows()
        logger.info("Track Done.")
        eval(args, [mot.sqm[index]], save=False)


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
                line = save_format.format(
                    frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval(args, seqs=None, save=True):
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
    print(strsummary)
    if save:
        with open(os.path.join(result_root, "eval_result.txt"), 'a') as f:
            f.write(str(vars(args)) + "\n")
            f.write(strsummary + "\n" * 3)
