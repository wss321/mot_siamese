import numpy as np
import cv2
import os, errno
from eval.eval_mot import extract, logger, load_network, Siamese, tqdm
import logging
from deep_sort.nn_matching import data_transforms
import torch
import argparse


def generate_detections(args):
    """Generate detections with features.
    Parameters
    ----------
    encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    data_dir : str
        Path to the MOTChallenge directory (can be either train or test).
    output_dir
        Path to the output directory. Will be created if it does not exist.
    detection_dir
        Path to custom detections. The directory structure should be the default
        MOTChallenge structure: `[sequence]/det/det.txt`. If None, uses the
        standard MOTChallenge detections.
    """
    detection_dir = args.data_dir
    try:
        os.makedirs(args.output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(args.output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % args.output_dir)
    logger.setLevel(logging.INFO)
    logger.info("Loading Model...")
    model = Siamese(751)
    model = load_network(model, args.model_dir)
    model = model.cuda()
    model = model.eval()

    for sequence in os.listdir(args.data_dir):
        logger.info("Processing %s" % sequence)
        sequence_dir = os.path.join(args.data_dir, sequence)

        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}

        detection_file = os.path.join(
            detection_dir, sequence, "det/{}".format(args.det_file))
        detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_out = []

        frame_indices = detections_in[:, 0].astype(np.int)
        min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()
        frame_idxs = tqdm(range(min_frame_idx, max_frame_idx + 1))
        frame_idxs.set_description(sequence)
        for frame_idx in frame_idxs:
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            if frame_idx not in image_filenames:
                print("WARNING could not find image for frame %d" % frame_idx)
                continue
            bgr_image = cv2.imread(image_filenames[frame_idx])
            # print(rows[:, 2:6])
            rois = extract(bgr_image, rows[:, 2:6].copy())
            # print(rois)
            # cv2.imshow("", rois[0])
            # cv2.waitKey(2000)
            with torch.no_grad():
                rois = np.asarray([data_transforms(src).numpy() for src in rois])
                rois = torch.Tensor(rois).cuda()
                f_rois = model.get_feature(rois).cpu().data.numpy()
            detections_out += [np.r_[(row, feature)] for row, feature
                               in zip(rows, f_rois)]

        output_filename = os.path.join(detection_dir, sequence, "det",
                                       "{}_{}.npy".format(sequence, args.det_file.split(".")[0]))
        np.save(
            output_filename, np.asarray(detections_out), allow_pickle=False)
        # np.savetxt(os.path.join(output_dir, "det.txt"), np.asarray(detections_out), delimiter=",",
        #            fmt="%d,%d,%d,%d,%d,%d,%.4f" + ",%.4f" * 2048)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluating')
    parser.add_argument('--data_dir', default="../datasets/MOT16/train/", type=str, help='dataset path')
    parser.add_argument('--output_dir', default='./output', type=str, help='output dir')
    parser.add_argument('--det_file', default='det.txt', type=str, help='detection file')
    parser.add_argument('--model_dir', default=r"E:\PyProjects\MOT\tracker\siamese\net_last.pth", type=str,
                        help='siamese model path')
    args = parser.parse_args()
    generate_detections(args)
    # python generate_det_feature.py --det_file det.txt --data_dir ../datasets/MOT16/train/
