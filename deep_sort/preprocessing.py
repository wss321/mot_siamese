import numpy as np


def non_max_suppression(boxes, max_bbox_overlap, scores=None):
    """Suppress overlapping detections.

    Original code from [1]_ has been adapted to include confidence score.

    .. [1] http://www.pyimagesearch.com/2015/02/16/
           faster-non-maximum-suppression-python/

    Examples
    --------

        # >>> boxes = [d.roi for d in detections]
        # >>> scores = [d.confidence for d in detections]
        # >>> indices = non_max_suppression(boxes, max_bbox_overlap, scores)
        # >>> detections = [detections[i] for i in indices]

    Parameters
    ----------
    boxes : ndarray
        Array of ROIs (x, y, width, height).
    max_bbox_overlap : float
        ROIs that overlap more than this values are suppressed.
    scores : Optional[array_like]
        Detector confidence score.

    Returns
    -------
    List[int]
        Returns indices of detections that have survived non-maxima suppression.

    """
    if len(boxes) == 0:
        return []

    boxes = boxes.astype(np.float)
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    if scores is not None:
        idxs = np.argsort(scores)
    else:
        idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(
            idxs, np.concatenate(
                ([last], np.where(overlap > max_bbox_overlap)[0])))

    return pick


def py_greedy_nms(dets, iou_th, scores=None):
    """Pure python implementation of traditional greedy NMS.
    Args:
        dets (numpy.array): Detection results with shape `(num, 5)`,
            data in second dimension are [x1, y1, x2, y2, score] respectively.
        iou_th (float): Drop the boxes that overlap with current
            maximum > thresh.
    Returns:
        numpy.array: Retained boxes.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2] + dets[:, 0]
    y2 = dets[:, 3] + dets[:, 1]
    if scores is None and len(dets[0]) > 4:
        scores = dets[:, 4]
    else:
        raise ValueError('scores must be given')
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    sorted_idx = scores.argsort()[::-1]

    keep = []
    while sorted_idx.size > 0:
        i = sorted_idx[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[sorted_idx[1:]])
        yy1 = np.maximum(y1[i], y1[sorted_idx[1:]])
        xx2 = np.minimum(x2[i], x2[sorted_idx[1:]])
        yy2 = np.minimum(y2[i], y2[sorted_idx[1:]])

        w = np.maximum(xx2 - xx1 + 1, 0.0)
        h = np.maximum(yy2 - yy1 + 1, 0.0)
        inter = w * h
        iou = inter / (areas[i] + areas[sorted_idx[1:]] - inter)

        retained_idx = np.where(iou <= iou_th)[0]
        sorted_idx = sorted_idx[retained_idx + 1]

    return keep


def soft_nms(dets, scores=None, method='linear', iou_th=0.3, sigma=0.5, score_th=0.001):
    """Pure python implementation of soft NMS as described in the paper
    `Improving Object Detection With One Line of Code`_.
    Args:
        dets (numpy.array): Detection results with shape `(num, 5)`,
            data in second dimension are [x1, y1, x2, y2, score] respectively.
        method (str): Rescore method. Only can be `linear`, `gaussian`
            or 'greedy'.
        iou_th (float): IOU threshold. Only work when method is `linear`
            or 'greedy'.
        sigma (float): Gaussian function parameter. Only work when method
            is `gaussian`.
        score_th (float): Boxes that score less than th.
    Returns:
        numpy.array: Retained boxes.
    .. _`Improving Object Detection With One Line of Code`:
        https://arxiv.org/abs/1704.04503
    """
    if scores is not None and len(dets[0]) == 4:
        scores = scores.reshape((-1, 1))
        dets = np.concatenate((dets, scores), axis=1)
    if method not in ('linear', 'gaussian', 'greedy'):
        raise ValueError('method must be linear, gaussian or greedy')

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2] + dets[:, 0]
    y2 = dets[:, 3] + dets[:, 1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # expand dets with areas, and the second dimension is
    # x1, y1, x2, y2, score, area
    dets = np.concatenate((dets, areas[:, None]), axis=1)

    # retained_box = []
    keep = []
    while dets.size > 0:
        max_idx = np.argmax(scores, axis=0)
        dets[[0, max_idx], :] = dets[[max_idx, 0], :]
        # retained_box.append(dets[0, :-1])

        xx1 = np.maximum(dets[0, 0], dets[1:, 0])
        yy1 = np.maximum(dets[0, 1], dets[1:, 1])
        xx2 = np.minimum(dets[0, 2], dets[1:, 2])
        yy2 = np.minimum(dets[0, 3], dets[1:, 3])

        w = np.maximum(xx2 - xx1 + 1, 0.0)
        h = np.maximum(yy2 - yy1 + 1, 0.0)
        inter = w * h
        iou = inter / (dets[0, 5] + dets[1:, 5] - inter)

        if method == 'linear':
            weight = np.ones_like(iou)
            weight[iou > iou_th] -= iou[iou > iou_th]
        elif method == 'gaussian':
            weight = np.exp(-(iou * iou) / sigma)
        else:  # traditional nms
            weight = np.ones_like(iou)
            weight[iou > iou_th] = 0

        dets[1:, 4] *= weight
        retained_idx = np.where(dets[1:, 4] >= score_th)[0]
        dets = dets[retained_idx + 1, :]

    # return np.vstack(retained_box)
    return keep


def soft(dets, confidence=None, ax=None):
    if confidence is not None and len(dets[0]) == 4:
        confidence = confidence.reshape((-1, 1))
        dets = np.concatenate((dets, confidence), axis=1)
    thresh = .9  # .6
    score_thresh = .7
    sigma = .5
    N = len(dets)
    x1 = dets[:, 0].copy()
    y1 = dets[:, 1].copy()
    x2 = dets[:, 2].copy()
    y2 = dets[:, 3].copy()
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    ious = np.zeros((N, N))
    for i in range(N):
        xx1 = np.maximum(x1[i], x1)
        yy1 = np.maximum(y1[i], y1)
        xx2 = np.minimum(x2[i], x2)
        yy2 = np.minimum(y2[i], y2)

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas - inter)
        ious[i, :] = ovr

    i = 0
    while i < N:
        maxpos = dets[i:N, 4].argmax()
        maxpos += i
        dets[[maxpos, i]] = dets[[i, maxpos]]
        areas[[maxpos, i]] = areas[[i, maxpos]]
        confidence[[maxpos, i]] = confidence[[i, maxpos]]
        ious[[maxpos, i]] = ious[[i, maxpos]]
        ious[:, [maxpos, i]] = ious[:, [i, maxpos]]

        ovr_bbox = np.where((ious[i, :N] > thresh))[0]
        avg_std_bbox = (dets[ovr_bbox, :4] / confidence[ovr_bbox]).sum(0) / (1 / confidence[ovr_bbox]).sum(0)
        # if cfg.STD_NMS:
        dets[i, :4] = avg_std_bbox
        # else:
        #     assert (False)

        areai = areas[i]
        pos = i + 1
        while pos < N:
            if ious[i, pos] > 0:
                ovr = ious[i, pos]
                dets[pos, 4] *= np.exp(-(ovr * ovr) / sigma)
                if dets[pos, 4] < 0.001:
                    dets[[pos, N - 1]] = dets[[N - 1, pos]]
                    areas[[pos, N - 1]] = areas[[N - 1, pos]]
                    confidence[[pos, N - 1]] = confidence[[N - 1, pos]]
                    ious[[pos, N - 1]] = ious[[N - 1, pos]]
                    ious[:, [pos, N - 1]] = ious[:, [N - 1, pos]]
                    N -= 1
                    pos -= 1
            pos += 1
        i += 1
    keep = [i for i in range(N)]
    return dets[keep], keep


def remove_overlay(boxes):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]
    # areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    index = [i for i in range(len(boxes))]
    # overlay = np.zeros((len(boxes), len(boxes)), dtype=int)
    remove = []
    for i in range(len(boxes)):
        for j in range(len(boxes)):
            if i >= j:
                continue
            if y1[i] < y1[j] and x1[i] < x1[j] and x2[i] > x2[j] and y2[i] > y2[j]:
                remove.append(i)
    # print(remove)
    return set(index) - set(remove)


if __name__ == '__main__':
    boxes = np.array([[655.79, 413.27, 120.26, 362.77],
                      [1247.5, 441.39, 36.321, 110.96],
                      [566.69, 453.55, 112.14, 338.41],
                      [755.53, 446.86, 84.742, 256.23],
                      [513.16, 473.76, 97.492, 294.47],
                      [770.33, 363.04, 112.14, 338.41],
                      [673., 321., 159., 479.],
                      [1014.4, 449.63, 25.39, 78.17],
                      [1097., 433., 39., 119.],
                      [1001., 441., 39., 119.],
                      [549.75, 412.56, 170.48, 513.45]], dtype=np.float32)

    # print('greedy result:')
    # print(py_greedy_nms(boxes, 0.7))
    # print('soft nms result:')
    # print(soft_nms(boxes, method='gaussian'))
    # print('soft nms result:')
    # print(soft_nms(boxes, method='linear'))

    print(remove_overlay(boxes))
