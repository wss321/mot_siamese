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


def ioa(bbox, candidate):
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidate[:2]
    candidates_br = candidate[:2] + candidate[2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[0]),
               np.maximum(bbox_tl[1], candidates_tl[1])]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[0]),
               np.minimum(bbox_br[1], candidates_br[1])]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod()
    # area_bbox = bbox[2:].prod()
    area_candidates = candidate[2:].prod()
    return area_intersection / area_candidates


def remove_area(boxes, ratio=[1.8, 0.2]):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2] + boxes[:, 0]
    y2 = boxes[:, 3] + boxes[:, 1]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    mean_area = areas.mean()
    keep = []
    for i in range(len(boxes)):
        if areas[i] > mean_area * ratio[0] or areas[i] < mean_area * ratio[1]:
            continue
        keep.append(i)
    return keep


def remove_h(boxes, thresh=30):
    return np.where(boxes[:, 3] > thresh)[0]


def remove_overlay(boxes, thresh=0.8):
    if len(boxes) == 0:
        return []
    remain = [i for i in range(len(boxes))]
    keep = []
    while len(boxes) > 0:
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2] + boxes[:, 0]
        y2 = boxes[:, 3] + boxes[:, 1]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        max_now = np.argmax(areas)
        L = len(boxes)
        flag = 0
        for j in range(L):
            # if y1[max_now] < y1[j] and x1[max_now] < x1[j] and x2[max_now] > x2[j] and y2[max_now] > y2[j]:
            #     boxes = np.delete(boxes, max_now, axis=0)
            #     del remain[max_now]
            #     flag = 1
            #     break
            if j == max_now:
                continue
            oa = ioa(boxes[max_now], boxes[j])
            if oa > thresh:
                boxes = np.delete(boxes, max_now, axis=0)
                del remain[max_now]
                flag = 1
                break
        if flag == 0:
            boxes = np.delete(boxes, max_now, axis=0)
            keep.append(remain[max_now])
            del remain[max_now]
    return keep

    # return set(index) - set(remove)


import numpy as np


def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=10):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1


def drawpoly(img, pts, color, thickness=1, style='dotted', ):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        drawline(img, s, e, color, thickness, style)


def drawrect(img, pt1, pt2, color, thickness=1, style='dotted'):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    drawpoly(img, pts, color, thickness, style)


if __name__ == '__main__':
    frame_id = 115
    img = f"E:/PyProjects/datasets/MOT16/train/MOT16-02/img1/000{frame_id}.jpg"
    det_file = r"E:\PyProjects\datasets\MOT16\train\MOT16-02\det\det.txt"
    import cv2
    from copy import copy

    img = cv2.imread(img)
    img0 = copy(img)
    raw = np.genfromtxt(det_file, delimiter=',', dtype=np.float32)
    idx = raw[:, 0] == frame_id
    boxes = raw[idx, 2:6]
    boxes0 = copy(boxes)
    # boxes0[:, 2:] += boxes0[:, :2]

    keep1 = remove_overlay(boxes0, 0.70)
    print(keep1)

    boxes = np.asarray([boxes0[i] for i in keep1])
    keep = remove_area(boxes, [100, 0])
    keep2 = [keep1[i] for i in keep]
    print(keep2)

    boxes = np.asarray([boxes0[i] for i in keep2])
    keep = remove_h(boxes, 30)
    keep3 = [keep2[i] for i in keep]
    print(keep3)

    remove = set([i for i in range(len(boxes0))]) - set(keep3)
    print(remove)
    boxes = np.asarray([boxes0[i] for i in keep3])
    boxes[:, 2:] += boxes[:, :2]
    for bbox in boxes:
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                      (0, 255, 0), 2)

    boxes0[:, 2:] += boxes0[:, :2]
    for re in remove:
        drawrect(img, (int(boxes0[re][0]), int(boxes0[re][1])), (int(boxes0[re][2]), int(boxes0[re][3])),
                 (0, 0, 255), 3, style='dotted')

    # for bbox in boxes0:
    #     cv2.rectangle(img0, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
    #                   (0, 255, 0), 2)
    #
    # img0 = cv2.resize(img0, (960, 540))
    img = cv2.resize(img, (960, 540))
    # img = np.vstack([img0, img])
    cv2.imwrite(f"../output/remove_overlay_{frame_id}.jpg", img)
    cv2.imshow("", img)
    if cv2.waitKey(1000) & 0xFF == ord('q'):
        pass
    # print('greedy result:')
    # print(py_greedy_nms(boxes, 0.7))
    # print('soft nms result:')
    # print(soft_nms(boxes, method='gaussian'))
    # print('soft nms result:')
    # print(soft_nms(boxes, method='linear'))

    # print(remove_overlay(boxes))
