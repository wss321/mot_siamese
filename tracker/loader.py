import numpy as np
import cv2


def extract_image_patch(image, bbox, patch_shape=None):
    """Extract image patch from bounding box.
    截图框

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    if patch_shape:
        image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image


def show_image(image):
    cv2.imshow('', image)
    if cv2.waitKey(10000) & 0xFF == ord('q'):
        pass


def load_detections(detections):
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
            dets.append({'bbox': [bb[0], bb[1], bb[2], bb[3]], 'score': s})
        data.append(dets)

    return data


def detection_from_frame_id(detections, frame_id):
    return [dets['bbox'] for dets in detections[frame_id - 1]], [dets['score'] for dets in detections[frame_id - 1]]


def to_tlbr(xywh):
    """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
    `(top left, bottom right)`.
    """
    ret = xywh.copy()
    ret[2:] += ret[:2]
    return ret


def show_bboxes(img, bboxes, txt=None):
    for i, bbox in enumerate(bboxes):
        bbox = to_tlbr(np.array(bbox))
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
                      (255, 255, 255), 2)
    if txt:
        cv2.putText(img, "{:.2f}".format(txt), (int(bbox[0]), int(bbox[1])), 0,
                    5e-3 * 200, (0, 255, 0), 2)
    show_image(img)


if __name__ == '__main__':
    path = "/home/wss/PycharmProjects/MOT/data/MOT16/train/MOT16-02/img1/000001.jpg"
    img = cv2.imread(path)
    data = load_detections("/home/wss/PycharmProjects/MOT/data/MOT16/train/MOT16-02/det/det.txt")
    bbox, score = detection_from_frame_id(data, 1)
    show_bboxes(img, bbox)
    img = extract_image_patch(img, bbox[0])
    show_image(img)
