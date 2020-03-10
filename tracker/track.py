import os
import cv2
import numpy as np
from .track_utils import extract_image_patch, detection_from_frame_id, load_detections


class TrackState(object):
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track(object):
    """
    单个目标跟踪器
    Parameters
    ----------
    track_id : int
        A unique track identifier.

    Attributes
    ----------
    track_id : int
        A unique track identifier.

    """

    def __init__(self, data_path, track_id):
        self.track_id = track_id
        self.frame_id = []  # 帧id
        self.bboxes = []  # 检测框
        self.data_path = data_path
        self.img_patch = None
        self.time_since_update = 0  # 多长时间没有更新预测值(检测)了
        self.age = 1
        self.hits = 1  # 预测的总次数
        self.color = (0, 255, 0)
        self.state = TrackState.Tentative
        self._n_init = 3
        self._max_age = 30

    def update(self, frame_id, bbox):
        """Perform Kalman filter measurement update step and update the feature
        cache.
        更新状态
        Parameters
        ----------
        bbox : (x,y,w,h)
        """
        self.hits += 1
        self.time_since_update = 0
        self.img_patch = self._get_img_patch(frame_id, np.array(bbox))
        self.frame_id.append(frame_id)
        self.bboxes.append(bbox)
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def _get_img_patch(self, frame_id, bbox):
        img_path = os.path.join(self.data_path, "img1", "{}".format(frame_id).zfill(6)) + ".jpg"
        frame1 = cv2.imread(img_path)
        return extract_image_patch(frame1, bbox)

    def to_tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.bboxes[-1].copy()
        ret[2:] += ret[:2]
        return ret
    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
