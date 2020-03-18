# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import

import cv2
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

from .track import Track
from .track_utils import get_around_bboxes, cost_matrix, load_network

from .loader import extract_image_patch


class Tracker(object):
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, model, fodler):
        self.model = model
        self.pre_frame, self.now_frame = None, None
        self.now_frame_id = 0
        self.folder = fodler
        self.tracks = []
        self._next_id = 1

    def update(self, frame, bboxes, now_frame_id):
        self.now_frame = frame
        self.now_frame_id = now_frame_id

        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = self._match(bboxes, self.now_frame)
        matches_td = []
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.now_frame_id, bboxes[detection_idx])
            matches_td.append([self.tracks[track_idx].track_id, detection_idx])
        # for track_idx in unmatched_tracks:
        #     self.tracks[track_idx].mark_missed()  # 删掉或者不更新匹配

        # 对最终没有匹配的检测框分配新的id
        unmatches_td = []
        for detection_idx in unmatched_detections:
            ni = self._initiate_track(self.folder, self.now_frame_id, bboxes[detection_idx])
            unmatches_td.append([ni, detection_idx])
        # self.tracks = [t for t in self.tracks if not t.is_deleted()]  # 丢掉被删除的id
        #
        # # Update distance metric.
        # active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        # features, targets = [], []
        # for track in self.tracks:
        #     if not track.is_confirmed():
        #         continue
        #     features += track.features
        #     targets += [track.track_id for _ in track.features]
        #     track.features = []
        #
        # self.metric.partial_fit(
        #     np.asarray(features), np.asarray(targets), active_targets)
        # return matches_td + unmatches_td

    def _match(self, bboxes, frame):
        srcs = []
        targets = []
        not_around_bboxes = []
        # confirmed_tracks_indices = [
        #     i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        # unconfirmed_tracks_indices = [
        #     i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        for idx in range(len(self.tracks)):
            srcs.append(self.tracks[idx].img_patch)
            not_around_bboxes.append(get_around_bboxes(self.tracks[idx].bboxes[-1], bboxes)[2])
        for bbox in bboxes:
            targets.append(extract_image_patch(frame, bbox))
        cost_m = cost_matrix(self.model, srcs, targets)
        for i, nabbs in enumerate(not_around_bboxes):
            for j in nabbs:
                cost_m[i, j] = 1.0
        # print(cost_m)
        indices = linear_assignment(cost_m)
        detection_indices = np.arange(len(targets))
        track_indices = np.arange(len(srcs))
        max_distance = 0.5
        matches, unmatched_tracks, unmatched_detections = [], [], []
        for col, detection_idx in enumerate(detection_indices):
            if col not in indices[:, 1]:
                unmatched_detections.append(detection_idx)
        for row, track_idx in enumerate(track_indices):
            if row not in indices[:, 0]:
                unmatched_tracks.append(track_idx)
        for row, col in indices:
            track_idx = track_indices[row]
            detection_idx = detection_indices[col]
            if cost_m[row, col] > max_distance:
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(detection_idx)
            else:
                matches.append((track_idx, detection_idx))
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()  # 删掉或者不更新匹配

        # 对最终没有匹配的检测框分配新的id
        unmatches_td = []
        for detection_idx in unmatched_detections:
            ni = self._initiate_track(self.folder, self.now_frame_id, bboxes[detection_idx])
            unmatches_td.append([ni, detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]  # 丢掉被删除的id
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, folder, frame_id, bbox):
        track = Track(folder, self._next_id)
        track.update(frame_id=frame_id, bbox=bbox)
        self.tracks.append(track)
        self._next_id += 1
        return self._next_id - 1


def to_tlbr(xywh):
    """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
    `(top left, bottom right)`.
    """
    ret = xywh.copy()
    ret[2:] += ret[:2]
    return ret
