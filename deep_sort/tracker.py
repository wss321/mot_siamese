# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


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

    def __init__(self, metric, img_shape, mean, max_iou_distance=0.6, max_age=60, n_init=3):
        # self.model = model
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.img_shape = img_shape
        self.now_frame_id = 1
        self.mean = mean

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
            框以及框的视觉特征

        """
        self.now_frame_id += 1
        matches, unmatched_tracks, unmatched_detections = \
            self._match_cascade(detections)
        matches_td = []
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx], self.now_frame_id)
            matches_td.append([self.tracks[track_idx].track_id, detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()  # 删掉或者不更新匹配

        # 对最终没有匹配的检测框分配新的id
        new_td = []
        for detection_idx in unmatched_detections:
            ni = self._initiate_track(detections[detection_idx])
            new_td.append([ni, detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]  # 丢掉被删除的id

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        last_bboxes = [t.last_bbox for t in self.tracks if t.is_confirmed()]
        patches, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            patches += track.patches
            targets += [track.track_id for _ in track.patches]

        self.metric.partial_fit(
            patches, targets, last_bboxes, active_targets)
        return matches_td  # + new_td

    def _match_cascade(self, detections):

        def gate_area(tracks, detections, track_indices=None,
                      detection_indices=None):
            if track_indices is None:
                track_indices = np.arange(len(tracks))
            if detection_indices is None:
                detection_indices = np.arange(len(detections))

            area_gate = np.zeros((len(track_indices), len(detection_indices)))
            area_candidates = np.asarray([detections[i].tlwh for i in detection_indices])[:, 2:].prod(axis=1)
            for row, track_idx in enumerate(track_indices):
                area_bbox = tracks[track_idx].last_bbox[2:].prod()
                for column, ac in enumerate(area_candidates):
                    if max(area_bbox / ac, ac / area_bbox) > 4:
                        area_gate[row, column] = linear_assignment.INFTY_COST

            return area_gate

        def gated_metric(tracks, dets, track_indices, detection_indices):
            patch = np.array([dets[i].patch for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            gate_matrix = gate_area([tracks[i] for i in track_indices], detections,
                                    detection_indices=detection_indices)
            cost_matrix = np.zeros((len(targets), len(patch)))
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)
            cost_matrix = self.metric.distance(patch, targets, self.mean, cost_matrix)
            # cost_matrix = linear_assignment.gate_cost_matrix(
            #     self.kf, cost_matrix, tracks, dets, track_indices,
            #     detection_indices)
            cost_matrix = cost_matrix + gate_matrix
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        # print("confirmed_tracks{}\tunconfirmed_tracks{}".format(confirmed_tracks, unconfirmed_tracks))
        # 分配id
        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                distance_metric=gated_metric, max_distance=self.metric.matching_threshold, cascade_depth=self.max_age,
                tracks=self.tracks, detections=detections, track_indices=confirmed_tracks, detection_indices=None)

        # # IoU匹配
        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        track = Track(mean, covariance, self._next_id, self.n_init, self.max_age,
                      detection.patch, detection.to_xyah(), max_patch_len=self.metric.budget,
                      frame_id=self.now_frame_id)
        self.tracks.append(track)
        self._next_id += 1
        return self._next_id - 1
