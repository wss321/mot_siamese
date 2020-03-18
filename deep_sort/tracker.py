# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


# from .orb import img_similarity
# from .generate_detections import extract_image_patch
# import cv2


class Tracker:
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

    def __init__(self, metric, img_shape, center_assingment, mean, max_iou_distance=0.7, max_age=30, n_init=3,
                 max_eu_dis=3461.12):
        # self.model = model
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.max_eu_dis = max_eu_dis
        self.img_shape = img_shape
        self.frame = []
        self.ca = center_assingment
        self.mean = mean

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections, frame):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
            框以及框的视觉特征

        """
        self.frame.append(frame)
        if len(self.frame) > 60:
            del self.frame[0]
        # Run matching cascade.级联匹配
        matches, unmatched_tracks, unmatched_detections = \
            self._match_cascade(detections)

        # print(matches, unmatched_tracks, unmatched_detections)
        # Update track set.
        # print([self.tracks[i].track_id for i in range(len(self.tracks))])

        # -------------------------center assignment-------------------------------------------
        if self.ca:
            def center_dis(x, y, centers, max_step):
                """
                计算中点加权欧式距离
                :param x:
                :param y:
                :param centers:
                :param max_step:
                :return:
                """
                dis = 0
                N = len(centers)
                if len(centers) > max_step:
                    N = max_step

                for n, center in enumerate(reversed(centers)):
                    if n > max_step:
                        break
                    w = 1 / N  # (2 * (n + 1) / (N ** 2 + N))
                    dis += w * np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
                return dis

            def margin_dis(x, y, v_w, v_h):
                """计算点到视频边缘的相对最小距离"""
                x_min = min(x, abs(x - v_w))
                y_min = min(y, abs(y - v_h))
                d = min(x_min, y_min)
                if d == x_min:
                    d /= v_w
                else:
                    d /= v_h
                return d

            # def ap_distance_metric(tracks, dets, track_indices, detection_indices):
            #     """外观度量"""
            #     features = np.array([dets[i].feature for i in detection_indices])
            #     targets = np.array([tracks[i].track_id for i in track_indices])
            #     cost_matrix = self.metric.distance(features, targets)
            #     # cost_matrix = linear_assignment.gate_cost_matrix(
            #     #     self.kf, cost_matrix, tracks, dets, track_indices,
            #     #     detection_indices)
            #
            #     return cost_matrix

            # def is_video_center(x, y, v_w, v_h, m=0.1):
            #     d = np.sqrt((x - v_w) ** 2 + (y - v_h) ** 2) / ((v_w + v_h) / 2)
            #     if d < m:
            #         return True
            #     return False

            if len(unmatched_detections) * len(unmatched_tracks) != 0:
                unmatched_tracks_temp = []
                unmatched_detections_temp = []
                centers_cost_matix = np.zeros((len(unmatched_detections), len(unmatched_tracks)))
                for i, detection_idx in enumerate(unmatched_detections):
                    xyah = detections[detection_idx].to_xyah()
                    for j, track_idx in enumerate(unmatched_tracks):
                        track_cnts = self.tracks[track_idx].centers()
                        dis = center_dis(xyah[0], xyah[1], track_cnts, max_step=15)
                        # patch1 = extract_image_patch(self.frame[-1], detections[detection_idx].tlwh, [416, 416])
                        # patch2 = extract_image_patch(self.frame[- self.tracks[track_idx].time_since_update - 1],
                        #                              detections[detection_idx].tlwh, [416, 416])
                        # dis /= img_similarity(patch1, patch2)
                        # cv2.imshow('', patch2)
                        # cv2.waitKey(100)
                        centers_cost_matix[i, j] = dis
                indices = linear_assignment.linear_assignment(centers_cost_matix)
                # print(unmatched_detections)
                try:
                    # print([unmatched_tracks[col] for col in indices[:, 0]],
                    #       [unmatched_detections[row] for row in indices[:, 1]])
                    # print(len(self.tracks), len(self.tracks))
                    # ap_cost_matrix = ap_distance_metric(self.tracks, detections,
                    #                                     [unmatched_tracks[col] for col in indices[:, 0]],
                    #                                     [unmatched_detections[row] for row in indices[:, 1]])
                    center_match_id = []
                    for row, col in indices:
                        detection_idx = unmatched_detections[row]
                        track_idx = unmatched_tracks[col]
                        # if ap_cost_matrix[track_idx, detection_idx] > 0.6 * self.metric.matching_threshold:
                        xyah = detections[detection_idx].to_xyah()
                        md = margin_dis(xyah[0], xyah[1], self.img_shape[0], self.img_shape[1])
                        # print(md)
                        if centers_cost_matix[row, col] > self.max_eu_dis or md < 0.05:
                            unmatched_tracks_temp.append(track_idx)
                            unmatched_detections_temp.append(detection_idx)
                        else:
                            matches.append((track_idx, detection_idx))
                            center_match_id.append(track_idx)
                            # print("match", self.tracks[track_idx].track_id)
                    unmatched_detections, unmatched_tracks = unmatched_detections_temp, unmatched_tracks_temp
                    for id in center_match_id:
                        if not self.tracks[id].color_change:
                            self.tracks[id].color_change = True
                            self.tracks[id].color = (0, 0, 255)
                except:
                    print('err')
                    pass
            # -------------------------------------------------------------------------
        matches_td = []
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
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
            # track.patches = []

        self.metric.partial_fit(
            patches, targets, last_bboxes, active_targets)
        return matches_td  # + new_td

    def _match_cascade(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            patch = np.array([dets[i].patch for i in detection_indices])
            bboxes = np.array([dets[i].tlwh for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(patch, bboxes, targets, self.mean)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        # print("confirmed_tracks{}\tunconfirmed_tracks{}".format(confirmed_tracks, unconfirmed_tracks))
        # 级联匹配已经确认的id
        # 分配id
        # Associate confirmed tracks using appearance features.
        # matches_a, unmatched_tracks_a, unmatched_detections = \
        #     linear_assignment.min_cost_matching(
        #         gated_metric, self.metric.matching_threshold, self.tracks,
        #         detections)
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                distance_metric=gated_metric, max_distance=self.metric.matching_threshold, cascade_depth=self.max_age,
                tracks=self.tracks, detections=detections, track_indices=confirmed_tracks, detection_indices=None)

        # print("级联匹配：matches_a{}\tunmatched_tracks_a{}\tunmatched_detections{}".format(matches_a, unmatched_tracks_a,
        #                                                                               unmatched_detections))

        # return matches_a, unmatched_tracks_a, unmatched_detections
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
        # print("a:{}\tb:{}".format(matches_a, matches_b))
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        track = Track(mean, covariance, self._next_id, self.n_init, self.max_age,
                      detection.patch, detection.to_xyah())
        self.tracks.append(track)
        self._next_id += 1
        return self._next_id - 1
