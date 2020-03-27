# vim: expandtab:ts=4:sw=4
import numpy as np
import torch
from torchvision.transforms import transforms
from .linear_assignment import INFTY_COST

data_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    # bbox = np.array(bbox)
    candidates = np.array(candidates)
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def _pdist(a, b):
    """Compute pair-wise squared distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a2, b2 = np.square(a).sum(axis=1), np.square(b).sum(axis=1)
    r2 = -2. * np.dot(a, b.T) + a2[:, None] + b2[None, :]
    r2 = np.clip(r2, 0., float(np.inf))
    return r2


def _cosine_distance(a, b, data_is_normalized=False):
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Parameters
    ----------
    a : array_like
        An NxM matrix of N samples of dimensionality M.
    b : array_like
        An LxM matrix of L samples of dimensionality M.
    data_is_normalized : Optional[bool]
        If True, assumes rows in a and b are unit length vectors.
        Otherwise, a and b are explicitly normalized to lenght 1.

    Returns
    -------
    ndarray
        Returns a matrix of size len(a), len(b) such that eleement (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1. - np.dot(a, b.T)


def _nn_euclidean_distance(x, y):
    """ Helper function for nearest neighbor distance metric (Euclidean).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest Euclidean distance to a sample in `x`.

    """
    distances = _pdist(x, y)
    return np.maximum(0.0, distances.min(axis=0))


def _nn_cosine_distance(x, y, mean=True):
    """ Helper function for nearest neighbor distance metric (cosine).

    Parameters
    ----------
    x : ndarray
        A matrix of N row-vectors (sample points).
    y : ndarray
        A matrix of M row-vectors (query points).

    Returns
    -------
    ndarray
        A vector of length M that contains for each entry in `y` the
        smallest cosine distance to a sample in `x`.

    """
    distances = _cosine_distance(x, y)
    if mean:
        return distances.mean(axis=0)
    return distances.min(axis=0)


def cost_matrix(model, srcs, targets, cost):
    """

    :param model:
    :param srcs: feature
    :param targets: feature
    :return:
    """
    assert cost.shape == (len(targets), len(srcs)), "shape doed not match"
    # cost = np.zeros((len(targets), len(srcs)))
    feature_shape = targets[0][0].shape[0]
    if feature_shape == 2048:
        with torch.no_grad():
            f_srcs = torch.Tensor(srcs).cuda()
            f_targets = []
            for tars in targets:
                tars = torch.Tensor(tars).cuda()
                f_targets.append(tars)
            for i, f_src in enumerate(f_srcs):
                for j, f_target in enumerate(f_targets):
                    if cost[j, i] == INFTY_COST:
                        continue
                    f_src_temp = torch.Tensor(np.asarray([f_src.cpu().data.numpy() for _ in f_target])).cuda()
                    sim = model.verify(f_src_temp, f_target)
                    sim = sim.mean(dim=0).cpu().data.numpy()[1]
                    cost[j, i] = -np.log(sim + 1e-5)  # np.log((1 - sim) / sim)
            return cost
    with torch.no_grad():
        srcs = np.asarray([data_transforms(src).numpy() for src in srcs])
        srcs = torch.Tensor(srcs).cuda()
        f_srcs = model.get_feature(srcs)
        f_targets = []
        for tars in targets:
            tars = np.asarray([data_transforms(tar).numpy() for tar in tars])
            tars = torch.Tensor(tars).cuda()
            f_targets.append(model.get_feature(tars))
        for i, f_src in enumerate(f_srcs):
            for j, f_target in enumerate(f_targets):
                f_src_temp = torch.Tensor(np.asarray([f_src.cpu().data.numpy() for _ in f_target])).cuda()
                sim = model.verify(f_src_temp, f_target)
                sim = sim.mean(dim=0).cpu().data.numpy()[1]
                cost[j, i] = -np.log(sim + 1e-5)  # np.log((1 - sim) / sim)

    return cost


class SimilarityMetric(object):
    """
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    Parameters
    ----------
    th: float
        The matching threshold. Samples with larger distance are considered an
        invalid match.
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.

    """

    def __init__(self, th, model, budget=16):
        # Remove the final fc layer and classifier layer

        # Change to test mode
        model = model.eval()
        model = model.cuda()
        self.matching_threshold = th
        self.budget = budget
        self.model = model
        self.samples = {}

    def partial_fit(self, patch, targets, bboxes, active_targets):
        """Update the distance metric with new data.

        Parameters
        ----------
        patch : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : ndarray
            An integer array of associated target identities.
        active_targets : List[int]
            A list of targets that are currently present in the scene.

        """
        self.bboxes = {}
        for i, k in enumerate(active_targets):
            self.bboxes.setdefault(k, bboxes[i])
        for feature, target in zip(patch, targets):
            self.samples.setdefault(target, []).append(feature)
            # if self.budget is not None and len(self.samples[target]) > self.budget:
            #     del self.samples[target][0]
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget:]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(self, src_patches, targets, mean=True, cost=None):
        """Compute distance between features and targets.

        Parameters
        ----------
        src_patches : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int] track_ids
            A list of targets to match the given `features` against.
        mean: if use mean distance

        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.

        """

        # get image patches to compute similarity
        if mean:
            tars_patches = [self.samples[target] for i, target in enumerate(targets)]
        else:
            tars_patches = [[self.samples[target][-1]] for i, target in enumerate(targets)]
        cost_m = cost_matrix(self.model, src_patches, tars_patches, cost)
        return cost_m
