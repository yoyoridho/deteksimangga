from dataclasses import dataclass
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from modules.schema import Link


@dataclass
class Features:
    points: np.ndarray
    descriptors: np.ndarray | None
    histograms: dict


def extract(image, observations):
    points, descriptors = cv2.SIFT_create(nfeatures=1400).detectAndCompute(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None)
    histograms = {}
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for obs in observations:
        x1, y1, x2, y2 = map(int, obs.box)
        patch = hsv[y1:y2, x1:x2]
        if patch.size == 0:
            raise ValueError("Bounding box kosong.")
        hist = cv2.calcHist([patch], [0, 1], None, [24, 16], [0, 180, 0, 256])
        histograms[obs.id] = cv2.normalize(hist, None).astype(np.float32)
        saturated = patch[:, :, 1] > 65
        yellow = (patch[:, :, 0] >= 15) & (patch[:, :, 0] <= 35) & saturated
        obs.yellow_fraction = float(yellow.sum() / max(int(saturated.sum()), 1))
    return Features(np.asarray([p.pt for p in points], dtype=np.float32).reshape(-1, 2), descriptors, histograms)


def sift_matches(first, second):
    if first.descriptors is None or second.descriptors is None or len(second.descriptors) < 2:
        return np.empty((0, 2), int)
    pairs = cv2.BFMatcher().knnMatch(first.descriptors, second.descriptors, k=2)
    good = [m for pair in pairs if len(pair) == 2 for m, n in [pair] if m.distance < 0.7*n.distance]
    # A descriptor in the second view can support only one correspondence.
    unique = {}
    for match in sorted(good, key=lambda m: m.distance):
        unique.setdefault(match.trainIdx, match)
    return np.asarray([(m.queryIdx, m.trainIdx) for m in unique.values()], dtype=int).reshape(-1, 2)


def contains(points, box):
    x1, y1, x2, y2 = box
    return (points[:, 0] >= x1) & (points[:, 0] <= x2) & (points[:, 1] >= y1) & (points[:, 1] <= y2)


def epipolar_distance(point_a, point_b, projection_a, projection_b):
    """Pixel distance to the epipolar line, using calibrated 3x4 projection matrices."""
    pa, pb = np.asarray(projection_a, float), np.asarray(projection_b, float)
    _, _, vh = np.linalg.svd(pa)
    epipole = pb @ vh[-1]
    x, y, z = epipole
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    line = skew @ pb @ np.linalg.pinv(pa) @ np.r_[point_a, 1]
    denominator = np.linalg.norm(line[:2])
    return float(abs(line @ np.r_[point_b, 1]) / denominator) if denominator > 1e-9 else float("inf")


def associate(first_obs, second_obs, first, second, adjacent=False, projections=None, spatial_radius=0.04):
    if not first_obs or not second_obs:
        return []
    matches = sift_matches(first, second)
    homography = None
    if len(matches) >= 12:
        candidate, mask = cv2.findHomography(first.points[matches[:, 0]], second.points[matches[:, 1]], cv2.RANSAC, 3.0)
        if candidate is not None and mask is not None and mask.mean() >= 0.55 and np.isfinite(candidate).all():
            homography = candidate
    costs = np.full((len(first_obs), len(second_obs)), 1e6)
    methods = {}
    for i, a in enumerate(first_obs):
        ac = (np.asarray(a.box[:2]) + a.box[2:]) / 2
        for j, b in enumerate(second_obs):
            bc = (np.asarray(b.box[:2]) + b.box[2:]) / 2
            diagonal = max(np.linalg.norm(np.asarray(b.box[2:]) - b.box[:2]), 1)
            appearance = cv2.compareHist(first.histograms[a.id], second.histograms[b.id], cv2.HISTCMP_BHATTACHARYYA)
            if a.xyz is not None and b.xyz is not None:
                distance = np.linalg.norm(np.asarray(a.xyz) - b.xyz)
                if distance <= spatial_radius and appearance < 0.45:
                    costs[i, j] = 0.15 + 0.25*distance/spatial_radius + 0.2*appearance
                    methods[i, j] = "3d-position"
                continue
            local = 0
            if len(matches):
                local = int((contains(first.points[matches[:, 0]], a.box) & contains(second.points[matches[:, 1]], b.box)).sum())
            geometric_error = float("inf")
            if homography is not None:
                projected = cv2.perspectiveTransform(ac.reshape(1, 1, 2).astype(np.float32), homography)[0, 0]
                geometric_error = float(np.linalg.norm(projected-bc)/diagonal)
            if projections is not None:
                if epipolar_distance(ac, bc, *projections) > 4:
                    continue
            if local >= 3 and appearance < 0.55 and (geometric_error < 0.8 or local >= 6):
                costs[i, j] = 0.12 + 0.3*appearance + 0.25/(local+1)
                methods[i, j] = "sift-hungarian"
            elif adjacent and a.track_id is not None and a.track_id == b.track_id and appearance < 0.25:
                area_a = (a.box[2]-a.box[0])*(a.box[3]-a.box[1])
                area_b = (b.box[2]-b.box[0])*(b.box[3]-b.box[1])
                motion = np.linalg.norm(ac-bc)/diagonal
                if 0.5 < area_a/max(area_b, 1) < 2 and min(motion, geometric_error) < 0.5:
                    costs[i, j] = 0.25 + 0.4*appearance + 0.2*min(motion, geometric_error)
                    methods[i, j] = "bytetrack-motion-appearance"
    rows, cols = linear_sum_assignment(costs)
    links = []
    for i, j in zip(rows, cols):
        if costs[i, j] >= 0.65:
            continue
        alternatives = np.r_[np.delete(costs[i], j), np.delete(costs[:, j], i)]
        if len(alternatives) and alternatives.min() - costs[i, j] < 0.035:
            continue
        links.append(Link(first_obs[i].id, second_obs[j].id, float(1-costs[i, j]), methods[i, j]))
    return links
