import numpy as np
import pytest
from detection.yolo import Detector
from duplicate_removal.grouping import group_observations, manual_merge
from modules.schema import Observation, Link
from multiview.association import extract, associate, epipolar_distance
from preprocessing.images import read_image


def obs(key, view, box=None, **kwargs):
    return Observation(key, view, box or [10, 10, 60, 80], 0.9, **kwargs)


def test_components_never_merge_simultaneous_fruit():
    items = [obs("a", "1"), obs("b", "1"), obs("c", "2")]
    links = [Link("a", "c", .9, "sift"), Link("b", "c", .8, "sift")]
    assert group_observations(items, links) == 2
    assert items[0].unique_id == items[2].unique_id != items[1].unique_id
    assert links[1].accepted is False
    with pytest.raises(ValueError):
        manual_merge(items, links, items[0].unique_id, items[1].unique_id)


def test_sift_association_and_broken_tracklet(synthetic):
    _, images = synthetic
    first, second = map(read_image, images[:2])
    detector = Detector()
    a = [obs(f"a{i}", "1", box, track_id=1+i) for i, (box, _, _) in enumerate(detector.predict(first))]
    b = [obs(f"b{i}", "2", box, track_id=50+i) for i, (box, _, _) in enumerate(detector.predict(second))]
    links = associate(a, b, extract(first, a), extract(second, b))
    assert len(links) == 3
    assert all(link.method == "sift-hungarian" for link in links)
    assert group_observations(a+b, links) == 3


def test_same_track_id_alone_is_insufficient():
    image = np.zeros((300, 300, 3), np.uint8)
    a, b = [obs("a", "1", track_id=1)], [obs("b", "2", [220, 220, 270, 290], track_id=1)]
    assert not associate(a, b, extract(image, a), extract(image, b), adjacent=True)


def test_spatial_association_and_camera_geometry():
    image = np.zeros((200, 200, 3), np.uint8)
    a = [obs("a", "1", xyz=[0, 0, 0])]
    b = [obs("b", "2", xyz=[.01, 0, 0])]
    assert associate(a, b, extract(image, a), extract(image, b))[0].method == "3d-position"
    pa = np.array([[100, 0, 50, 0], [0, 100, 50, 0], [0, 0, 1, 0]], float)
    pb = pa.copy()
    pb[0, 3] = -100
    assert epipolar_distance([50, 50], [30, 50], pa, pb) < 1e-6
    assert epipolar_distance([50, 50], [30, 70], pa, pb) > 10


def test_manual_merge_and_rejected_link():
    items = [obs("a", "1"), obs("b", "2")]
    link = Link("a", "b", .9, "sift", False)
    assert group_observations(items, [link]) == 2
    links = [link]
    assert manual_merge(items, links, "M0001", "M0002") == 1
