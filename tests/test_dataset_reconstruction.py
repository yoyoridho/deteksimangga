import numpy as np
import pytest
from dataset.prepare import prepare
from multiview.reconstruction import reconstruct, load_points


def test_dataset_split_preserves_capture_groups(tmp_path, synthetic):
    _, images = synthetic
    source = tmp_path / "raw"
    (source / "images").mkdir(parents=True)
    (source / "labels").mkdir()
    for i, original in enumerate(images):
        for angle in range(2):
            stem = f"tree{i}__angle{angle}"
            (source / "images" / f"{stem}.jpg").write_bytes(original.read_bytes())
            (source / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    destination = tmp_path / "prepared"
    counts = prepare(source, destination)
    assert sum(counts.values()) == 6 and min(counts.values()) > 0
    split_groups = [{p.stem.split("__")[0] for p in (destination / "images" / split).glob("*")} for split in ("train", "val", "test")]
    assert not (split_groups[0] & split_groups[1] or split_groups[1] & split_groups[2] or split_groups[0] & split_groups[2])
    assert "0: mango" in (destination / "data.yaml").read_text()
    with pytest.raises(ValueError, match="sudah ada"):
        prepare(source, destination)


def test_colmap_unavailable_is_explicit_and_real_text_parser(tmp_path):
    with pytest.raises(ValueError, match="COLMAP tidak ditemukan"):
        reconstruct(tmp_path, tmp_path / "out", executable="nonexistent_colmap_binary")
    points = tmp_path / "points3D.txt"
    points.write_text("# synthetic fixture\n1 0 1 2 255 128 0 0.2 1 4\n")
    xyz, rgb = load_points(points)
    assert np.allclose(xyz, [[0, 1, 2]]) and np.allclose(rgb[0], [1, 128/255, 0])
