import hashlib
import json
import numpy as np
import pytest
from dataset.build_real import Sample, remove_leakage, yolo_rows
from training.model_info import trained_model_info
from training.run_when_ready import enough_memory


def test_real_annotations_convert_without_inventing_boxes():
    rows = yolo_rows([[20, 10, 60, 30]], 100, 50)
    assert np.allclose(list(map(float, rows[0].split())), [0, .4, .4, .4, .4])
    assert yolo_rows([], 100, 100) == []
    with pytest.raises(ValueError):
        yolo_rows([[0, 0, 1010, 10]], 100, 100)
    with pytest.raises(ValueError):
        yolo_rows([[-.5, 0, -.1, 10]], 100, 100)


def test_training_waits_for_memory_without_closing_other_programs():
    gib = 1024**3
    assert not enough_memory(gib, 8*gib, True)
    assert not enough_memory(8*gib, gib, True)
    assert not enough_memory(8*gib, None, True)
    assert not enough_memory(4*gib, 2*gib, True)
    assert enough_memory(4*gib, 4*gib, True)
    assert enough_memory(4*gib, None, False)


def test_large_training_photo_scales_image_and_labels_together():
    sample = Sample("large", "postharvest", "g", "train", np.zeros((1000, 2000, 3), np.uint8),
                    [[200, 100, 1000, 800]], "large.jpg")
    assert sample.image.shape == (320, 640, 3)
    assert np.allclose(sample.boxes, [[64, 32, 320, 256]])
    assert np.allclose(list(map(float, yolo_rows(sample.boxes, 640, 320)[0].split())), [0, .3, .45, .4, .7])


def test_leakage_removal_retains_holdout_over_training_copy():
    image = np.random.default_rng(42).integers(0, 255, (60, 80, 3), dtype=np.uint8)
    samples = [Sample("train", "orchard", "a", "train", image.copy(), [[0, 0, 20, 20]], "train.jpg"),
               Sample("test", "orchard", "b", "test", image.copy(), [[0, 0, 20, 20]], "test.jpg")]
    kept, removed = remove_leakage(samples)
    assert [sample.name for sample in kept] == ["test"]
    assert removed == [{"name": "train", "reason": "exact-duplicate", "split": "train"}]


def test_trained_model_identity_requires_matching_checkpoint(tmp_path):
    assert trained_model_info(tmp_path) is None
    data = b"test-only-not-a-model"
    (tmp_path / "best.pt").write_bytes(data)
    summary = {"model_sha256": hashlib.sha256(data).hexdigest(), "confidence_threshold": .4}
    (tmp_path / "training_summary.json").write_text(json.dumps(summary))
    assert trained_model_info(tmp_path) == summary
    (tmp_path / "best.pt").write_bytes(b"changed")
    assert trained_model_info(tmp_path) is None


def test_local_dataset_has_disjoint_capture_groups_and_hashes():
    from collections import defaultdict
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "dataset"
    manifest = next((p for p in [root / "mango_real_v3" / "manifest.json", root / "mango_real" / "manifest.json"] if p.exists()), None)
    if manifest is None:
        pytest.skip("Real dataset is downloaded locally, not committed to Git")
    records = json.loads(manifest.read_text(encoding="utf-8"))["samples"]
    groups, hashes = defaultdict(set), defaultdict(set)
    for record in records:
        groups[record["group"]].add(record["split"])
        hashes[record["sha256"]].add(record["split"])
    assert all(len(splits) == 1 for splits in groups.values())
    assert all(len(splits) == 1 for splits in hashes.values())
