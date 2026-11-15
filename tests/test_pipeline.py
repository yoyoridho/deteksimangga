from pathlib import Path
from io import BytesIO
import json
import zipfile
import cv2
import numpy as np
import pytest

from detection.yolo import Detector
from modules.pipeline import process_images, process_video
from modules.schema import Run
from preprocessing.images import PreprocessConfig, read_image
from reports.export import artifact_zip, csv_bytes
from tracking.byte import VideoTracker


def test_full_images_exports_and_roundtrip(synthetic, tmp_path):
    root, images = synthetic
    run = process_images([("bad.jpg", b"broken")] + [(p.name, p.read_bytes()) for p in images], Detector(), tmp_path, PreprocessConfig(enhance=False))
    assert len(run.views) == 3 and len(run.observations) == 9 and len(run.fruits) == 3
    assert len(run.observations)-len(run.fruits) == 6 and run.warnings
    loaded = Run.load(Path(run.directory) / "run.json")
    assert loaded.observations[0].unique_id == run.observations[0].unique_id
    assert (Path(run.directory) / "report.pdf").read_bytes().startswith(b"%PDF")
    with zipfile.ZipFile(BytesIO(artifact_zip(run))) as archive:
        assert {"report.pdf", "fruits.csv", "observations.csv", "run.json", "annotated/V0001.jpg"} <= set(archive.namelist())
    assert b"'=bad" in csv_bytes([{"value": "=bad"}], ["value"])


def test_real_bytetrack_continuity_and_reset():
    detections = [([20., 20., 60., 90.], .9, "synthetic-shape")]
    tracker = VideoTracker(12)
    first = tracker.update(detections, (100, 100, 3))
    second = tracker.update([([22., 20., 62., 90.], .9, "synthetic-shape")], (100, 100, 3))
    assert first[0] == second[0]
    assert tracker.update([], (100, 100, 3)) == {}
    assert VideoTracker(12).update(detections, (100, 100, 3))[0] == first[0]


def test_video_pipeline(synthetic, tmp_path):
    root, _ = synthetic
    run = process_video((root / "SYNTHETIC_orbit.mp4").read_bytes(), ".mp4", Detector(), tmp_path,
                        PreprocessConfig(enhance=False), stride=2, max_frames=12)
    assert len(run.views) == 12 and len(run.observations) == 36 and len(run.fruits) == 3
    assert all(o.track_id is not None for o in run.observations)
    assert Path(run.video_path).stat().st_size > 1000
    capture = cv2.VideoCapture(run.video_path)
    assert capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 12
    capture.release()


def test_empty_detection_and_invalid_inputs(tmp_path):
    success, encoded = cv2.imencode(".png", np.zeros((200, 200, 3), np.uint8))
    run = process_images([("blank.png", encoded.tobytes())], Detector(), tmp_path)
    assert not run.observations and not run.fruits
    with pytest.raises(ValueError):
        process_images([], Detector(), tmp_path)
    with pytest.raises(ValueError):
        process_video(b"broken", ".mp4", Detector(), tmp_path)
    with pytest.raises(ValueError):
        Detector("mango", tmp_path / "missing.pt")


def test_portrait_photo_pdf_fits_page(tmp_path):
    success, encoded = cv2.imencode(".png", np.zeros((1500, 150, 3), np.uint8))
    run = process_images([("portrait.png", encoded.tobytes())], Detector(), tmp_path)
    assert (Path(run.directory) / "report.pdf").stat().st_size > 1000


@pytest.mark.integration
def test_real_yolo_inference(synthetic):
    weights = Path(__file__).resolve().parents[1] / "models" / "yolo11n.pt"
    if not weights.is_file():
        pytest.skip("Download models/yolo11n.pt to run real YOLO integration")
    _, images = synthetic
    detector = Detector("yolo-demo", weights)
    result = detector.predict(read_image(images[0]))
    assert isinstance(result, list)
    assert all(label in {"apple", "banana", "orange"} and len(box) == 4 and 0 <= score <= 1 for box, score, label in result)
    with pytest.raises(ValueError, match="kelas mango"):
        Detector("mango", weights)
