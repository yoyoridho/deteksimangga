"""Generate deterministic SYNTHETIC fixtures; never field measurements."""
from pathlib import Path
import json
import cv2
import numpy as np

from preprocessing.images import write_image


def generate(destination: Path, video=True):
    destination.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    background = rng.integers(35, 100, (520, 800), dtype=np.uint8)
    base = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
    centers = [(180, 210), (400, 270), (620, 190)]
    colors = [(40, 155, 65), (25, 180, 195), (25, 190, 240)]
    for fruit, ((cx, cy), color) in enumerate(zip(centers, colors)):
        mask = np.zeros(base.shape[:2], np.uint8)
        cv2.ellipse(mask, (cx, cy), (43, 64), 0, 0, 360, 255, -1)
        base[mask > 0] = color
        for _ in range(120):
            dx, dy = rng.integers(-36, 37), rng.integers(-55, 56)
            if (dx/38)**2+(dy/58)**2 < 0.9:
                shade = int(rng.integers(20, 245))
                cv2.circle(base, (int(cx+dx), int(cy+dy)), int(rng.integers(1, 3)), (shade, shade, shade), -1)
        cv2.putText(base, str(fruit+1), (cx-12, cy+8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 40), 2)
    images, truth = [], {}
    for index, (dx, dy) in enumerate([(0, 0), (22, 8), (-15, -5)]):
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        image = cv2.warpAffine(base, matrix, (800, 520), borderMode=cv2.BORDER_REFLECT)
        path = destination / f"SYNTHETIC_view_{index+1}.jpg"
        write_image(path, image)
        images.append(path)
        truth[f"V{index+1:04d}"] = [[cx-43+dx, cy-64+dy, cx+44+dx, cy+65+dy] for cx, cy in centers]
    metadata = {"kind": "SYNTHETIC TEST DATA - NOT FIELD DATA", "unique_fruits": 3, "ground_truth_boxes": truth,
                "description": "Three textured ellipses, translated views; no real tree or ripeness labels."}
    (destination / "ground_truth.json").write_text(json.dumps(truth, indent=2), encoding="utf-8")
    (destination / "README.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if video:
        writer = cv2.VideoWriter(str(destination / "SYNTHETIC_orbit.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 12, (800, 520))
        if not writer.isOpened():
            raise OSError("Encoder MP4 tidak tersedia untuk fixture.")
        try:
            for frame in range(24):
                dx = round(15*np.sin(frame/24*2*np.pi))
                writer.write(cv2.warpAffine(base, np.float32([[1, 0, dx], [0, 1, 0]]), (800, 520), borderMode=cv2.BORDER_REFLECT))
        finally:
            writer.release()
    return images


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent / "synthetic")
    print("SYNTHETIC fixtures written to dataset/synthetic; not field data.")
