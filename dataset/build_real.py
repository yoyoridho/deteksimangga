"""Build audited orchard/postharvest/negative mango detection data. No generated positive labels."""
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
import zipfile

import cv2
import numpy as np
import pyarrow.parquet as pq

from preprocessing.images import decode_image


@dataclass
class Sample:
    name: str
    source: str
    group: str
    split: str
    image: np.ndarray
    boxes: list
    origin: str

    def __post_init__(self):
        height, width = self.image.shape[:2]
        if max(height, width) > 640:
            scale = 640/max(height, width)
            size = (round(width*scale), round(height*scale))
            self.image = cv2.resize(self.image, size, interpolation=cv2.INTER_AREA)
            factors = np.array([size[0]/width, size[1]/height]*2)
            self.boxes = [(np.asarray(box)*factors).tolist() for box in self.boxes]


def fingerprint(image):
    gray = cv2.cvtColor(cv2.resize(image, (32, 32)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    dct = cv2.dct(gray)[:8, :8].flatten()[1:]
    bits = dct > np.median(dct)
    value = sum(int(bit) << i for i, bit in enumerate(bits))
    return value, cv2.resize(image, (24, 24)).astype(np.float32)


def remove_leakage(samples):
    """Prefer held-out data and drop exact/very near duplicates from lower-priority partitions."""
    priority = {"test": 0, "val": 1, "train": 2}
    kept, hashes, dropped = [], [], []
    exact = set()
    for sample in sorted(samples, key=lambda s: (priority[s.split], s.name)):
        digest = hashlib.sha256(sample.image.tobytes()).hexdigest()
        phash, thumb = fingerprint(sample.image)
        reason = "exact-duplicate" if digest in exact else None
        if reason is None:
            for old_hash, old_thumb, old_split in hashes:
                if old_split != sample.split and (phash ^ old_hash).bit_count() <= 4 and np.sqrt(np.mean((thumb-old_thumb)**2)) < 12:
                    reason = "near-duplicate-across-splits"
                    break
        if reason:
            dropped.append({"name": sample.name, "reason": reason, "split": sample.split})
        else:
            exact.add(digest)
            kept.append(sample)
            hashes.append((phash, thumb, sample.split))
    return kept, dropped


def yolo_rows(boxes, width, height):
    rows = []
    for box in boxes:
        values = np.asarray(box, float)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError("Invalid annotation: expected finite xyxy.")
        x1, y1, x2, y2 = values
        if x1 < -1 or y1 < -1 or x2 > width+1 or y2 > height+1 or x2 <= x1 or y2 <= y1:
            raise ValueError(f"Annotation outside image: {box}, image {width}x{height}.")
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Annotation is empty after clipping.")
        rows.append(f"0 {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
    return rows


def read_sources(source):
    samples = []
    for row in pq.read_table(source / "orchard.parquet").to_pylist():
        name = Path(row["image"]["path"]).stem
        image = decode_image(row["image"]["bytes"])
        tree_row = re.match(r"gear(\d+)", name)
        if not tree_row:
            raise ValueError(f"Unknown orchard capture group: {name}")
        group = "orchard-row-"+tree_row[1]
        split = "val" if tree_row[1] == "2" else "test" if tree_row[1] == "3" else "train"
        objects = row["objects"]
        if any(category != 0 for category in objects["categories"]):
            raise ValueError("Unknown orchard class.")
        boxes = [[x, y, x+w, y+h] for x, y, w, h in objects["bbox"]]
        samples.append(Sample("orchard_"+name, "orchard", group, split, image, boxes, row["image"]["path"]))
    growth_rows = pq.read_table(source / "growth.parquet").to_pylist()
    original_names = set()
    for row in sorted(growth_rows, key=lambda r: r["image"]["path"]):
        original = row["image"]["path"].split(".rf.")[0]
        # Keep one export per original photo rather than treating offline augmentations as new evidence.
        if original in original_names:
            continue
        original_names.add(original)
        capture = re.search(r"^(.*?)(\d+)", original)
        if capture is None:
            raise ValueError(f"Unknown smartphone capture name: {original}")
        group = f"phone-{capture[1]}-{int(capture[2])//10}"
        bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10
        split = "val" if bucket == 0 else "test" if bucket == 1 else "train"
        objects = row["objects"]
        if any(category not in (0, 1, 2, 3) for category in objects["categories"]):
            raise ValueError("Unknown smartphone mango class.")
        image = decode_image(row["image"]["bytes"])
        boxes = [[x, y, x+w, y+h] for x, y, w, h in objects["bbox"]]
        name = "phone_"+hashlib.sha256(original.encode()).hexdigest()[:20]
        samples.append(Sample(name, "smartphone-orchard", group, split, image, boxes, row["image"]["path"]))
    with zipfile.ZipFile(source / "postharvest.zip") as archive:
        paths = archive.namelist()
        for path in sorted(paths):
            if not re.search(r"/data/4-class/images/(train|test)/[^/]+\.(jpg|jpeg|png)$", path, re.I):
                continue
            name = Path(path).stem
            label_path = path.replace("/images/", "/labels/").rsplit(".", 1)[0]+".txt"
            image = decode_image(archive.read(path))
            height, width = image.shape[:2]
            boxes = []
            for line in archive.read(label_path).decode("utf-8-sig").splitlines():
                cls, x, y, w, h = map(float, line.split())
                if cls not in (0, 1, 2, 3) or not all(0 <= v <= 1 for v in (x, y, w, h)):
                    raise ValueError("Unknown postharvest annotation class / coordinates.")
                boxes.append([(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height])
            # Keep author's test partition untouched; validation is drawn only from training.
            group = re.sub(r"^[a-f0-9]{8}-", "", name)
            group = re.sub(r"\.rf\..*", "", group)
            split = "test" if "/images/test/" in path else "val" if int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 7 == 0 else "train"
            samples.append(Sample("post_"+name, "postharvest", "post-"+group, split, image, boxes, path))
    negatives = json.loads((source / "negative_sources.json").read_text(encoding="utf-8"))
    for record in negatives["records"]:
        path = source / record["path"]
        name = "negative_" + hashlib.sha256(record["path"].encode()).hexdigest()[:16]
        split = "train" if record["partition"] == "Training" else "val" if int(hashlib.sha256(record["category"].encode()).hexdigest()[:8], 16) % 2 == 0 else "test"
        samples.append(Sample(name, "non-mango", "fruit360-"+record["category"]+"-"+record["partition"],
                              split, decode_image(path.read_bytes()), [], record["path"]))
    return samples


def build(source, destination):
    if destination.exists():
        raise ValueError("Dataset destination already exists; choose a new directory.")
    samples = read_sources(source)
    group_splits = {}
    priority = {"test": 0, "val": 1, "train": 2}
    for sample in samples:
        previous = group_splits.get(sample.group, sample.split)
        group_splits[sample.group] = min(previous, sample.split, key=lambda value: priority[value])
    for sample in samples:
        sample.split = group_splits[sample.group]
    samples, dropped = remove_leakage(samples)
    # A balanced fine-tuning set prevents night-orchard images from dominating phone/postharvest images.
    balanced, totals, omitted = [], Counter(), Counter()
    for sample in sorted(samples, key=lambda s: hashlib.sha256(s.name.encode()).hexdigest()):
        if sample.split == "train" and totals[sample.source] >= 400:
            omitted[sample.source] += 1
            continue
        balanced.append(sample)
        if sample.split == "train":
            totals[sample.source] += 1
    samples = balanced
    manifest = []
    for sample in samples:
        height, width = sample.image.shape[:2]
        labels = yolo_rows(sample.boxes, width, height)
        # Store actual source images, resized with unchanged aspect ratio; labels stay normalized.
        scale = min(640/max(height, width), 1)
        image = cv2.resize(sample.image, (round(width*scale), round(height*scale))) if scale < 1 else sample.image
        image_path = destination / "images" / sample.split / (sample.name+".jpg")
        label_path = destination / "labels" / sample.split / (sample.name+".txt")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            raise OSError("JPEG encoding failed.")
        image_path.write_bytes(encoded.tobytes())
        label_path.write_text("\n".join(labels)+( "\n" if labels else ""), encoding="utf-8")
        manifest.append({"name": sample.name, "source": sample.source, "group": sample.group, "split": sample.split,
                         "origin": sample.origin, "objects": len(labels), "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest()})
    config = {"path": destination.resolve().as_posix(), "train": "images/train", "val": "images/val", "test": "images/test", "names": {0: "mango"}}
    # JSON is valid YAML and safely represents paths containing quotes or non-ASCII characters.
    (destination / "data.yaml").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = {"samples": manifest, "dropped": dropped, "counts": dict(Counter(f"{s.split}/{s.source}" for s in samples)),
              "objects": dict(Counter({split: sum(len(s.boxes) for s in samples if s.split == split) for split in ("train", "val", "test")})),
              "omitted_for_training_balance": dict(omitted),
              "split_policy": "orchard entire acquisition rows 2=val,3=test; smartphone original photos grouped in sequential blocks of 10 before hashing; postharvest author's test retained; non-mango author's Test held out; exact/near leakage removed",
              "limitations": "Original specimen identities are not provided for all postharvest/negative photos; unseen specimens/varieties still need local validation."}
    (destination / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"counts": report["counts"], "objects": report["objects"], "dropped": len(dropped)}, indent=2))
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    build(root / "sources", root / "mango_real")
