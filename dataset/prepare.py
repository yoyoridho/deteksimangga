"""Prepare one-class YOLO data without overwriting source or leaking capture groups."""
from pathlib import Path
from collections import defaultdict
import argparse
import hashlib
import random
import re
import shutil
import numpy as np

from preprocessing.images import decode_image


def prepare(source: Path, destination: Path, seed=42):
    if destination.exists():
        raise ValueError("Folder tujuan sudah ada. Pilih folder baru agar data tidak ditimpa.")
    images = sorted(p for p in (source / "images").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        raise ValueError("Struktur sumber harus images/* dan labels/<nama>.txt.")
    if len({p.stem for p in images}) != len(images):
        raise ValueError("Nama dasar gambar harus unik agar label tidak tertimpa.")
    groups = defaultdict(list)
    digests = {}
    for image in images:
        data = image.read_bytes()
        decode_image(data)
        label = source / "labels" / f"{image.stem}.txt"
        if not label.is_file():
            raise ValueError(f"Label hilang: {label.name}; gunakan file kosong untuk gambar negatif.")
        for line in label.read_text(encoding="utf-8-sig").splitlines():
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Format label tidak valid: {label.name}")
            values = np.asarray([float(v) for v in parts])
            if not np.isfinite(values).all() or values[0] != 0 or not np.all((values[1:] >= 0) & (values[1:] <= 1)) or min(values[3:]) <= 0:
                raise ValueError(f"Label harus kelas 0 dan xywh normalisasi 0..1: {label.name}")
            x, y, w, h = values[1:]
            if x-w/2 < -1e-6 or y-h/2 < -1e-6 or x+w/2 > 1+1e-6 or y+h/2 > 1+1e-6:
                raise ValueError(f"Bounding box keluar gambar: {label.name}")
        # Capture names tree01__angle1 or video01_frame001 share a split.
        group = re.split(r"__|_frame", image.stem, maxsplit=1)[0]
        digest = hashlib.sha256(data).hexdigest()
        if digest in digests and digests[digest] != group:
            raise ValueError("Gambar identik di grup berbeda; rapikan data agar tidak bocor antar split.")
        digests[digest] = group
        groups[group].append((image, label))
    keys = sorted(groups)
    if len(keys) < 3:
        raise ValueError("Minimal tiga grup pohon/perekaman independen untuk train, val, test.")
    random.Random(seed).shuffle(keys)
    n_test, n_val = max(1, round(len(keys)*0.1)), max(1, round(len(keys)*0.2))
    partitions = {"test": keys[:n_test], "val": keys[n_test:n_test+n_val], "train": keys[n_test+n_val:]}
    for split, group_keys in partitions.items():
        for folder in ("images", "labels"):
            (destination / folder / split).mkdir(parents=True, exist_ok=True)
        for key in group_keys:
            for image, label in groups[key]:
                shutil.copy2(image, destination / "images" / split / image.name)
                shutil.copy2(label, destination / "labels" / split / label.name)
    dataset_path = destination.resolve().as_posix().replace("'", "''")
    (destination / "data.yaml").write_text(f"path: '{dataset_path}'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: mango\n", encoding="utf-8")
    return {split: sum(len(groups[k]) for k in items) for split, items in partitions.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(prepare(args.source, args.destination))
