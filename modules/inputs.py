import csv
from io import StringIO
import json
import numpy as np
from modules.schema import finite_number


def csv_records(data: bytes, required):
    if len(data) > 5_000_000:
        raise ValueError("CSV melebihi 5 MB.")
    reader = csv.DictReader(StringIO(data.decode("utf-8-sig")))
    if not reader.fieldnames or not set(required) <= set(reader.fieldnames):
        raise ValueError(f"Kolom wajib: {', '.join(required)}")
    rows = list(reader)
    if len(rows) > 50_000:
        raise ValueError("CSV melebihi 50.000 baris.")
    return rows


def field_records(data):
    rows = csv_records(data, ["unique_id"])
    result = {}
    for row in rows:
        uid = row.pop("unique_id").strip()
        if not uid or uid in result:
            raise ValueError("unique_id kosong atau duplikat dalam data lapangan.")
        result[uid] = {k: v.strip() for k, v in row.items() if k and v is not None and v.strip()}
    return result


def projections_from_json(data):
    if len(data) > 1_000_000:
        raise ValueError("Metadata kamera melebihi 1 MB.")
    raw = json.loads(data)
    if not isinstance(raw, dict):
        raise ValueError("Metadata kamera harus berupa object view_id -> matriks 3x4.")
    result = {}
    for view, matrix in raw.items():
        array = np.asarray(matrix, dtype=float)
        if array.shape != (3, 4) or not np.isfinite(array).all() or np.linalg.matrix_rank(array) != 3:
            raise ValueError(f"Matriks kamera {view} tidak valid.")
        result[view] = array
    return result


def spatial_records(data, observations):
    lookup = {o.id: o for o in observations}
    updates = {}
    for row in csv_records(data, ["observation_id", "x", "y", "z"]):
        key = row["observation_id"]
        if key not in lookup or key in updates:
            raise ValueError(f"ID spasial tidak tersedia atau duplikat: {key}")
        point = [float(row[axis]) for axis in ("x", "y", "z")]
        if not np.isfinite(point).all():
            raise ValueError("Koordinat 3D harus bernilai hingga.")
        updates[key] = point
    for key, point in updates.items():
        lookup[key].xyz = point


def count_records(data):
    rows = csv_records(data, ["tree_id", "actual", "predicted"])
    ids = [row["tree_id"] for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Satu baris per tree_id yang unik diperlukan.")
    return rows
