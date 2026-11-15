import numpy as np
from modules.schema import finite_number


def count_metrics(actual, predicted):
    if len(actual) != len(predicted) or not len(actual):
        raise ValueError("Pasangan jumlah aktual dan prediksi harus tersedia dengan panjang sama.")
    actual = np.array([finite_number(v, "actual") for v in actual])
    predicted = np.array([finite_number(v, "predicted") for v in predicted])
    if np.any(actual != np.floor(actual)) or np.any(predicted != np.floor(predicted)):
        raise ValueError("Jumlah buah harus bilangan bulat.")
    error = predicted - actual
    positive = actual > 0
    return {"MAE": float(np.abs(error).mean()), "RMSE": float(np.sqrt(np.square(error).mean())),
            "MAPE": float((np.abs(error[positive])/actual[positive]).mean()*100) if positive.any() else None,
            "MAPE_excluded_zero_actual": int((~positive).sum()),
            "Counting Error": float(error.sum()), "Absolute Counting Error": float(np.abs(error).sum()),
            "samples": len(actual)}


def box_iou(box, boxes):
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    if not len(boxes):
        return np.array([])
    box = np.asarray(box, float)
    top = np.maximum(box[:2], boxes[:, :2])
    bottom = np.minimum(box[2:], boxes[:, 2:])
    intersect = np.prod(np.maximum(bottom-top, 0), axis=1)
    a = np.prod(box[2:]-box[:2])
    b = np.prod(boxes[:, 2:]-boxes[:, :2], axis=1)
    return intersect/np.maximum(a+b-intersect, 1e-12)


def detection_metrics(predictions, ground_truth):
    """Single mango class, confidence-ranked matching; 101-point interpolated AP50."""
    if not ground_truth:
        raise ValueError("Ground truth harus mencakup setiap view evaluasi, termasuk daftar kosong.")
    if set(p["view_id"] for p in predictions) - set(ground_truth):
        raise ValueError("Ground truth belum mencakup semua view prediksi.")
    for boxes in ground_truth.values():
        for box in boxes:
            if len(box) != 4 or not np.isfinite(box).all() or min(box) < 0 or box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("Ground truth xyxy tidak valid.")
    used = {view: set() for view in ground_truth}
    tp, fp = [], []
    for pred in sorted(predictions, key=lambda item: item["confidence"], reverse=True):
        overlaps = box_iou(pred["box"], ground_truth[pred["view_id"]])
        candidates = [i for i, overlap in enumerate(overlaps) if i not in used[pred["view_id"]] and overlap >= 0.5]
        match = max(candidates, key=lambda i: overlaps[i]) if candidates else None
        tp.append(int(match is not None))
        fp.append(int(match is None))
        if match is not None:
            used[pred["view_id"]].add(match)
    total = sum(map(len, ground_truth.values()))
    tps, fps = np.cumsum(tp), np.cumsum(fp)
    precision_curve = tps/np.maximum(tps+fps, 1)
    recall_curve = tps/max(total, 1)
    precision = float(tps[-1]/(tps[-1]+fps[-1])) if len(tps) else 0.0
    recall = float(tps[-1]/total) if len(tps) and total else 0.0
    ap = float(np.mean([precision_curve[recall_curve >= r].max() if np.any(recall_curve >= r) else 0.0
                        for r in np.linspace(0, 1, 101)])) if total else None
    return {"Precision": precision if len(tp) else None, "Recall": recall if total else None,
            "F1": 2*precision*recall/(precision+recall) if precision+recall else 0.0,
            "mAP@50": ap, "TP": int(sum(tp)), "FP": int(sum(fp)), "FN": total-int(sum(tp)),
            "note": "AP satu kelas mangga pada confidence tersimpan; untuk seluruh kurva gunakan YOLO val conf=0.001."}
