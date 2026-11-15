from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import logging
import platform
import shutil
import sys

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics import __version__ as ultralytics_version
from ultralytics.utils import LOGGER

from evaluation.metrics import detection_metrics


ROOT = Path(__file__).resolve().parents[1]


def training_device():
    if torch.cuda.is_available():
        try:
            if torch.cuda.mem_get_info()[0] >= 2*1024**3:
                return 0
        except RuntimeError:
            pass
    return "cpu"


def evaluate(model, dataset, destination, imgsz=512):
    device = training_device()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    validation = model.val(data=str(dataset / "data.yaml"), split="val", imgsz=imgsz, conf=.001,
                           device=device, batch=8, workers=0, plots=True, project=str(destination), name="validation")
    curve = validation.box.f1_curve.mean(axis=0)
    threshold = round(float(np.clip(validation.box.px[int(np.argmax(curve))], .15, .85)), 2)
    results = {"validation": {k: float(v) for k, v in validation.results_dict.items()}, "confidence_threshold": threshold}
    domains = {s["name"]: s["source"] for s in manifest["samples"]}
    for domain in ("overall", "orchard", "smartphone-orchard", "postharvest"):
        selected = [s for s in manifest["samples"] if s["split"] == "test" and (domain == "overall" or s["source"] == domain)]
        test_list = destination / f"{domain}_test.txt"
        paths = [dataset / "images" / "test" / (s["name"]+".jpg") for s in selected]
        test_list.write_text("\n".join(str(path.resolve()) for path in paths), encoding="utf-8")
        config = json.loads((dataset / "data.yaml").read_text(encoding="utf-8"))
        config["test"] = str(test_list.resolve())
        config_path = destination / f"{domain}.yaml"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        metrics = model.val(data=str(config_path), split="test", imgsz=imgsz, conf=.001, device=device, batch=8,
                            workers=0, plots=True, project=str(destination), name=domain)
        ground_truth, predictions = {}, []
        for path, result in zip(paths, model.predict([str(p) for p in paths], stream=True, imgsz=imgsz,
                                                     conf=threshold, device=device, verbose=False)):
            height, width = result.orig_shape
            boxes = []
            label_path = dataset / "labels" / "test" / (path.stem+".txt")
            for line in label_path.read_text().splitlines():
                _, x, y, w, h = map(float, line.split())
                boxes.append([(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height])
            ground_truth[path.stem] = boxes
            for box, confidence in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                predictions.append({"view_id": path.stem, "box": box, "confidence": confidence})
        results[domain] = {"images": len(selected), "objects": sum(s["objects"] for s in selected),
                           "standard_metrics": {k: float(v) for k, v in metrics.results_dict.items()},
                           "at_selected_threshold": detection_metrics(predictions, ground_truth)}
        if domain == "overall":
            (destination / "test_predictions.json").write_text(json.dumps({"confidence": threshold, "predictions": predictions,
                                                                           "ground_truth": ground_truth}, indent=2), encoding="utf-8")
            with (destination / "test_errors.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, ["image", "domain", "actual", "detected", "TP", "FP", "FN"])
                writer.writeheader()
                for key, boxes in ground_truth.items():
                    items = [p for p in predictions if p["view_id"] == key]
                    metrics_image = detection_metrics(items, {key: boxes})
                    writer.writerow({"image": key, "domain": domains[key], "actual": len(boxes), "detected": len(items),
                                     **{name: metrics_image[name] for name in ("TP", "FP", "FN")}})
    negatives = [s for s in manifest["samples"] if s["split"] == "test" and s["source"] == "non-mango"]
    paths = [str(dataset / "images" / "test" / (s["name"]+".jpg")) for s in negatives]
    counts = [len(result.boxes) for result in model.predict(paths, stream=True, imgsz=imgsz, conf=threshold, device=device, verbose=False)]
    results["non_mango"] = {"images": len(counts), "images_with_false_detection": sum(n > 0 for n in counts),
                             "false_positive_image_rate": sum(n > 0 for n in counts)/len(counts) if counts else None,
                             "false_detections": sum(counts)}
    results["dataset_counts"] = manifest["counts"]
    results["dataset_manifest_sha256"] = hashlib.sha256((dataset / "manifest.json").read_bytes()).hexdigest()
    return results


def run(args):
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    dataset = args.data.resolve().parent
    if not args.data.is_file() or not (dataset / "manifest.json").is_file():
        raise ValueError("Prepare dataset/mango_real first with python -m dataset.build_real.")
    project = ROOT / "output" / "training"
    destination = project / args.name
    if destination.exists():
        raise ValueError("Training run already exists. Choose another --name to preserve previous results.")
    destination.mkdir(parents=True)
    with (destination / "console.log").open("w", encoding="utf-8") as log:
        handler = logging.StreamHandler(log)
        original_handlers = LOGGER.handlers[:]
        LOGGER.handlers = [handler]
        try:
            with redirect_stdout(log), redirect_stderr(log):
                model = YOLO(str(args.weights))
                model.add_callback("on_train_start", lambda trainer: torch.set_num_threads(4))
                model.add_callback("on_val_start", lambda validator: torch.set_num_threads(4))
                def progress(trainer):
                    record = {"epoch": trainer.epoch+1, "epochs": args.epochs,
                              "metrics": {k: float(v) for k, v in trainer.metrics.items()}}
                    print(json.dumps(record), file=sys.__stdout__, flush=True)
                model.add_callback("on_fit_epoch_end", progress)
                model.train(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                            device=training_device(), workers=0, cache=False, optimizer="AdamW", lr0=.001, lrf=.05,
                            weight_decay=.0005, warmup_epochs=2, patience=10, freeze=args.freeze,
                            seed=42, deterministic=True, amp=False, hsv_h=.015, hsv_s=.5, hsv_v=.35,
                            degrees=15, translate=.1, scale=.4, fliplr=.5, mosaic=.7, close_mosaic=5,
                            mixup=0, copy_paste=0, project=str(project), name=args.name, exist_ok=True,
                            plots=True, save=True, save_period=5)
                best = destination / "weights" / "best.pt"
                if not best.is_file():
                    raise RuntimeError("Training produced no best.pt.")
                if args.evaluate:
                    model = YOLO(str(best))
                    evaluation = destination / "evaluation"
                    evaluation.mkdir()
                    report = evaluate(model, dataset, evaluation, args.imgsz)
                    stages = []
                    for stage in [args.weights.resolve().parent.parent, destination]:
                        history = stage / "results.csv"
                        if history.is_file():
                            with history.open(newline="", encoding="utf-8") as stream:
                                rows = list(csv.DictReader(stream))
                            stages.append({"name": stage.name, "epochs_completed": len(rows),
                                           "training_seconds": float(rows[-1]["time"]) if rows else 0})
                    report.update({"created_at": datetime.now(timezone.utc).isoformat(), "training_run": str(destination),
                                   "imgsz": args.imgsz, "seed": 42,
                                   "epochs_completed": len((destination / "results.csv").read_text().splitlines())-1,
                                   "initial_checkpoint": str(args.weights),
                                   "training_stages": stages,
                                   "software": {"python": platform.python_version(), "torch": torch.__version__,
                                                "ultralytics": ultralytics_version, "opencv": cv2.__version__,
                                                "cuda": torch.version.cuda},
                                   "hardware": {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
                                   "model_sha256": hashlib.sha256(best.read_bytes()).hexdigest(),
                                   "task": "single-class mango detection, orchard and postharvest",
                                   "limitations": ["Finite public test set, not proof of perfect accuracy or generalization to every cultivar.",
                                                    "Source postharvest annotations released for research; upstream image rights still apply.",
                                                    "Does not learn physiological maturity or guarantee duplicate removal."]})
                    (evaluation / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
                    target = ROOT / "models" / "best.pt"
                    if target.exists():
                        shutil.copy2(target, target.with_name(f"previous_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}.pt"))
                    shutil.copy2(best, target)
                    (ROOT / "models" / "training_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
                    print(json.dumps({"model": str(target), "report": str(evaluation / "metrics.json"), "summary": report}), file=sys.__stdout__, flush=True)
        finally:
            LOGGER.handlers = original_handlers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train on real annotated mangoes and evaluate on untouched held-out data.")
    parser.add_argument("--data", type=Path, default=ROOT / "dataset" / "mango_real" / "data.yaml")
    parser.add_argument("--weights", type=Path, default=ROOT / "models" / "yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--name", default="mango_yolo11n")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate on held-out data and install models/best.pt after successful training")
    run(parser.parse_args())
