from pathlib import Path
import hashlib
import cv2
import numpy as np


class Detector:
    def __init__(self, mode="synthetic", model_path=None, class_ids=None, confidence=0.25):
        if mode not in {"synthetic", "yolo-demo", "mango"}:
            raise ValueError("Mode detektor tidak dikenal.")
        if not 0.01 <= confidence <= 1:
            raise ValueError("Confidence harus 0.01..1.")
        self.mode, self.confidence, self.class_ids = mode, confidence, class_ids
        self.model = None
        if mode != "synthetic":
            from ultralytics import YOLO
            if mode == "mango" and (not model_path or not Path(model_path).is_file()):
                raise ValueError("Model mangga tidak ditemukan. Masukkan models/best.pt atau unggah model.")
            self.model = YOLO(str(model_path or "yolo11n.pt"), task="detect")
            if self.model.task != "detect":
                raise ValueError("Gunakan model YOLO object detection.")
            if mode == "yolo-demo":
                self.class_ids = [i for i, name in self.model.names.items() if name in {"apple", "banana", "orange"}]
                if not self.class_ids:
                    raise ValueError("Model demo harus memiliki kelas COCO apple, banana, orange.")
            elif not class_ids:
                self.class_ids = [i for i, name in self.model.names.items() if name.lower() in {"mango", "mangga"}]
                if not self.class_ids:
                    raise ValueError("Model tidak memiliki kelas mango/mangga. Tentukan ID kelas mangga secara eksplisit.")
            if any(i not in self.model.names for i in self.class_ids):
                raise ValueError("ID kelas tidak tersedia pada model.")
        self.metadata = {"mode": mode, "class_ids": self.class_ids, "confidence": confidence}
        if self.model is not None:
            checkpoint = Path(self.model.ckpt_path)
            with checkpoint.open("rb") as stream:
                self.metadata["model_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            self.metadata["model"] = checkpoint.name
            self.metadata["class_names"] = [self.model.names[i] for i in self.class_ids]
            self.metadata["imgsz"] = self.model.overrides.get("imgsz", 640)

    def predict(self, image):
        """Return xyxy, score, original class name. No invented YOLO predictions."""
        if self.mode == "synthetic":
            # shortcut: HSV recognizes only the supplied synthetic shapes; use a trained model for field photos.
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, (18, 115, 85), (82, 255, 255))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            boxes = []
            for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
                if cv2.contourArea(contour) < 250:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                # The synthetic score is a shape fill ratio, not model confidence.
                score = min(1.0, cv2.contourArea(contour) / (w*h))
                if score >= self.confidence:
                    boxes.append(([float(x), float(y), float(x+w), float(y+h)], float(score), "synthetic-shape"))
            return sorted(boxes, key=lambda item: item[0][0])
        result = self.model.predict(image, conf=self.confidence, classes=self.class_ids, verbose=False, device="cpu")[0]
        return [(box.tolist(), float(score), result.names[int(cls)]) for box, score, cls in
                zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy())]
