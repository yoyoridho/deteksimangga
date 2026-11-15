from types import SimpleNamespace
import numpy as np


class VideoTracker:
    """A fresh ByteTrack instance per video; track IDs are association hints."""
    def __init__(self, fps=25, confidence=0.25):
        from ultralytics.trackers.byte_tracker import BYTETracker
        self.tracker = BYTETracker(SimpleNamespace(
            track_high_thresh=confidence, track_low_thresh=min(0.1, confidence),
            new_track_thresh=confidence, track_buffer=max(1, round(fps)), match_thresh=0.8, fuse_score=True,
        ))

    def update(self, detections, shape):
        from ultralytics.engine.results import Boxes
        rows = np.asarray([[*box, score, 0] for box, score, _ in detections], dtype=np.float32).reshape(-1, 6)
        tracks = self.tracker.update(Boxes(rows, shape[:2]))
        # ByteTrack returns xyxy, track_id, score, class_id, input_index.
        ids = {}
        for track in tracks:
            ids[int(track[-1])] = int(track[4])
        return ids
