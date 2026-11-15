from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import logging
import shutil
import subprocess

import cv2
import numpy as np

from modules.schema import Observation, Run, View
from preprocessing.images import PreprocessConfig, decode_image, preprocess, write_image, read_image
from multiview.association import associate, extract
from duplicate_removal.grouping import group_observations
from maturity.assessment import assess
from tracking.byte import VideoTracker

logger = logging.getLogger(__name__)


def add_view(run, image, source, detector, config, frame=None, seconds=None, tracker=None):
    image, blur = preprocess(image, config)
    view_id = f"V{len(run.views)+1:04d}"
    path = Path(run.directory) / "views" / f"{view_id}.jpg"
    write_image(path, image)
    view = View(view_id, str(path), source, image.shape[1], image.shape[0], blur, frame, seconds)
    predictions = detector.predict(image)
    track_ids = tracker.update(predictions, image.shape) if tracker else {}
    for index, (box, score, label) in enumerate(predictions):
        box = np.asarray(box, float)
        if box.shape != (4,) or not np.isfinite(box).all() or not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Output detektor tidak valid.")
        box[[0, 2]] = np.clip(box[[0, 2]], 0, image.shape[1])
        box[[1, 3]] = np.clip(box[[1, 3]], 0, image.shape[0])
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        run.observations.append(Observation(f"{view_id}-D{index+1:03d}", view_id, box.tolist(), score, label, track_ids.get(index)))
    run.views.append(view)
    if blur < 50:
        run.warnings.append(f"{view_id}: kemungkinan blur (variansi Laplacian {blur:.1f}); ambang perlu kalibrasi kamera.")


def process_images(inputs, detector, output_root, config=None, progress=None):
    if not 1 <= len(inputs) <= 40:
        raise ValueError("Masukkan 1 sampai 40 gambar per pohon.")
    if sum(len(data) for _, data in inputs) > 200*1024*1024:
        raise ValueError("Total ukuran foto melebihi 200 MB. Kurangi jumlah atau resolusi foto.")
    config = config or PreprocessConfig()
    run = Run.create(Path(output_root), detector.mode, {"preprocessing": asdict(config), "confidence": detector.confidence,
                                                       "detector": detector.metadata})
    for index, (name, data) in enumerate(inputs):
        try:
            image = decode_image(data)
        except ValueError as exc:
            run.warnings.append(f"{Path(name).name}: {exc}")
            logger.warning("Image decode rejected: %s", exc)
            continue
        add_view(run, image, Path(name).name, detector, config)
        if progress:
            progress(0.55*(index+1)/len(inputs), f"Deteksi gambar {index+1}/{len(inputs)}")
    if not run.views:
        raise ValueError("Tidak ada gambar valid untuk diproses.")
    finalize(run, progress=progress)
    return run


def process_video(data, suffix, detector, output_root, config=None, stride=1, max_frames=300, progress=None):
    if not data or len(data) > 200*1024*1024 or suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise ValueError("Video kosong, format tidak didukung, atau melebihi 200 MB.")
    if not 1 <= stride <= 30 or not 1 <= max_frames <= 1000:
        raise ValueError("Stride 1..30 dan batas frame 1..1000 diperlukan.")
    config = config or PreprocessConfig()
    run = Run.create(Path(output_root), detector.mode, {"preprocessing": asdict(config), "confidence": detector.confidence,
                                                       "stride": stride, "max_frames": max_frames, "detector": detector.metadata})
    source = Path(run.directory) / f"input{suffix.lower()}"
    source.write_bytes(data)
    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            raise ValueError("Video tidak dapat dibuka. Coba ekspor sebagai MP4 H.264.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0 or fps > 240:
            raise ValueError("Frame rate video tidak valid.")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        run.settings["fps"] = fps/stride
        tracker = VideoTracker(fps/stride, detector.confidence)
        index = 0
        while len(run.views) < max_frames:
            success, image = capture.read()
            if not success:
                if total > 0 and index < total-2:
                    run.warnings.append(f"Dekode video berhenti di frame {index}/{total}; hasil hanya mencakup bagian terbaca.")
                break
            if index % stride == 0:
                add_view(run, image, source.name, detector, config, frame=index, seconds=index/fps, tracker=tracker)
                if progress:
                    progress(0.5*len(run.views)/min(max_frames, max(total//stride, 1)), f"ByteTrack: frame sumber {index}")
            index += 1
        if len(run.views) == max_frames:
            run.warnings.append(f"Batas {max_frames} frame tercapai; video mungkin belum diproses seluruhnya.")
    finally:
        capture.release()
        source.unlink(missing_ok=True)
    if not run.views:
        raise ValueError("Video tidak memiliki frame yang dapat dibaca.")
    finalize(run, progress=progress)
    return run


def finalize(run, projections=None, spatial_radius=0.04, progress=None):
    if not np.isfinite(spatial_radius) or spatial_radius <= 0:
        raise ValueError("Radius 3D harus positif dalam satuan metadata.")
    if projections and set(projections) - {v.id for v in run.views}:
        raise ValueError("Metadata kamera memiliki view_id yang tidak ada pada hasil.")
    by_view = defaultdict(list)
    for obs in run.observations:
        by_view[obs.view_id].append(obs)
    features = {}
    manual_links = [link for link in run.links if link.method == "manual-review"]
    run.links = []
    for j, second in enumerate(run.views):
        features[second.id] = extract(read_image(second.path), by_view[second.id])
        if second.frame is None:
            previous = range(j)
        else:
            # shortcut: video compares the last 3 sampled views plus the first for loop closure; widen for long occlusions.
            previous = sorted(set(range(max(0, j-3), j)) | ({0} if j > 3 else set()))
        for i in previous:
            first = run.views[i]
            pair_projections = (projections[first.id], projections[second.id]) if projections and first.id in projections and second.id in projections else None
            run.links.extend(associate(by_view[first.id], by_view[second.id], features[first.id], features[second.id],
                                       adjacent=second.frame is not None and j-i == 1,
                                       projections=pair_projections, spatial_radius=spatial_radius))
        if progress:
            progress(0.55+0.3*(j+1)/len(run.views), f"Asosiasi SIFT {j+1}/{len(run.views)}")
        if second.frame is not None and j > 3:
            features.pop(run.views[j-3].id, None)
    run.links.extend(manual_links)
    group_observations(run.observations, run.links)
    run.fruits = assess(run.observations)
    run.settings.pop("calibration", None)
    run.settings.pop("field_records", None)
    run.evaluations = []
    run.warnings = list(dict.fromkeys(run.warnings))
    refresh_artifacts(run)
    if progress:
        progress(1.0, "Analisis dan laporan tersimpan")


def annotate(image, observations, unique=True):
    result = image.copy()
    for obs in observations:
        x1, y1, x2, y2 = map(int, obs.box)
        color = (75, 153, 37) if unique else (25, 140, 235)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        score = f"shape {obs.confidence:.2f}" if obs.label == "synthetic-shape" else f"{obs.confidence:.2f}"
        name = obs.unique_id if unique else obs.id.split("-")[-1]
        track = f" T{obs.track_id}" if obs.track_id is not None else ""
        cv2.putText(result, f"{name} {obs.label} {score}{track}", (x1, max(16, y1-7)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return result


def refresh_artifacts(run):
    by_view = defaultdict(list)
    for obs in run.observations:
        by_view[obs.view_id].append(obs)
    writer = None
    destination = Path(run.directory) / "tracking.mp4"
    try:
        for view in run.views:
            annotated = annotate(read_image(view.path), by_view[view.id])
            write_image(Path(run.directory) / "annotated" / f"{view.id}.jpg", annotated)
            if view.frame is not None:
                if writer is None:
                    height, width = annotated.shape[:2]
                    # MPEG-4 encoders require even dimensions.
                    shape = (width//2*2, height//2*2)
                    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), run.settings["fps"], shape)
                    if not writer.isOpened():
                        raise OSError("Codec MP4 tidak tersedia untuk menyimpan tracking.")
                writer.write(cv2.resize(annotated, shape))
    finally:
        if writer is not None:
            writer.release()
    if writer is not None:
        run.video_path = str(destination)
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            browser_video = destination.with_name("tracking_h264.mp4")
            try:
                subprocess.run([ffmpeg, "-y", "-i", str(destination), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                "-movflags", "+faststart", str(browser_video)], check=True, capture_output=True, timeout=180)
                run.video_path = str(browser_video)
            except (subprocess.SubprocessError, OSError):
                logger.exception("H264 conversion failed")
                run.warnings.append("Konversi H.264 gagal; unduh tracking.mp4 untuk diputar di pemutar video.")
        else:
            warning = "FFmpeg tidak ditemukan: tracking MP4 tersedia untuk diunduh; preview browser mungkin tidak mendukung codec mp4v."
            if warning not in run.warnings:
                run.warnings.append(warning)
    run.save()
    from reports.export import write_reports
    write_reports(run)
