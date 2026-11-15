"""Optional COLMAP adapter; reconstruction is never marked active without a real result."""
from pathlib import Path
import argparse
import shutil
import subprocess
import numpy as np


def reconstruct(images: Path, workspace: Path, executable="colmap"):
    binary = shutil.which(executable)
    if not binary:
        raise ValueError("COLMAP tidak ditemukan di PATH. Instal COLMAP terlebih dahulu.")
    if not images.is_dir() or not any(p.suffix.lower() in {".jpg", ".png", ".jpeg"} for p in images.iterdir()):
        raise ValueError("Folder gambar COLMAP kosong / tidak tersedia.")
    if workspace.exists():
        raise ValueError("Pilih workspace COLMAP baru agar rekonstruksi lama tidak tertimpa.")
    workspace.mkdir(parents=True)
    sparse = workspace / "sparse"
    sparse.mkdir()
    database = workspace / "database.db"
    commands = [
        ["feature_extractor", "--database_path", str(database), "--image_path", str(images)],
        ["exhaustive_matcher", "--database_path", str(database)],
        ["mapper", "--database_path", str(database), "--image_path", str(images), "--output_path", str(sparse)],
    ]
    log = workspace / "colmap.log"
    with log.open("w", encoding="utf-8") as stream:
        for command in commands:
            subprocess.run([binary, *command], stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=1800)
        candidates = sorted(p for p in sparse.iterdir() if p.is_dir())
        if not candidates:
            raise ValueError(f"COLMAP tidak menghasilkan model. Periksa {log}.")
        text = workspace / "text"
        text.mkdir()
        subprocess.run([binary, "model_converter", "--input_path", str(candidates[0]), "--output_path", str(text),
                        "--output_type", "TXT"], stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=180)
    return text


def load_points(path: Path):
    points, colors = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        values = line.split()
        if len(values) < 8:
            raise ValueError("Format COLMAP points3D.txt tidak valid.")
        points.append([float(v) for v in values[1:4]])
        colors.append([float(v)/255 for v in values[4:7]])
    xyz, rgb = np.asarray(points).reshape(-1, 3), np.asarray(colors).reshape(-1, 3)
    if not len(xyz) or not np.isfinite(xyz).all() or not np.isfinite(rgb).all() or np.any((rgb < 0) | (rgb > 1)):
        raise ValueError("Point cloud kosong atau tidak valid.")
    return xyz, rgb


def export_open3d(points, colors, destination: Path, show=False):
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ValueError("Open3D belum terpasang. Instal requirements-3d.txt dengan Python yang didukung.") from exc
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    if not o3d.io.write_point_cloud(str(destination), cloud):
        raise OSError("Gagal menyimpan point cloud PLY.")
    if show:
        o3d.visualization.draw_geometries([cloud], window_name="Mango Smart Counting - COLMAP point cloud")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--show", action="store_true", help="Open desktop Open3D visualizer after reconstruction")
    args = parser.parse_args()
    text = reconstruct(args.images, args.workspace)
    xyz, rgb = load_points(text / "points3D.txt")
    print(export_open3d(xyz, rgb, args.workspace / "cloud.ply", show=args.show))
