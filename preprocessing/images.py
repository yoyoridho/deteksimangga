from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass
class PreprocessConfig:
    max_side: int = 960
    enhance: bool = True
    brightness: float = 1.0
    crop: tuple[float, float, float, float] | None = None


def decode_image(data: bytes) -> np.ndarray:
    if not data or len(data) > 40 * 1024 * 1024:
        raise ValueError("Gambar kosong atau melebihi 40 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as source:
                if source.width * source.height > 40_000_000:
                    raise ValueError("Gambar melebihi 40 megapiksel.")
                rgb = np.asarray(ImageOps.exif_transpose(source).convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    except (OSError, Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ValueError("Format gambar rusak atau tidak didukung.") from exc


def preprocess(image: np.ndarray, config: PreprocessConfig):
    if image is None or image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Gambar harus BGR uint8 dengan tiga kanal.")
    if not 128 <= config.max_side <= 4096 or not 0.3 <= config.brightness <= 2.5:
        raise ValueError("Ukuran / pencahayaan di luar batas.")
    image = image.copy()
    if config.crop is not None:
        x1, y1, x2, y2 = config.crop
        if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            raise ValueError("Crop harus berupa koordinat proporsi 0..1 yang berurutan.")
        h, w = image.shape[:2]
        image = image[int(y1*h):int(y2*h), int(x1*w):int(x2*w)]
    if min(image.shape[:2]) < 8:
        raise ValueError("Area gambar terlalu kecil.")
    h, w = image.shape[:2]
    scale = min(config.max_side / max(h, w), 1.0)
    if scale < 1:
        image = cv2.resize(image, (max(1, round(w*scale)), max(1, round(h*scale))), interpolation=cv2.INTER_AREA)
    blur = float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
    if config.enhance:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[:, :, 0])
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    image = np.clip(image.astype(np.float32) * config.brightness, 0, 255).astype(np.uint8)
    return image, blur


def normalize(image):
    """CHW RGB float32 in 0..1; YOLO's own predictor normalizes internally."""
    return np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255


def write_image(path: Path, image):
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not success:
        raise OSError("Gagal mengenkode gambar.")
    path.write_bytes(encoded.tobytes())


def read_image(path: str | Path):
    image = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Tidak dapat membaca gambar {Path(path).name}.")
    return image
