from pathlib import Path
import hashlib
import json


def trained_model_info(directory: Path):
    """Only identify our locally trained checkpoint when its recorded checksum matches."""
    model = directory / "best.pt"
    report = directory / "training_summary.json"
    if not model.is_file() or not report.is_file():
        return None
    try:
        info = json.loads(report.read_text(encoding="utf-8"))
        with model.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if info.get("model_sha256") != checksum:
            return None
        return info
    except (OSError, ValueError, AttributeError):
        return None
