from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import uuid


@dataclass
class Observation:
    id: str
    view_id: str
    box: list[float]
    confidence: float
    label: str = "mango"
    track_id: int | None = None
    unique_id: str = ""
    yellow_fraction: float = 0.0
    xyz: list[float] | None = None


@dataclass
class View:
    id: str
    path: str
    source: str
    width: int
    height: int
    blur_score: float
    frame: int | None = None
    seconds: float | None = None


@dataclass
class Link:
    a: str
    b: str
    score: float
    method: str
    accepted: bool = True


@dataclass
class Run:
    id: str
    created_at: str
    directory: str
    mode: str
    settings: dict
    views: list[View] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    fruits: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    evaluations: list[dict] = field(default_factory=list)
    video_path: str | None = None

    @classmethod
    def create(cls, root: Path, mode: str, settings: dict):
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
        directory = root / run_id
        directory.mkdir(parents=True)
        (directory / "views").mkdir()
        (directory / "annotated").mkdir()
        return cls(run_id, datetime.now(timezone.utc).isoformat(), str(directory), mode, settings)

    def save(self):
        target = Path(self.directory) / "run.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(target)

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, record in [("views", View), ("observations", Observation), ("links", Link)]:
            data[key] = [record(**item) for item in data[key]]
        return cls(**data)


def finite_number(value, name: str, minimum=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} harus berupa angka.") from exc
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name} harus bernilai hingga dan >= {minimum}.")
    return result
