"""Run the two training stages when RAM/VRAM are available, without closing user applications."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import psutil

ROOT = Path(__file__).resolve().parents[1]
GIB = 1024**3


def enough_memory(ram, vram, gpu):
    return ram >= 4*GIB and (not gpu or vram is not None and vram >= 4*GIB)


def free_gpu_memory():
    if not shutil.which("nvidia-smi"):
        return None
    result = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True, timeout=20)
    return int(result.stdout.strip().splitlines()[0])*1024**2


def execute(args):
    directory = ROOT / "output" / "training"
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / "queued_training.json"
    if status_path.exists():
        previous = json.loads(status_path.read_text(encoding="utf-8"))
        pid = previous.get("pid")
        if pid and psutil.pid_exists(pid) and "training.run_when_ready" in psutil.Process(pid).cmdline():
            raise RuntimeError("A training coordinator is already running.")
    job = "mango_joint_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    state = {"job": job, "pid": os.getpid(), "created_at": datetime.now(timezone.utc).isoformat(), "state": "starting"}
    def save(**updates):
        state.update(updates, updated_at=datetime.now(timezone.utc).isoformat())
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temporary.replace(status_path)
        print(json.dumps(state), flush=True)
    def wait_for_memory(stage):
        while True:
            ram = psutil.virtual_memory().available
            vram = free_gpu_memory() if args.gpu else None
            if enough_memory(ram, vram, args.gpu):
                return
            save(state="waiting_resources", stage=stage, ram_free_gib=round(ram/GIB, 2),
                 vram_free_gib=round(vram/GIB, 2) if vram is not None else None,
                 message="Menunggu RAM kosong 4 GiB dan VRAM 4 GiB untuk GPU. Aplikasi lain tidak ditutup otomatis.")
            time.sleep(30)
    try:
        if not args.python.is_file() or not args.data.is_file():
            raise ValueError("Python interpreter or real dataset config not found.")
        weights = ROOT / "models" / "yolo11n.pt"
        for stage, epochs, size, freeze in [("warmup", 8, 384, 10), ("finetune", 30, 640, 0)]:
            wait_for_memory(stage)
            name = f"{job}_{stage}"
            run_path = directory / name
            save(state="training", stage=stage, epochs=epochs, run_path=str(run_path),
                 message="Training nyata sedang berjalan; hasil test baru ditulis setelah training selesai.")
            command = [str(args.python.resolve()), "-u", "-m", "training.train", "--data", str(args.data.resolve()),
                       "--weights", str(weights), "--epochs", str(epochs), "--imgsz", str(size), "--freeze", str(freeze),
                       "--batch", "4", "--name", name]
            if stage == "finetune":
                command.append("--evaluate")
            with (directory / f"{name}_job.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            weights = run_path / "weights" / "best.pt"
        save(state="complete", message="Training, evaluasi, dan pemasangan models/best.pt selesai.")
    except Exception as exc:
        save(state="failed", message=str(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--data", type=Path, default=ROOT / "dataset" / "mango_real" / "data.yaml")
    parser.add_argument("--gpu", action="store_true")
    execute(parser.parse_args())
