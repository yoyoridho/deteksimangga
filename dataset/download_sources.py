"""Download versioned public research datasets locally; never execute their code."""
from pathlib import Path
import hashlib
import json
import requests
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent / "sources"
SOURCES = {
    "orchard.parquet": "https://huggingface.co/datasets/Project-AgML/mango_detection_australia/resolve/5dded49774d27723649380e5301d75428c5b3545/data/train-00000-of-00001.parquet",
    "postharvest.zip": "https://codeload.github.com/nhasanati/Mango-YOLO/zip/40ad503d265fc041d07b0da539301cd5182ae546",
    "growth.parquet": "https://huggingface.co/datasets/Project-AgML/mango_growth_detection/resolve/25a94aa0f2cf8759d5c102d8c38dfc292e0301dc/data/train-00000-of-00001.parquet",
}


def download(url, destination):
    temporary = destination.with_suffix(destination.suffix + ".part")
    if destination.exists():
        return
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for chunk in response.iter_content(1024*1024):
                stream.write(chunk)
    temporary.replace(destination)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    previous_manifest = json.loads((ROOT / "downloads.json").read_text(encoding="utf-8")) if (ROOT / "downloads.json").exists() else {}
    manifest = {}
    for filename, url in SOURCES.items():
        path = ROOT / filename
        print(f"Downloading {filename}...", flush=True)
        download(url, path)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        previous = previous_manifest.get(filename, {})
        actual_url = previous.get("url", url) if previous.get("sha256") == digest else url
        manifest[filename] = {"url": actual_url, "pinned_source_url": url, "bytes": path.stat().st_size, "sha256": digest}
        print(f"Ready {filename}: {path.stat().st_size:,} bytes", flush=True)
    (ROOT / "downloads.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def download_negatives():
    """Author-provided Training/Test partitions; all selected categories are non-mango."""
    repository = "fruits-360/fruits-360-100x100"
    api = "https://api.github.com/repos/" + repository
    revision = requests.get(api + "/commits/main", timeout=30)
    revision.raise_for_status()
    revision = revision.json()["sha"]
    categories = ["Apple Granny Smith 1", "Apple Red 1", "Avocado 1", "Avocado ripe 1", "Banana 1",
                  "Guava 1", "Kiwi 1", "Lemon 1", "Mangostan 1", "Orange 1", "Papaya 1", "Pear 1"]
    records = []
    for partition in ("Training", "Test"):
        listing = requests.get(f"{api}/contents/{partition}?ref={revision}", timeout=30)
        listing.raise_for_status()
        available = {item["name"] for item in listing.json()}
        for category in categories:
            if category not in available:
                continue
            listing = requests.get(f"{api}/contents/{partition}/{quote(category)}?ref={revision}", timeout=30)
            listing.raise_for_status()
            files = sorted((item for item in listing.json() if item["name"].endswith(".jpg")), key=lambda x: x["name"])
            count = 16 if partition == "Training" else 12
            indices = [round(i*(len(files)-1)/(count-1)) for i in range(count)]
            for index in sorted(set(indices)):
                item = files[index]
                relative = f"{partition}/{category}/{item['name']}"
                destination = ROOT / "negatives" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                records.append({"path": str(destination.relative_to(ROOT)), "url": item["download_url"],
                                "partition": partition, "category": category})
    def fetch(record):
        download(record["url"], ROOT / record["path"])
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, records))
    (ROOT / "negative_sources.json").write_text(json.dumps({"revision": revision, "license": "CC BY-SA 4.0", "records": records}, indent=2), encoding="utf-8")
    print(f"Ready {len(records)} real non-mango images", flush=True)


if __name__ == "__main__":
    main()
    download_negatives()
