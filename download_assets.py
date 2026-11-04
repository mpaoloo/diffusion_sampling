"""Загрузка предобученных весов и test-части CIFAR-10"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parent
REVISION = "267b167dc01f0e4e61923ea244e8b988f84deb80"
MODEL_BASE = f"https://huggingface.co/google/ddpm-cifar10-32/resolve/{REVISION}/"
ASSETS = [
    (MODEL_BASE + "config.json", "cache/model/config.json"),
    (MODEL_BASE + "scheduler_config.json", "cache/model/scheduler_config.json"),
    (MODEL_BASE + "diffusion_pytorch_model.safetensors", "cache/model/diffusion_pytorch_model.safetensors"),
    (MODEL_BASE + "README.md", "cache/model/README.md"),
    ("https://huggingface.co/datasets/uoft-cs/cifar10/resolve/0b2714987fa478483af9968de7c934580d0bb9a2/plain_text/test-00000-of-00001.parquet",
     "cache/data/cifar10-test.parquet"),
    ("https://github.com/toshas/torch-fidelity/releases/download/v0.2.0/weights-inception-2015-12-05-6726825d.pth",
     "cache/inception/weights-inception-2015-12-05-6726825d.pth"),
]


def file_hash(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(item):
    url, relative = item
    target = ROOT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        temporary = Path(str(target) + ".part")
        subprocess.run(["curl", "--fail", "--location", "--show-error", "--silent", "--retry", "2",
                        "--max-time", "900", url, "-o", str(temporary)], check=True)
        temporary.replace(target)
    if target.name == "cifar-10-python.tar.gz":
        assert file_hash(target, "md5") == "c58f30108f718f92721af3b95e74349a", "CIFAR-10 повреждён"
    sha = file_hash(target)
    if target.name == "cifar10-test.parquet":
        assert sha == "841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de", "CIFAR test не совпал"
    if target.suffix == ".json":
        json.loads(target.read_text())
    if target.name.startswith("weights-inception"):
        assert sha.startswith("6726825d"), "Не совпал hash Inception"
    return {"path": relative, "url": url, "bytes": target.stat().st_size, "sha256": sha}


def main():
    with ThreadPoolExecutor(max_workers=3) as pool:
        assets = list(pool.map(download, ASSETS))
    target = ROOT / "results" / "assets.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"revision": REVISION, "assets": assets}, indent=2), encoding="utf-8")
    for a in assets:
        print(a["path"], a["bytes"], a["sha256"], flush=True)


if __name__ == "__main__":
    main()
