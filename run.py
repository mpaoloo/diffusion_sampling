"""РК1. DDIM на CIFAR-10: генерация и оценка FID, NFE и времени

Генерация с предобученной моделью на MPS или CPU
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
import platform
import random
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache" / "matplotlib"))
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(ROOT / "cache" / "torch"))

import numpy as np
import psutil
import torch
from diffusers import DDIMScheduler, UNet2DModel
from PIL import Image, ImageDraw


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def save_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def hash_array(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def current_memory(device):
    data = {"rss_gib": psutil.Process().memory_info().rss / 1024**3}
    if device.type == "mps":
        data["mps_allocated_gib"] = torch.mps.current_allocated_memory() / 1024**3
        data["mps_driver_gib"] = torch.mps.driver_allocated_memory() / 1024**3
    return data


def select_device(choice):
    if choice == "auto":
        choice = "mps" if torch.backends.mps.is_available() else "cpu"
    if choice == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS недоступен. Для запуска на CPU укажите --device cpu.")
    return torch.device(choice)


def environment(config, device):
    versions = {p: version(p) for p in ("torch", "torchvision", "diffusers", "numpy", "scipy",
                                      "torch-fidelity", "Pillow", "huggingface-hub")}
    return {"python": sys.version, "platform": platform.platform(), "architecture": platform.machine(),
            "processor": platform.processor(), "ram_gib": psutil.virtual_memory().total / 1024**3,
            "logical_cpus": os.cpu_count(), "device": str(device), "dtype": "float32",
            "mps_built": torch.backends.mps.is_built(), "mps_available": torch.backends.mps.is_available(),
            "threads": torch.get_num_threads(), "versions": versions, "config": config}


def load_model(config, device):
    start = time.perf_counter()
    model = UNet2DModel.from_pretrained(str(ROOT / "cache" / "model"), use_safetensors=True,
                                       torch_dtype=torch.float32, local_files_only=True)
    model = model.to(device).eval().requires_grad_(False)
    assert model.config.sample_size == 32
    assert model.config.in_channels == model.config.out_channels == 3
    synchronize(device)
    return model, time.perf_counter() - start


def make_scheduler(config):
    original = json.loads((ROOT / "cache/model/scheduler_config.json").read_text())
    scheduler = DDIMScheduler.from_config(original, prediction_type="epsilon",
                                          timestep_spacing=config["timestep_spacing"],
                                          clip_sample=config["clip_sample"])
    scheduler.set_timesteps(config["steps"])
    return scheduler


def prediction_to_x0(x_t, prediction, alpha_bar, prediction_type):
    """alpha_bar - накопленное произведение коэффициентов alpha"""
    a = torch.as_tensor(alpha_bar, dtype=x_t.dtype, device=x_t.device).sqrt()
    s = (1 - torch.as_tensor(alpha_bar, dtype=x_t.dtype, device=x_t.device)).sqrt()
    if prediction_type == "epsilon":
        return (x_t - s * prediction) / a
    if prediction_type == "sample":
        return prediction
    if prediction_type == "v_prediction":
        return a * x_t - s * prediction
    raise ValueError(prediction_type)


def make_noise(config):
    """Создание и сохранение начальных шумов на CPU независимо от размера батча"""
    target = ROOT / "results" / "initial_noise.npy"
    manifest = ROOT / "results" / "initial_noise.json"
    shape = (config["samples"], 3, 32, 32)
    if target.exists():
        noise = np.load(target)
        if noise.shape != shape or noise.dtype != np.float32:
            raise ValueError("Параметры сохранённых начальных шумов не совпадают с конфигурацией.")
        if manifest.exists():
            saved = json.loads(manifest.read_text())
            if saved["seed"] != config["seed"] or saved["sha256"] != hash_array(noise):
                raise ValueError("Изменился seed или содержимое сохранённого шума")
        else:
            expected = torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(config["seed"])).numpy()
            if not np.array_equal(noise, expected):
                raise ValueError("Существующий шум не совпал с заданным seed")
            save_json(manifest, {"seed": config["seed"], "shape": list(shape), "sha256": hash_array(noise)})
        return noise
    noise = torch.randn(shape, generator=torch.Generator(device="cpu").manual_seed(config["seed"]),
                        dtype=torch.float32).numpy()
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target, noise)
    save_json(manifest, {"seed": config["seed"], "shape": list(shape), "sha256": hash_array(noise)})
    return noise


@torch.inference_mode()
def sample_batch(model, noise_cpu, config, device, capture=False):
    """Генерация через DDIMScheduler с подсчётом вызовов U-Net"""
    scheduler = make_scheduler(config)
    sample = torch.as_tensor(noise_cpu, device=device).clone() * scheduler.init_noise_sigma
    calls = [0]
    def count_forward(module, args, output):
        calls[0] += 1
    handle = model.register_forward_hook(count_forward)
    trajectory = []
    if capture:
        trajectory.append(("noise", to_uint8(sample)))
    synchronize(device)
    start = time.perf_counter()
    try:
        for index, timestep in enumerate(scheduler.timesteps):
            # В данном DDIM scale_model_input — тождественное преобразование
            model_input = scheduler.scale_model_input(sample, timestep)
            predicted_epsilon = model(model_input, timestep).sample
            sample = scheduler.step(predicted_epsilon, timestep, sample, eta=config["eta"]).prev_sample
            if capture and index in {0, len(scheduler.timesteps)//2, len(scheduler.timesteps)-1}:
                trajectory.append((f"step {index+1}", to_uint8(sample)))
        synchronize(device)
        elapsed = time.perf_counter() - start
    finally:
        handle.remove()
    assert calls[0] == config["steps"], (calls[0], config["steps"])
    if not torch.isfinite(sample).all():
        raise RuntimeError("NaN/Inf в генерации")
    return to_uint8(sample), {"seconds": elapsed, "network_calls": calls[0],
                              "batch_size": len(noise_cpu), **current_memory(device)}, trajectory


def to_uint8(tensor):
    return ((tensor.detach().float().cpu().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).numpy()


def grid(images, path, columns=8, scale=3, ids=None):
    """Сетка изображений NCHW с масштабированием nearest-neighbour"""
    images = np.asarray(images)
    rows = (len(images) + columns - 1) // columns
    cell = 32 * scale
    label = 15 if ids is not None else 0
    canvas = Image.new("RGB", (columns * cell, rows * (cell + label)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, arr in enumerate(images):
        picture = Image.fromarray(arr.transpose(1, 2, 0)).resize((cell, cell), Image.Resampling.NEAREST)
        x, y = (i % columns) * cell, (i // columns) * (cell + label)
        canvas.paste(picture, (x, y))
        if ids is not None:
            draw.text((x + 2, y + cell), str(ids[i]), fill="black")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def get_cifar_reference(config):
    """Стратифицированная выборка из test-части CIFAR-10: n/10 изображений каждого класса

    Pickle читается только из официального архива после проверки опубликованного MD5
    """
    from download_assets import file_hash
    parquet = ROOT / "cache/data/cifar10-test.parquet"
    if parquet.exists():
        import pyarrow.parquet as pq
        from io import BytesIO
        assert file_hash(parquet) == "841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de"
        table = pq.read_table(parquet)
        labels = np.asarray(table.column("label").to_pylist())
        images = np.stack([np.asarray(Image.open(BytesIO(row["img"]["bytes"])).convert("RGB")).transpose(2, 0, 1)
                           for row in table.to_pylist()])
        source = "uoft-cs/cifar10 parquet, revision 0b2714987fa478483af9968de7c934580d0bb9a2"
    else:
        archive = ROOT / "cache/data/cifar-10-python.tar.gz"
        assert file_hash(archive, "md5") == "c58f30108f718f92721af3b95e74349a"
        with tarfile.open(archive) as tar:
            with tar.extractfile("cifar-10-batches-py/test_batch") as stream:
                batch = pickle.load(stream, encoding="bytes")
        images = batch[b"data"].reshape(-1, 3, 32, 32)
        labels = np.asarray(batch[b"labels"])
        source = "original Toronto python archive"
    assert images.shape == (10000, 3, 32, 32)
    assert images.dtype == np.uint8
    assert all((labels == c).sum() == 1000 for c in range(10))
    n = config["reference_samples"]
    if n % 10:
        raise ValueError("reference_samples должен делиться на 10")
    rng = np.random.default_rng(config["seed"] + 1)
    indices = np.concatenate([rng.permutation(np.where(labels == c)[0])[:n//10] for c in range(10)])
    indices = rng.permutation(indices)
    save_json(ROOT / "results/reference_indices.json", {"split": "test", "source": source, "indices": indices.tolist(),
              "class_counts": {str(c): int((labels[indices] == c).sum()) for c in range(10)}})
    return images[indices].copy()


def sanity(config, device):
    model, load_seconds = load_model(config, device)
    noise = make_noise(config)
    scheduler = make_scheduler(config)
    # Проверяем преобразование epsilon -> x0 на выходе сети
    x = torch.from_numpy(noise[:2]).to(device)
    t = scheduler.timesteps[0]
    with torch.inference_mode():
        eps = model(x, t).sample
        manual = prediction_to_x0(x, eps, scheduler.alphas_cumprod[int(t)], "epsilon")
        reference = scheduler.step(eps, t, x, eta=0.0).pred_original_sample
        error = float((manual.clamp(-1, 1) - reference).abs().max().cpu())
    assert error < 1e-4, error
    # Прогрев коротким прогоном не включается в измерения
    short = {**config, "steps": 4}
    sample_batch(model, noise[:1], short, device)
    benchmarks = []
    for batch in (1, 4, 8, 16):
        _, stats, _ = sample_batch(model, noise[:batch], {**config, "steps": 10}, device)
        stats["steps"] = 10
        stats["seconds_per_image"] = stats["seconds"] / batch
        benchmarks.append(stats)
        print("TIMING", stats, flush=True)
    images, stats, trajectory = sample_batch(model, noise[:16], config, device, capture=True)
    images2, _, _ = sample_batch(model, noise[:2], config, device)
    # Проверка повторяемости при одинаковом размере батча
    images3, _, _ = sample_batch(model, noise[:2], config, device)
    reproducible = bool(np.array_equal(images2, images3))
    grid(images, ROOT / "results/figures/sanity_grid.png", columns=4, scale=4, ids=list(range(16)))
    grid(np.concatenate([im[:4] for _, im in trajectory]), ROOT / "results/figures/trajectory.png",
         columns=4, scale=3, ids=[label for label, im in trajectory for _ in range(min(4, len(im)))])
    data = {"device": str(device), "load_seconds": load_seconds, "epsilon_to_x0_max_abs_error": error,
            "same_noise_same_batch_bitwise_equal_uint8": reproducible,
            "benchmarks": benchmarks, "full_50_step_batch": stats,
            "noise_sha256": hash_array(noise), "model_parameters": sum(p.numel() for p in model.parameters()),
            "all_weights_frozen": all(not p.requires_grad for p in model.parameters())}
    save_json(ROOT / "results/sanity.json", data)
    print(json.dumps(data, indent=2), flush=True)


def baseline(config, device):
    existing_progress = ROOT / "results/progress.json"
    existing_metrics = ROOT / "results/baseline_metrics.json"
    if existing_progress.exists() and existing_metrics.exists():
        saved = json.loads(existing_progress.read_text())
        if saved["config"] != config:
            raise ValueError("Изменились параметры завершённого baseline")
        if saved["completed"] == config["samples"]:
            print("Baseline завершён. Используются сохранённые результаты.")
            return
    model, load_seconds = load_model(config, device)
    noise = make_noise(config)
    save_json(ROOT / "results/model_contract.json", {
        "model_config": dict(model.config), "scheduler_config": dict(make_scheduler(config).config),
        "timesteps": make_scheduler(config).timesteps.tolist(), "input_shape": ["B", 3, 32, 32],
        "output": "epsilon prediction; Bx3x32x32; unbounded float32", "conditioning": "none"})
    output_path = ROOT / "results/samples.npy"
    progress_path = ROOT / "results/progress.json"
    existing = json.loads(progress_path.read_text()) if progress_path.exists() else None
    if existing:
        if existing["config"] != config or existing["noise_sha256"] != hash_array(noise):
            raise ValueError("Изменился протокол уже начатого baseline")
        output = np.lib.format.open_memmap(output_path, mode="r+")
        completed, batches = existing["completed"], existing["batches"]
    else:
        output = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.uint8,
                                         shape=(len(noise), 3, 32, 32))
        completed, batches = 0, []
    if completed < len(noise):
        # Полный прогрев выбранного размера батча перед замером baseline
        sample_batch(model, noise[:config["batch_size"]], config, device)
    for start in range(completed, len(noise), config["batch_size"]):
        stop = min(len(noise), start + config["batch_size"])
        pictures, stats, _ = sample_batch(model, noise[start:stop], config, device)
        output[start:stop] = pictures
        output.flush()
        stats.update({"start": start, "stop": stop})
        batches.append(stats)
        progress = {"config": config, "noise_sha256": hash_array(noise), "completed": stop, "batches": batches}
        save_json(progress_path, progress)
        print(f"BASELINE {stop}/{len(noise)} | {stats['seconds']:.2f}s | NFE {stats['network_calls']}", flush=True)
        if stop == len(noise) or start == 0:
            grid(np.asarray(output[:min(stop, 64)]), ROOT / "results/figures/generated_grid.png",
                 columns=8, scale=3, ids=list(range(min(stop, 64))))
    if not batches:
        raise ValueError("Нет результатов baseline")
    save_csv(ROOT / "results/batch_metrics.csv", batches)
    seconds = sum(b["seconds"] for b in batches)
    summary = {"sampler": "DDIM", "steps": config["steps"], "eta": config["eta"], "device": str(device),
               "samples": len(noise), "batch_size": config["batch_size"], "precision": "float32",
               "nfe_per_image": config["steps"], "actual_batched_network_calls": sum(b["network_calls"] for b in batches),
               "image_weighted_network_evaluations": sum(b["network_calls"] * b["batch_size"] for b in batches),
               "generation_seconds": seconds, "seconds_per_image": seconds / len(noise),
               "images_per_second": len(noise) / seconds, "model_load_seconds": load_seconds,
               "max_observed_rss_gib": max(b["rss_gib"] for b in batches),
               "samples_sha256": hash_array(np.asarray(output)), "noise_sha256": hash_array(noise),
               "time_excludes": "download, model load, warmup, saving images, FID; includes denoising loop and sync"}
    if device.type == "mps":
        summary["max_observed_mps_allocated_gib"] = max(b["mps_allocated_gib"] for b in batches)
        summary["max_observed_mps_driver_gib"] = max(b["mps_driver_gib"] for b in batches)
    # Повторы времени — один и тот же фиксированный батч, не независимые FID-выборки
    repeated = []
    for i in range(config["timing_repeats"]):
        _, stats, _ = sample_batch(model, noise[:config["batch_size"]], config, device)
        stats["repeat"] = i
        repeated.append(stats)
    save_csv(ROOT / "results/timing_repeats.csv", repeated)
    summary["timing_batch_mean_seconds"] = float(np.mean([r["seconds"] for r in repeated]))
    summary["timing_batch_std_seconds"] = float(np.std([r["seconds"] for r in repeated], ddof=1))
    save_json(ROOT / "results/baseline_metrics.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def evaluate(config, device):
    from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
    from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
    import gc
    generated = np.load(ROOT / "results/samples.npy", mmap_mode="r")
    progress = json.loads((ROOT / "results/progress.json").read_text())
    if progress["completed"] != config["samples"]:
        raise ValueError("Генерация не завершена")
    real = get_cifar_reference(config)
    grid(real[:64], ROOT / "results/figures/real_grid.png", columns=8, scale=3)
    weights = ROOT / "cache/inception/weights-inception-2015-12-05-6726825d.pth"
    extractor = FeatureExtractorInceptionV3("inception-v3-compat", ["2048"],
                                          feature_extractor_weights_path=str(weights)).to(device).eval()
    for p in extractor.parameters():
        p.requires_grad_(False)
    features = {}
    extraction_times = {}
    from download_assets import file_hash
    for name, images in (("generated", generated), ("real", real)):
        target = ROOT / "results" / f"features_{name}.npy"
        meta_path = target.with_suffix(".json")
        signature = {"image_sha256": hash_array(images), "shape": list(images.shape),
                     "extractor_sha256": file_hash(weights), "device": str(device), "feature_dimension": 2048}
        if target.exists() and meta_path.exists():
            metadata = json.loads(meta_path.read_text())
            if metadata["signature"] != signature:
                raise ValueError("Сохранённые признаки не соответствуют входным данным или устройству.")
            features[name] = np.load(target)
            assert features[name].shape == (len(images), 2048)
            extraction_times[name] = metadata["seconds"]
            continue
        chunks = []
        synchronize(device)
        start = time.perf_counter()
        for i in range(0, len(images), config["feature_batch_size"]):
            batch = torch.from_numpy(np.array(images[i:i + config["feature_batch_size"]], copy=True)).to(device)
            with torch.inference_mode():
                chunk = extractor(batch)[0].cpu().numpy()
            chunks.append(chunk)
            print(f"FEATURES {name} {min(i+len(batch),len(images))}/{len(images)}", flush=True)
        synchronize(device)
        extraction_times[name] = time.perf_counter() - start
        features[name] = np.concatenate(chunks)
        np.save(target, features[name])
        save_json(meta_path, {"signature": signature, "seconds": extraction_times[name]})
    del extractor
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    if not all(np.isfinite(a).all() for a in features.values()):
        raise ValueError("Неконечные Inception признаки")
    # FID — numpy float64 на CPU. MPS не поддерживает float64
    stats_gen = fid_features_to_statistics(torch.from_numpy(features["generated"].astype(np.float64)))
    stats_real = fid_features_to_statistics(torch.from_numpy(features["real"].astype(np.float64)))
    start = time.perf_counter()
    metric = fid_statistics_to_metric(stats_gen, stats_real, verbose=True)
    fid = float(metric["frechet_inception_distance"])
    if not np.isfinite(fid) or fid < -1e-6:
        raise ValueError(f"Некорректный FID {fid}")
    summary = {"FID": fid, "n_generated": len(generated), "n_reference": len(real),
               "real_split": "CIFAR-10 test; stratified", "reference_class_count": len(real)//10,
               "feature_extractor": "torch-fidelity 0.3.0 inception-v3-compat", "feature_dimension": 2048,
               "feature_device": str(device), "fid_covariance_dtype": "float64 CPU",
               "feature_extraction_seconds": extraction_times, "fid_matrix_seconds": time.perf_counter() - start,
               "fid_runs": 1, "limitations": "Small-sample FID; not directly comparable to published 50k-sample FID"}
    save_json(ROOT / "results/fid_metrics.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("sanity", "baseline", "evaluate"))
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    (ROOT / "results" / "logs").mkdir(parents=True, exist_ok=True)
    class Tee:
        def __init__(self, stream, file):
            self.stream, self.file = stream, file
        def write(self, value):
            self.stream.write(value)
            self.file.write(value)
            self.file.flush()
        def flush(self):
            self.stream.flush()
            self.file.flush()
    logfile = (ROOT / "results" / "logs" / f"{args.stage}.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout, sys.stderr = Tee(sys.stdout, logfile), Tee(sys.stderr, logfile)
    config = json.loads((ROOT / "config.json").read_text())
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    device = select_device(args.device)
    (ROOT / "results" / "logs").mkdir(parents=True, exist_ok=True)
    save_json(ROOT / "results" / "logs" / f"environment_{args.stage}.json", environment(config, device))
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (ROOT / "requirements-lock.txt").write_text(freeze, encoding="utf-8")
    {"sanity": sanity, "baseline": baseline, "evaluate": evaluate}[args.stage](config, device)


if __name__ == "__main__":
    main()
