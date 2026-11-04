"""Построение рисунков для отчёта по сохранённым результатам"""
from pathlib import Path
import csv
import json
import os
import numpy as np

ROOT = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def draw_grid(images, labels):
    canvas = Image.new("RGB", (384, 436), "white")
    draw = ImageDraw.Draw(canvas)
    for i, image in enumerate(images[:16]):
        x, y = i % 4 * 96, i // 4 * 109
        canvas.paste(Image.fromarray(image.transpose(1, 2, 0)).resize((96, 96), Image.Resampling.NEAREST), (x, y))
        draw.text((x+3, y+96), str(labels[i]), fill="black")
    return canvas


def main():
    from run import get_cifar_reference
    config = json.loads((ROOT / "config.json").read_text())
    output = ROOT / "results/figures"
    output.mkdir(parents=True, exist_ok=True)
    with (ROOT / "results/batch_metrics.csv").open() as f:
        batches = list(csv.DictReader(f))
    x = [int(r["stop"]) for r in batches]
    y = [float(r["seconds"]) / int(r["batch_size"]) for r in batches]
    fig, ax = plt.subplots(figsize=(9.2, 2.8))
    ax.plot(x, y, color="#222222", linewidth=1.1)
    ax.set(xlabel="Накопленное число изображений", ylabel="Денойзинг, с/изображение",
           title="DDIM, 50 NFE, Apple M1 MPS: измерения по батчам")
    ax.grid(axis="y", color="#dddddd", linewidth=.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output / "latency.png", dpi=170)
    plt.close(fig)
    generated = np.load(ROOT / "results/samples.npy", mmap_mode="r")
    real = get_cifar_reference(config)
    indices = json.loads((ROOT / "results/reference_indices.json").read_text())["indices"]
    both = Image.new("RGB", (790, 465), "white")
    both.paste(draw_grid(generated[:16], list(range(16))), (0, 27))
    both.paste(draw_grid(real[:16], indices[:16]), (406, 27))
    font_dir = next(ROOT.glob(".venv/lib/python*/site-packages/matplotlib/mpl-data/fonts/ttf"))
    font = ImageFont.truetype(str(font_dir / "DejaVuSans.ttf"), 15)
    draw = ImageDraw.Draw(both)
    draw.text((0, 2), "Генерации: первые 16", fill="black", font=font)
    draw.text((406, 2), "Референсы CIFAR-10 test", fill="black", font=font)
    both.save(output / "visual_comparison.png")
    print("Рисунки сохранены:", output)


if __name__ == "__main__":
    main()
