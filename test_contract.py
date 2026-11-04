"""Проверки параметризаций, DDIMScheduler, NFE и начального шума"""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

import run


@pytest.mark.parametrize("alpha_bar", [0.001, 0.25, 0.9])
def test_three_parameterizations_recover_same_clean_image(alpha_bar):
    g = torch.Generator().manual_seed(17)
    clean, epsilon = torch.randn((2, 3, 4, 4), generator=g), torch.randn((2, 3, 4, 4), generator=g)
    a, s = alpha_bar**0.5, (1-alpha_bar)**0.5
    noisy = a * clean + s * epsilon
    velocity = a * epsilon - s * clean
    for name, output in (("epsilon", epsilon), ("v_prediction", velocity), ("sample", clean)):
        actual = run.prediction_to_x0(noisy, output, alpha_bar, name)
        assert torch.allclose(actual, clean, atol=1e-5)


def test_conversion_agrees_with_diffusers_scheduler():
    config = json.loads((run.ROOT / "config.json").read_text())
    scheduler = run.make_scheduler(config)
    x, eps = torch.randn(2, 3, 32, 32), torch.randn(2, 3, 32, 32)
    for timestep in scheduler.timesteps[[0, 20, -1]]:
        raw = run.prediction_to_x0(x, eps, scheduler.alphas_cumprod[int(timestep)], "epsilon")
        reference = scheduler.step(eps, timestep, x, eta=0.0).pred_original_sample
        assert torch.allclose(raw.clamp(-1, 1), reference, atol=1e-5)


def test_nfe_counts_actual_forward_calls_and_repeatability():
    class ZeroPredictor(torch.nn.Module):
        def forward(self, image, timestep):
            return SimpleNamespace(sample=torch.zeros_like(image))
    config = {**json.loads((run.ROOT / "config.json").read_text()), "steps": 5}
    noise = np.random.default_rng(7).normal(size=(3, 3, 32, 32)).astype(np.float32)
    first, info, _ = run.sample_batch(ZeroPredictor(), noise, config, torch.device("cpu"))
    second, _, _ = run.sample_batch(ZeroPredictor(), noise, config, torch.device("cpu"))
    assert info["network_calls"] == 5
    assert info["batch_size"] == 3
    assert first.shape == noise.shape and first.dtype == np.uint8
    assert np.array_equal(first, second)


def test_saved_noise_cannot_silently_change_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    config = {"samples": 8, "seed": 77}
    first = run.make_noise(config)
    assert np.array_equal(first, run.make_noise(config))
    with pytest.raises(ValueError):
        run.make_noise({**config, "seed": 78})


def test_completed_baseline_preserves_original_measurements(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "ROOT", tmp_path)
    config = {"samples": 10}
    run.save_json(tmp_path / "results/progress.json", {"config": config, "completed": 10})
    metrics = tmp_path / "results/baseline_metrics.json"
    original = '{"generation_seconds": 123.456}'
    metrics.write_text(original)
    # Завершённый baseline использует сохранённые измерения без загрузки модели
    run.baseline(config, torch.device("cpu"))
    assert metrics.read_text() == original
