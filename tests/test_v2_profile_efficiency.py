import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image


def _script():
    spec = importlib.util.spec_from_file_location(
        "profile_v2", Path("scripts/v2/profile_efficiency_v2.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeEncoder:
    """Mimics V2Encoder: CPU prepare, to_device, normalized forward."""

    def __init__(self):
        self.device, self.dtype = torch.device("cpu"), torch.float32
        self.processor = None
        self.attention_method = "not applicable"
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1),  # 3*3*3*8 MACs per output pixel
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(8, 12),
        )

    def transform(self, image):
        array = np.asarray(image.convert("RGB").resize((16, 16)), dtype=np.float32) / 255
        return torch.from_numpy(array).permute(2, 0, 1)

    def prepare(self, images):
        tensors = []
        for path in images:
            with Image.open(path) as image:
                tensors.append(self.transform(image))
        return {"pixel_values": torch.stack(tensors)}

    def to_device(self, inputs):
        return {k: v.to(self.device, self.dtype) for k, v in inputs.items()}

    def forward(self, inputs):
        return F.normalize(self.model(inputs["pixel_values"]).float(), dim=-1)

    def image_parameter_count(self):
        return sum(p.numel() for p in self.model.parameters())


def test_cpu_profile_degrades_gracefully(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    paths = []
    for i in range(5):
        path = tmp_path / f"{i}.jpg"
        Image.new("RGB", (40, 30), (i * 40, 10, 10)).save(path)
        paths.append(path)
    script = _script()
    result = script.profile_encoder(FakeEncoder(), paths, model_key="fake")
    assert result["image_tower_parameters"] == 3 * 3 * 3 * 8 + 8 + 8 * 12 + 12
    conv_flops = 2 * 16 * 16 * 8 * 27  # 1 MAC = 2 FLOPs
    linear_flops = 2 * 8 * 12
    assert result["flops"]["flops_per_image"] == conv_flops + linear_flops
    assert result["flops"]["input_shapes"]["pixel_values"] == [1, 3, 16, 16]
    assert result["latency"] == {"1": None, "64": None}
    assert result["throughput_images_per_second"] is None
    assert result["allocator_peak_bytes"] is None and result["energy"] is None
    assert result["gpu_skipped_reason"].startswith("skipped")
    assert result["preprocessing"]["per_image_seconds"]["n"] == 5
    assert result["storage"]["native_dim"] == 12
    assert result["storage"]["native_float32_mb_per_100k"] == pytest.approx(4.8)
    assert result["storage"]["pca256_float32_mb_per_100k"] == pytest.approx(102.4)
    json.dumps(result, allow_nan=False)


def test_power_integration_and_energy_summary():
    script = _script()
    samples = [(0.0, 100.0), (0.5, 100.0), (1.0, 200.0)]
    assert script.integrate_power(samples) == pytest.approx(50 + 75)
    assert script.integrate_power(samples[:1]) is None
    summary = script.energy_summary(
        {"joules": 3600.0, "seconds": 10.0, "images": 1000, "idle_watts": 60.0},
        "test",
        grid_intensity=0.5,
    )
    assert summary["wh_per_1k_images"] == pytest.approx(1.0)
    assert summary["net_of_idle_wh_per_1k_images"] == pytest.approx((3600 - 600) / 3600)
    assert summary["kgco2e_per_1k_images"] == pytest.approx(0.0005)
    q = script.quantiles([1, 2, 3, 4, 5])
    assert q["median"] == 3 and q["iqr"] == 2


def test_main_refuses_overwrite(tmp_path):
    output = tmp_path / "profile.json"
    output.write_text("{}")
    with pytest.raises(FileExistsError):
        _script().main(
            ["--model", "resnet50", "--images-dir", str(tmp_path), "--output", str(output)]
        )
