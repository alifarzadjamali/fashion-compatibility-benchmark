"""v2 efficiency profiler, one encoder per process (HANDOFF_V2 §6.6).

The v1 profiler (scripts/profile_efficiency.py) is kept unchanged for history. This
one separates the measurements v1 conflated:

- image-tower parameters (M);
- GFLOPs/image: torch.utils.flop_counter.FlopCounterMode over one forward() at the
  encoder's native preprocessing resolution, batch 1 (1 MAC = 2 FLOPs, PyTorch's
  convention); covers whatever forward() runs for extraction (for Qwen embedding
  models, the whole MLLM forward);
- GPU-only latency: preprocessed tensors already on the device, CUDA events,
  10 warm-up + 50 timed forward passes, median + IQR, batch 1 and 64 (uniform bf16 +
  SDPA via V2Encoder); throughput from batch 64;
- allocator peak memory per batch size (torch.cuda.max_memory_allocated; this is the
  allocator peak, not device memory; nvidia-smi memory may be N/A on unified memory);
- CPU preprocessing (decode + resize + normalize from disk) measured separately;
- energy: NVML total-energy counter if supported, else ~10 Hz power sampling
  (NVML power, then nvidia-smi power.draw) integrated by the trapezoid rule; the
  method used is recorded. Idle power is sampled first and net values are reported;
- storage MB (10^6 bytes) per 100k items, native float32/float16 vs PCA-256 float32.

Without CUDA all GPU measurements are null with the reason recorded (unit tests).
Output: one JSON per encoder, exclusive create (never overwritten).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import random
import subprocess
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

WARMUP, TIMED = 10, 50
BATCH_SIZES = (1, 64)
PCA_DIM = 256
ITEMS = 100_000


def quantiles(values):
    values = np.asarray(values, dtype=float)
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    return {
        "median": float(median),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "n": len(values),
    }


def storage_mb(dim):
    return {
        "native_dim": int(dim),
        "native_float32_mb_per_100k": dim * 4 * ITEMS / 1e6,
        "native_float16_mb_per_100k": dim * 2 * ITEMS / 1e6,
        "pca256_float32_mb_per_100k": PCA_DIM * 4 * ITEMS / 1e6,
        "unit": "MB = 10^6 bytes; vectors only, no ids/index",
    }


# ------------------------------------------------------------------------- energy


class NVML:
    """Minimal ctypes NVML binding (pynvml is not a project dependency)."""

    def __init__(self, index=0):
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        if self.lib.nvmlInit_v2() != 0:
            raise RuntimeError("nvmlInit failed")
        self.handle = ctypes.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(index, ctypes.byref(self.handle)) != 0:
            raise RuntimeError("NVML device handle unavailable")

    def total_energy_joules(self):
        value = ctypes.c_ulonglong()
        code = self.lib.nvmlDeviceGetTotalEnergyConsumption(self.handle, ctypes.byref(value))
        return None if code != 0 else value.value / 1000.0  # mJ -> J

    def power_watts(self):
        value = ctypes.c_uint()
        code = self.lib.nvmlDeviceGetPowerUsage(self.handle, ctypes.byref(value))
        return None if code != 0 else value.value / 1000.0  # mW -> W


def integrate_power(samples):
    """Trapezoid integral of [(t_seconds, watts), ...] in joules."""
    if len(samples) < 2:
        return None
    t = np.asarray([s[0] for s in samples], dtype=float)
    w = np.asarray([s[1] for s in samples], dtype=float)
    return float(np.sum((w[1:] + w[:-1]) / 2 * np.diff(t)))


class PowerSampler:
    """Samples power at ~10 Hz in a thread from a callable returning watts or None."""

    def __init__(self, read_watts, interval=0.1):
        self.read_watts, self.interval = read_watts, interval
        self.samples, self._stop = [], threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            watts = self.read_watts()
            if watts is not None:
                self.samples.append((time.perf_counter(), watts))
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


class NvidiaSmiPower:
    """Streams `nvidia-smi --query-gpu=power.draw -lms 100`; latest value on demand."""

    def __init__(self, index=0):
        self.latest = None
        self.process = subprocess.Popen(
            [
                "nvidia-smi",
                "-i",
                str(index),
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
                "-lms",
                "100",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.process.stdout:
            try:
                self.latest = float(line.strip())
            except ValueError:
                self.latest = None

    def __call__(self):
        return self.latest

    def close(self):
        self.process.terminate()


def energy_method():
    """Return (name, measure) where measure(run, idle_seconds) -> dict."""
    nvml = None
    try:
        nvml = NVML()
    except Exception:  # noqa: BLE001 - library absent or not initializable
        nvml = None
    if nvml is not None and nvml.total_energy_joules() is not None:

        def counter(run, idle_seconds):
            idle_start = nvml.total_energy_joules()
            time.sleep(idle_seconds)
            idle_watts = (nvml.total_energy_joules() - idle_start) / idle_seconds
            start, began = nvml.total_energy_joules(), time.perf_counter()
            images = run()
            seconds = time.perf_counter() - began
            return {
                "joules": nvml.total_energy_joules() - start,
                "seconds": seconds,
                "images": images,
                "idle_watts": idle_watts,
            }

        return "NVML nvmlDeviceGetTotalEnergyConsumption counter", counter
    sources = []
    if nvml is not None and nvml.power_watts() is not None:
        sources.append(
            ("NVML nvmlDeviceGetPowerUsage sampled at ~10 Hz, trapezoid", nvml.power_watts)
        )
    else:
        try:
            smi = NvidiaSmiPower()
            time.sleep(1.0)
            if smi() is not None:
                sources.append(("nvidia-smi power.draw sampled at ~10 Hz, trapezoid", smi))
            else:
                smi.close()
        except FileNotFoundError:
            pass
    if not sources:
        return "unavailable (no NVML energy/power and no nvidia-smi power.draw)", None
    name, read = sources[0]

    def sampled(run, idle_seconds):
        with PowerSampler(read) as idle:
            time.sleep(idle_seconds)
        idle_watts = float(np.mean([w for _, w in idle.samples])) if idle.samples else None
        with PowerSampler(read) as sampler:
            began = time.perf_counter()
            images = run()
            seconds = time.perf_counter() - began
        return {
            "joules": integrate_power(sampler.samples),
            "seconds": seconds,
            "images": images,
            "idle_watts": idle_watts,
            "power_samples": len(sampler.samples),
        }

    return name, sampled


def energy_summary(raw, method, grid_intensity):
    joules, images = raw["joules"], raw["images"]
    if joules is None or not images:
        return {"method": method, **raw, "wh_per_1k_images": None}
    wh_per_1k = joules / 3600 / images * 1000
    net = None
    if raw.get("idle_watts") is not None:
        net = max(0.0, joules - raw["idle_watts"] * raw["seconds"]) / 3600 / images * 1000
    result = {
        "method": method,
        **raw,
        "scope": "GPU/SoC as reported by the driver; excludes host CPU, memory and PSU loss",
        "wh_per_1k_images": wh_per_1k,
        "net_of_idle_wh_per_1k_images": net,
        "grid_intensity_kg_per_kwh": grid_intensity,
        "kgco2e_per_1k_images": None
        if grid_intensity is None
        else wh_per_1k / 1000 * grid_intensity,
    }
    return result


# ------------------------------------------------------------------------- profile


def _batch(paths, size):
    return [paths[i % len(paths)] for i in range(size)]


def _shapes(inputs):
    return {k: list(v.shape) for k, v in inputs.items() if hasattr(v, "shape")}


def preprocessing_cost(encoder, paths, repeats=1):
    per_image = []
    for _ in range(repeats):
        for path in paths:
            started = time.perf_counter()
            encoder.prepare([path])
            per_image.append(time.perf_counter() - started)
    started = time.perf_counter()
    encoder.prepare(list(paths))
    batched = (time.perf_counter() - started) / len(paths)
    processor = encoder.processor if encoder.processor is not None else encoder.transform
    import torch

    return {
        "per_image_seconds": quantiles(per_image),
        "batched_seconds_per_image": batched,
        "images": len(paths),
        "includes": "file decode + RGB convert + resize + normalize on CPU; no device transfer",
        "processor": type(processor).__name__,
        "fast_processor": bool(getattr(processor, "is_fast", False)),
        "torch_cpu_threads": torch.get_num_threads(),
    }


def count_flops(encoder, path):
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    inputs = encoder.to_device(encoder.prepare([path]))
    try:
        with torch.inference_mode(), FlopCounterMode(display=False) as counter:
            encoder.forward(inputs)
        flops = int(counter.get_total_flops())
        return {
            "gflops_per_image": flops / 1e9,
            "flops_per_image": flops,
            "input_shapes": _shapes(inputs),
            "method": "torch.utils.flop_counter.FlopCounterMode, batch 1, native preprocessing "
            "resolution; 1 MAC = 2 FLOPs; counts matmul/conv/attention ops only",
        }
    except Exception as exc:  # noqa: BLE001 - keep other measurements
        return {"gflops_per_image": None, "error": f"{type(exc).__name__}: {exc}"}


def gpu_latency(encoder, paths, batch_size, warmup=WARMUP, timed=TIMED):
    import torch

    inputs = encoder.to_device(encoder.prepare(_batch(paths, batch_size)))
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    times = []
    with torch.inference_mode():
        for _ in range(warmup):
            encoder.forward(inputs)
        torch.cuda.synchronize()
        for _ in range(timed):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            encoder.forward(inputs)
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))  # ms
    stats = quantiles(times)
    return {
        "batch_size": batch_size,
        "warmup": warmup,
        "timed": timed,
        "latency_ms": stats,
        "latency_ms_per_image_median": stats["median"] / batch_size,
        "throughput_images_per_second": batch_size / (stats["median"] / 1000),
        "allocator_peak_bytes": int(torch.cuda.max_memory_allocated()),
        "allocator_peak_label": "torch.cuda.max_memory_allocated (allocator peak; not device "
        "total; includes weights and inputs already resident)",
        "input_shapes": _shapes(inputs),
    }


def profile_encoder(
    encoder,
    paths,
    *,
    model_key,
    energy_seconds=30.0,
    idle_seconds=5.0,
    grid_intensity=None,
    warmup=WARMUP,
    timed=TIMED,
):
    import torch

    cuda = torch.cuda.is_available() and encoder.device.type == "cuda"
    result = {
        "model_key": model_key,
        "utc": datetime.now(UTC).isoformat(),
        "host": platform.node(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name() if cuda else "cpu",
        "dtype": str(encoder.dtype),
        "attention_method": getattr(encoder, "attention_method", None),
        "images_sampled": [str(p) for p in paths],
    }
    result["image_tower_parameters"] = int(encoder.image_parameter_count())
    result["image_tower_parameters_m"] = result["image_tower_parameters"] / 1e6
    result["preprocessing"] = preprocessing_cost(encoder, paths)
    result["flops"] = count_flops(encoder, paths[0])
    with torch.inference_mode():
        probe = encoder.forward(encoder.to_device(encoder.prepare([paths[0]])))
    result["storage"] = storage_mb(probe.shape[-1])
    if not cuda:
        reason = "skipped: CUDA unavailable"
        result["latency"] = {str(b): None for b in BATCH_SIZES}
        result["throughput_images_per_second"] = None
        result["allocator_peak_bytes"] = None
        result["energy"] = None
        result["gpu_skipped_reason"] = reason
        return result
    result["cuda"] = {
        "capability": list(torch.cuda.get_device_capability()),
        "cuda_runtime": torch.version.cuda,
        "sdpa_flash": torch.backends.cuda.flash_sdp_enabled(),
        "sdpa_mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    result["latency"] = {
        str(b): gpu_latency(encoder, paths, b, warmup=warmup, timed=timed) for b in BATCH_SIZES
    }
    big = result["latency"][str(BATCH_SIZES[-1])]
    result["throughput_images_per_second"] = big["throughput_images_per_second"]
    result["allocator_peak_bytes"] = max(
        v["allocator_peak_bytes"] for v in result["latency"].values()
    )
    method, measure = energy_method()
    if measure is None or energy_seconds <= 0:
        result["energy"] = {"method": method, "wh_per_1k_images": None}
        return result
    inputs = encoder.to_device(encoder.prepare(_batch(paths, BATCH_SIZES[-1])))

    def run():
        images = 0
        deadline = time.perf_counter() + energy_seconds
        with torch.inference_mode():
            while time.perf_counter() < deadline:
                encoder.forward(inputs)
                torch.cuda.synchronize()
                images += BATCH_SIZES[-1]
        return images

    result["energy"] = energy_summary(measure(run, idle_seconds), method, grid_intensity)
    result["energy"]["workload"] = "repeated batch-64 forward on device-resident inputs"
    return result


def sample_paths(args):
    if args.image_list:
        paths = [Path(line) for line in args.image_list.read_text().split() if line]
    else:
        paths = sorted(
            p
            for p in args.images_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
    if not paths:
        raise SystemExit("No images")
    return sorted(random.Random(args.seed).sample(paths, min(args.n_images, len(paths))))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images-dir", type=Path)
    source.add_argument("--image-list", type=Path, help="Whitespace-separated image paths")
    parser.add_argument("--n-images", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--energy-seconds", type=float, default=30.0)
    parser.add_argument("--idle-seconds", type=float, default=5.0)
    parser.add_argument(
        "--grid-intensity",
        type=float,
        default=None,
        help="kg CO2e per kWh; state the source in the paper",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--g2", type=Path, help="G2 gate record (required on CUDA)")
    parser.add_argument("--skip-budget-check", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"{args.output} exists; profiles are never overwritten")
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    from repbench.utils.seeds import seed_everything
    from repbench.v2.encoders import V2Encoder
    from repbench.v2.registry import V2_ENCODERS
    from repbench.v2.state import require_budget, require_gate, write_json

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        if args.g2 is None:
            raise SystemExit("--g2 is required for GPU profiling")
        require_gate(args.g2, "G2")
        if not args.skip_budget_check:
            require_budget(1200)
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = False
    paths = sample_paths(args)
    spec = V2_ENCODERS[args.model]
    started = time.perf_counter()
    encoder = V2Encoder(
        spec, device=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32
    )
    if device == "cuda":
        torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    result = profile_encoder(
        encoder,
        paths,
        model_key=args.model,
        energy_seconds=args.energy_seconds,
        idle_seconds=args.idle_seconds,
        grid_intensity=args.grid_intensity,
    )
    result["load_seconds"] = load_seconds
    result["spec"] = {k: v for k, v in vars(spec).items()}
    result["seed"] = args.seed
    write_json(args.output, result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("model_key", "image_tower_parameters_m", "throughput_images_per_second")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
