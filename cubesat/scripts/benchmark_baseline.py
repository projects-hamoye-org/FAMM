
"""
Phase 1 — Baseline Model Characterisation
Measures the original MobileNet V3 before any optimisation.
Saves results to data/results/baseline_metrics.csv
No TIF files needed — uses synthetic dummy input.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import psutil
import time
import os
from pathlib import Path
from torchvision import models

# ── Constants 
CHECKPOINT_PATH = "models/mobilenetv3_best.pth"
RESULTS_DIR     = "data/results"
NUM_BANDS       = 10
PATCH_SIZE      = 224
N_WARMUP        = 10    # warmup runs (not measured)
N_MEASURE       = 100   # measured runs
device          = torch.device("cpu")

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)


# ── Load model 
def load_model():
    model = models.mobilenet_v3_large(weights=None)
    model.features[0][0] = nn.Conv2d(
        NUM_BANDS, 16, kernel_size=3, stride=2, padding=1, bias=False
    )
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


# ── Metrics 
def measure_model_size():
    size_bytes = os.path.getsize(CHECKPOINT_PATH)
    return size_bytes / 1_048_576   # MB


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def measure_latency(model, n_warmup=N_WARMUP, n_measure=N_MEASURE):
    dummy = torch.randn(1, NUM_BANDS, PATCH_SIZE, PATCH_SIZE)
    # Warmup — not measured
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)
    # Measured runs
    latencies = []
    with torch.no_grad():
        for _ in range(n_measure):
            t0 = time.perf_counter()
            _ = model(dummy)
            latencies.append((time.perf_counter() - t0) * 1000)  # ms
    return {
        "mean_ms":   round(np.mean(latencies), 2),
        "median_ms": round(np.median(latencies), 2),
        "p95_ms":    round(np.percentile(latencies, 95), 2),
        "min_ms":    round(np.min(latencies), 2),
        "max_ms":    round(np.max(latencies), 2),
    }


def measure_ram(model):
    process = psutil.Process()
    dummy   = torch.randn(1, NUM_BANDS, PATCH_SIZE, PATCH_SIZE)
    before  = process.memory_info().rss / 1_048_576
    with torch.no_grad():
        _ = model(dummy)
    after = process.memory_info().rss / 1_048_576
    return {
        "before_mb": round(before, 1),
        "after_mb":  round(after, 1),
        "delta_mb":  round(after - before, 1),
    }


def estimate_throughput(mean_latency_ms):
    patches_per_second = 1000 / mean_latency_ms
    # Ghana ROI: 60 tiles, each roughly 1000x1000px → ~25 patches per tile at stride 224
    patches_total = 60 * 25
    total_seconds = patches_total / patches_per_second
    return {
        "patches_per_second": round(patches_per_second, 1),
        "ghana_patches_total": patches_total,
        "ghana_runtime_min":  round(total_seconds / 60, 1),
    }


# ── Main 
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 1 — Baseline Model Characterisation")
    print("=" * 60)

    print("\n Loading model...")
    model, checkpoint = load_model()
    print(f"   Epoch          : {checkpoint.get('epoch')}")
    print(f"   Best test acc  : {checkpoint.get('best_test_acc', 0):.4f}")

    print("\n Model size...")
    size_mb = measure_model_size()
    total_params, trainable_params = count_parameters(model)
    print(f"   Checkpoint size : {size_mb:.1f} MB")
    print(f"   Total params    : {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")

    print(f"\n  Latency ({N_MEASURE} runs, CPU only)...")
    latency = measure_latency(model)
    for k, v in latency.items():
        print(f"   {k:<12}: {v} ms")

    print("\n RAM usage...")
    ram = measure_ram(model)
    for k, v in ram.items():
        print(f"   {k:<12}: {v} MB")

    print("\n Throughput estimate...")
    throughput = estimate_throughput(latency["mean_ms"])
    for k, v in throughput.items():
        print(f"   {k}: {v}")

    # ── CubeSat constraint comparison 
    print("\n CubeSat constraint check (original model):")
    cubesat_ram_mb   = 256
    cubesat_power_w  = 2.0
    cubesat_latency_ms = 500

    ram_ok     = ram["delta_mb"] < cubesat_ram_mb
    latency_ok = latency["mean_ms"] < cubesat_latency_ms

    print(f"   RAM delta {ram['delta_mb']} MB < {cubesat_ram_mb} MB budget : {'' if ram_ok else ''}")
    print(f"   Latency {latency['mean_ms']} ms < {cubesat_latency_ms} ms budget : {'' if latency_ok else ''}")
    print(f"   Checkpoint {size_mb:.1f} MB — needs to fit in 50 MB flash : {'' if size_mb < 50 else ''}")

    # ── Save results 
    results = {
        "model":            "MobileNetV3-Large (original)",
        "epoch":            checkpoint.get("epoch"),
        "best_test_acc":    round(checkpoint.get("best_test_acc", 0), 4),
        "size_mb":          round(size_mb, 2),
        "total_params":     total_params,
        "mean_latency_ms":  latency["mean_ms"],
        "p95_latency_ms":   latency["p95_ms"],
        "ram_delta_mb":     ram["delta_mb"],
        "patches_per_sec":  throughput["patches_per_second"],
        "ghana_runtime_min": throughput["ghana_runtime_min"],
    }

    df = pd.DataFrame([results])
    out = f"{RESULTS_DIR}/baseline_metrics.csv"
    df.to_csv(out, index=False)
    print(f"\n✅ Results saved → {out}")
    print("=" * 60)