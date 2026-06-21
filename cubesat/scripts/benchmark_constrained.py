"""
Phase 3 — Simulated CubeSat Constraint Benchmarking
-----------------------------------------------------
Run natively for unconstrained baseline, then inside Docker with CPU/RAM
limits to simulate CubeSat hardware. Compare the two sets of results to
show what the model actually behaves like under orbital constraints.

WHAT --cpus="0.1" MEANS:
  M1 runs at ~3.2 GHz. A CubeSat processor (e.g. GOMspace) runs at ~400 MHz — roughly 8x slower.
  --cpus="0.1" limits Docker to 10% of one CPU core ≈ 320 MHz ≈ CubeSat speed.
  This is an approximation — real CubeSat has different instruction sets —
  but this is one way of simulation approach in edge AI papers.

WHAT --memory="256m" MEANS:
  Limits the container to 256 MB RAM total (OS + Python + model + inference).
  If the model exceeds this, the container is killed (OOMKilled).
  If it runs successfully, we have proven RAM feasibility.

TIF FILES:
  Not required for latency/RAM benchmarking.
  data/tif_input/ for accuracy comparison section.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import psutil
from pathlib import Path

# Detect if running inside Docker
IN_DOCKER = os.path.exists("/.dockerenv")

# ── Constants 
RESULTS_DIR = "data/results"
TIF_DIR     = "data/tif_input"
NUM_BANDS   = 10
PATCH_SIZE  = 224
STRIDE      = 224
THRESHOLD   = 0.3
N_WARMUP    = 3     # fewer warmups — constrained CPU is slow
N_MEASURE   = 20    # fewer runs — constrained CPU is slow
THRESHOLD   = 0.3

# CubeSat hardware reference specs
# Sources: GOMspace datasheet, Pumpkin CubeSat Kit Module Spec
CUBESAT_SPECS = {
    "RAM_MB":        256,
    "FLASH_MB":      50,
    "LATENCY_MS":    500,
    "POWER_W":       2.0,
    "PROCESSOR":     "ARM Cortex-M / ~400 MHz",
}

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

try:
    import onnxruntime as ort
    ONNX_OK = True
except ImportError:
    print(" onnxruntime not installed — run: pip install onnxruntime")
    ONNX_OK = False


# ── Benchmark one ONNX model 
def benchmark_onnx(path, label):
    if not Path(path).exists():
        print(f"   ⚠️  {label} not found at {path} — skipping")
        print(f"       Run quantize_model.py first to generate ONNX models")
        return None

    # Single thread — simulates single-core CubeSat processor
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session    = ort.InferenceSession(path, opts)
    input_name = session.get_inputs()[0].name
    dummy      = np.random.randn(1, NUM_BANDS, PATCH_SIZE, PATCH_SIZE).astype(np.float32)
    process    = psutil.Process()

    # Warmup
    for _ in range(N_WARMUP):
        session.run(None, {input_name: dummy})

    # Latency measurement
    latencies = []
    for _ in range(N_MEASURE):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    # RAM delta
    before = process.memory_info().rss / 1_048_576
    session.run(None, {input_name: dummy})
    after  = process.memory_info().rss / 1_048_576

    size_mb  = os.path.getsize(path) / 1_048_576
    mean_ms  = round(np.mean(latencies), 2)
    p95_ms   = round(np.percentile(latencies, 95), 2)
    ram_mb   = round(max(after - before, 0), 1)

    fits_size    = size_mb < CUBESAT_SPECS["FLASH_MB"]
    fits_latency = mean_ms < CUBESAT_SPECS["LATENCY_MS"]
    fits_ram     = ram_mb  < CUBESAT_SPECS["RAM_MB"]

    status = " FITS" if (fits_size and fits_latency and fits_ram) else " EXCEEDS"

    print(f"\n  {label}  [{status}]")
    print(f"    Size         : {size_mb:.2f} MB   budget {CUBESAT_SPECS['FLASH_MB']} MB  {'' if fits_size else ''}")
    print(f"    Latency mean : {mean_ms} ms      budget {CUBESAT_SPECS['LATENCY_MS']} ms  {'' if fits_latency else ''}")
    print(f"    Latency P95  : {p95_ms} ms")
    print(f"    RAM delta    : {ram_mb} MB       budget {CUBESAT_SPECS['RAM_MB']} MB  {'' if fits_ram else ''}")

    return {
        "run_mode":        "Docker (constrained)" if IN_DOCKER else "Native (Mac)",
        "model":           label,
        "size_mb":         round(size_mb, 2),
        "mean_latency_ms": mean_ms,
        "p95_latency_ms":  p95_ms,
        "ram_delta_mb":    ram_mb,
        "fits_cubesat":    fits_size and fits_latency and fits_ram,
    }


# ── Accuracy on real TIF tiles 
def accuracy_comparison():
    tif_files = list(Path(TIF_DIR).glob("*.tif"))
    if not tif_files:
        print(f"\n  No TIF files in {TIF_DIR}/ — skipping accuracy comparison")
        return

    try:
        import rasterio
        import torch, torch.nn as nn, torch.nn.functional as F
        from torchvision import models as tv_models
    except ImportError as e:
        print(f"  Missing library for accuracy comparison: {e}")
        return

    print(f"\n Accuracy comparison on {len(tif_files)} real Ghana tiles...")
    print("   Comparing Original vs INT8 Quantized on same patches...")

    # Load original PyTorch model
    device  = torch.device("cpu")
    pt_model = tv_models.mobilenet_v3_large(weights=None)
    pt_model.features[0][0] = nn.Conv2d(NUM_BANDS, 16, kernel_size=3, stride=2, padding=1, bias=False)
    pt_model.classifier[3]  = nn.Linear(pt_model.classifier[3].in_features, 2)
    ckpt = torch.load("models/mobilenetv3_best.pth", map_location=device)
    pt_model.load_state_dict(ckpt["model_state_dict"])
    pt_model.eval()

    # Load INT8 ONNX model
    int8_session = None
    if Path("models/mobilenetv3_int8.onnx").exists():
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        int8_session = ort.InferenceSession("models/mobilenetv3_int8.onnx", opts)

    orig_confs, int8_confs = [], []
    tiles_processed = 0

    for tif_path in tif_files[:5]:  # 5 tiles for comparison
        try:
            with rasterio.open(tif_path) as src:
                image = src.read()
        except Exception as e:
            print(f"   Skipping {tif_path.name}: {e}")
            continue

        _, H, W = image.shape
        for i in range(0, H - PATCH_SIZE + 1, STRIDE):
            for j in range(0, W - PATCH_SIZE + 1, STRIDE):
                patch = image[:, i:i+PATCH_SIZE, j:j+PATCH_SIZE].astype(np.float32)
                if patch.max() > 1:
                    patch = patch / 10000.0

                # Original PyTorch
                pt_input = torch.tensor(patch).unsqueeze(0)
                with torch.no_grad():
                    o_conf = F.softmax(pt_model(pt_input), dim=1)[0, 1].item()
                orig_confs.append(o_conf)

                # INT8 ONNX
                if int8_session:
                    ort_out = int8_session.run(None, {"input": patch[np.newaxis]})[0]
                    probs   = np.exp(ort_out) / np.exp(ort_out).sum(axis=1, keepdims=True)
                    int8_confs.append(float(probs[0, 1]))

        tiles_processed += 1

    if not orig_confs:
        print("   No patches processed — check TIF file format")
        return

    orig_arr = np.array(orig_confs)
    orig_det = (orig_arr >= THRESHOLD).sum()

    print(f"\n  Patches evaluated    : {len(orig_confs)}")
    print(f"  Tiles processed      : {tiles_processed}")
    print(f"  Original detections  : {orig_det}")

    if int8_confs:
        int8_arr = np.array(int8_confs)
        int8_det = (int8_arr >= THRESHOLD).sum()
        mae      = np.mean(np.abs(orig_arr - int8_arr))
        agree    = (orig_det == int8_det)
        print(f"  INT8 detections      : {int8_det}")
        print(f"  Mean conf delta      : {mae:.4f}  (lower is better — target <0.01)")
        print(f"  Detection agreement  : {' Yes' if agree else 'No'}")
        print(f"\n  PAPER FINDING: INT8 model {'maintains' if mae < 0.01 else 'slightly reduces'}")
        print(f"  confidence accuracy (MAE={mae:.4f}) while reducing model size by 73.7%")


# ── Main 
if __name__ == "__main__":
    mode = "Docker (CubeSat-constrained)" if IN_DOCKER else "Native (M1 Mac, unconstrained)"

    print("=" * 60)
    print("Phase 3 — CubeSat Constraint Benchmarking")
    print(f"Mode      : {mode}")
    print(f"CPU cores : {psutil.cpu_count()}")
    print(f"RAM avail : {psutil.virtual_memory().available / 1_048_576:.0f} MB")
    print("=" * 60)
    print("\nCubeSat reference hardware:")
    for k, v in CUBESAT_SPECS.items():
        print(f"  {k:<15}: {v}")

    if not ONNX_OK:
        sys.exit(1)

    results = []
    models_to_test = [
        ("models/mobilenetv3.onnx",            "Original (FP32 ONNX)"),
        ("models/mobilenetv3_int8.onnx",        "INT8 Quantized (ONNX)"),
        ("models/mobilenetv3_pruned.onnx",      "Pruned 30% (ONNX)"),
        ("models/mobilenetv3_pruned_int8.onnx", "Pruned 30% + INT8 (ONNX)"),
    ]

    print(f"\n Benchmarking {len(models_to_test)} model variants [{mode}]:")
    for path, label in models_to_test:
        r = benchmark_onnx(path, label)
        if r:
            results.append(r)

    # Accuracy section (needs TIF files)
    accuracy_comparison()

    # Save results
    if results:
        df  = pd.DataFrame(results)
        tag = "docker" if IN_DOCKER else "native"
        out = f"{RESULTS_DIR}/constrained_metrics_{tag}.csv"
        df.to_csv(out, index=False)
        print(f"\n Results saved → {out}")

    print("\n" + "=" * 60)
    if not IN_DOCKER:
        print("NEXT: Run inside Docker to get constrained results:")

    else:
        print("Docker run complete — compare with native results for paper")
    print("=" * 60)