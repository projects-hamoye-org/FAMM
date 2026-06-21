"""
Phase 2 — Model Optimisation (M1 Mac compatible)
-------------------------------------------------
Uses ONNX export + ONNX Runtime quantization instead of PyTorch's
quantize_dynamic (which requires x86 FBGEMM engine, unavailable on M1).

WHY ONNX:
  ONNX (Open Neural Network Exchange) is the standard format for deploying
  models on edge hardware including CubeSats. Flight computers like the
  Unibap iX5-100 and Ubotica CogniSat use ONNX Runtime for inference.

THREE OPTIMISATION STEPS:
  1. ONNX export          — converts PyTorch model to portable format
  2. ONNX INT8 quantization — shrinks weights from float32 to int8
  3. Structural pruning   — removes redundant channels before ONNX export

OUTPUT FILES:
  models/mobilenetv3.onnx          — FP32 ONNX (baseline portable)
  models/mobilenetv3_int8.onnx     — INT8 quantized ONNX
  models/mobilenetv3_pruned.pth    — pruned PyTorch (before ONNX export)
  models/mobilenetv3_pruned_int8.onnx — pruned + quantized ONNX

No TIF files needed.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
import pandas as pd
import psutil
import time
import os
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from torchvision import models

# ONNX Runtime for inference + quantization
try:
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    ONNX_AVAILABLE = True
except ImportError:
    print(" onnx or onnxruntime not installed")
    print("   Run: pip install onnx onnxruntime")
    ONNX_AVAILABLE = False

# ── Constants 
CHECKPOINT_PATH        = "models/mobilenetv3_best.pth"
ONNX_PATH              = "models/mobilenetv3.onnx"
ONNX_INT8_PATH         = "models/mobilenetv3_int8.onnx"
PRUNED_PTH_PATH        = "models/mobilenetv3_pruned.pth"
PRUNED_ONNX_PATH       = "models/mobilenetv3_pruned.onnx"
PRUNED_INT8_ONNX_PATH  = "models/mobilenetv3_pruned_int8.onnx"
RESULTS_DIR            = "data/results"
NUM_BANDS              = 10
PATCH_SIZE             = 224
PRUNE_AMOUNT           = 0.3   # remove 30% of lowest-magnitude weights
N_WARMUP               = 5
N_MEASURE              = 50
device                 = torch.device("cpu")

Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path("models").mkdir(parents=True, exist_ok=True)


# ── Load model 
def load_model(path=CHECKPOINT_PATH):
    """Load model — architecture + weights from checkpoint."""
    model = models.mobilenet_v3_large(weights=None)
    model.features[0][0] = nn.Conv2d(
        NUM_BANDS, 16, kernel_size=3, stride=2, padding=1, bias=False
    )
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    return model.to(device).eval()


# ── Step 1: Export to ONNX 
def export_to_onnx(model, output_path):
    dummy_input = torch.randn(1, NUM_BANDS, PATCH_SIZE, PATCH_SIZE)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params    = True,
        opset_version    = 17,
        do_constant_folding = True,
        input_names      = ["input"],
        output_names     = ["output"],
        dynamic_axes     = {"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        verbose          = False,
    )
    # Validate the exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"   ONNX model validated  → {output_path}")


# ── Step 2: INT8 quantization via ONNX Runtime 
def quantize_onnx(input_path, output_path):
    quantize_dynamic(
        model_input  = input_path,
        model_output = output_path,
        weight_type  = QuantType.QInt8,
    )
    print(f"   INT8 quantized ✅ → {output_path}")


# ── Step 3: Structural pruning
def apply_pruning(amount=PRUNE_AMOUNT):
    """
    Why prune before ONNX export?
    ONNX quantization works better when sparse weights are already zeroed —
    the quantizer can represent more values in the int8 range accurately.
    """
    model = load_model()
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=amount)
            prune.remove(module, "weight")

    # Count sparsity
    zero_params = sum(
        (p == 0).sum().item() for p in model.parameters()
    )
    total_params = sum(p.numel() for p in model.parameters())
    sparsity = zero_params / total_params * 100
    print(f"   Sparsity after pruning: {sparsity:.1f}% of weights zeroed")

    return model


# ── Benchmark ONNX model 
def benchmark_onnx(onnx_path, label):
    """
    Benchmark an ONNX model using ONNX Runtime.
    """
    if not Path(onnx_path).exists():
        print(f"   ⚠️  {label} — file not found: {onnx_path}")
        return None

    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1   # Simulate single-core CubeSat CPU
    session = ort.InferenceSession(onnx_path, sess_options=sess_options)
    input_name = session.get_inputs()[0].name

    dummy = np.random.randn(1, NUM_BANDS, PATCH_SIZE, PATCH_SIZE).astype(np.float32)
    process = psutil.Process()

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
    after = process.memory_info().rss / 1_048_576

    size_mb = os.path.getsize(onnx_path) / 1_048_576

    # === FIXED: Safe input size calculation ===
    def safe_prod(shape):
        prod = 1
        for dim in shape:
            if isinstance(dim, (int, np.integer)):
                prod *= int(dim)
            # Skip string dimensions like 'batch_size'
        return prod

    input_size = sum(safe_prod(inp.shape) for inp in session.get_inputs())

    result = {
        "model":           label,
        "format":          "ONNX",
        "size_mb":         round(size_mb, 2),
        "mean_latency_ms": round(np.mean(latencies), 2),
        "median_latency_ms": round(np.median(latencies), 2),
        "p95_latency_ms":  round(np.percentile(latencies, 95), 2),
        "ram_delta_mb":    round(after - before, 1),
        "total_params":    input_size,          # Fixed
    }

    # CubeSat constraint check
    fits_size    = size_mb              < 50
    fits_latency = result["mean_latency_ms"] < 500
    fits_ram     = result["ram_delta_mb"]    < 256

    print(f"\n  📊 {label}")
    print(f"     Size     : {size_mb:.2f} MB  {'' if fits_size else ''}")
    print(f"     Latency  : {result['mean_latency_ms']} ms  {'' if fits_latency else ''}")
    print(f"     RAM delta: {result['ram_delta_mb']} MB  {'' if fits_ram else ''}")
    result["fits_cubesat"] = fits_size and fits_latency and fits_ram

    return result


# ── Main 
if __name__ == "__main__":
    print("=" * 60)
    print("Phase 2 — Model Optimisation (ONNX-based, M1 compatible)")
    print("=" * 60)

    if not ONNX_AVAILABLE:
        exit(1)

    all_results = []

    # ── Step 1: Export original model to ONNX 
    print("\n Step 1: Exporting original model to ONNX...")
    original = load_model()
    export_to_onnx(original, ONNX_PATH)
    r1 = benchmark_onnx(ONNX_PATH, "Original (FP32 ONNX)")
    if r1:
        all_results.append(r1)

    # ── Step 2: INT8 quantization 
    print("\n Step 2: Applying INT8 dynamic quantization...")
    quantize_onnx(ONNX_PATH, ONNX_INT8_PATH)
    r2 = benchmark_onnx(ONNX_INT8_PATH, "INT8 Quantized (ONNX)")
    if r2:
        all_results.append(r2)

    # ── Step 3: Pruning 
    print(f"\n Step 3: Pruning ({int(PRUNE_AMOUNT*100)}% of Conv2d weights)...")
    pruned = apply_pruning(PRUNE_AMOUNT)
    # Save pruned PyTorch model
    torch.save({"model_state_dict": pruned.state_dict()}, PRUNED_PTH_PATH)
    print(f"   Pruned PyTorch model saved → {PRUNED_PTH_PATH}")
    # Export pruned to ONNX
    export_to_onnx(pruned, PRUNED_ONNX_PATH)
    r3 = benchmark_onnx(PRUNED_ONNX_PATH, f"Pruned {int(PRUNE_AMOUNT*100)}% (ONNX)")
    if r3:
        all_results.append(r3)

    # ── Step 4: Pruned + INT8 
    print(f"\n  Step 4: Pruned + INT8 quantization (combined)...")
    quantize_onnx(PRUNED_ONNX_PATH, PRUNED_INT8_ONNX_PATH)
    r4 = benchmark_onnx(PRUNED_INT8_ONNX_PATH, f"Pruned {int(PRUNE_AMOUNT*100)}% + INT8 (ONNX)")
    if r4:
        all_results.append(r4)

    # ── Summary 
    if all_results:
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        df = pd.DataFrame(all_results)
        base_size = df["size_mb"].iloc[0]
        base_lat  = df["mean_latency_ms"].iloc[0]
        df["size_reduction_%"] = ((base_size - df["size_mb"]) / base_size * 100).round(1)
        df["speedup_x"]        = (base_lat / df["mean_latency_ms"]).round(2)

        print(df[["model", "size_mb", "mean_latency_ms",
                  "size_reduction_%", "speedup_x", "fits_cubesat"]].to_string(index=False))

        out = f"{RESULTS_DIR}/optimisation_metrics.csv"
        df.to_csv(out, index=False)
        print(f"\n Results saved → {out}")

    print("\n" + "=" * 60)
    print("ONNX models saved — these are the deployment artefacts for the paper:")
    for p in [ONNX_PATH, ONNX_INT8_PATH, PRUNED_ONNX_PATH, PRUNED_INT8_ONNX_PATH]:
        if Path(p).exists():
            mb = os.path.getsize(p) / 1_048_576
            print(f"  {p:<45} {mb:.2f} MB")
    print("=" * 60) 