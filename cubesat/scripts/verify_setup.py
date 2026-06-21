import torch
import torch.nn as nn
import torchinfo
import onnx
import rasterio
import psutil
import pandas
import matplotlib
import os
from pathlib import Path
from torchvision import models

print("=== Environment Check ===")
print(f"PyTorch   : {torch.__version__}")
print(f"MPS (M1)  : {torch.backends.mps.is_available()}")
print(f"rasterio  : {rasterio.__version__}")
print(f"pandas    : {pandas.__version__}")

# ── Model settings
NUM_BANDS = 10
device    = torch.device("cpu")   # always CPU for benchmarking

print("\n=== Model Check ===")

checkpoint_path = "models/mobilenetv3_best.pth"

if not Path(checkpoint_path).exists():
    print(f" Model not found at {checkpoint_path}")
    print("   Copy it from your FAMM dashboard repo:")
    print("   cp ~/projects/famm-dashboard/models/mobilenetv3_best.pth models/")
else:
    # ── Rebuild the architecture 
    model = models.mobilenet_v3_large(weights=None)
    model.features[0][0] = nn.Conv2d(
        NUM_BANDS, 16, kernel_size=3, stride=2, padding=1, bias=False
    )
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)

    # ── Load weights from the checkpoint dict 
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    print(" Model loaded and ready")
    print(f"   Checkpoint epoch      : {checkpoint.get('epoch', 'N/A')}")
    print(f"   Best test accuracy    : {checkpoint.get('best_test_acc', 'N/A')}")

    # ── Size and parameter count 
    size_mb     = os.path.getsize(checkpoint_path) / 1_048_576
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters            : {total_params:,}")
    print(f"   Checkpoint size       : {size_mb:.1f} MB")

    # ── Quick forward pass to confirm model works
    dummy = torch.randn(1, NUM_BANDS, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    print(f"   Output shape          : {out.shape}  (expected: [1, 2])")
    print(f"   Output (raw logits)   : {out[0].tolist()}")

print("\n=== TIF Check ===")
tif_dir  = "data/tif_input"
tif_files = list(Path(tif_dir).glob("*.tif"))
print(f"TIF tiles found: {len(tif_files)}")

if tif_files:
    with rasterio.open(tif_files[0]) as src:
        print(f"   Sample tile : {tif_files[0].name}")
        print(f"   Bands       : {src.count}  (expected: 10)")
        print(f"   CRS         : {src.crs}")
        print(f"   Size        : {src.width} x {src.height} px")
else:
    print("   No .tif files found — copy them from your March pipeline run:")


print("\n=== RAM Check ===")
mem = psutil.virtual_memory()
print(f"Total RAM : {mem.total / 1_073_741_824:.1f} GB")
print(f"Available : {mem.available / 1_073_741_824:.1f} GB")

print("\n Setup verified — ready to start Phase 1 benchmarking")