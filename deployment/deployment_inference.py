import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import rasterio
import json
import os
import geopandas as gpd
from torchvision import models
from pathlib import Path
from shapely.geometry import Point
from datetime import datetime
from tqdm import tqdm


# SETTINGS & PATHS

BASE_PATH = "./"
CHECKPOINT_PATH = os.path.join(BASE_PATH, "models/mobilenetv3_best.pth")
INPUT_DIR = os.path.join(BASE_PATH, "data/composite tif")
OUTPUT_DIR = os.path.join(BASE_PATH, "deployment/deployment_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_SIZE = 224          
STRIDE = 224               
NUM_BANDS = 10
THRESHOLD = 0.3           

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# LOAD MODEL & DUAL BOUNDARIES

def load_model(checkpoint_path):
    model = models.mobilenet_v3_large(weights=None)
    model.features[0][0] = nn.Conv2d(NUM_BANDS, 16, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model

# LOAD BOTH ADMINISTRATIVE LAYERS
ADM1_PATH = os.path.join(BASE_PATH, "deployment/geoBoundaries-GHA-ADM1.geojson")
ADM2_PATH = os.path.join(BASE_PATH, "deployment/geoBoundaries-GHA-ADM2.geojson")

ghana_regions = gpd.read_file(ADM1_PATH) if os.path.exists(ADM1_PATH) else None
ghana_districts = gpd.read_file(ADM2_PATH) if os.path.exists(ADM2_PATH) else None

def get_location_details(longitude, latitude):
    # Standard WGS84 point
    point = Point(longitude, latitude)
    
    district = "Unknown"
    region = "Unknown"

    # 1. Get Region from ADM1
    if ghana_regions is not None:
        # Ensure point is in the same CRS as the boundary file
        r_match = ghana_regions[ghana_regions.contains(point)]
        if not r_match.empty:
            region = r_match.iloc[0].get('shapeName', 'Unknown')
            
    # 2. Get District from ADM2
    if ghana_districts is not None:
        d_match = ghana_districts[ghana_districts.contains(point)]
        if not d_match.empty:
            district = d_match.iloc[0].get('shapeName', 'Unknown')
            
    return district, region


# INFERENCE (DUAL-JOIN OPTIMIZED)

def process_shard(model, tif_path):
    features = []
    tile_id = os.path.basename(tif_path)
    area_ha = (MODEL_SIZE * 10 * MODEL_SIZE * 10) / 10000.0

    with rasterio.open(tif_path) as src:
        image = src.read()
        transform = src.transform
        _, H, W = image.shape
        
        for i in range(0, H - MODEL_SIZE + 1, STRIDE):
            for j in range(0, W - MODEL_SIZE + 1, STRIDE):
                patch = image[:, i:i+MODEL_SIZE, j:j+MODEL_SIZE]
                patch_tensor = torch.tensor(patch, dtype=torch.float32)
                
                if patch_tensor.max() > 1: patch_tensor /= 10000.0
                
                patch_tensor = patch_tensor.unsqueeze(0).to(device)
                with torch.no_grad():
                    output = model(patch_tensor)
                    probs = F.softmax(output, dim=1)
                    confidence = probs[0, 1].item()

                if confidence >= THRESHOLD:
                    alert = "HIGH" if confidence > 0.8 else "MEDIUM" if confidence > 0.5 else "LOW"
                    longitude, latitude = transform * (j + (MODEL_SIZE // 2), i + (MODEL_SIZE // 2))
                    
                    # TWO-STEP DYNAMIC LOOKUP
                    district, region = get_location_details(longitude, latitude)

                    features.append({
                        "type": "Feature",
                        "geometry": {"type": "Point", "coordinates": [float(longitude), float(latitude)]},
                        "properties": {
                            "confidence": round(float(confidence), 4),
                            "district": district,
                            "region": region,
                            "date": datetime.now().strftime("%Y-%m-%d"),
                            "area_ha": round(area_ha, 2),
                            "tile_id": tile_id,
                            "alert_level": alert
                        }
                    })
    return features

if __name__ == "__main__":
    print(f" Running ASM Pipeline with Dual-Layer Boundaries (ADM1 + ADM2)...")
    model = load_model(CHECKPOINT_PATH)
    tif_files = list(Path(INPUT_DIR).glob("*.tif"))
    all_features = []

    for tif_path in tqdm(tif_files, desc="Processing"):
        all_features.extend(process_shard(model, str(tif_path)))

    output_path = os.path.join(OUTPUT_DIR, "asm_monitoring_results.geojson")
    with open(output_path, "w") as f:
        json.dump({"type": "FeatureCollection", "features": all_features}, f, indent=2)
    
    print(f"\n Pipeline Complete. Region (ADM1) and District (ADM2) successfully mapped.")