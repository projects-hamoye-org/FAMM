#!/usr/bin/env python3
"""
FAMM GeoJSON Validator & Cleaner (Append Mode)

"""

import json
import sys
from datetime import datetime
from pathlib import Path

# ── Thresholds —
def remap_alert_level(confidence: float) -> str:
    if confidence > 0.8:
        return "HIGH"
    elif confidence > 0.5:
        return "MEDIUM"
    return "LOW"


REQUIRED_FIELDS = [
    "confidence", "district", "region",
    "date", "area_ha", "tile_id", "alert_level"
]

DISTRICT_TO_REGION = {
    "Kwabre":       "Ashanti Region",
    "Obuasi":       "Ashanti Region",
    "Amansie":      "Ashanti Region",
    "Adansi":       "Ashanti Region",
    "Bekwai":       "Ashanti Region",
    "Tarkwa":       "Western Region",
    "Prestea":      "Western Region",
    "Wassa":        "Western Region",
    "Bibiani":      "Western Region",
    "Mfantseman":   "Central Region",
    "Komenda":      "Central Region",
    "Cape Coast":   "Central Region",
    "Abura":        "Central Region",
    "Atiwa":        "Eastern Region",
    "Birim":        "Eastern Region",
    "Denkyembour":  "Eastern Region",
    "West Akim":    "Eastern Region",
    "East Akim":    "Eastern Region",
}


def title_case_district(name: str) -> str:
    def cap(part):
        return part[0].upper() + part[1:] if part else part
    return " ".join(
        "-".join(cap(p) for p in word.split("-"))
        for word in name.split(" ")
    )


def fallback_region(district: str):
    for key, region in DISTRICT_TO_REGION.items():
        if key.lower() in district.lower():
            return region
    return None


def validate_date(s: str) -> bool:
    try:
        datetime.strptime(s, "%Y-%m-%d")
        return True
    except (ValueError, TypeError):
        return False


def load_existing(output_file: str) -> tuple[list, set]:
    """
    Load existing features from the output file.
    Returns (features list, set of (tile_id, date) keys already present).
    """
    if not Path(output_file).exists():
        return [], set()
    try:
        with open(output_file) as f:
            data = json.load(f)
        existing = data.get("features", [])
        keys = {
            (ft["properties"].get("tile_id", ""),
             ft["properties"].get("date", ""))
            for ft in existing
        }
        return existing, keys
    except (json.JSONDecodeError, KeyError):
        print(f"  ⚠️  Could not read existing file — starting fresh.")
        return [], set()


def clean_feature(feature: dict, index: int) -> tuple[dict | None, str | None]:
    """
    Validate and clean a single feature.
    Returns (cleaned_feature, None) on success or (None, reason) on failure.
    """
    props = feature.get("properties", {})
    geom  = feature.get("geometry", {})

    missing = [f for f in REQUIRED_FIELDS if f not in props]
    if missing:
        return None, f"Missing fields: {missing}"

    coords = geom.get("coordinates", [])
    if len(coords) < 2:
        return None, "Invalid coordinates"

    confidence = float(props["confidence"])
    district   = str(props["district"]).strip()
    region     = str(props["region"]).strip()
    date_str   = str(props["date"]).strip()
    area_ha    = float(props["area_ha"])
    tile_id    = str(props.get("tile_id", ""))
    alert_in   = str(props.get("alert_level", "")).strip()

    if not validate_date(date_str):
        return None, f"Bad date format: {date_str!r}"

    alert_out      = remap_alert_level(confidence)
    district_fixed = title_case_district(district)

    if region and "Region" not in region and region not in ("Unknown", ""):
        region += " Region"

    if not region or region in ("Unknown", "Unknown Region", ""):
        fb     = fallback_region(district_fixed)
        region = fb if fb else "Unknown Region"

    return {
        "type": "Feature",
        "geometry": geom,
        "properties": {
            "confidence":  round(confidence, 4),
            "district":    district_fixed,
            "region":      region,
            "date":        date_str,
            "area_ha":     round(area_ha, 2),
            "tile_id":     tile_id,
            "alert_level": alert_out,
        }
    }, None


def validate_and_clean(input_file: str, output_file: str,
                        overwrite: bool = False) -> bool:
    print("=" * 60)
    print("FAMM GeoJSON Validator (Append Mode)")
    print(f"  Input  : {input_file}")
    print(f"  Output : {output_file}")
    print(f"  Mode   : {'OVERWRITE' if overwrite else 'APPEND'}")
    print("=" * 60)

    # ── Load new input ────────────────────────────────────────────────────────
    try:
        with open(input_file) as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Input not found: {input_file}")
        return False
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON: {e}")
        return False

    raw = data.get("features", [])
    print(f"\n📥 New features from inference  : {len(raw)}")

    # ── Load existing history (unless overwrite) ───────────────────────────────
    if overwrite:
        existing_features, existing_keys = [], set()
        print(f"📂 Existing history             : skipped (overwrite mode)")
    else:
        existing_features, existing_keys = load_existing(output_file)
        print(f"📂 Existing features in history : {len(existing_features)}")

    # ── Clean new features ────────────────────────────────────────────────────
    cleaned  = []
    skipped  = []
    dupes    = 0

    for i, feat in enumerate(raw):
        result, reason = clean_feature(feat, i)
        if result is None:
            skipped.append((i, reason))
            continue

        # Deduplication check
        key = (result["properties"]["tile_id"],
               result["properties"]["date"])
        if key in existing_keys:
            dupes += 1
            continue

        cleaned.append(result)
        existing_keys.add(key)   # prevent dupes within this batch too

    # ── Combine and save ──────────────────────────────────────────────────────
    all_features = existing_features + cleaned
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({"type": "FeatureCollection", "features": all_features}, f,
                  indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    alert_counts  = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    region_counts = {}
    week_counts   = {}

    for feat in all_features:
        p = feat["properties"]
        alert_counts[p["alert_level"]] = alert_counts.get(p["alert_level"], 0) + 1
        region_counts[p["region"]]     = region_counts.get(p["region"], 0) + 1
        week_counts[p["date"]]         = week_counts.get(p["date"], 0) + 1

    print(f"\n{'─'*60}")
    print("RESULTS")
    print(f"{'─'*60}")
    print(f"  New detections added     : {len(cleaned)}")
    print(f"  Duplicates skipped       : {dupes}")
    print(f"  Invalid features skipped : {len(skipped)}")
    print(f"  Total in history now     : {len(all_features)}")
    print(f"\n  Alert levels (Thresholds  HIGH>0.8, MEDIUM>0.5):")
    for lvl in ("HIGH", "MEDIUM", "LOW"):
        print(f"    {lvl:6s}: {alert_counts.get(lvl, 0)}")
    print(f"\n  Regions:")
    for r, c in sorted(region_counts.items()):
        print(f"    {r}: {c}")
    print(f"\n  Detections by date (all history):")
    for d, c in sorted(week_counts.items()):
        marker = " ← this run" if d == max(week_counts) else ""
        print(f"    {d}: {c} detection(s){marker}")
    if skipped:
        print(f"\n  Skipped features:")
        for idx, reason in skipped:
            print(f"    Feature {idx}: {reason}")
    print(f"\n✅ Saved to: {output_file}")
    print("=" * 60)
    return True


if __name__ == "__main__":
    overwrite = "--overwrite" in sys.argv
    args      = [a for a in sys.argv[1:] if not a.startswith("--")]

    inp = args[0] if len(args) > 0 else "asm_monitoring_results.geojson"
    out = args[1] if len(args) > 1 else "data/geojson/latest_detections.geojson"

    sys.exit(0 if validate_and_clean(inp, out, overwrite=overwrite) else 1)