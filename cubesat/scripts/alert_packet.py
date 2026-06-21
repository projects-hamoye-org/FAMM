"""
Phase 4 — Alert Packet Design and Communication Protocol
Defines the low-bandwidth alert packet format for CubeSat downlink.
Demonstrates compression ratio vs raw GeoTIFF.
No TIF files or model needed.
"""

import json
import struct
import hashlib
import time
import pandas as pd
import zlib
from pathlib import Path
from datetime import datetime, timezone

RESULTS_DIR   = "data/results"
OUTPUTS_DIR   = "outputs/packets"
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(OUTPUTS_DIR).mkdir(parents=True, exist_ok=True)


# ── Alert packet formats ──────────────────────────────────────────────────────

def create_binary_packet(detection: dict) -> bytes:
    """
    Minimal binary packet.
    Designed for constrained satellite downlink (S-band, ~9.6 kbps).

    Format (big-endian):
      4 bytes  — latitude  (float32)
      4 bytes  — longitude (float32)
      4 bytes  — confidence (float32)
      4 bytes  — area_ha (float32)
      4 bytes  — unix timestamp (uint32)
      1 byte   — alert level (0=LOW, 1=MEDIUM, 2=HIGH)
      1 byte   — region code (0-15 for Ghana's 16 regions)
      6 bytes  — reserved / padding
    Total: 28 bytes
    """
    alert_map  = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
    region_map = {
        "Ashanti Region": 0, "Brong-Ahafo Region": 1, "Central Region": 2,
        "Eastern Region": 3, "Greater Accra Region": 4, "Northern Region": 5,
        "Upper East Region": 6, "Upper West Region": 7, "Volta Region": 8,
        "Western Region": 9, "Ahafo Region": 10, "Bono Region": 11,
        "Bono East Region": 12, "Oti Region": 13, "Savannah Region": 14,
        "Western North Region": 15,
    }
    return struct.pack(
        ">ffffIBBxxxxxx",
        detection["latitude"],
        detection["longitude"],
        detection["confidence"],
        detection["area_ha"],
        int(datetime.now(timezone.utc).timestamp()),
        alert_map.get(detection["alert_level"], 0),
        region_map.get(detection["region"], 255),
    )


def create_json_packet(detection: dict) -> str:
    alert_short = {"LOW": "L", "MEDIUM": "M", "HIGH": "H"}
    packet = {
        "v":    1,
        "ts":   int(datetime.now(timezone.utc).timestamp()),
        "lat":  round(detection["latitude"], 4),
        "lon":  round(detection["longitude"], 4),
        "conf": round(detection["confidence"], 3),
        "alrt": alert_short.get(detection["alert_level"], "L"),
        "area": round(detection["area_ha"], 1),
        "reg":  detection["region"][:3].upper(),
    }
    return json.dumps(packet, separators=(",", ":"))


def create_batch_packet(detections: list) -> bytes:
    """
    Batch packet — multiple detections in one downlink transmission.
    """
    magic     = b"FAMM"
    count     = len(detections)
    timestamp = int(datetime.now(timezone.utc).timestamp())
    header    = struct.pack(">4sHH I", magic, count, 0, timestamp)  # 12 bytes
    body      = b"".join(create_binary_packet(d) for d in detections)
    checksum  = zlib.crc32(header + body) & 0xFFFFFFFF
    footer    = struct.pack(">I", checksum)   # 4 bytes
    return header + body + footer


def decode_binary_packet(data: bytes) -> dict:
    """Decode a 28-byte binary packet back to human-readable dict."""
    alert_map  = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    region_map = {v: k for k, v in {
        "Ashanti Region": 0, "Brong-Ahafo Region": 1, "Central Region": 2,
        "Eastern Region": 3, "Greater Accra Region": 4, "Northern Region": 5,
        "Upper East Region": 6, "Upper West Region": 7, "Volta Region": 8,
        "Western Region": 9, "Ahafo Region": 10, "Bono Region": 11,
        "Bono East Region": 12, "Oti Region": 13, "Savannah Region": 14,
        "Western North Region": 15,
    }.items()}
    lat, lon, conf, area, ts, alert_code, region_code = struct.unpack(">ffffIBBxxxxxx", data)
    return {
        "latitude":    round(lat, 4),
        "longitude":   round(lon, 4),
        "confidence":  round(conf, 3),
        "area_ha":     round(area, 1),
        "timestamp":   datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
        "alert_level": alert_map.get(alert_code, "UNKNOWN"),
        "region":      region_map.get(region_code, f"code_{region_code}"),
    }


# ── Compression analysis 
def analyse_compression(detections: list):
    """
    Compare packet sizes against raw GeoTIFF transmission.
    """
    raw_tif_bytes  = 75 * 1_048_576   # 75 MB per tile
    tiles_per_run  = 60               # full Ghana run

    binary_total   = len(create_batch_packet(detections))
    json_total     = sum(len(create_json_packet(d).encode()) for d in detections)
    raw_total      = raw_tif_bytes * tiles_per_run

    results = {
        "detections_count":        len(detections),
        "binary_batch_bytes":      binary_total,
        "binary_batch_kb":         round(binary_total / 1024, 2),
        "json_total_bytes":        json_total,
        "json_total_kb":           round(json_total / 1024, 2),
        "raw_tif_total_mb":        round(raw_total / 1_048_576, 0),
        "compression_vs_raw_x":    round(raw_total / binary_total, 0),
        "downlink_time_binary_s":  round(binary_total * 8 / 9600, 2),  # 9.6 kbps S-band
        "downlink_time_raw_hours": round(raw_total * 8 / 9600 / 3600, 1),
    }
    return results


# ── Main 
if __name__ == "__main__":

    print("Phase 4 — Alert Packet Design")


    # Sample detections matching the March 2026 pipeline run
    sample_detections = [
        {"latitude": 6.41,  "longitude": -1.62, "confidence": 0.87, "area_ha": 501.76, "alert_level": "HIGH",   "region": "Ashanti Region"},
        {"latitude": 5.89,  "longitude": -1.23, "confidence": 0.73, "area_ha": 501.76, "alert_level": "MEDIUM", "region": "Eastern Region"},
        {"latitude": 9.31,  "longitude": -0.87, "confidence": 0.91, "area_ha": 501.76, "alert_level": "HIGH",   "region": "Savannah Region"},
        {"latitude": 6.12,  "longitude": -0.45, "confidence": 0.42, "area_ha": 501.76, "alert_level": "LOW",    "region": "Volta Region"},
        {"latitude": 7.74,  "longitude": -1.05, "confidence": 0.66, "area_ha": 501.76, "alert_level": "MEDIUM", "region": "Northern Region"},
    ]

    print("\n Single binary packet (28 bytes):")
    single = create_binary_packet(sample_detections[0])
    print(f"   Raw bytes  : {single.hex()}")
    print(f"   Size       : {len(single)} bytes")

    print("\n Decode and verify round-trip:")
    decoded = decode_binary_packet(single)
    for k, v in decoded.items():
        print(f"   {k:<15}: {v}")

    print("\n JSON packet:")
    json_pkt = create_json_packet(sample_detections[0])
    print(f"   {json_pkt}")
    print(f"   Size: {len(json_pkt.encode())} bytes")

    print(f"\n Batch packet ({len(sample_detections)} detections):")
    batch = create_batch_packet(sample_detections)
    print(f"   Total size : {len(batch)} bytes ({len(batch)/1024:.2f} KB)")

    print("\n Compression analysis (vs raw GeoTIFF downlink):")
    compression = analyse_compression(sample_detections)
    for k, v in compression.items():
        print(f"   {k:<30}: {v}")

    print(f"   {compression['compression_vs_raw_x']:,.0f}x compression ratio")
    print(f"   Binary alert packet: {compression['binary_batch_kb']} KB")
    print(f"   Raw GeoTIFF total : {compression['raw_tif_total_mb']:.0f} MB")
    print(f"   Downlink time (binary): {compression['downlink_time_binary_s']} seconds")
    print(f"   Downlink time (raw)   : {compression['downlink_time_raw_hours']} hours")

    # Save outputs
    pd.DataFrame([compression]).to_csv(f"{RESULTS_DIR}/compression_analysis.csv", index=False)

    with open(f"{OUTPUTS_DIR}/sample_batch_packet.bin", "wb") as f:
        f.write(batch)
    with open(f"{OUTPUTS_DIR}/sample_detections.json", "w") as f:
        json.dump([create_json_packet(d) for d in sample_detections], f, indent=2)

    print(f"\n Outputs saved to {OUTPUTS_DIR}/")
    print("=" * 60)