#!/usr/bin/env python3
"""
FAMM Demo Data Generator
Creates realistic sample detection data for dashboard demo
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

# Realistic coordinates for Ghana mining regions
REGIONS_DATA = {
    'Western Region': {
        'center': [-2.0, 6.0],
        'districts': {
            'Tarkwa-Nsuaem': {'center': [-1.99, 5.30], 'radius': 0.15},
            'Prestea-Huni Valley': {'center': [-2.15, 5.45], 'radius': 0.12},
            'Wassa East': {'center': [-2.05, 5.65], 'radius': 0.10},
            'Bibiani-Anhwiaso-Bekwai': {'center': [-2.32, 6.46], 'radius': 0.13}
        }
    },
    'Ashanti Region': {
        'center': [-1.5, 6.7],
        'districts': {
            'Obuasi': {'center': [-1.67, 6.19], 'radius': 0.12},
            'Amansie Central': {'center': [-1.86, 6.43], 'radius': 0.10},
            'Amansie West': {'center': [-2.05, 6.35], 'radius': 0.11},
            'Adansi South': {'center': [-1.83, 6.10], 'radius': 0.09}
        }
    },
    'Eastern Region': {
        'center': [-0.4, 6.4],
        'districts': {
            'Atiwa West': {'center': [-0.74, 6.16], 'radius': 0.08},
            'Birim North': {'center': [-0.91, 6.14], 'radius': 0.10},
            'Denkyembour': {'center': [-0.96, 6.08], 'radius': 0.09},
            'West Akim': {'center': [-0.72, 5.98], 'radius': 0.11}
        }
    }
}

def random_point_near(center, radius):
    """Generate random point within radius of center"""
    # Add random offset
    lon_offset = random.uniform(-radius, radius)
    lat_offset = random.uniform(-radius, radius)
    
    return [
        round(center[0] + lon_offset, 6),
        round(center[1] + lat_offset, 6)
    ]

def generate_detection(region, district, district_info, date, confidence_bias='medium'):
    """Generate a single detection feature"""
    
    # Generate confidence based on bias
    if confidence_bias == 'high':
        confidence = random.uniform(0.85, 0.98)
    elif confidence_bias == 'medium':
        confidence = random.uniform(0.70, 0.89)
    else:  # low
        confidence = random.uniform(0.60, 0.74)
    
    # Generate coordinates near district center
    coordinates = random_point_near(district_info['center'], district_info['radius'])
    
    # Generate area (hectares) - correlated with confidence
    # Higher confidence often means more visible/larger sites
    if confidence > 0.9:
        area = round(random.uniform(1.5, 4.5), 1)
    elif confidence > 0.75:
        area = round(random.uniform(0.8, 2.5), 1)
    else:
        area = round(random.uniform(0.3, 1.5), 1)
    
    # Determine alert level
    if confidence >= 0.85:
        alert_level = 'HIGH'
    elif confidence >= 0.70:
        alert_level = 'MEDIUM'
    else:
        alert_level = 'LOW'
    
    # Create tile ID (realistic format)
    tile_id = f"S2A_TILE_{date.strftime('%Y%m%d')}_{random.randint(100000, 999999)}"
    
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": coordinates
        },
        "properties": {
            "confidence": round(confidence, 3),
            "district": district,
            "region": region,
            "date": date.strftime('%Y-%m-%d'),
            "area_ha": area,
            "tile_id": tile_id,
            "alert_level": alert_level
        }
    }

def generate_demo_data(
    num_detections=45,
    days_back=30,
    high_risk_ratio=0.20,
    medium_risk_ratio=0.50
):
    """
    Generate realistic demo detection data
    
    Args:
        num_detections: Total number of detections to generate
        days_back: How many days back to generate data
        high_risk_ratio: Proportion of high-risk detections
        medium_risk_ratio: Proportion of medium-risk detections
    """
    
    features = []
    end_date = datetime.now()
    
    # Calculate detection counts by risk level
    num_high = int(num_detections * high_risk_ratio)
    num_medium = int(num_detections * medium_risk_ratio)
    num_low = num_detections - num_high - num_medium
    
    detection_schedule = (
        ['high'] * num_high +
        ['medium'] * num_medium +
        ['low'] * num_low
    )
    random.shuffle(detection_schedule)
    
    # Weight towards Western and Ashanti regions (historically active areas)
    region_weights = {
        'Western Region': 0.50,
        'Ashanti Region': 0.35,
        'Eastern Region': 0.15
    }
    
    for i, confidence_bias in enumerate(detection_schedule):
        # Random date within range (more recent = more likely)
        # Weight towards recent detections
        days_ago = int(random.triangular(0, days_back, days_back * 0.2))
        detection_date = end_date - timedelta(days=days_ago)
        
        # Select region based on weights
        region = random.choices(
            list(region_weights.keys()),
            weights=list(region_weights.values())
        )[0]
        
        # Select random district in that region
        district = random.choice(list(REGIONS_DATA[region]['districts'].keys()))
        district_info = REGIONS_DATA[region]['districts'][district]
        
        # Generate detection
        feature = generate_detection(
            region,
            district,
            district_info,
            detection_date,
            confidence_bias
        )
        
        features.append(feature)
    
    # Create GeoJSON FeatureCollection
    geojson = {
        "type": "FeatureCollection",
        "metadata": {
            "generated": datetime.now().isoformat(),
            "generator": "FAMM Demo Data Generator v1.0",
            "purpose": "Demo/Testing",
            "total_features": len(features)
        },
        "features": features
    }
    
    return geojson

def save_demo_data(geojson, filepath='data/geojson/latest_detections.geojson'):
    """Save GeoJSON to file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    print(f"✅ Demo data saved to: {filepath}")
    print(f"   Total detections: {len(geojson['features'])}")
    
    # Print summary statistics
    high_risk = sum(1 for f in geojson['features'] if f['properties']['alert_level'] == 'HIGH')
    medium_risk = sum(1 for f in geojson['features'] if f['properties']['alert_level'] == 'MEDIUM')
    low_risk = sum(1 for f in geojson['features'] if f['properties']['alert_level'] == 'LOW')
    
    print(f"\n   Alert Level Breakdown:")
    print(f"   - HIGH: {high_risk} ({high_risk/len(geojson['features'])*100:.1f}%)")
    print(f"   - MEDIUM: {medium_risk} ({medium_risk/len(geojson['features'])*100:.1f}%)")
    print(f"   - LOW: {low_risk} ({low_risk/len(geojson['features'])*100:.1f}%)")
    
    # Print regional distribution
    print(f"\n   Regional Distribution:")
    for region in REGIONS_DATA.keys():
        count = sum(1 for f in geojson['features'] if f['properties']['region'] == region)
        print(f"   - {region}: {count} ({count/len(geojson['features'])*100:.1f}%)")
    
    # Print date range
    dates = [f['properties']['date'] for f in geojson['features']]
    print(f"\n   Date Range: {min(dates)} to {max(dates)}")

def main():
    """Main function"""
    print("="*60)
    print("FAMM Demo Data Generator")
    print("="*60)
    print()
    
    # Generate demo data
    print("Generating realistic demo data...")
    geojson = generate_demo_data(
        num_detections=45,  # Good number for demo
        days_back=30,       # Last 30 days
        high_risk_ratio=0.22,   # 22% high risk
        medium_risk_ratio=0.53  # 53% medium risk
    )
    
    # Save to file
    save_demo_data(geojson)
    
    print()
    print("="*60)
    print("✅ Demo data ready!")
    print("   Run the dashboard: streamlit run streamlit_app.py")
    print("="*60)

if __name__ == '__main__':
    main()
