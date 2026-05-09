"""
Utility functions for FAMM Dashboard
"""

import ee
import geemap
import geopandas as gpd
import pandas as pd
import json
from datetime import datetime, timedelta
from pathlib import Path
import config

def initialize_earth_engine():
    """Initialize Earth Engine (must be authenticated first)"""
    try:
        ee.Initialize()
        return True
    except Exception as e:
        print(f"Earth Engine initialization failed: {e}")
        return False

def load_latest_detections(filepath='data/geojson/latest_detections.geojson'):
    """Load the latest detection results from GeoJSON"""
    try:
        if Path(filepath).exists():
            gdf = gpd.read_file(filepath)
            return gdf
        else:
            # Return empty GeoDataFrame with expected schema
            return gpd.GeoDataFrame(columns=['geometry', 'confidence', 'district', 'date', 'area_ha'])
    except Exception as e:
        print(f"Error loading detections: {e}")
        return gpd.GeoDataFrame(columns=['geometry', 'confidence', 'district', 'date', 'area_ha'])

def filter_by_date(gdf, start_date, end_date):
    """Filter GeoDataFrame by date range"""
    if 'date' in gdf.columns:
        gdf['date'] = pd.to_datetime(gdf['date'])
        mask = (gdf['date'] >= pd.to_datetime(start_date)) & (gdf['date'] <= pd.to_datetime(end_date))
        return gdf[mask]
    return gdf

def filter_by_region(gdf, region):
    """Filter GeoDataFrame by region"""
    if region == 'All Regions' or 'region' not in gdf.columns:
        return gdf
    return gdf[gdf['region'] == region]

def filter_by_district(gdf, district):
    """Filter GeoDataFrame by district"""
    if district == 'All Districts' or 'district' not in gdf.columns:
        return gdf
    return gdf[gdf['district'] == district]

def calculate_metrics(gdf):
    """Calculate summary metrics from detection data"""
    metrics = {
        'total_sites': len(gdf),
        'new_sites_7d': 0,
        'high_risk_sites': 0,
        'total_area_ha': 0
    }
    
    if len(gdf) == 0:
        return metrics
    
    # Count new sites in last 7 days
    if 'date' in gdf.columns:
        week_ago = datetime.now() - timedelta(days=7)
        gdf['date'] = pd.to_datetime(gdf['date'])
        metrics['new_sites_7d'] = len(gdf[gdf['date'] >= week_ago])
    
    # Count high-risk sites
    if 'confidence' in gdf.columns:
        metrics['high_risk_sites'] = len(gdf[gdf['confidence'] >= config.CONFIDENCE_THRESHOLDS['HIGH']])
    
    # Calculate total area
    if 'area_ha' in gdf.columns:
        metrics['total_area_ha'] = gdf['area_ha'].sum()
    
    return metrics

def get_alert_level(confidence):
    """Determine alert level based on confidence score"""
    if confidence >= config.CONFIDENCE_THRESHOLDS['HIGH']:
        return 'HIGH'
    elif confidence >= config.CONFIDENCE_THRESHOLDS['MEDIUM']:
        return 'MEDIUM'
    else:
        return 'LOW'

def export_to_geojson(gdf, filepath):
    """Export GeoDataFrame to GeoJSON file"""
    try:
        gdf.to_file(filepath, driver='GeoJSON')
        return True
    except Exception as e:
        print(f"Error exporting to GeoJSON: {e}")
        return False

def create_sample_geojson():
    """Create sample GeoJSON for testing"""
    sample_features = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [-2.0, 6.0]
                },
                "properties": {
                    "confidence": 0.92,
                    "district": "Tarkwa-Nsuaem",
                    "region": "Western Region",
                    "date": "2025-01-28",
                    "area_ha": 2.3
                }
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [-1.8, 6.2]
                },
                "properties": {
                    "confidence": 0.88,
                    "district": "Obuasi",
                    "region": "Ashanti Region",
                    "date": "2025-01-27",
                    "area_ha": 1.8
                }
            },
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [-2.1, 5.9]
                },
                "properties": {
                    "confidence": 0.85,
                    "district": "Prestea-Huni Valley",
                    "region": "Western Region",
                    "date": "2025-01-26",
                    "area_ha": 3.1
                }
            }
        ]
    }
    return sample_features

def run_weekly_detection(aoi_geometry, start_date, end_date):
    """
    Run weekly detection using Earth Engine (placeholder)
    This should integrate with your actual MobileNet model
    """
    # TODO: Implement actual Earth Engine + model inference pipeline
    # This is a placeholder that returns sample data
    
    print("Running weekly detection...")
    print(f"AOI: {aoi_geometry}")
    print(f"Date range: {start_date} to {end_date}")
    
    # In production, this would:
    # 1. Query Sentinel-2 imagery for the date range
    # 2. Run your MobileNet classifier on tiles
    # 3. Generate GeoJSON with detections
    # 4. Return the results
    
    return create_sample_geojson()
