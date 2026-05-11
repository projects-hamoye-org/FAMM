"""
Configuration settings for FAMM Dashboard
"""

# Map settings
MAP_CENTER_GHANA = [7.9465, -1.0232]  # Western/Ashanti region
DEFAULT_ZOOM = 8

# Regions and districts
REGIONS = {
    'Western Region': {
        'center': [5.5557, -2.6957],
        'districts': ['Tarkwa-Nsuaem', 'Prestea-Huni Valley', 'Wassa East', 'Bibiani-Anhwiaso-Bekwai']
    },
    'Ashanti Region': {
        'center': [6.7460, -1.5220],
        'districts': ['Obuasi', 'Amansie Central', 'Amansie West', 'Adansi South']
    },
    'Eastern Region': {
        'center': [6.3696, -0.4275],
        'districts': ['Atiwa West', 'Birim North', 'Denkyembour', 'West Akim']
    }
}

# Alert thresholds
CONFIDENCE_THRESHOLDS = {
    'HIGH': 0.85,
    'MEDIUM': 0.70,
    'LOW': 0.0
}

# Colors for map markers
ALERT_COLORS = {
    'HIGH': '#f44336',     # Red
    'MEDIUM': '#ff9800',   # Orange
    'LOW': '#fdd835'       # Yellow
}

# Model settings
MODEL_F1_TARGET = 0.85
MODEL_FP_TARGET = 0.05  # 5% false positive rate

# Data paths (update these based on your actual structure)
GEOJSON_DIR = 'data/geojson'
MODEL_DIR = 'models'

# Earth Engine settings
EE_PROJECT_ID = 'your-project-id'  # Update with your actual project ID
EE_COLLECTION = 'COPERNICUS/S2_SR'  # Sentinel-2 Surface Reflectance

# Update frequency
UPDATE_INTERVAL_DAYS = 7  # Weekly updates
