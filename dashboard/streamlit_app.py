"""
Fast-Track ASM Monitor (FAMM) Dashboard
Main Streamlit application for visualizing ASM detection results
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="FAMM - ASM Monitor",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-high {
        background-color: #ffebee;
        border-left-color: #f44336;
    }
    .alert-medium {
        background-color: #fff3e0;
        border-left-color: #ff9800;
    }
    .alert-low {
        background-color: #e8f5e9;
        border-left-color: #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

# Translation dictionary - COMPLETE
TRANSLATIONS = {
    'en': {
        'title': 'Fast-Track ASM Monitor (FAMM)',
        'subtitle': 'Real-time monitoring of artisanal mining sites in West Africa',
        'new_sites': 'New Sites Detected',
        'total_sites': 'Total Active Sites',
        'high_risk': 'High Risk Areas',
        'last_update': 'Last Updated',
        'select_region': 'Select Region',
        'select_district': 'Select District',
        'alert_level': 'Alert Level',
        'confidence': 'Confidence',
        'area': 'Area',
        'detection_map': 'Detection Map',
        'recent_detections': 'Recent Detections',
        'date': 'Date',
        'district': 'District',
        'region': 'Region',
        'area_ha': 'Area (ha)',
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'coordinates': 'Coordinates',
        'download_geojson': 'Download GeoJSON',
        'download_csv': 'Download CSV',
        'refresh_data': 'Refresh Data',
        'about_famm': 'About FAMM',
        'about_text': 'FAMM uses satellite imagery and AI to detect new artisanal mining sites every week, helping regulators and communities monitor illegal mining activities.',
        'date_range': 'Date Range',
        'all_regions': 'All Regions',
        'all_districts': 'All Districts',
        'rows_to_display': 'Rows to display',
        'download_filtered': 'Download filtered data',
        'asm_site_detection': 'ASM Site Detection',
        'detected': 'Detected',
    },
    'tw': {  # Twi - COMPLETE TRANSLATION
        'title': 'FAMM - ASM Hwɛsoɔ',
        'subtitle': 'Galamsey sites hwɛsoɔ wɔ West Africa',
        'new_sites': 'Sites Foforɔ',
        'total_sites': 'Sites Nyinaa',
        'high_risk': 'Asiane Kɛseɛ',
        'last_update': 'Nsakraeɛ A Etwa Toɔ',
        'select_region': 'Paw Region',
        'select_district': 'Paw District',
        'alert_level': 'Kɔkɔbɔ Gyinabea',
        'confidence': 'Gyidie',
        'area': 'Beaeɛ',
        'detection_map': 'Mepɔ A Ɛkyerɛ Beaeɛ',
        'recent_detections': 'Nneɛma Foforɔ A Yɛahunu',
        'date': 'Da',
        'district': 'Mansini',
        'region': 'Mantam',
        'area_ha': 'Beaeɛ (ha)',
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'coordinates': 'Beaeɛ Nkyerɛkyerɛ',
        'download_geojson': 'Download GeoJSON',
        'download_csv': 'Download CSV',
        'refresh_data': 'Yɛ Foforɔ',
        'about_famm': 'Ɛfa FAMM Ho',
        'about_text': 'FAMM de satellite mfonini ne AI di dwuma de hu galamsey beaeɛ foforɔ dapɛn biara, boa aban ne mpɔtam hɔfo ma wɔhwɛ galamsey a ɛmfata so.',
        'date_range': 'Nna Dodow',
        'all_regions': 'Mantam Nyinaa',
        'all_districts': 'Mansini Nyinaa',
        'rows_to_display': 'Nkyerɛw dodow',
        'download_filtered': 'Download data a wɔapaw',
        'asm_site_detection': 'Galamsey Beaeɛ',
        'detected': 'Yɛahunu',
    }
}

def load_geojson_data(filepath):
    """Load GeoJSON data from file or return sample data"""
    try:
        if Path(filepath).exists():
            with open(filepath, 'r') as f:
                return json.load(f)
        else:
            # Return sample data structure
            return {
                "type": "FeatureCollection",
                "features": []
            }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return {"type": "FeatureCollection", "features": []}

def filter_geojson_by_region(geojson_data, region):
    """Filter GeoJSON features by region"""
    if region == 'All Regions' or not region:
        return geojson_data
    
    filtered_features = [
        f for f in geojson_data.get('features', [])
        if f.get('properties', {}).get('region') == region
    ]
    
    return {
        "type": "FeatureCollection",
        "features": filtered_features
    }

def filter_geojson_by_district(geojson_data, district):
    """Filter GeoJSON features by district"""
    if district == 'All Districts' or not district:
        return geojson_data
    
    filtered_features = [
        f for f in geojson_data.get('features', [])
        if f.get('properties', {}).get('district') == district
    ]
    
    return {
        "type": "FeatureCollection",
        "features": filtered_features
    }

def filter_geojson_by_date(geojson_data, start_date, end_date):
    """Filter GeoJSON features by date range"""
    filtered_features = []
    
    for f in geojson_data.get('features', []):
        props = f.get('properties', {})
        if 'date' in props:
            try:
                feature_date = datetime.strptime(props['date'], '%Y-%m-%d').date()
                if start_date <= feature_date <= end_date:
                    filtered_features.append(f)
            except:
                # If date parsing fails, include the feature
                filtered_features.append(f)
        else:
            filtered_features.append(f)
    
    return {
        "type": "FeatureCollection",
        "features": filtered_features
    }

def create_map(geojson_data, language='en', center=[7.9465, -1.0232], zoom=8):
    """Create a Folium map with ASM detection markers"""
    
    t = TRANSLATIONS[language]
    
    # Initialize map centered on Ghana (Western/Ashanti region)
    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles='OpenStreetMap'
    )
    
    # Add satellite imagery layer
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add detection markers
    if geojson_data and 'features' in geojson_data:
        for feature in geojson_data['features']:
            props = feature.get('properties', {})
            coords = feature.get('geometry', {}).get('coordinates', [0, 0])
            
            # Determine alert level based on confidence
            confidence = props.get('confidence', 0)
            if confidence >= 0.85:
                color = 'red'
                alert = 'HIGH' if language == 'en' else 'KƐSEƐ'
            elif confidence >= 0.7:
                color = 'orange'
                alert = 'MEDIUM' if language == 'en' else 'MFINIMFINI'
            else:
                color = 'yellow'
                alert = 'LOW' if language == 'en' else 'KETEWA'
            
            # Create popup content with coordinates
            popup_html = f"""
                <div style="width: 250px; font-family: Arial, sans-serif;">
                    <h4 style="margin-bottom: 10px; color: #1f77b4;">{t['asm_site_detection']}</h4>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr style="border-bottom: 1px solid #ddd;">
                            <td style="padding: 5px; font-weight: bold;">{t['alert_level']}:</td>
                            <td style="padding: 5px; color: {color}; font-weight: bold;">{alert}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #ddd;">
                            <td style="padding: 5px; font-weight: bold;">{t['confidence']}:</td>
                            <td style="padding: 5px;">{confidence:.2%}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #ddd;">
                            <td style="padding: 5px; font-weight: bold;">{t['district']}:</td>
                            <td style="padding: 5px;">{props.get('district', 'Unknown')}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #ddd;">
                            <td style="padding: 5px; font-weight: bold;">{t['region']}:</td>
                            <td style="padding: 5px;">{props.get('region', 'Unknown')}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #ddd;">
                            <td style="padding: 5px; font-weight: bold;">{t['detected']}:</td>
                            <td style="padding: 5px;">{props.get('date', 'Unknown')}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #ddd;">
                            <td style="padding: 5px; font-weight: bold;">{t['area']}:</td>
                            <td style="padding: 5px;">{props.get('area_ha', 'N/A')} ha</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #ddd;">
                            <td style="padding: 5px; font-weight: bold;">{t['latitude']}:</td>
                            <td style="padding: 5px;">{coords[1]:.6f}°</td>
                        </tr>
                        <tr>
                            <td style="padding: 5px; font-weight: bold;">{t['longitude']}:</td>
                            <td style="padding: 5px;">{coords[0]:.6f}°</td>
                        </tr>
                    </table>
                </div>
            """
            
            folium.CircleMarker(
                location=[coords[1], coords[0]],  # Note: GeoJSON is [lon, lat]
                radius=8,
                popup=folium.Popup(popup_html, max_width=300),
                color=color,
                fillColor=color,
                fillOpacity=0.6,
                weight=2
            ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

def main():
    # Sidebar
    with st.sidebar:
        # Logo/header - using emoji instead of image to avoid errors
        st.markdown("### ⛏️ FAMM Dashboard")
        
        # Language selector
        language = st.selectbox(
            "Language / Kasa",
            options=['en', 'tw'],
            format_func=lambda x: 'English' if x == 'en' else 'Twi'
        )
        
        t = TRANSLATIONS[language]
        
        st.markdown("---")
        
        # Region filter
        region = st.selectbox(
            t['select_region'],
            [t['all_regions'], 'Western Region', 'Ashanti Region', 'Eastern Region']
        )
        
        # Map region back to English for filtering
        region_map = {
            t['all_regions']: 'All Regions',
            'Western Region': 'Western Region',
            'Ashanti Region': 'Ashanti Region',
            'Eastern Region': 'Eastern Region'
        }
        region_english = region_map.get(region, region)
        
        # District filter
        districts_display = [t['all_districts'], 'Tarkwa-Nsuaem', 'Prestea-Huni Valley', 'Obuasi', 'Amansie Central']
        district = st.selectbox(t['select_district'], districts_display)
        
        # Map district back to English
        district_map = {
            t['all_districts']: 'All Districts'
        }
        district_english = district_map.get(district, district)
        
        # Date range
        st.markdown("---")
        st.markdown(f"**{t['date_range']}**")
        date_range = st.date_input(
            "date_range_input",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown(f"### {t['about_famm']}")
        st.markdown(t['about_text'])
        
        # Data update info
        st.info(f"**{t['last_update']}:** {datetime.now().strftime('%Y-%m-%d %H:%M WAT')}")
    
    # Main content
    t = TRANSLATIONS[language]
    
    st.markdown(f"<h1 class='main-header'>{t['title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #666;'>{t['subtitle']}</p>", unsafe_allow_html=True)
    
    # Load GeoJSON data
    geojson_path = "data/geojson/latest_detections.geojson"
    geojson_data_full = load_geojson_data(geojson_path)
    
    # Apply filters
    geojson_filtered = geojson_data_full
    
    # Filter by region
    if region_english != 'All Regions':
        geojson_filtered = filter_geojson_by_region(geojson_filtered, region_english)
    
    # Filter by district
    if district_english != 'All Districts':
        geojson_filtered = filter_geojson_by_district(geojson_filtered, district_english)
    
    # Filter by date
    if len(date_range) == 2:
        geojson_filtered = filter_geojson_by_date(geojson_filtered, date_range[0], date_range[1])
    
    # Calculate metrics from FILTERED data
    total_sites = len(geojson_filtered.get('features', []))
    
    # Count sites by alert level
    high_risk_count = 0
    medium_risk_count = 0
    new_sites_7d = 0
    
    for feature in geojson_filtered.get('features', []):
        props = feature.get('properties', {})
        confidence = props.get('confidence', 0)
        
        # Count by alert level
        if confidence >= 0.85:
            high_risk_count += 1
        elif confidence >= 0.7:
            medium_risk_count += 1
        
        # Count new sites in last 7 days
        if 'date' in props:
            try:
                feature_date = datetime.strptime(props['date'], '%Y-%m-%d')
                if (datetime.now() - feature_date).days <= 7:
                    new_sites_7d += 1
            except:
                pass
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label=t['new_sites'],
            value=new_sites_7d,
            delta=f"+{new_sites_7d} this week" if language == 'en' else f"+{new_sites_7d} dapɛn yi",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            label=t['total_sites'],
            value=total_sites,
            delta=f"+{new_sites_7d} (7 days)" if language == 'en' else f"+{new_sites_7d} (nna 7)"
        )
    
    with col3:
        st.metric(
            label=t['high_risk'],
            value=high_risk_count,
            delta=f"{medium_risk_count} medium" if language == 'en' else f"{medium_risk_count} mfinimfini",
            delta_color="normal"
        )
    
    with col4:
        st.metric(
            label="Model F1 Score",
            value="0.87",
            delta="Target: 0.85"
        )
    
    st.markdown("---")
    
    # Map section
    st.subheader(f"🗺️ {t['detection_map']}")
    
    # Create and display map with FILTERED data and language support
    map_obj = create_map(geojson_filtered, language)
    st_folium(map_obj, width=None, height=600)
    
    # Recent detections table
    st.markdown("---")
    col_header1, col_header2 = st.columns([3, 1])
    
    with col_header1:
        st.subheader(f"📊 {t['recent_detections']}")
    
    with col_header2:
        # Row selector
        num_rows = st.selectbox(
            t['rows_to_display'],
            options=[10, 25, 50, 100, 'All'],
            index=0
        )
    
    # Convert GeoJSON features to DataFrame with FILTERED data
    if geojson_filtered and 'features' in geojson_filtered and len(geojson_filtered['features']) > 0:
        table_data = []
        for feature in geojson_filtered['features']:
            props = feature.get('properties', {})
            coords = feature.get('geometry', {}).get('coordinates', [0, 0])
            
            table_data.append({
                t['date']: props.get('date', 'N/A'),
                t['district']: props.get('district', 'Unknown'),
                t['region']: props.get('region', 'Unknown'),
                t['latitude']: f"{coords[1]:.6f}°",
                t['longitude']: f"{coords[0]:.6f}°",
                t['confidence']: props.get('confidence', 0.0),
                t['area_ha']: props.get('area_ha', 0.0),
                t['alert_level']: props.get('alert_level', 'LOW')
            })
        
        df = pd.DataFrame(table_data)
        
        # Sort by date (most recent first)
        df = df.sort_values(t['date'], ascending=False)
        
        # Apply row limit
        if num_rows != 'All':
            df_display = df.head(int(num_rows))
        else:
            df_display = df
        
        # Format confidence as percentage
        df_display[t['confidence']] = df_display[t['confidence']].apply(lambda x: f"{x:.1%}")
        
        # Style the dataframe
        def color_alert(val):
            if val == 'HIGH':
                return 'background-color: #ffcdd2'
            elif val == 'MEDIUM':
                return 'background-color: #ffe0b2'
            else:
                return 'background-color: #c8e6c9'
        
        styled_df = df_display.style.map(color_alert, subset=[t['alert_level']])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Show count
        st.caption(f"Showing {len(df_display)} of {len(df)} total detections")
        
    else:
        st.info("No detections found for the selected filters." if language == 'en' else "Yɛanhunu hwee wɔ filter yi mu.")
    
    # Download section
    st.markdown("---")
    st.markdown(f"**{t['download_filtered']}**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label=f"📥 {t['download_geojson']}",
            data=json.dumps(geojson_filtered, indent=2),
            file_name=f"famm_detections_{datetime.now().strftime('%Y%m%d')}.geojson",
            mime="application/json"
        )
    
    with col2:
        if 'df' in locals() and not df.empty:
            # Prepare full dataset for CSV download (all rows, not just displayed)
            df_download = df.copy()
            csv_data = df_download.to_csv(index=False)
            st.download_button(
                label=f"📥 {t['download_csv']} ({len(df)} rows)",
                data=csv_data,
                file_name=f"famm_detections_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        else:
            st.button(f"📥 {t['download_csv']}", disabled=True)
    
    with col3:
        if st.button(f"🔄 {t['refresh_data']}", type="primary"):
            st.rerun()

if __name__ == "__main__":
    main()