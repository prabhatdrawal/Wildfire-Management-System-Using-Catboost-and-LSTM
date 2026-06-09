"""
Interactive Wildfire Risk Assessment Web Application
Select district, zone, and input environmental parameters for real-time fire risk prediction

to run the cod e :streamlit run /Users/prabhatrawal/Minor_project_code/code/streamlit_ffopm.py 
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import geopandas as gpd
from datetime import datetime, timedelta
import sys
import math
from shapely.geometry import box
from shapely.ops import unary_union
import folium
from streamlit_folium import st_folium

# Add the deployment module path
sys.path.append('/Users/prabhatrawal/Minor_project_code/code')
from catboost_deployment_tuned import CatBoostPredictor

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_DIR = Path("/Users/prabhatrawal/Minor_project_code/data/integrated_data/catboost_tuning")
POLYGON_PATH = Path("/Users/prabhatrawal/Minor_project_code/polygon_file/actual_timezone_designated_district_using_EPSG_32644.shp")

# District mapping
DISTRICT_MAPPING = {
    'District_0': 'Banke',
    'District_1': 'Bardiya',
    'District_2': 'Surkhet',
    'District_3': 'Dang',
    'District_4': 'Salyan'
}

# Zone counts per district
ZONE_COUNTS = {
    'Banke': 14,
    'Bardiya': 15,
    'Surkhet': 19,
    'Dang': 17,
    'Salyan': 14
}

# Top 20 most important features (from your output)
TOP_FEATURES = [
    'vapor_pressure_deficit_kpa_roll14_mean',
    'dewpoint_2m_celsius',
    'mtpi_min',
    'clear_day_coverage',
    'relative_humidity_pct',
    'landsat_savi',
    'vapor_pressure_deficit_kpa',
    'lst_day_c',
    'relative_humidity_pct_lag1',
    'precipitation_mm_lag1',
    'relative_humidity_pct_lag3',
    'aspect_stddev_deg',
    'slope_stddev_deg',
    'slope_max_deg',
    'vapor_pressure_deficit_kpa_lag7',
    'relative_humidity_pct_lag7',
    'precipitation_mm_lag5',
    'elevation_range_m',
    'mtpi_mean',
    'vapor_pressure_deficit_kpa_lag1'
]

# Feature metadata for better UX
FEATURE_METADATA = {
    # Weather - Current
    'dewpoint_2m_celsius': {
        'display_name': '🌡️ Dew Point Temperature',
        'unit': '°C',
        'min': -20, 'max': 35, 'default': 15,
        'help': 'Temperature at which air becomes saturated with moisture'
    },
    'lst_day_c': {
        'display_name': '🌡️ Land Surface Temperature (Day)',
        'unit': '°C',
        'min': 0, 'max': 60, 'default': 30,
        'help': 'Temperature of the land surface during daytime'
    },
    'relative_humidity_pct': {
        'display_name': '💧 Relative Humidity',
        'unit': '%',
        'min': 0, 'max': 100, 'default': 50,
        'help': 'Current moisture content in air'
    },
    'vapor_pressure_deficit_kpa': {
        'display_name': '🌫️ Vapor Pressure Deficit',
        'unit': 'kPa',
        'min': 0, 'max': 8, 'default': 2,
        'help': 'Difference between actual and saturated vapor pressure (dryness indicator)'
    },
    
    # Weather - Historical (Lag features)
    'relative_humidity_pct_lag1': {
        'display_name': '💧 Humidity (Yesterday)',
        'unit': '%',
        'min': 0, 'max': 100, 'default': 50,
        'help': 'Relative humidity from 1 day ago'
    },
    'relative_humidity_pct_lag3': {
        'display_name': '💧 Humidity (3 Days Ago)',
        'unit': '%',
        'min': 0, 'max': 100, 'default': 50,
        'help': 'Relative humidity from 3 days ago'
    },
    'relative_humidity_pct_lag7': {
        'display_name': '💧 Humidity (7 Days Ago)',
        'unit': '%',
        'min': 0, 'max': 100, 'default': 50,
        'help': 'Relative humidity from 7 days ago'
    },
    'precipitation_mm_lag1': {
        'display_name': '🌧️ Rainfall (Yesterday)',
        'unit': 'mm',
        'min': 0, 'max': 200, 'default': 0,
        'help': 'Precipitation from 1 day ago'
    },
    'precipitation_mm_lag5': {
        'display_name': '🌧️ Rainfall (5 Days Ago)',
        'unit': 'mm',
        'min': 0, 'max': 200, 'default': 0,
        'help': 'Precipitation from 5 days ago'
    },
    'vapor_pressure_deficit_kpa_lag1': {
        'display_name': '🌫️ VPD (Yesterday)',
        'unit': 'kPa',
        'min': 0, 'max': 8, 'default': 2,
        'help': 'Vapor pressure deficit from 1 day ago'
    },
    'vapor_pressure_deficit_kpa_lag7': {
        'display_name': '🌫️ VPD (7 Days Ago)',
        'unit': 'kPa',
        'min': 0, 'max': 8, 'default': 2,
        'help': 'Vapor pressure deficit from 7 days ago'
    },
    
    # Weather - Rolling averages
    'vapor_pressure_deficit_kpa_roll14_mean': {
        'display_name': '🌫️ VPD (14-Day Average)',
        'unit': 'kPa',
        'min': 0, 'max': 8, 'default': 2,
        'help': 'Mean vapor pressure deficit over past 14 days'
    },
    
    # Vegetation
    'landsat_savi': {
        'display_name': '🌿 Soil Adjusted Vegetation Index',
        'unit': '',
        'min': -1, 'max': 1, 'default': 0.5,
        'help': 'Vegetation health indicator (higher = healthier vegetation)'
    },
    
    # Terrain
    'mtpi_min': {
        'display_name': '⛰️ Terrain Position (Minimum)',
        'unit': '',
        'min': -500, 'max': 500, 'default': 0,
        'help': 'Minimum multi-scale topographic position index'
    },
    'mtpi_mean': {
        'display_name': '⛰️ Terrain Position (Mean)',
        'unit': '',
        'min': -500, 'max': 500, 'default': 0,
        'help': 'Mean multi-scale topographic position index'
    },
    'aspect_stddev_deg': {
        'display_name': '🧭 Aspect Variability',
        'unit': '°',
        'min': 0, 'max': 180, 'default': 45,
        'help': 'Variability in slope direction'
    },
    'slope_stddev_deg': {
        'display_name': '📐 Slope Variability',
        'unit': '°',
        'min': 0, 'max': 45, 'default': 10,
        'help': 'Variability in terrain steepness'
    },
    'slope_max_deg': {
        'display_name': '📐 Maximum Slope',
        'unit': '°',
        'min': 0, 'max': 90, 'default': 30,
        'help': 'Steepest slope in the area'
    },
    'elevation_range_m': {
        'display_name': '🏔️ Elevation Range',
        'unit': 'm',
        'min': 0, 'max': 2000, 'default': 200,
        'help': 'Difference between highest and lowest elevations'
    },
    
    # Other
    'clear_day_coverage': {
        'display_name': '☀️ Clear Sky Coverage (Day)',
        'unit': '%',
        'min': 0, 'max': 100, 'default': 70,
        'help': 'Percentage of clear sky during daytime'
    }
}

# ============================================================================
# CACHING & INITIALIZATION
# ============================================================================

@st.cache_resource
def load_model():
    """Load the CatBoost model (cached)"""
    return CatBoostPredictor.load(MODEL_DIR)

@st.cache_data
def load_polygon_data():
    """Load district polygon data (cached)"""
    try:
        gdf = gpd.read_file(POLYGON_PATH)
        return gdf
    except Exception as e:
        st.error(f"Could not load polygon file: {e}")
        return None

@st.cache_data
def generate_zones_for_all_districts():
    """Generate zones for all districts using uniform grid (cached)"""
    gdf = load_polygon_data()
    if gdf is None:
        return None
    
    # Compute union and grid parameters
    union_poly = gdf.geometry.unary_union
    minx, miny, maxx, maxy = union_poly.bounds
    width = maxx - minx
    height = maxy - miny
    total_area = gdf.geometry.area.sum()
    
    # Target approximately 10 zones per district
    n_districts = len(gdf)
    target_n_total = 10 * n_districts
    cell_size = math.sqrt(total_area / target_n_total)
    
    # Number of columns and rows for the uniform grid
    n_cols = math.ceil(width / cell_size)
    n_rows = math.ceil(height / cell_size)
    dx = width / n_cols
    dy = height / n_rows
    
    # Create uniform grid boxes
    grid_boxes = []
    for i in range(n_cols):
        for j in range(n_rows):
            xmin = minx + i * dx
            xmax = xmin + dx
            ymin = miny + j * dy
            ymax = ymin + dy
            grid_boxes.append(box(xmin, ymin, xmax, ymax))
    
    # Process each district
    district_zones = {}
    for idx, row in gdf.iterrows():
        district_name = row['NAME'] if 'NAME' in gdf.columns else f'District_{idx}'
        polygon = row['geometry']
        zones_gdf = assign_and_merge_zones(polygon, grid_boxes, gdf.crs)
        if zones_gdf is not None:
            zones_gdf['district'] = district_name
            district_zones[district_name] = zones_gdf
    
    # Combine all zones
    if district_zones:
        all_zones_gdf = gpd.pd.concat(district_zones.values(), ignore_index=True)
        return all_zones_gdf, district_zones
    
    return None, None

def assign_and_merge_zones(district_poly, grid_boxes, crs, target_n=10, merge_threshold=0.25):
    """
    Assigns parts of the uniform grid to the district by intersecting, then merges small zones.
    """
    zones = []
    for gb in grid_boxes:
        inter = gb.intersection(district_poly)
        if not inter.is_empty:
            zones.append({'geometry': inter})
    
    if not zones:
        return None
    
    zones_gdf = gpd.GeoDataFrame(zones, crs=crs)
    zones_gdf['area'] = zones_gdf.geometry.area
    
    district_area = district_poly.area
    target_area = district_area / target_n
    threshold_area = merge_threshold * target_area
    
    # Merge loop
    merged = True
    while merged:
        merged = False
        small_zones = zones_gdf[zones_gdf['area'] < threshold_area].copy()
        if small_zones.empty:
            break
        
        for idx, small_row in small_zones.iterrows():
            if idx not in zones_gdf.index:
                continue
            
            # Find touching neighbors
            touches = zones_gdf.geometry.touches(small_row.geometry)
            neighbors = zones_gdf[touches & (zones_gdf.index != idx)]
            
            if neighbors.empty:
                continue
            
            # Select the neighbor with the smallest area
            smallest_neighbor = neighbors.loc[neighbors['area'].idxmin()]
            smallest_idx = smallest_neighbor.name
            
            # Merge geometries
            merged_geom = unary_union([small_row.geometry, smallest_neighbor.geometry])
            
            # Update the smallest neighbor
            zones_gdf.at[smallest_idx, 'geometry'] = merged_geom
            zones_gdf.at[smallest_idx, 'area'] = merged_geom.area
            
            # Remove the small zone
            zones_gdf = zones_gdf.drop(idx)
            
            merged = True
            break  # Restart after each merge
    
    # Renumber zones
    zones_gdf = zones_gdf.reset_index(drop=True)
    zones_gdf['zone'] = zones_gdf.index + 1
    
    return zones_gdf

def create_interactive_map(all_zones_gdf, district_zones, selected_district=None, selected_zone=None):
    """Create an interactive Folium map with clickable zones"""
    
    # Convert to WGS84 for folium
    all_zones_wgs84 = all_zones_gdf.to_crs(epsg=4326)
    
    # Calculate center
    bounds = all_zones_wgs84.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=9,
        tiles='OpenStreetMap'
    )
    
    # Add zones with colors and click functionality
    for idx, row in all_zones_wgs84.iterrows():
        district = row['district']
        zone = row['zone']
        
        # Color based on selection
        if selected_district and district == selected_district:
            if selected_zone and zone == selected_zone:
                fill_color = '#DC143C'  # Red for selected zone
                fill_opacity = 0.7
            else:
                fill_color = '#FFA500'  # Orange for district zones
                fill_opacity = 0.5
        else:
            fill_color = '#90EE90'  # Light green for other zones
            fill_opacity = 0.3
        
        # Create polygon
        folium.GeoJson(
            row['geometry'].__geo_interface__,
            style_function=lambda x, fc=fill_color, fo=fill_opacity: {
                'fillColor': fc,
                'color': 'black',
                'weight': 1,
                'fillOpacity': fo
            },
            tooltip=f"<b>{district}</b><br>Zone {zone}<br>Click to select",
            popup=folium.Popup(
                f"""
                <div style='width: 150px'>
                    <h4>{district}</h4>
                    <p><b>Zone:</b> {zone}</p>
                    <p><b>Area:</b> {row['area']/1e6:.2f} km²</p>
                </div>
                """,
                max_width=200
            )
        ).add_to(m)
    
    # Add district boundaries
    gdf = load_polygon_data()
    if gdf is not None:
        gdf_wgs84 = gdf.to_crs(epsg=4326)
        folium.GeoJson(
            gdf_wgs84,
            style_function=lambda x: {
                'color': 'black',
                'weight': 3,
                'fillOpacity': 0
            }
        ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 200px; height: 130px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
        <p><b>Zone Selection</b></p>
        <p><span style="background-color: #DC143C; padding: 3px 10px;">▮</span> Selected Zone</p>
        <p><span style="background-color: #FFA500; padding: 3px 10px;">▮</span> Selected District</p>
        <p><span style="background-color: #90EE90; padding: 3px 10px;">▮</span> Other Zones</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_default_feature_values():
    """Get default values for all 81 features"""
    # This should match the exact feature list from your model
    # For now, we'll use zeros/defaults for features not in top 20
    defaults = {}
    
    # All features with safe defaults
    all_features = [
        'lst_day_c', 's2_ndvi', 's2_gndvi', 's2_ndwi', 's2_ndsi', 's2_evi', 's2_savi',
        's2_cloud_cover_percent', 'landsat_ndvi', 'landsat_gndvi', 'landsat_nbr',
        'landsat_savi', 'veg_data_quality', 'dewpoint_2m_celsius', 'skin_temperature_celsius',
        'soil_moisture_m3m3', 'precipitation_mm', 'u_wind_component_ms', 'v_wind_component_ms',
        'wind_speed_ms', 'wind_direction_deg', 'relative_humidity_pct',
        'vapor_pressure_deficit_kpa', 'elevation_min_m', 'elevation_stddev_m',
        'elevation_range_m', 'mtpi_mean', 'mtpi_min', 'mtpi_max', 'mtpi_stddev',
        'slope_min_deg', 'slope_max_deg', 'slope_stddev_deg', 'aspect_mean_deg',
        'aspect_stddev_deg', 'lst_missing_flag', 's2_ndvi_lag1', 's2_ndvi_lag3',
        's2_ndvi_lag7', 's2_ndvi_lag14', 'landsat_ndvi_lag1', 'landsat_ndvi_lag3',
        'landsat_ndvi_lag7', 'landsat_ndvi_lag14', 'precipitation_mm_lag1',
        'precipitation_mm_lag5', 'precipitation_mm_lag10', 'precipitation_mm_lag30',
        'vapor_pressure_deficit_kpa_lag1', 'vapor_pressure_deficit_kpa_lag3',
        'vapor_pressure_deficit_kpa_lag7', 'vapor_pressure_deficit_kpa_lag14',
        'soil_temperature_celsius_lag1', 'soil_temperature_celsius_lag3',
        'soil_temperature_celsius_lag7', 'soil_moisture_m3m3_lag1', 'soil_moisture_m3m3_lag3',
        'soil_moisture_m3m3_lag7', 'soil_moisture_m3m3_lag14', 'temperature_2m_celsius_lag1',
        'temperature_2m_celsius_lag3', 'temperature_2m_celsius_lag7',
        'skin_temperature_celsius_lag1', 'skin_temperature_celsius_lag3',
        'skin_temperature_celsius_lag7', 'relative_humidity_pct_lag1',
        'relative_humidity_pct_lag3', 'relative_humidity_pct_lag7',
        'precipitation_mm_roll7_sum', 'precipitation_mm_roll14_sum',
        'precipitation_mm_roll30_sum', 's2_ndvi_roll7_mean', 's2_ndvi_roll14_mean',
        'landsat_ndvi_roll7_mean', 'landsat_ndvi_roll14_mean',
        'temperature_2m_celsius_roll7_mean', 'temperature_2m_celsius_roll14_mean',
        'vapor_pressure_deficit_kpa_roll7_mean', 'vapor_pressure_deficit_kpa_roll14_mean',
        'clear_day_coverage', 'clear_night_coverage'
    ]
    
    for feat in all_features:
        if feat in FEATURE_METADATA:
            defaults[feat] = FEATURE_METADATA[feat]['default']
        else:
            # Safe defaults for unlisted features
            if 'humidity' in feat.lower():
                defaults[feat] = 50
            elif 'temperature' in feat.lower() or 'lst' in feat.lower():
                defaults[feat] = 25
            elif 'precipitation' in feat.lower():
                defaults[feat] = 0
            elif 'ndvi' in feat.lower() or 'savi' in feat.lower():
                defaults[feat] = 0.5
            elif 'elevation' in feat.lower():
                defaults[feat] = 500
            elif 'slope' in feat.lower():
                defaults[feat] = 15
            elif 'aspect' in feat.lower():
                defaults[feat] = 180
            elif 'coverage' in feat.lower():
                defaults[feat] = 70
            else:
                defaults[feat] = 0
    
    return defaults

def create_risk_gauge(probability, threshold):
    """Create a gauge chart for fire risk"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Fire Risk Probability", 'font': {'size': 24}},
        delta={'reference': threshold * 100, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkred" if probability >= threshold else "orange"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#90EE90'},
                {'range': [20, 40], 'color': '#FFD700'},
                {'range': [40, 60], 'color': '#FFA500'},
                {'range': [60, 80], 'color': '#FF6347'},
                {'range': [80, 100], 'color': '#DC143C'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def get_risk_color(risk_level):
    """Get color for risk level"""
    colors = {
        'Low': '#90EE90',
        'Moderate': '#FFD700',
        'High': '#FF6347',
        'Extreme': '#DC143C'
    }
    return colors.get(risk_level, '#808080')

def get_alert_color(alert_priority):
    """Get color for alert priority"""
    colors = {
        'Monitor': '#90EE90',
        'Watch': '#FFD700',
        'Medium': '#FFA500',
        'High': '#FF6347',
        'Critical': '#DC143C'
    }
    return colors.get(alert_priority, '#808080')

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Page config
    st.set_page_config(
        page_title="Wildfire Risk Assessment",
        page_icon="🔥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            color: #DC143C;
            text-align: center;
            font-weight: bold;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #555;
            text-align: center;
            margin-bottom: 2rem;
        }
        .risk-box {
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
            text-align: center;
            font-size: 1.2rem;
            font-weight: bold;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 8px;
            margin: 5px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="main-header">🔥 Wildfire Risk Assessment System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Mid-Western Nepal - Real-time Fire Risk Prediction</p>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner('Loading AI model and generating zones...'):
        predictor = load_model()
        all_zones_gdf, district_zones = generate_zones_for_all_districts()
    
    if all_zones_gdf is None:
        st.error("Failed to load polygon data or generate zones")
        return
    
    # Initialize session state for selections
    if 'selected_district' not in st.session_state:
        st.session_state.selected_district = None
    if 'selected_zone' not in st.session_state:
        st.session_state.selected_zone = None
    
    # Sidebar - Location Selection
    st.sidebar.title("📍 Location Selection")
    
    # Selection method
    selection_method = st.sidebar.radio(
        "Selection Method",
        ["🗺️ Interactive Map", "📋 Dropdown Menu"],
        help="Choose how to select district and zone"
    )
    
    if selection_method == "📋 Dropdown Menu":
        # Traditional dropdown selection
        district_options = sorted(list(set(all_zones_gdf['district'])))
        selected_district = st.sidebar.selectbox(
            "Select District",
            district_options,
            help="Choose the district for risk assessment"
        )
        
        # Get zones for selected district
        district_data = all_zones_gdf[all_zones_gdf['district'] == selected_district]
        zone_options = sorted(district_data['zone'].unique())
        
        selected_zone = st.sidebar.selectbox(
            "Select Zone",
            zone_options,
            help=f"{selected_district} has {len(zone_options)} zones"
        )
        
        # Update session state
        st.session_state.selected_district = selected_district
        st.session_state.selected_zone = selected_zone
        
    else:
        # Map-based selection
        st.sidebar.info("👆 Click on a zone in the map below to select it")
        
        if st.session_state.selected_district:
            st.sidebar.success(f"**Selected:** {st.session_state.selected_district} - Zone {st.session_state.selected_zone}")
    
    st.sidebar.markdown("---")
    
    # Date selection
    st.sidebar.title("📅 Assessment Date")
    assessment_date = st.sidebar.date_input(
        "Select Date",
        datetime.now(),
        help="Date for the fire risk assessment"
    )
    
    st.sidebar.markdown("---")
    
    # Quick presets
    st.sidebar.title("⚡ Quick Presets")
    preset = st.sidebar.selectbox(
        "Load Scenario",
        ["Custom", "High Risk (Dry Season)", "Moderate Risk (Normal)", "Low Risk (Monsoon)"],
        help="Load pre-configured scenarios"
    )
    
    # Main content - Show map if in map mode
    if selection_method == "🗺️ Interactive Map":
        st.subheader("🗺️ Interactive Zone Selection Map")
        st.info("Click on any zone to select it for fire risk assessment")
        
        # Create and display map
        m = create_interactive_map(
            all_zones_gdf, 
            district_zones,
            st.session_state.selected_district,
            st.session_state.selected_zone
        )
        
        # Display map and capture clicks
        map_data = st_folium(m, width=1400, height=600, returned_objects=["last_object_clicked"])
        
        # Process map clicks
        if map_data and map_data.get("last_object_clicked"):
            clicked_coords = map_data["last_object_clicked"]
            if clicked_coords:
                # Convert clicked point to determine which zone was clicked
                from shapely.geometry import Point
                click_point = Point(clicked_coords["lng"], clicked_coords["lat"])
                
                # Find which zone contains this point
                all_zones_wgs84 = all_zones_gdf.to_crs(epsg=4326)
                for idx, row in all_zones_wgs84.iterrows():
                    if row['geometry'].contains(click_point):
                        st.session_state.selected_district = row['district']
                        st.session_state.selected_zone = row['zone']
                        st.rerun()
                        break
        
        st.markdown("---")
    
    # Show selected location
    if st.session_state.selected_district and st.session_state.selected_zone:
        selected_district = st.session_state.selected_district
        selected_zone = st.session_state.selected_zone
        
        # Get zone info
        zone_info = all_zones_gdf[
            (all_zones_gdf['district'] == selected_district) & 
            (all_zones_gdf['zone'] == selected_zone)
        ].iloc[0]
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("📍 District", selected_district)
        with col_info2:
            st.metric("🔢 Zone", f"Zone {selected_zone}")
        with col_info3:
            st.metric("📐 Area", f"{zone_info['area']/1e6:.2f} km²")
        
        st.markdown("---")
    else:
        st.warning("⚠️ Please select a district and zone to proceed with risk assessment")
        return
    
    # Main content area - Parameters and Results
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📊 Environmental Parameters - {selected_district}, Zone {selected_zone}")
        
        # Get default values
        feature_values = get_default_feature_values()
        
        # Apply presets
        if preset == "High Risk (Dry Season)":
            feature_values.update({
                'relative_humidity_pct': 20,
                'vapor_pressure_deficit_kpa': 5,
                'precipitation_mm_lag1': 0,
                'precipitation_mm_lag5': 0,
                'lst_day_c': 45,
                'dewpoint_2m_celsius': 5
            })
        elif preset == "Low Risk (Monsoon)":
            feature_values.update({
                'relative_humidity_pct': 85,
                'vapor_pressure_deficit_kpa': 0.5,
                'precipitation_mm_lag1': 50,
                'precipitation_mm_lag5': 30,
                'lst_day_c': 25,
                'dewpoint_2m_celsius': 20
            })
        
        # Create input form for top features
        st.markdown("#### 🌡️ Weather Conditions")
        
        weather_col1, weather_col2 = st.columns(2)
        
        with weather_col1:
            for feat in ['dewpoint_2m_celsius', 'lst_day_c', 'relative_humidity_pct', 
                        'vapor_pressure_deficit_kpa', 'clear_day_coverage']:
                if feat in FEATURE_METADATA:
                    meta = FEATURE_METADATA[feat]
                    feature_values[feat] = st.slider(
                        meta['display_name'],
                        min_value=float(meta['min']),
                        max_value=float(meta['max']),
                        value=float(feature_values[feat]),
                        help=meta['help']
                    )
        
        with weather_col2:
            for feat in ['relative_humidity_pct_lag1', 'relative_humidity_pct_lag3',
                        'relative_humidity_pct_lag7', 'precipitation_mm_lag1',
                        'precipitation_mm_lag5']:
                if feat in FEATURE_METADATA:
                    meta = FEATURE_METADATA[feat]
                    feature_values[feat] = st.slider(
                        meta['display_name'],
                        min_value=float(meta['min']),
                        max_value=float(meta['max']),
                        value=float(feature_values[feat]),
                        help=meta['help']
                    )
        
        st.markdown("#### 🌿 Vegetation & Terrain")
        
        terrain_col1, terrain_col2 = st.columns(2)
        
        with terrain_col1:
            for feat in ['landsat_savi', 'elevation_range_m', 'slope_max_deg']:
                if feat in FEATURE_METADATA:
                    meta = FEATURE_METADATA[feat]
                    feature_values[feat] = st.slider(
                        meta['display_name'],
                        min_value=float(meta['min']),
                        max_value=float(meta['max']),
                        value=float(feature_values[feat]),
                        help=meta['help']
                    )
        
        with terrain_col2:
            for feat in ['mtpi_min', 'mtpi_mean', 'aspect_stddev_deg', 'slope_stddev_deg']:
                if feat in FEATURE_METADATA:
                    meta = FEATURE_METADATA[feat]
                    feature_values[feat] = st.slider(
                        meta['display_name'],
                        min_value=float(meta['min']),
                        max_value=float(meta['max']),
                        value=float(feature_values[feat]),
                        help=meta['help']
                    )
        
        st.markdown("#### 📈 Historical Trends (14-Day)")
        
        for feat in ['vapor_pressure_deficit_kpa_roll14_mean', 'vapor_pressure_deficit_kpa_lag1',
                    'vapor_pressure_deficit_kpa_lag7']:
            if feat in FEATURE_METADATA:
                meta = FEATURE_METADATA[feat]
                feature_values[feat] = st.slider(
                    meta['display_name'],
                    min_value=float(meta['min']),
                    max_value=float(meta['max']),
                    value=float(feature_values[feat]),
                    help=meta['help']
                )
    
    with col2:
        st.subheader("🎯 Risk Assessment")
        
        # Predict button
        if st.button("🔮 Assess Fire Risk", type="primary", use_container_width=True):
            
            with st.spinner('Analyzing fire risk...'):
                # Create feature dataframe
                X_input = pd.DataFrame([feature_values])
                
                # Make prediction
                result = predictor.predict_with_risk_levels(X_input)
                
                probability = result['probability'].values[0]
                prediction = result['prediction'].values[0]
                risk_level = result['risk_level'].values[0]
                alert_priority = result['alert_priority'].values[0]
                threshold = result['threshold_used'].values[0]
                
                # Display gauge
                st.plotly_chart(
                    create_risk_gauge(probability, threshold),
                    use_container_width=True
                )
                
                # Risk level box
                risk_color = get_risk_color(risk_level)
                st.markdown(
                    f'<div class="risk-box" style="background-color: {risk_color};">'
                    f'Risk Level: {risk_level}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Alert priority box
                alert_color = get_alert_color(alert_priority)
                st.markdown(
                    f'<div class="risk-box" style="background-color: {alert_color};">'
                    f'Alert: {alert_priority}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                
                # Metrics
                st.markdown("### 📊 Detailed Metrics")
                
                st.metric("Fire Probability", f"{probability*100:.2f}%")
                st.metric("Decision Threshold", f"{threshold*100:.2f}%")
                
                if prediction == 1:
                    st.error("🔥 **FIRE RISK DETECTED**")
                else:
                    st.success("✅ **No Fire Risk**")
                
                # Recommendations
                st.markdown("### 💡 Recommendations")
                
                if alert_priority == "Critical":
                    st.error("""
                    **⚠️ CRITICAL ALERT**
                    - Immediate action required
                    - Deploy fire crews to area
                    - Establish monitoring stations
                    - Restrict public access
                    """)
                elif alert_priority == "High":
                    st.warning("""
                    **⚠️ HIGH ALERT**
                    - Enhanced monitoring required
                    - Pre-position fire resources
                    - Issue public warnings
                    - Prepare evacuation plans
                    """)
                elif alert_priority == "Medium":
                    st.info("""
                    **ℹ️ MEDIUM ALERT**
                    - Regular monitoring
                    - Review fire prevention measures
                    - Update community awareness
                    """)
                elif alert_priority == "Watch":
                    st.info("""
                    **👁️ WATCH STATUS**
                    - Monitor conditions closely
                    - Conditions approaching threshold
                    - Review preparedness plans
                    """)
                else:
                    st.success("""
                    **✅ MONITOR STATUS**
                    - Routine monitoring sufficient
                    - Maintain normal operations
                    - Continue regular patrols
                    """)
        
        else:
            st.info("👆 Set parameters and click 'Assess Fire Risk' to get prediction")
    
    # Bottom section - Feature importance
    st.markdown("---")
    st.subheader("📈 Model Feature Importance (Top 20)")
    
    importance_data = predictor.get_feature_importance(top_n=20)
    
    if importance_data is not None:
        fig = px.bar(
            importance_data,
            x='importance',
            y='feature',
            orientation='h',
            title='Feature Importance in Fire Risk Prediction',
            labels={'importance': 'Importance Score', 'feature': 'Feature'},
            color='importance',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=600, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
        <div style='text-align: center; color: #666;'>
            <p><strong>Wildfire Risk Assessment System</strong> | Powered by S-Tier CatBoost ML Model</p>
            <p>Model Performance: ROC-AUC: 0.963 | PR-AUC: 0.242 | Recall: 88.9% @ 5% FAR</p>
            <p>📍 Total Zones: {len(all_zones_gdf)} across 5 Districts (Mid-Western Nepal)</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()