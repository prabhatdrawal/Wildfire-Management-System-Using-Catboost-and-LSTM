import ee
import geopandas as gpd
import pandas as pd
from pathlib import Path
import json
from datetime import datetime, timedelta
import time
import math

# ------------------------------
# Configuration
# ------------------------------
# Paths
SHAPEFILE_PATH = Path("/Users/prabhatrawal/Minor_project_code/polygon_file/actual_timezone_designated_district_using_EPSG_32644.shp")
GEE_KEY_PATH = Path("/Users/prabhatrawal/Minor_project_code/keys/gee_project_id.txt")
OUTPUT_BASE = Path("/Users/prabhatrawal/Minor_project_code/data/era5_data")

# Date range - ERA5-Land daily aggregated starts from 1950, but let's use 2000-2025
START_YEAR = 2000
END_YEAR = 2025

# Create output directory
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("ERA5-Land Climate Data Extraction from Google Earth Engine")
print("=" * 60)
print("OPTIMIZED VERSION - Processing year by year")
print(f"Date range: {START_YEAR} to {END_YEAR}")

# ------------------------------
# Initialize Google Earth Engine
# ------------------------------
print("\n[1/6] Initializing Google Earth Engine...")
try:
    # Read project ID from file
    with open(GEE_KEY_PATH, 'r') as f:
        project_id = f.read().strip()
    
    # Initialize Earth Engine
    ee.Initialize(project=project_id)
    print(f"✓ GEE initialized with project: {project_id}")
except Exception as e:
    print(f"✗ Error initializing GEE: {e}")
    print("Please run 'earthengine authenticate' in terminal first")
    exit(1)

# ------------------------------
# Load and Process Shapefile
# ------------------------------
print("\n[2/6] Loading shapefile and creating zones...")
gdf = gpd.read_file(SHAPEFILE_PATH)
print(f"✓ Loaded shapefile with {len(gdf)} districts")

# Import zone creation code
from shapely.geometry import box
from shapely.ops import unary_union

union_poly = gdf.geometry.union_all()
minx, miny, maxx, maxy = union_poly.bounds
width = maxx - minx
height = maxy - miny
total_area = gdf.geometry.area.sum()

n_districts = len(gdf)
target_n_total = 10 * n_districts
cell_size = math.sqrt(total_area / target_n_total)

n_cols = math.ceil(width / cell_size)
n_rows = math.ceil(height / cell_size)
dx = width / n_cols
dy = height / n_rows

grid_boxes = []
for i in range(n_cols):
    for j in range(n_rows):
        xmin = minx + i * dx
        xmax = xmin + dx
        ymin = miny + j * dy
        ymax = ymin + dy
        grid_boxes.append(box(xmin, ymin, xmax, ymax))

def assign_and_merge_zones(district_poly, grid_boxes, target_n=10, merge_threshold=0.25):
    zones = []
    for gb in grid_boxes:
        inter = gb.intersection(district_poly)
        if not inter.is_empty:
            zones.append({'geometry': inter})
    
    if not zones:
        return None
    
    zones_gdf = gpd.GeoDataFrame(zones, crs=gdf.crs)
    zones_gdf['area'] = zones_gdf.geometry.area
    
    district_area = district_poly.area
    target_area = district_area / target_n
    threshold_area = merge_threshold * target_area
    
    merged = True
    while merged:
        merged = False
        small_zones = zones_gdf[zones_gdf['area'] < threshold_area].copy()
        if small_zones.empty:
            break
        
        for idx, small_row in small_zones.iterrows():
            if idx not in zones_gdf.index:
                continue
            
            touches = zones_gdf.geometry.touches(small_row.geometry)
            neighbors = zones_gdf[touches & (zones_gdf.index != idx)]
            
            if neighbors.empty:
                continue
            
            smallest_neighbor = neighbors.loc[neighbors['area'].idxmin()]
            smallest_idx = smallest_neighbor.name
            
            merged_geom = unary_union([small_row.geometry, smallest_neighbor.geometry])
            
            zones_gdf.at[smallest_idx, 'geometry'] = merged_geom
            zones_gdf.at[smallest_idx, 'area'] = merged_geom.area
            
            zones_gdf = zones_gdf.drop(idx)
            
            merged = True
            break
    
    zones_gdf = zones_gdf.reset_index(drop=True)
    zones_gdf['zone'] = zones_gdf.index + 1
    
    return zones_gdf

# Process each district
district_zones = {}
for idx, row in gdf.iterrows():
    district_name = row['NAME'] if 'NAME' in gdf.columns else f'District_{idx}'
    polygon = row['geometry']
    zones_gdf = assign_and_merge_zones(polygon, grid_boxes)
    if zones_gdf is not None:
        zones_gdf['district'] = district_name
        district_zones[district_name] = zones_gdf
        print(f"  ✓ {district_name}: {len(zones_gdf)} zones")

# ------------------------------
# Convert geometries to GEE format
# ------------------------------
print("\n[3/6] Converting geometries to GEE format...")

def shapely_to_ee_geometry(shapely_geom, source_crs):
    """Convert Shapely geometry to Earth Engine geometry"""
    # Reproject to WGS84 (EPSG:4326) for GEE
    gdf_temp = gpd.GeoDataFrame([1], geometry=[shapely_geom], crs=source_crs)
    gdf_wgs84 = gdf_temp.to_crs('EPSG:4326')
    geom_wgs84 = gdf_wgs84.geometry.iloc[0]
    
    # Convert to GEE format
    if geom_wgs84.geom_type == 'Polygon':
        coords = [list(geom_wgs84.exterior.coords)]
        return ee.Geometry.Polygon(coords)
    elif geom_wgs84.geom_type == 'MultiPolygon':
        all_coords = []
        for poly in geom_wgs84.geoms:
            all_coords.append(list(poly.exterior.coords))
        return ee.Geometry.MultiPolygon(all_coords)
    else:
        raise ValueError(f"Unsupported geometry type: {geom_wgs84.geom_type}")

# ------------------------------
# ERA5-Land processing functions
# ------------------------------
print("\n[4/6] Setting up ERA5-Land processing...")

def calculate_derived_variables(image):
    """Calculate derived meteorological variables"""
    # Get bands
    temp_2m = image.select('temperature_2m')
    dewpoint = image.select('dewpoint_temperature_2m')
    u_wind = image.select('u_component_of_wind_10m')
    v_wind = image.select('v_component_of_wind_10m')
    
    # Wind speed = sqrt(u^2 + v^2)
    wind_speed = u_wind.pow(2).add(v_wind.pow(2)).sqrt().rename('wind_speed_10m')
    
    # Wind direction = atan2(u, v) * 180/pi + 180
    wind_direction = u_wind.atan2(v_wind).multiply(180).divide(math.pi).add(180).rename('wind_direction_10m')
    
    # Relative Humidity approximation using Magnus formula
    # RH = 100 * exp((17.625 * Td) / (243.04 + Td)) / exp((17.625 * T) / (243.04 + T))
    # where T and Td are in Celsius
    temp_celsius = temp_2m.subtract(273.15)
    dewpoint_celsius = dewpoint.subtract(273.15)
    
    rh = dewpoint_celsius.multiply(17.625).divide(dewpoint_celsius.add(243.04)).exp() \
        .divide(temp_celsius.multiply(17.625).divide(temp_celsius.add(243.04)).exp()) \
        .multiply(100).rename('relative_humidity')
    
    # Vapor Pressure Deficit (VPD) - important for fire risk!
    # VPD = es - ea, where es is saturation vapor pressure and ea is actual vapor pressure
    es = temp_celsius.multiply(17.625).divide(temp_celsius.add(243.04)).exp().multiply(0.6108)
    ea = dewpoint_celsius.multiply(17.625).divide(dewpoint_celsius.add(243.04)).exp().multiply(0.6108)
    vpd = es.subtract(ea).rename('vapor_pressure_deficit_kpa')
    
    return image.addBands([wind_speed, wind_direction, rh, vpd])

print("✓ ERA5-Land processing functions ready")
print("  Bands to extract:")
print("    - temperature_2m (K)")
print("    - dewpoint_temperature_2m (K)")
print("    - skin_temperature (K)")
print("    - soil_temperature_level_1 (K)")
print("    - volumetric_soil_water_layer_1 (m³/m³)")
print("    - total_precipitation_sum (m)")
print("    - u_component_of_wind_10m (m/s)")
print("    - v_component_of_wind_10m (m/s)")
print("    - surface_pressure (Pa)")
print("  Derived variables:")
print("    - wind_speed_10m (m/s)")
print("    - wind_direction_10m (degrees)")
print("    - relative_humidity (%)")
print("    - vapor_pressure_deficit_kpa (kPa)")

# ------------------------------
# OPTIMIZED extraction function
# ------------------------------
def extract_zone_data_optimized(zone_geometry, zone_id, district_name, year):
    """Extract ERA5-Land data for a specific zone and year - OPTIMIZED"""
    try:
        # Convert zone geometry to EE
        ee_geom = shapely_to_ee_geometry(zone_geometry, gdf.crs)
        
        # Date range for this year
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31' if year < 2025 else '2025-01-19'
        
        # Load ERA5-Land collection
        era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR") \
            .filterDate(start_date, end_date) \
            .filterBounds(ee_geom) \
            .select([
                'temperature_2m',
                'dewpoint_temperature_2m',
                'skin_temperature',
                'soil_temperature_level_1',
                'volumetric_soil_water_layer_1',
                'total_precipitation_sum',
                'u_component_of_wind_10m',
                'v_component_of_wind_10m',
                'surface_pressure'
            ])
        
        # Get image count
        count = era5.size().getInfo()
        
        if count == 0:
            return pd.DataFrame()
        
        # Calculate derived variables
        era5_processed = era5.map(calculate_derived_variables)
        
        # OPTIMIZED: Process all images at once using server-side operations
        def process_image(img):
            date = ee.Date(img.get('system:time_start'))
            
            # Reduce region to get mean values
            stats = img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee_geom,
                scale=11132,  # ~11km native resolution for ERA5-Land
                maxPixels=1e9
            )
            
            # Return feature with all info
            return ee.Feature(None, {
                'date': date.format('YYYY-MM-dd'),
                'temp_2m_k': stats.get('temperature_2m'),
                'dewpoint_2m_k': stats.get('dewpoint_temperature_2m'),
                'skin_temp_k': stats.get('skin_temperature'),
                'soil_temp_k': stats.get('soil_temperature_level_1'),
                'soil_moisture': stats.get('volumetric_soil_water_layer_1'),
                'precip_m': stats.get('total_precipitation_sum'),
                'u_wind_ms': stats.get('u_component_of_wind_10m'),
                'v_wind_ms': stats.get('v_component_of_wind_10m'),
                'pressure_pa': stats.get('surface_pressure'),
                'wind_speed_ms': stats.get('wind_speed_10m'),
                'wind_dir_deg': stats.get('wind_direction_10m'),
                'rel_humidity_pct': stats.get('relative_humidity'),
                'vpd_kpa': stats.get('vapor_pressure_deficit_kpa')
            })
        
        # Map over all images
        features = era5_processed.map(process_image)
        
        # Get all data at once
        feature_list = features.toList(count).getInfo()
        
        # Convert to DataFrame
        records = []
        for feature in feature_list:
            props = feature['properties']
            
            # Convert Kelvin to Celsius for temperature
            temp_2m_c = props.get('temp_2m_k', 273.15) - 273.15
            dewpoint_c = props.get('dewpoint_2m_k', 273.15) - 273.15
            skin_temp_c = props.get('skin_temp_k', 273.15) - 273.15
            soil_temp_c = props.get('soil_temp_k', 273.15) - 273.15
            
            # Convert precipitation from m to mm
            precip_mm = props.get('precip_m', 0) * 1000
            
            # Convert pressure from Pa to hPa
            pressure_hpa = props.get('pressure_pa', 0) / 100
            
            records.append({
                'date': props['date'],
                'zone': zone_id,
                'district': district_name,
                'temperature_2m_celsius': round(temp_2m_c, 2),
                'dewpoint_2m_celsius': round(dewpoint_c, 2),
                'skin_temperature_celsius': round(skin_temp_c, 2),
                'soil_temperature_celsius': round(soil_temp_c, 2),
                'soil_moisture_m3m3': round(props.get('soil_moisture', 0), 4),
                'precipitation_mm': round(precip_mm, 2),
                'u_wind_component_ms': round(props.get('u_wind_ms', 0), 2),
                'v_wind_component_ms': round(props.get('v_wind_ms', 0), 2),
                'wind_speed_ms': round(props.get('wind_speed_ms', 0), 2),
                'wind_direction_deg': round(props.get('wind_dir_deg', 0), 1),
                'surface_pressure_hpa': round(pressure_hpa, 2),
                'relative_humidity_pct': round(props.get('rel_humidity_pct', 0), 1),
                'vapor_pressure_deficit_kpa': round(props.get('vpd_kpa', 0), 3)
            })
        
        if records:
            df = pd.DataFrame(records)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"  ✗ Error extracting data for zone {zone_id}, year {year}: {e}")
        return pd.DataFrame()

# ------------------------------
# Process all districts and zones - BY YEAR
# ------------------------------
print("\n[5/6] Extracting ERA5-Land data for all zones (year by year)...")

for district_name, zones_gdf in district_zones.items():
    print(f"\n{'='*60}")
    print(f"Processing District: {district_name}")
    print(f"{'='*60}")
    
    # Create district folder
    district_folder = OUTPUT_BASE / district_name.replace(' ', '_')
    district_folder.mkdir(exist_ok=True)
    
    # Store all data for this district
    all_district_data = []
    
    # Process each zone
    for idx, zone_row in zones_gdf.iterrows():
        zone_id = zone_row['zone']
        zone_geom = zone_row['geometry']
        
        print(f"\n  Zone {zone_id}/{len(zones_gdf)}:")
        
        zone_all_years_data = []
        
        # Process year by year
        for year in range(START_YEAR, END_YEAR + 1):
            try:
                print(f"    Year {year}...", end=' ')
                year_df = extract_zone_data_optimized(zone_geom, zone_id, district_name, year)
                
                if not year_df.empty:
                    zone_all_years_data.append(year_df)
                    avg_temp = year_df['temperature_2m_celsius'].mean()
                    total_precip = year_df['precipitation_mm'].sum()
                    print(f"✓ {len(year_df)} days (avg temp: {avg_temp:.1f}°C, precip: {total_precip:.1f}mm)")
                else:
                    print("✓ No data")
                
                time.sleep(0.3)  # Small delay between years
                
            except Exception as e:
                print(f"✗ Error: {e}")
                continue
        
        # Combine all years for this zone
        if zone_all_years_data:
            zone_complete_df = pd.concat(zone_all_years_data, ignore_index=True)
            all_district_data.append(zone_complete_df)
            
            # Save individual zone file
            zone_file = district_folder / f"zone_{zone_id}_data.csv"
            zone_complete_df.to_csv(zone_file, index=False)
            
            avg_temp = zone_complete_df['temperature_2m_celsius'].mean()
            total_precip = zone_complete_df['precipitation_mm'].sum()
            print(f"  ✓ Zone {zone_id} complete: {len(zone_complete_df)} days (avg: {avg_temp:.1f}°C, total precip: {total_precip:.0f}mm)")
    
    # Combine all zones for this district
    if all_district_data:
        district_df = pd.concat(all_district_data, ignore_index=True)
        
        # Save combined district file
        output_file = district_folder / f"{district_name.replace(' ', '_')}_era5_data.csv"
        district_df.to_csv(output_file, index=False)
        
        print(f"\n✓ District complete: {len(district_df)} total records saved to {output_file}")
        
        # Create summary statistics
        summary = district_df.groupby('zone').agg({
            'temperature_2m_celsius': ['mean', 'min', 'max', 'std'],
            'precipitation_mm': ['sum', 'mean', 'max'],
            'soil_moisture_m3m3': ['mean', 'min', 'max'],
            'wind_speed_ms': ['mean', 'max'],
            'relative_humidity_pct': 'mean',
            'vapor_pressure_deficit_kpa': 'mean',
            'date': 'count'
        }).round(2)
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary_file = district_folder / f"{district_name.replace(' ', '_')}_summary.csv"
        summary.to_csv(summary_file)
        print(f"✓ Summary statistics saved to {summary_file}")
    else:
        print(f"\n⚠ No data extracted for {district_name}")

# ------------------------------
# Summary
# ------------------------------
print("\n" + "="*60)
print("[6/6] EXTRACTION COMPLETE!")
print("="*60)
print(f"\nData saved to: {OUTPUT_BASE}")
print("\nFolder structure:")
print("  era5_data/")
for district_name in district_zones.keys():
    print(f"    ├── {district_name.replace(' ', '_')}/")
    print(f"    │   ├── {district_name.replace(' ', '_')}_era5_data.csv")
    print(f"    │   ├── {district_name.replace(' ', '_')}_summary.csv")
    print(f"    │   └── zone_*.csv")

print("\n" + "="*60)
print("Data Description:")
print("="*60)
print("Temperature Variables (converted to Celsius):")
print("  - temperature_2m_celsius: Air temperature at 2m")
print("  - dewpoint_2m_celsius: Dewpoint temperature")
print("  - skin_temperature_celsius: Surface/skin temperature")
print("  - soil_temperature_celsius: Soil temperature (top layer)")
print("\nMoisture & Precipitation:")
print("  - soil_moisture_m3m3: Volumetric soil water content")
print("  - precipitation_mm: Total daily precipitation (converted to mm)")
print("\nWind Variables:")
print("  - u_wind_component_ms, v_wind_component_ms: Wind components")
print("  - wind_speed_ms: Calculated wind speed")
print("  - wind_direction_deg: Wind direction (0-360°)")
print("\nPressure & Humidity:")
print("  - surface_pressure_hpa: Surface pressure (hPa)")
print("  - relative_humidity_pct: Calculated relative humidity")
print("  - vapor_pressure_deficit_kpa: VPD (important for fire risk!)")
print("\nResolution: ~11km (ERA5-Land native)")
print("Temporal: Daily aggregated values")
print("\n✓ All done!")