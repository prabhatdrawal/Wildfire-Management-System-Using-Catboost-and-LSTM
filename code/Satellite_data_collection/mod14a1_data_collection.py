import ee
import geopandas as gpd
import pandas as pd
from pathlib import Path
import json
from datetime import datetime, timedelta
import time

# ------------------------------
# Configuration
# ------------------------------
# Paths
SHAPEFILE_PATH = Path("/Users/prabhatrawal/Minor_project_code/polygon_file/actual_timezone_designated_district_using_EPSG_32644.shp")
GEE_KEY_PATH = Path("/Users/prabhatrawal/Minor_project_code/keys/gee_project_id.txt")
OUTPUT_BASE = Path("/Users/prabhatrawal/Minor_project_code/data/mod14a1_data")

# Date range - PROCESS BY YEAR to avoid timeouts
START_YEAR = 2000
END_YEAR = 2025

# Create output directory
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("MOD14A1 Fire Mask Data Extraction from Google Earth Engine")
print("=" * 60)
print("OPTIMIZED VERSION - Processing year by year")

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
import math
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
# Extract MOD14A1 data - OPTIMIZED
# ------------------------------
print("\n[4/6] Loading MOD14A1 collection...")
print("✓ MOD14A1 collection: MODIS/061/MOD14A1")
print("  Fire mask values: 7=Low confidence, 8=Nominal confidence, 9=High confidence")

# ------------------------------
# OPTIMIZED extraction function
# ------------------------------
def extract_zone_data_optimized(zone_geometry, zone_id, district_name, year):
    """Extract MOD14A1 FireMask data for a specific zone and year - OPTIMIZED"""
    try:
        # Convert zone geometry to EE
        ee_geom = shapely_to_ee_geometry(zone_geometry, gdf.crs)
        
        # Date range for this year
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31' if year < 2025 else '2025-01-19'
        
        # Load collection for this year
        mod14a1 = ee.ImageCollection("MODIS/061/MOD14A1").select(['FireMask'])
        zone_collection = mod14a1.filterDate(start_date, end_date).filterBounds(ee_geom)
        
        # Get image count
        count = zone_collection.size().getInfo()
        
        if count == 0:
            return pd.DataFrame()
        
        # OPTIMIZED: Process all images at once using server-side operations
        def process_image(img):
            date = ee.Date(img.get('system:time_start'))
            fire_mask = img.select('FireMask')
            
            # Create masks for different fire confidence levels
            low_conf = fire_mask.eq(7)
            nominal_conf = fire_mask.eq(8)
            high_conf = fire_mask.eq(9)
            any_fire = fire_mask.gte(7).And(fire_mask.lte(9))
            
            # Count pixels (using 1km scale)
            stats = ee.Image.cat([
                low_conf,
                nominal_conf,
                high_conf,
                any_fire,
                ee.Image.constant(1)
            ]).rename([
                'low_conf',
                'nominal_conf', 
                'high_conf',
                'any_fire',
                'total'
            ]).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=ee_geom,
                scale=1000,
                maxPixels=1e9
            )
            
            # Return feature with all info
            return ee.Feature(None, {
                'date': date.format('YYYY-MM-dd'),
                'low_conf': stats.get('low_conf'),
                'nominal_conf': stats.get('nominal_conf'),
                'high_conf': stats.get('high_conf'),
                'total_fire': stats.get('any_fire'),
                'total_pixels': stats.get('total')
            })
        
        # Map over all images
        features = zone_collection.map(process_image)
        
        # Get all data at once
        feature_list = features.toList(count).getInfo()
        
        # Convert to DataFrame
        records = []
        for feature in feature_list:
            props = feature['properties']
            total_pixels = props.get('total_pixels', 0)
            fire_pixels = props.get('total_fire', 0)
            fire_percentage = (fire_pixels / total_pixels * 100) if total_pixels > 0 else 0
            
            records.append({
                'date': props['date'],
                'zone': zone_id,
                'district': district_name,
                'low_confidence_fire_pixels': props.get('low_conf', 0),
                'nominal_confidence_fire_pixels': props.get('nominal_conf', 0),
                'high_confidence_fire_pixels': props.get('high_conf', 0),
                'total_fire_pixels': fire_pixels,
                'total_pixels': total_pixels,
                'fire_percentage': round(fire_percentage, 4)
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
print("\n[5/6] Extracting MOD14A1 data for all zones (year by year)...")

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
                    fire_days = len(year_df[year_df['total_fire_pixels'] > 0])
                    print(f"✓ {len(year_df)} days ({fire_days} with fire)")
                else:
                    print("✓ No data")
                
                time.sleep(0.2)  # Small delay between years
                
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
            
            total_fire_days = len(zone_complete_df[zone_complete_df['total_fire_pixels'] > 0])
            print(f"  ✓ Zone {zone_id} complete: {len(zone_complete_df)} total days, {total_fire_days} with fire")
    
    # Combine all zones for this district
    if all_district_data:
        district_df = pd.concat(all_district_data, ignore_index=True)
        
        # Save combined district file
        output_file = district_folder / f"{district_name.replace(' ', '_')}_mod14a1_data.csv"
        district_df.to_csv(output_file, index=False)
        
        print(f"\n✓ District complete: {len(district_df)} total records saved to {output_file}")
        
        # Create summary statistics
        summary = district_df.groupby('zone').agg({
            'total_fire_pixels': ['sum', 'mean', 'max'],
            'low_confidence_fire_pixels': 'sum',
            'nominal_confidence_fire_pixels': 'sum',
            'high_confidence_fire_pixels': 'sum',
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
print("  mod14a1_data/")
for district_name in district_zones.keys():
    print(f"    ├── {district_name.replace(' ', '_')}/")
    print(f"    │   ├── {district_name.replace(' ', '_')}_mod14a1_data.csv")
    print(f"    │   ├── {district_name.replace(' ', '_')}_summary.csv")
    print(f"    │   └── zone_*.csv")
print("\n✓ All done!")