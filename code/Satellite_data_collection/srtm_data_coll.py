import ee
import geopandas as gpd
import pandas as pd
from pathlib import Path
import json
import numpy as np
import time

# ------------------------------
# Configuration
# ------------------------------
# Paths
SHAPEFILE_PATH = Path("/Users/prabhatrawal/code-day2/polygon_file/actual_timezone_designated_district_using_EPSG_32644.shp")
GEE_KEY_PATH = Path("/Users/prabhatrawal/Minor_project_code/keys/gee_project_id.txt")
OUTPUT_BASE = Path("/Users/prabhatrawal/Minor_project_code/data/srtm_data")

# Create output directory
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("SRTM Elevation Data Extraction from Google Earth Engine")
print("=" * 60)

# ------------------------------
# Initialize Google Earth Engine
# ------------------------------
print("\n[1/5] Initializing Google Earth Engine...")
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
print("\n[2/5] Loading shapefile and creating zones...")
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
print("\n[3/5] Converting geometries to GEE format...")

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
# Load SRTM datasets
# ------------------------------
print("\n[4/5] Loading SRTM datasets from GEE...")

# Load both SRTM datasets
srtm_usgs = ee.Image("USGS/SRTMGL1_003").select('elevation')
srtm_mtpi = ee.Image("CSP/ERGo/1_0/Global/SRTM_mTPI").select('elevation')

print("✓ Loaded SRTM datasets:")
print("  1. USGS/SRTMGL1_003 - Standard elevation (30m resolution)")
print("  2. CSP/ERGo SRTM_mTPI - Multi-scale Topographic Position Index")

# Calculate additional terrain metrics from USGS SRTM
terrain = ee.Terrain.products(srtm_usgs)

print("✓ Calculated terrain derivatives:")
print("  - Slope (degrees)")
print("  - Aspect (degrees)")
print("  - Hillshade")

# ------------------------------
# Extract SRTM data for a zone
# ------------------------------
def extract_zone_data(zone_geometry, zone_id, district_name):
    """Extract SRTM elevation and terrain data for a specific zone"""
    try:
        # Convert zone geometry to EE
        ee_geom = shapely_to_ee_geometry(zone_geometry, gdf.crs)
        
        # Extract elevation statistics from USGS SRTM
        usgs_stats = srtm_usgs.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                reducer2=ee.Reducer.min(),
                sharedInputs=True
            ).combine(
                reducer2=ee.Reducer.max(),
                sharedInputs=True
            ).combine(
                reducer2=ee.Reducer.stdDev(),
                sharedInputs=True
            ).combine(
                reducer2=ee.Reducer.median(),
                sharedInputs=True
            ),
            geometry=ee_geom,
            scale=30,  # 30m resolution
            maxPixels=1e9
        ).getInfo()
        
        # Extract mTPI statistics
        mtpi_stats = srtm_mtpi.reduceRegion(
            reducer=ee.Reducer.mean().combine(
                reducer2=ee.Reducer.min(),
                sharedInputs=True
            ).combine(
                reducer2=ee.Reducer.max(),
                sharedInputs=True
            ).combine(
                reducer2=ee.Reducer.stdDev(),
                sharedInputs=True
            ),
            geometry=ee_geom,
            scale=30,
            maxPixels=1e9
        ).getInfo()
        
        # Extract terrain statistics (slope, aspect)
        terrain_stats = terrain.select(['slope', 'aspect']).reduceRegion(
            reducer=ee.Reducer.mean().combine(
                reducer2=ee.Reducer.min(),
                sharedInputs=True
            ).combine(
                reducer2=ee.Reducer.max(),
                sharedInputs=True
            ).combine(
                reducer2=ee.Reducer.stdDev(),
                sharedInputs=True
            ),
            geometry=ee_geom,
            scale=30,
            maxPixels=1e9
        ).getInfo()
        
        # Calculate elevation percentiles for better understanding
        elevation_percentiles = srtm_usgs.reduceRegion(
            reducer=ee.Reducer.percentile([10, 25, 50, 75, 90]),
            geometry=ee_geom,
            scale=30,
            maxPixels=1e9
        ).getInfo()
        
        # Compile all statistics
        record = {
            'zone': zone_id,
            'district': district_name,
            # USGS SRTM Elevation
            'elevation_mean_m': round(usgs_stats.get('elevation_mean', 0), 2),
            'elevation_min_m': round(usgs_stats.get('elevation_min', 0), 2),
            'elevation_max_m': round(usgs_stats.get('elevation_max', 0), 2),
            'elevation_median_m': round(usgs_stats.get('elevation_median', 0), 2),
            'elevation_stddev_m': round(usgs_stats.get('elevation_stdDev', 0), 2),
            'elevation_range_m': round(usgs_stats.get('elevation_max', 0) - usgs_stats.get('elevation_min', 0), 2),
            # Elevation percentiles
            'elevation_p10_m': round(elevation_percentiles.get('elevation_p10', 0), 2),
            'elevation_p25_m': round(elevation_percentiles.get('elevation_p25', 0), 2),
            'elevation_p75_m': round(elevation_percentiles.get('elevation_p75', 0), 2),
            'elevation_p90_m': round(elevation_percentiles.get('elevation_p90', 0), 2),
            # mTPI (Topographic Position Index)
            'mtpi_mean': round(mtpi_stats.get('elevation_mean', 0), 2),
            'mtpi_min': round(mtpi_stats.get('elevation_min', 0), 2),
            'mtpi_max': round(mtpi_stats.get('elevation_max', 0), 2),
            'mtpi_stddev': round(mtpi_stats.get('elevation_stdDev', 0), 2),
            # Terrain derivatives
            'slope_mean_deg': round(terrain_stats.get('slope_mean', 0), 2),
            'slope_min_deg': round(terrain_stats.get('slope_min', 0), 2),
            'slope_max_deg': round(terrain_stats.get('slope_max', 0), 2),
            'slope_stddev_deg': round(terrain_stats.get('slope_stdDev', 0), 2),
            'aspect_mean_deg': round(terrain_stats.get('aspect_mean', 0), 2),
            'aspect_stddev_deg': round(terrain_stats.get('aspect_stdDev', 0), 2),
        }
        
        return record
        
    except Exception as e:
        print(f"  ✗ Error extracting data for zone {zone_id}: {e}")
        return None

# ------------------------------
# Process all districts and zones
# ------------------------------
print("\n[5/5] Extracting SRTM data for all zones...")

all_data = []

for district_name, zones_gdf in district_zones.items():
    print(f"\n{'='*60}")
    print(f"Processing District: {district_name}")
    print(f"{'='*60}")
    
    # Create district folder
    district_folder = OUTPUT_BASE / district_name.replace(' ', '_')
    district_folder.mkdir(exist_ok=True)
    
    district_records = []
    
    for idx, zone_row in zones_gdf.iterrows():
        zone_id = zone_row['zone']
        zone_geom = zone_row['geometry']
        
        print(f"  Zone {zone_id}/{len(zones_gdf)}...", end=' ')
        
        # Extract data
        record = extract_zone_data(zone_geom, zone_id, district_name)
        
        if record:
            district_records.append(record)
            all_data.append(record)
            print(f"✓ Elev: {record['elevation_mean_m']}m (range: {record['elevation_min_m']}-{record['elevation_max_m']}m), Slope: {record['slope_mean_deg']}°")
        else:
            print("✗ Failed")
        
        time.sleep(0.1)  # Small delay to avoid rate limits
    
    # Save district data
    if district_records:
        district_df = pd.DataFrame(district_records)
        
        # Save combined district file
        output_file = district_folder / f"{district_name.replace(' ', '_')}_srtm_data.csv"
        district_df.to_csv(output_file, index=False)
        
        print(f"\n✓ Saved {len(district_df)} zones to {output_file}")
        
        # Create summary
        print(f"\n  District Summary:")
        print(f"    Elevation: {district_df['elevation_mean_m'].mean():.1f}m avg, "
              f"{district_df['elevation_min_m'].min():.1f}-{district_df['elevation_max_m'].max():.1f}m range")
        print(f"    Slope: {district_df['slope_mean_deg'].mean():.1f}° avg, "
              f"max {district_df['slope_max_deg'].max():.1f}°")
    else:
        print(f"\n⚠ No data extracted for {district_name}")

# ------------------------------
# Save combined dataset
# ------------------------------
if all_data:
    print(f"\n{'='*60}")
    print("Saving combined dataset...")
    print(f"{'='*60}")
    
    all_df = pd.DataFrame(all_data)
    combined_file = OUTPUT_BASE / "all_districts_srtm_data.csv"
    all_df.to_csv(combined_file, index=False)
    
    print(f"✓ Saved all {len(all_df)} zones to {combined_file}")
    
    # Create overall summary by district
    summary = all_df.groupby('district').agg({
        'elevation_mean_m': ['mean', 'min', 'max'],
        'elevation_range_m': 'mean',
        'slope_mean_deg': ['mean', 'max'],
        'mtpi_mean': 'mean',
        'zone': 'count'
    }).round(2)
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    summary_file = OUTPUT_BASE / "district_summary.csv"
    summary.to_csv(summary_file)
    print(f"✓ Saved district summary to {summary_file}")

# ------------------------------
# Final Summary
# ------------------------------
print("\n" + "="*60)
print("EXTRACTION COMPLETE!")
print("="*60)
print(f"\nData saved to: {OUTPUT_BASE}")
print("\nFolder structure:")
print("  srtm_data/")
print("    ├── all_districts_srtm_data.csv (all zones combined)")
print("    ├── district_summary.csv (summary by district)")
for district_name in district_zones.keys():
    print(f"    ├── {district_name.replace(' ', '_')}/")
    print(f"    │   └── {district_name.replace(' ', '_')}_srtm_data.csv")

print("\n" + "="*60)
print("Data Description:")
print("="*60)
print("Elevation Metrics (from USGS SRTM):")
print("  - elevation_mean/min/max/median/stddev_m: Basic statistics")
print("  - elevation_range_m: Difference between max and min")
print("  - elevation_p10/p25/p75/p90_m: Percentiles")
print("\nTopographic Position Index (from CSP/ERGo):")
print("  - mtpi_mean/min/max/stddev: Multi-scale TPI")
print("  - Positive values = ridges/peaks, Negative = valleys")
print("\nTerrain Derivatives:")
print("  - slope_mean/min/max/stddev_deg: Terrain slope")
print("  - aspect_mean/stddev_deg: Direction of slope")
print("\nResolution: 30 meters (SRTM native)")
print("\n✓ All done!")