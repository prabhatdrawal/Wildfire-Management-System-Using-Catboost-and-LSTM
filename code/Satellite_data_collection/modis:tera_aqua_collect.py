import ee
import geopandas as gpd
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
from shapely.geometry import box
from shapely.ops import unary_union
import warnings

# Suppress FutureWarning for concat
warnings.filterwarnings('ignore', category=FutureWarning)

# ------------------------------
# Configuration
# ------------------------------
SHAPEFILE_PATH = Path("/Users/prabhatrawal/Minor_project_code/polygon_file/actual_timezone_designated_district_using_EPSG_32644.shp")
GEE_KEY_PATH = Path("/Users/prabhatrawal/Minor_project_code/keys/gee_project_id.txt")
OUTPUT_BASE = Path("/Users/prabhatrawal/Minor_project_code/data/modis_lst_data")

START_YEAR = 2000
END_YEAR = 2025
YEARS_PER_BATCH = 5  # Process 5 years at once
MAX_WORKERS = 4  # Parallel processing threads

# Resume capability - skip already processed zones
RESUME_MODE = True  # Set to False to reprocess everything

OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("MODIS LST Data Extraction - OPTIMIZED VERSION v2")
print("=" * 70)
print(f"Date range: {START_YEAR} to {END_YEAR}")
print(f"Batch size: {YEARS_PER_BATCH} years")
print(f"Parallel workers: {MAX_WORKERS}")
print(f"Resume mode: {'ON' if RESUME_MODE else 'OFF'}")
print("=" * 70)

# ------------------------------
# Initialize Google Earth Engine
# ------------------------------
print("\n[1/6] Initializing Google Earth Engine...")
try:
    with open(GEE_KEY_PATH, 'r') as f:
        project_id = f.read().strip()
    ee.Initialize(project=project_id)
    print(f"✓ GEE initialized with project: {project_id}")
except Exception as e:
    print(f"✗ Error initializing GEE: {e}")
    exit(1)

# ------------------------------
# Load and Process Shapefile
# ------------------------------
print("\n[2/6] Loading shapefile and creating zones...")
gdf = gpd.read_file(SHAPEFILE_PATH)
print(f"✓ Loaded shapefile with {len(gdf)} districts")

# Zone creation
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
# GEE Geometry Conversion
# ------------------------------
print("\n[3/6] Converting geometries to GEE format...")

def shapely_to_ee_geometry(shapely_geom, source_crs):
    """Convert Shapely geometry to Earth Engine geometry"""
    gdf_temp = gpd.GeoDataFrame([1], geometry=[shapely_geom], crs=source_crs)
    gdf_wgs84 = gdf_temp.to_crs('EPSG:4326')
    geom_wgs84 = gdf_wgs84.geometry.iloc[0]
    
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
# OPTIMIZED MODIS LST Processing
# ------------------------------
print("\n[4/6] Setting up optimized MODIS LST processing...")
print("Collections:")
print("  - MODIS/061/MOD11A1 (Terra)")
print("  - MODIS/061/MYD11A1 (Aqua)")
print("Bands:")
print("  - LST_Day_1km, LST_Night_1km")
print("Quality Control:")
print("  - QC_Day (Bits 0-1 = 00 for good quality)")
print("  - QC_Night (Bits 0-1 = 00 for good quality)")
print("  - Clear_day_cov, Clear_night_cov")

def process_lst_collection_optimized(collection, satellite_name, ee_geom):
    """
    Ultra-optimized LST processing
    - Applies QC masking (Bits 0-1 = 00)
    - Converts Kelvin to Celsius
    - Reduces to mean per geometry
    """
    
    def process_image(img):
        # Quality masking - only keep good quality pixels (bits 0-1 = 00)
        qc_day = img.select('QC_Day')
        qc_night = img.select('QC_Night')
        
        day_quality = qc_day.bitwiseAnd(3).eq(0)
        night_quality = qc_night.bitwiseAnd(3).eq(0)
        
        # Convert to Celsius: LST * 0.02 - 273.15
        lst_day = img.select('LST_Day_1km') \
            .updateMask(day_quality) \
            .multiply(0.02) \
            .subtract(273.15)
        
        lst_night = img.select('LST_Night_1km') \
            .updateMask(night_quality) \
            .multiply(0.02) \
            .subtract(273.15)
        
        # Reduce to mean statistics
        stats = ee.Image([
            lst_day,
            lst_night,
            img.select('Clear_day_cov'),
            img.select('Clear_night_cov')
        ]).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=ee_geom,
            scale=1000,  # 1km resolution
            maxPixels=1e9,
            bestEffort=True
        )
        
        return ee.Feature(None, {
            'date': ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),
            'satellite': satellite_name,
            'LST_Day_C': stats.get('LST_Day_1km'),
            'LST_Night_C': stats.get('LST_Night_1km'),
            'Clear_Day': stats.get('Clear_day_cov'),
            'Clear_Night': stats.get('Clear_night_cov')
        })
    
    return collection.map(process_image)

def extract_zone_data_batch(zone_info):
    """
    Extract data for a single zone across all year batches
    
    **FIX FOR FutureWarning**: 
    - Filters out empty DataFrames BEFORE concat
    - Ensures consistent dtypes
    
    Returns: (zone_id, district_name, dataframe)
    """
    zone_id, district_name, zone_geom, year_batches, zone_file = zone_info
    
    # Resume mode: skip if already exists
    if RESUME_MODE and zone_file.exists():
        try:
            existing_df = pd.read_csv(zone_file)
            if not existing_df.empty:
                return (zone_id, district_name, existing_df, True)  # True = skipped
        except:
            pass  # If file is corrupted, reprocess
    
    try:
        ee_geom = shapely_to_ee_geometry(zone_geom, gdf.crs)
        batch_dataframes = []  # Store non-empty DataFrames only
        
        for batch_start, batch_end in year_batches:
            try:
                start_date = f'{batch_start}-01-01'
                end_date = f'{batch_end}-12-31' if batch_end < 2025 else '2025-01-19'
                
                # Load MODIS collections
                terra = ee.ImageCollection("MODIS/061/MOD11A1") \
                    .filterDate(start_date, end_date) \
                    .filterBounds(ee_geom)
                
                aqua = ee.ImageCollection("MODIS/061/MYD11A1") \
                    .filterDate(start_date, end_date) \
                    .filterBounds(ee_geom)
                
                # Process both satellites
                all_features = ee.FeatureCollection([])
                
                terra_count = terra.size().getInfo()
                if terra_count > 0:
                    terra_features = process_lst_collection_optimized(terra, 'Terra', ee_geom)
                    all_features = all_features.merge(terra_features)
                
                aqua_count = aqua.size().getInfo()
                if aqua_count > 0:
                    aqua_features = process_lst_collection_optimized(aqua, 'Aqua', ee_geom)
                    all_features = all_features.merge(aqua_features)
                
                # Get all data in ONE call (key to speed!)
                total_count = all_features.size().getInfo()
                if total_count > 0:
                    feature_list = all_features.toList(total_count).getInfo()
                    
                    batch_records = []
                    for feature in feature_list:
                        props = feature['properties']
                        # Only include records with valid LST data
                        if props.get('LST_Day_C') is not None or props.get('LST_Night_C') is not None:
                            batch_records.append({
                                'date': props['date'],
                                'zone': zone_id,
                                'district': district_name,
                                'satellite': props.get('satellite', 'Unknown'),
                                'LST_Day_C': round(props.get('LST_Day_C', float('nan')), 2) if props.get('LST_Day_C') else None,
                                'LST_Night_C': round(props.get('LST_Night_C', float('nan')), 2) if props.get('LST_Night_C') else None,
                                'Clear_Day_Coverage': round(props.get('Clear_Day', 0), 2),
                                'Clear_Night_Coverage': round(props.get('Clear_Night', 0), 2)
                            })
                    
                    # **FIX**: Only create DataFrame if we have records
                    if batch_records:
                        batch_df = pd.DataFrame(batch_records)
                        batch_dataframes.append(batch_df)
                
                time.sleep(0.05)  # Minimal rate limiting
                
            except Exception as e:
                print(f"    ⚠ Zone {zone_id}, years {batch_start}-{batch_end}: {e}")
                continue
        
        # **FIX FOR FutureWarning**: Only concat if we have non-empty DataFrames
        if batch_dataframes:
            # All DataFrames are guaranteed to be non-empty and have same structure
            final_df = pd.concat(batch_dataframes, ignore_index=True)
            
            # Sort and format
            final_df['date'] = pd.to_datetime(final_df['date'])
            final_df = final_df.sort_values(['date', 'satellite']).reset_index(drop=True)
            final_df['date'] = final_df['date'].dt.strftime('%Y-%m-%d')
            
            return (zone_id, district_name, final_df, False)  # False = processed
        else:
            return (zone_id, district_name, pd.DataFrame(), False)
            
    except Exception as e:
        print(f"    ✗ Zone {zone_id} error: {e}")
        return (zone_id, district_name, pd.DataFrame(), False)

print("✓ Optimized processing functions ready")

# ------------------------------
# Parallel Processing with Resume
# ------------------------------
print("\n[5/6] Extracting MODIS LST data (PARALLEL + RESUME MODE)...")

# Prepare year batches
year_batches = []
for year in range(START_YEAR, END_YEAR + 1, YEARS_PER_BATCH):
    batch_end = min(year + YEARS_PER_BATCH - 1, END_YEAR)
    year_batches.append((year, batch_end))

print(f"Year batches: {year_batches}")

for district_name, zones_gdf in district_zones.items():
    print(f"\n{'='*70}")
    print(f"Processing District: {district_name}")
    print(f"{'='*70}")
    
    district_folder = OUTPUT_BASE / district_name.replace(' ', '_')
    district_folder.mkdir(exist_ok=True)
    
    # Prepare zone tasks
    zone_tasks = []
    for idx, zone_row in zones_gdf.iterrows():
        zone_file = district_folder / f"zone_{zone_row['zone']}_lst_data.csv"
        zone_tasks.append((
            zone_row['zone'],
            district_name,
            zone_row['geometry'],
            year_batches,
            zone_file
        ))
    
    # Process zones in parallel
    district_dataframes = []  # Store non-empty DataFrames only
    skipped_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_zone = {executor.submit(extract_zone_data_batch, task): task[0] 
                          for task in zone_tasks}
        
        completed = 0
        for future in as_completed(future_to_zone):
            zone_id = future_to_zone[future]
            completed += 1
            
            try:
                zone_id, district_name, zone_df, was_skipped = future.result()
                
                if was_skipped:
                    skipped_count += 1
                    print(f"  ⏭ Zone {zone_id} [{completed}/{len(zone_tasks)}]: Skipped (already exists)")
                    if not zone_df.empty:
                        district_dataframes.append(zone_df)
                    continue
                
                if not zone_df.empty:
                    district_dataframes.append(zone_df)
                    
                    # Save individual zone file
                    zone_file = district_folder / f"zone_{zone_id}_lst_data.csv"
                    zone_df.to_csv(zone_file, index=False)
                    
                    avg_day = zone_df['LST_Day_C'].mean()
                    avg_night = zone_df['LST_Night_C'].mean()
                    terra_count = len(zone_df[zone_df['satellite'] == 'Terra'])
                    aqua_count = len(zone_df[zone_df['satellite'] == 'Aqua'])
                    
                    print(f"  ✓ Zone {zone_id} [{completed}/{len(zone_tasks)}]: "
                          f"T:{terra_count} A:{aqua_count} "
                          f"(Day:{avg_day:.1f}°C Night:{avg_night:.1f}°C)")
                else:
                    print(f"  ⚠ Zone {zone_id} [{completed}/{len(zone_tasks)}]: No data")
                    
            except Exception as e:
                print(f"  ✗ Zone {zone_id} [{completed}/{len(zone_tasks)}]: {e}")
    
    # **FIX FOR FutureWarning**: Combine district data
    if district_dataframes:
        district_df = pd.concat(district_dataframes, ignore_index=True)
        
        output_file = district_folder / f"{district_name.replace(' ', '_')}_modis_lst_data.csv"
        district_df.to_csv(output_file, index=False)
        
        print(f"\n✓ District complete: {len(district_df)} total records")
        if skipped_count > 0:
            print(f"  ({skipped_count} zones skipped - already processed)")
        
        # Summary statistics
        summary = district_df.groupby(['zone', 'satellite']).agg({
            'LST_Day_C': ['mean', 'min', 'max', 'std', 'count'],
            'LST_Night_C': ['mean', 'min', 'max', 'std', 'count'],
            'Clear_Day_Coverage': 'mean',
            'Clear_Night_Coverage': 'mean'
        }).round(2)
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary_file = district_folder / f"{district_name.replace(' ', '_')}_lst_summary.csv"
        summary.to_csv(summary_file)
        print(f"✓ Summary saved to {summary_file}")
    else:
        print(f"\n⚠ No data for {district_name}")

# ------------------------------
# Final Summary
# ------------------------------
print("\n" + "="*70)
print("[6/6] EXTRACTION COMPLETE!")
print("="*70)
print(f"\nOptimizations Applied:")
print(f"  ✓ Parallel processing ({MAX_WORKERS} workers)")
print(f"  ✓ Larger batch size ({YEARS_PER_BATCH} years)")
print(f"  ✓ Resume capability (skip existing files)")
print(f"  ✓ Fixed FutureWarning (filter empty DataFrames)")
print(f"  ✓ Optimized memory management")
print(f"  ✓ Reduced API calls")
print(f"\nMODIS LST Details:")
print(f"  - Collections: MOD11A1 (Terra) + MYD11A1 (Aqua)")
print(f"  - Bands: LST_Day_1km, LST_Night_1km")
print(f"  - Resolution: 1km daily")
print(f"  - QC masking: Bits 0-1 = 00 (good quality only)")
print(f"  - Clear sky coverage tracked")
print(f"\nData saved to: {OUTPUT_BASE}")
print("\n✓ All done!")