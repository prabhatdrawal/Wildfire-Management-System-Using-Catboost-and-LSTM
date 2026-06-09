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
OUTPUT_BASE = Path("/Users/prabhatrawal/Minor_project_code/data/landsat_data")

# Date range - Landsat 8 started April 2013, Landsat 9 started October 2021
START_YEAR = 2013
END_YEAR = 2025

# Parallel processing configuration - 4 BATCH SYSTEM
YEARS_PER_BATCH = 3  # Process 3 years at once (4 batches: 2013-2015, 2016-2018, 2019-2021, 2022-2025)
MAX_WORKERS = 4  # Process 4 batches simultaneously
ZONE_BATCH_SIZE = 4  # Process 4 zones in parallel per district

# Resume capability
RESUME_MODE = True

# Cloud masking threshold
CLOUD_COVER_MAX = 20  # Maximum cloud cover percentage

OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("USGS LANDSAT Data Extraction - 4-BATCH PARALLEL SYSTEM")
print("=" * 70)
print(f"Date range: {START_YEAR} to {END_YEAR}")
print(f"Batch configuration: {YEARS_PER_BATCH} years × {MAX_WORKERS} parallel batches")
print(f"Cloud cover filter: <{CLOUD_COVER_MAX}%")
print(f"Resume mode: {'ON' if RESUME_MODE else 'OFF'}")
print("=" * 70)
print("\nCollections:")
print("  - LANDSAT/LC08/C02/T1 (Landsat 8 - Historical: 2013-2021)")
print("  - LANDSAT/LC09/C02/T1 (Landsat 9 - Real-time: 2021-present)")
print("\nBands: B2, B3, B4, B8, B11, B12")
print("Indices: NDVI, GNDVI, NBR, NDWI, NDSI, EVI, SAVI")
print("=" * 70)

# ------------------------------
# Initialize Google Earth Engine
# ------------------------------
print("\n[1/7] Initializing Google Earth Engine...")
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
print("\n[2/7] Loading shapefile and creating zones...")
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
print("\n[3/7] Converting geometries to GEE format...")

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
# LANDSAT Processing Functions
# ------------------------------
print("\n[4/7] Setting up LANDSAT processing functions...")

def mask_landsat_clouds(image):
    """
    Cloud masking for Landsat Collection 2
    Uses QA_PIXEL band for cloud masking
    """
    qa = image.select('QA_PIXEL')
    
    # Bits 3 and 4 are cloud and cloud shadow
    cloud_shadow_bit = 1 << 3
    cloud_bit = 1 << 4
    
    # Both flags should be zero (clear conditions)
    mask = qa.bitwiseAnd(cloud_shadow_bit).eq(0).And(
           qa.bitwiseAnd(cloud_bit).eq(0))
    
    return image.updateMask(mask)

def calculate_indices(image):
    """
    Calculate spectral indices matching Sentinel-2 methodology
    
    Indices:
    - NDVI: (B8 - B4) / (B8 + B4)
    - GNDVI: (B8 - B3) / (B8 + B3)
    - NBR: (B8 - B11) / (B8 + B11)
    - NDWI: (B3 - B8) / (B3 + B8)
    - NDSI: (B11 - B12) / (B11 + B12)
    - EVI: 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1)
    - SAVI: ((B8 - B4) / (B8 + B4 + 0.5)) * 1.5
    """
    # For Landsat 8/9, the bands are: B2=Blue, B3=Green, B4=Red, B5=NIR, B6=SWIR1, B7=SWIR2
    # Mapping: B2→B2, B3→B3, B4→B4, B8→B5(NIR), B11→B6(SWIR1), B12→B7(SWIR2)
    
    nir = image.select('SR_B5')  # NIR band (B8 equivalent)
    red = image.select('SR_B4')  # Red band
    green = image.select('SR_B3')  # Green band
    blue = image.select('SR_B2')  # Blue band
    swir1 = image.select('SR_B6')  # SWIR1 band (B11 equivalent)
    swir2 = image.select('SR_B7')  # SWIR2 band (B12 equivalent)
    
    # Apply scale factor (Landsat Collection 2 scale factor is 0.0000275, offset -0.2)
    nir = nir.multiply(0.0000275).add(-0.2)
    red = red.multiply(0.0000275).add(-0.2)
    green = green.multiply(0.0000275).add(-0.2)
    blue = blue.multiply(0.0000275).add(-0.2)
    swir1 = swir1.multiply(0.0000275).add(-0.2)
    swir2 = swir2.multiply(0.0000275).add(-0.2)
    
    # Calculate indices
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    gndvi = nir.subtract(green).divide(nir.add(green)).rename('GNDVI')
    nbr = nir.subtract(swir1).divide(nir.add(swir1)).rename('NBR')
    ndwi = green.subtract(nir).divide(green.add(nir)).rename('NDWI')
    ndsi = swir1.subtract(swir2).divide(swir1.add(swir2)).rename('NDSI')
    
    # EVI: 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1)
    evi = nir.subtract(red).divide(
        nir.add(red.multiply(6)).subtract(blue.multiply(7.5)).add(1)
    ).multiply(2.5).rename('EVI')
    
    # SAVI: ((NIR - Red) / (NIR + Red + 0.5)) * 1.5
    savi = nir.subtract(red).divide(
        nir.add(red).add(0.5)
    ).multiply(1.5).rename('SAVI')
    
    return image.addBands([ndvi, gndvi, nbr, ndwi, ndsi, evi, savi])

def process_landsat_collection(collection, satellite_name, ee_geom):
    """
    Process Landsat collection with cloud masking and index calculation
    """
    def process_image(img):
        # Cloud masking
        masked = mask_landsat_clouds(img)
        
        # Calculate indices
        with_indices = calculate_indices(masked)
        
        # Reduce to statistics
        bands_to_reduce = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7',
                          'NDVI', 'GNDVI', 'NBR', 'NDWI', 'NDSI', 'EVI', 'SAVI']
        
        stats = with_indices.select(bands_to_reduce).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=ee_geom,
            scale=30,  # Landsat resolution
            maxPixels=1e9,
            bestEffort=True
        )
        
        # Get cloud cover
        cloud_cover = img.get('CLOUD_COVER')
        
        return ee.Feature(None, {
            'date': ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),
            'satellite': satellite_name,
            'cloud_cover': cloud_cover,
            'B2': stats.get('SR_B2'),
            'B3': stats.get('SR_B3'),
            'B4': stats.get('SR_B4'),
            'B5_NIR': stats.get('SR_B5'),
            'B6_SWIR1': stats.get('SR_B6'),
            'B7_SWIR2': stats.get('SR_B7'),
            'NDVI': stats.get('NDVI'),
            'GNDVI': stats.get('GNDVI'),
            'NBR': stats.get('NBR'),
            'NDWI': stats.get('NDWI'),
            'NDSI': stats.get('NDSI'),
            'EVI': stats.get('EVI'),
            'SAVI': stats.get('SAVI')
        })
    
    return collection.map(process_image)

def extract_zone_batch_data(zone_info):
    """
    Extract Landsat data for a single zone across all year batches
    Uses 4-BATCH PARALLEL SYSTEM for maximum speed
    
    Returns: (zone_id, district_name, dataframe, was_skipped)
    """
    zone_id, district_name, zone_geom, year_batches, zone_file = zone_info
    
    # Resume mode: skip if already exists
    if RESUME_MODE and zone_file.exists():
        try:
            existing_df = pd.read_csv(zone_file)
            if not existing_df.empty:
                return (zone_id, district_name, existing_df, True)
        except:
            pass
    
    try:
        ee_geom = shapely_to_ee_geometry(zone_geom, gdf.crs)
        batch_dataframes = []
        
        for batch_start, batch_end in year_batches:
            try:
                start_date = f'{batch_start}-01-01'
                end_date = f'{batch_end}-12-31' if batch_end < 2025 else '2025-01-21'
                
                # Determine which satellite to use
                if batch_end < 2021:
                    # Use only Landsat 8 for historical data (2013-2020)
                    collection = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
                        .filterDate(start_date, end_date) \
                        .filterBounds(ee_geom) \
                        .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_COVER_MAX))
                    
                    satellite_name = 'Landsat-8'
                    
                elif batch_start >= 2021:
                    # Use Landsat 9 for real-time data (2021-present)
                    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
                        .filterDate(start_date, end_date) \
                        .filterBounds(ee_geom) \
                        .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_COVER_MAX))
                    
                    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2") \
                        .filterDate(start_date, end_date) \
                        .filterBounds(ee_geom) \
                        .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_COVER_MAX))
                    
                    # Merge both collections
                    collection = l8.merge(l9)
                    satellite_name = 'Landsat-8/9'
                    
                else:
                    # Transition period (2021)
                    l8 = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2") \
                        .filterDate(start_date, end_date) \
                        .filterBounds(ee_geom) \
                        .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_COVER_MAX))
                    
                    l9 = ee.ImageCollection("LANDSAT/LC09/C02/T1_L2") \
                        .filterDate(start_date, end_date) \
                        .filterBounds(ee_geom) \
                        .filter(ee.Filter.lt('CLOUD_COVER', CLOUD_COVER_MAX))
                    
                    collection = l8.merge(l9)
                    satellite_name = 'Landsat-8/9'
                
                # Check if data exists
                count = collection.size().getInfo()
                
                if count > 0:
                    # Process collection
                    features = process_landsat_collection(collection, satellite_name, ee_geom)
                    
                    # Get all data in ONE call
                    feature_list = features.toList(count).getInfo()
                    
                    batch_records = []
                    for feature in feature_list:
                        props = feature['properties']
                        
                        # Only include if we have valid data
                        if props.get('NDVI') is not None:
                            batch_records.append({
                                'date': props['date'],
                                'zone': zone_id,
                                'district': district_name,
                                'satellite': props.get('satellite', 'Unknown'),
                                'cloud_cover': round(props.get('cloud_cover', 0), 2),
                                'B2_Blue': round(props.get('B2', float('nan')), 4) if props.get('B2') else None,
                                'B3_Green': round(props.get('B3', float('nan')), 4) if props.get('B3') else None,
                                'B4_Red': round(props.get('B4', float('nan')), 4) if props.get('B4') else None,
                                'B5_NIR': round(props.get('B5_NIR', float('nan')), 4) if props.get('B5_NIR') else None,
                                'B6_SWIR1': round(props.get('B6_SWIR1', float('nan')), 4) if props.get('B6_SWIR1') else None,
                                'B7_SWIR2': round(props.get('B7_SWIR2', float('nan')), 4) if props.get('B7_SWIR2') else None,
                                'NDVI': round(props.get('NDVI', float('nan')), 4) if props.get('NDVI') else None,
                                'GNDVI': round(props.get('GNDVI', float('nan')), 4) if props.get('GNDVI') else None,
                                'NBR': round(props.get('NBR', float('nan')), 4) if props.get('NBR') else None,
                                'NDWI': round(props.get('NDWI', float('nan')), 4) if props.get('NDWI') else None,
                                'NDSI': round(props.get('NDSI', float('nan')), 4) if props.get('NDSI') else None,
                                'EVI': round(props.get('EVI', float('nan')), 4) if props.get('EVI') else None,
                                'SAVI': round(props.get('SAVI', float('nan')), 4) if props.get('SAVI') else None
                            })
                    
                    if batch_records:
                        batch_df = pd.DataFrame(batch_records)
                        batch_dataframes.append(batch_df)
                
                time.sleep(0.05)
                
            except Exception as e:
                print(f"    ⚠ Zone {zone_id}, years {batch_start}-{batch_end}: {e}")
                continue
        
        # Combine all batches
        if batch_dataframes:
            final_df = pd.concat(batch_dataframes, ignore_index=True)
            final_df['date'] = pd.to_datetime(final_df['date'])
            final_df = final_df.sort_values('date').reset_index(drop=True)
            final_df['date'] = final_df['date'].dt.strftime('%Y-%m-%d')
            
            return (zone_id, district_name, final_df, False)
        else:
            return (zone_id, district_name, pd.DataFrame(), False)
            
    except Exception as e:
        print(f"    ✗ Zone {zone_id} error: {e}")
        return (zone_id, district_name, pd.DataFrame(), False)

print("✓ Landsat processing functions ready")
print("  - Cloud masking with QA_PIXEL")
print("  - 7 spectral indices calculated")
print("  - 30m resolution")

# ------------------------------
# Prepare Year Batches (4-BATCH SYSTEM)
# ------------------------------
print("\n[5/7] Preparing 4-batch parallel system...")

year_batches = []
for year in range(START_YEAR, END_YEAR + 1, YEARS_PER_BATCH):
    batch_end = min(year + YEARS_PER_BATCH - 1, END_YEAR)
    year_batches.append((year, batch_end))

print(f"Batch configuration: {len(year_batches)} batches")
for i, (start, end) in enumerate(year_batches, 1):
    print(f"  Batch {i}: {start}-{end} ({end-start+1} years)")

# ------------------------------
# Parallel Zone Processing
# ------------------------------
print("\n[6/7] Extracting Landsat data (4-BATCH PARALLEL MODE)...")

for district_name, zones_gdf in district_zones.items():
    print(f"\n{'='*70}")
    print(f"Processing District: {district_name}")
    print(f"{'='*70}")
    
    district_folder = OUTPUT_BASE / district_name.replace(' ', '_')
    district_folder.mkdir(exist_ok=True)
    
    # Prepare zone tasks
    zone_tasks = []
    for idx, zone_row in zones_gdf.iterrows():
        zone_file = district_folder / f"zone_{zone_row['zone']}_landsat_data.csv"
        zone_tasks.append((
            zone_row['zone'],
            district_name,
            zone_row['geometry'],
            year_batches,
            zone_file
        ))
    
    # Process zones in parallel (4 at a time)
    district_dataframes = []
    skipped_count = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_zone = {executor.submit(extract_zone_batch_data, task): task[0] 
                          for task in zone_tasks}
        
        completed = 0
        for future in as_completed(future_to_zone):
            zone_id = future_to_zone[future]
            completed += 1
            
            try:
                zone_id, district_name, zone_df, was_skipped = future.result()
                
                if was_skipped:
                    skipped_count += 1
                    print(f"  ⏭ Zone {zone_id} [{completed}/{len(zone_tasks)}]: Skipped (exists)")
                    if not zone_df.empty:
                        district_dataframes.append(zone_df)
                    continue
                
                if not zone_df.empty:
                    district_dataframes.append(zone_df)
                    
                    # Save zone file
                    zone_file = district_folder / f"zone_{zone_id}_landsat_data.csv"
                    zone_df.to_csv(zone_file, index=False)
                    
                    l8_count = len(zone_df[zone_df['satellite'] == 'Landsat-8'])
                    l89_count = len(zone_df[zone_df['satellite'] == 'Landsat-8/9'])
                    avg_ndvi = zone_df['NDVI'].mean()
                    avg_cloud = zone_df['cloud_cover'].mean()
                    
                    print(f"  ✓ Zone {zone_id} [{completed}/{len(zone_tasks)}]: "
                          f"L8:{l8_count} L8/9:{l89_count} "
                          f"(NDVI:{avg_ndvi:.3f} Cloud:{avg_cloud:.1f}%)")
                else:
                    print(f"  ⚠ Zone {zone_id} [{completed}/{len(zone_tasks)}]: No data")
                    
            except Exception as e:
                print(f"  ✗ Zone {zone_id} [{completed}/{len(zone_tasks)}]: {e}")
    
    # Combine district data
    if district_dataframes:
        district_df = pd.concat(district_dataframes, ignore_index=True)
        
        output_file = district_folder / f"{district_name.replace(' ', '_')}_landsat_data.csv"
        district_df.to_csv(output_file, index=False)
        
        print(f"\n✓ District complete: {len(district_df)} total records")
        if skipped_count > 0:
            print(f"  ({skipped_count} zones skipped)")
        
        # Summary statistics
        summary = district_df.groupby(['zone', 'satellite']).agg({
            'NDVI': ['mean', 'min', 'max', 'std'],
            'GNDVI': ['mean', 'std'],
            'NBR': ['mean', 'std'],
            'NDWI': ['mean', 'std'],
            'EVI': ['mean', 'std'],
            'SAVI': ['mean', 'std'],
            'cloud_cover': 'mean',
            'date': 'count'
        }).round(4)
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary_file = district_folder / f"{district_name.replace(' ', '_')}_landsat_summary.csv"
        summary.to_csv(summary_file)
        print(f"✓ Summary saved to {summary_file}")
    else:
        print(f"\n⚠ No data for {district_name}")

# ------------------------------
# Final Summary
# ------------------------------
print("\n" + "="*70)
print("[7/7] EXTRACTION COMPLETE!")
print("="*70)
print(f"\n4-Batch Parallel System Performance:")
print(f"  ✓ {MAX_WORKERS} batches processed simultaneously")
print(f"  ✓ {YEARS_PER_BATCH} years per batch")
print(f"  ✓ Total batches: {len(year_batches)}")
print(f"  ✓ Resume capability enabled")
print(f"\nLandsat Collections:")
print(f"  - LANDSAT/LC08/C02/T1_L2 (Landsat 8: 2013-present)")
print(f"  - LANDSAT/LC09/C02/T1_L2 (Landsat 9: 2021-present)")
print(f"\nData Extracted:")
print(f"  - 6 spectral bands (B2, B3, B4, B5/NIR, B6/SWIR1, B7/SWIR2)")
print(f"  - 7 spectral indices (NDVI, GNDVI, NBR, NDWI, NDSI, EVI, SAVI)")
print(f"  - Cloud cover metadata")
print(f"  - 30m spatial resolution")
print(f"  - 13 years of data (2013-2025)")
print(f"\nData saved to: {OUTPUT_BASE}")
print("\n✓ All done! Ready for ML training with 13 years of Landsat data!")