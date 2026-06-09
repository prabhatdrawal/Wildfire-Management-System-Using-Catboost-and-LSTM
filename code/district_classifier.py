import geopandas as gpd
from pathlib import Path
import pandas as pd
import math
from shapely.geometry import box
from shapely.ops import unary_union

# ------------------------------
# Configuration
# ------------------------------
SHAPEFILE_PATH = Path("/Users/prabhatrawal/Minor_project_code/polygon_file/actual_timezone_designated_district_using_EPSG_32644.shp")

print("=" * 60)
print("District Name & Zone Identifier")
print("=" * 60)

# Load shapefile
print("\nLoading shapefile...")
gdf = gpd.read_file(SHAPEFILE_PATH)

print(f"✓ Loaded {len(gdf)} districts\n")

# Display all available columns
print("Available columns in shapefile:")
print(gdf.columns.tolist())
print()

# ------------------------------
# Zone Division Logic (Same as in extraction scripts)
# ------------------------------
print("Calculating zone divisions...")

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

# ------------------------------
# Process each district and count zones
# ------------------------------
print("Processing districts and counting zones...\n")

district_zones = {}
district_mapping = []

for idx, row in gdf.iterrows():
    # Get district name
    if 'NAME_3' in gdf.columns:
        district_name = row['NAME_3']
    elif 'NAME' in gdf.columns:
        district_name = row['NAME']
    elif 'NAME_2' in gdf.columns:
        district_name = row['NAME_2']
    else:
        district_name = f'District_{idx}'
    
    # Get district polygon
    polygon = row['geometry']
    
    # Calculate zones for this district
    zones_gdf = assign_and_merge_zones(polygon, grid_boxes)
    
    if zones_gdf is not None:
        num_zones = len(zones_gdf)
        zones_gdf['district'] = district_name
        district_zones[district_name] = zones_gdf
    else:
        num_zones = 0
    
    # Store mapping information
    district_mapping.append({
        'District_Index': idx,
        'District_Name': district_name,
        'Code_Name': f'District_{idx}',
        'Folder_Name': district_name.replace(' ', '_'),
        'Number_of_Zones': num_zones,
        'Zone_Range': f'1-{num_zones}' if num_zones > 0 else 'N/A',
        'Region': row.get('NAME_2', 'N/A'),
        'Province': row.get('NAME_1', 'N/A')
    })

# ------------------------------
# Display Results
# ------------------------------
print("=" * 60)
print("DISTRICT MAPPING WITH ZONE COUNTS")
print("=" * 60)

for mapping in district_mapping:
    print(f"\nDistrict_{mapping['District_Index']} = {mapping['District_Name']}")
    print(f"  Number of Zones: {mapping['Number_of_Zones']}")
    print(f"  Zone Range: {mapping['Zone_Range']}")
    print(f"  Folder Name: {mapping['Folder_Name']}/")
    print(f"  Region: {mapping['Region']}")

print()

# Create a DataFrame for easy reference
mapping_df = pd.DataFrame(district_mapping)

# Display as a nice table
print("=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print()
print(mapping_df[['District_Index', 'District_Name', 'Number_of_Zones', 'Folder_Name']].to_string(index=False))
print()

# Save to CSV for reference
output_file = Path("/Users/prabhatrawal/Minor_project_code/data/district_mapping.csv")
output_file.parent.mkdir(parents=True, exist_ok=True)
mapping_df.to_csv(output_file, index=False)

print("=" * 60)
print(f"✓ District mapping saved to: {output_file}")
print("=" * 60)

# Create a detailed zone breakdown
print("\n" + "=" * 60)
print("DETAILED ZONE BREAKDOWN")
print("=" * 60)

for district_name, zones_gdf in district_zones.items():
    print(f"\n{district_name}:")
    print(f"  Total Zones: {len(zones_gdf)}")
    print(f"  Zones: {', '.join([f'Zone_{z}' for z in zones_gdf['zone']])}")
    
    # Calculate some statistics
    total_area = zones_gdf['area'].sum()
    avg_area = zones_gdf['area'].mean()
    min_area = zones_gdf['area'].min()
    max_area = zones_gdf['area'].max()
    
    print(f"  Total Area: {total_area:,.0f} sq meters")
    print(f"  Average Zone Size: {avg_area:,.0f} sq meters")
    print(f"  Min Zone Size: {min_area:,.0f} sq meters")
    print(f"  Max Zone Size: {max_area:,.0f} sq meters")

# Create a quick reference guide
print("\n" + "=" * 60)
print("QUICK REFERENCE FOR YOUR DATA FOLDERS")
print("=" * 60)
print("\nWhen you see these folder names in your extracted data:\n")

for mapping in district_mapping:
    folder_name = mapping['Folder_Name']
    district_name = mapping['District_Name']
    num_zones = mapping['Number_of_Zones']
    
    print(f"📁 {folder_name}/")
    print(f"   → District: {district_name}")
    print(f"   → Contains: {num_zones} zones (zone_1_data.csv to zone_{num_zones}_data.csv)")
    print()

# Save detailed zone information
zone_details = []
for district_name, zones_gdf in district_zones.items():
    for idx, zone_row in zones_gdf.iterrows():
        zone_details.append({
            'District_Name': district_name,
            'Zone_Number': zone_row['zone'],
            'Zone_Area_sqm': zone_row['area'],
            'File_Name': f"zone_{zone_row['zone']}_data.csv"
        })

zone_df = pd.DataFrame(zone_details)
zone_output_file = Path("/Users/prabhatrawal/Minor_project_code/data/zone_details.csv")
zone_df.to_csv(zone_output_file, index=False)

print("=" * 60)
print(f"✓ Zone details saved to: {zone_output_file}")
print("=" * 60)

# Summary statistics
print("\n" + "=" * 60)
print("OVERALL STATISTICS")
print("=" * 60)
print(f"\nTotal Districts: {len(district_mapping)}")
print(f"Total Zones: {sum([m['Number_of_Zones'] for m in district_mapping])}")
print(f"Average Zones per District: {sum([m['Number_of_Zones'] for m in district_mapping]) / len(district_mapping):.1f}")
print(f"\nZone Distribution:")
for mapping in district_mapping:
    print(f"  {mapping['District_Name']}: {mapping['Number_of_Zones']} zones")

print("\n" + "=" * 60)
print("✓ Complete!")
print("=" * 60)