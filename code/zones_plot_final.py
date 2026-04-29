import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import math
from shapely.ops import unary_union

# Load shapefile
from pathlib import Path

shapefile_path = Path("/Users/prabhatrawal/Minor_project_code/polygon_file/actual_timezone_designated_district_using_EPSG_32644.shp")
gdf = gpd.read_file(shapefile_path)


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

# Create uniform grid boxes over the union bounds
grid_boxes = []
for i in range(n_cols):
    for j in range(n_rows):
        xmin = minx + i * dx
        xmax = xmin + dx
        ymin = miny + j * dy
        ymax = ymin + dy
        grid_boxes.append(box(xmin, ymin, xmax, ymax))

# Function to assign and merge zones for a district

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


# Process each district
district_zones = {}
for idx, row in gdf.iterrows():
    district_name = row['NAME'] if 'NAME' in gdf.columns else f'District_{idx}'
    polygon = row['geometry']
    zones_gdf = assign_and_merge_zones(polygon, grid_boxes)
    if zones_gdf is not None:
        zones_gdf['district'] = district_name
        district_zones[district_name] = zones_gdf
        print(f"{district_name} has {len(zones_gdf)} zones after merging.")


# Combine all zones for plotting
if district_zones:
    all_zones_gdf = gpd.pd.concat(district_zones.values(), ignore_index=True)

    # Plot all districts combined
    fig, ax = plt.subplots(figsize=(12, 12))
    gdf.boundary.plot(ax=ax, edgecolor='black', linewidth=1.5, label='District Boundaries')
    all_zones_gdf.plot(ax=ax, column='zone', cmap='tab20', alpha=0.5, edgecolor='k', legend=True)
    plt.title("All Districts with Uniform Grid Zones (Merged Small Ones)")
    plt.xlabel("Easting (meters)")
    plt.ylabel("Northing (meters)")
    plt.legend()
    plt.show()

    # Plot individual districts
    if 'NAME' in gdf.columns:
        for district, zones_gdf in district_zones.items():
            district_poly = gdf[gdf['NAME'] == district]
            
            fig, ax = plt.subplots(figsize=(10, 10))
            district_poly.boundary.plot(ax=ax, edgecolor='black', linewidth=1.5)
            zones_gdf.plot(ax=ax, column='zone', cmap='tab20', alpha=0.5, edgecolor='k', legend=True)
            plt.title(f"{district} with Uniform Grid Zones (Merged Small Ones)")
            plt.xlabel("Easting (meters)")
            plt.ylabel("Northing (meters)")
            plt.show()
    else:
        print("No 'NAME' column in shapefile; skipping individual district plots.")

else:
    print("No zones generated.")