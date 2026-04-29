# Wildfire_Management_System_Phase1_5district
## Spatial analysis
Required software - QGIS (Quantum Geographic Information System) ( downloadable for all os platform - windows , mac , linux)
# Polygon file
Import the polygong file of whole in the .shp format then clip the designated vector layer and download at required time zone.
# Zone creation using python - matplotlib
The district are divided into different zones in python.

* library used :  matplotlib, shapely - (.geometry , .ops ), math

* Logic used :
   a. GLobal grid calibration
   - Bounding Box: The algorithm determines the total spatial extent (min/max X and Y) of the entire dataset.

   - Density Calculation: It   calculates the required area per zone A_target using the formula:
      S = sqrt((sum(Area_district))/(N*10)) where S is the side length of each square grid cell.
     
   - Grid Generation: A uniform tessellation of square boxes considered as zone is generated across the entire bounding box.

   b. Spatial Intersection - Cookie Cutter
   - Geometric Intersection: For every district, the algorithm identifies which global grid cells overlap with it.

   - Clipping: It calculates the intersection between the district polygon and the grid squares. This ensures the resulting sub-zones            follow the exact administrative boundaries of the district.
 
   c. Iterative Sliver Refinement (The Merging Algorithm)
   Useful for statistical modeling, the algorithm removes tiny, irregular polygons (slivers) through an iterative cleanup process.

   - Area Thresholding: A zone is flagged as a "sliver" if its area is less than 25% of the target zone area.

  - Adjacency Search: For each flagged sliver, the algorithm identifies all neighboring zones that share a boundary (using the .touches()       predicate).

  - Smallest-Neighbor Merge: * The algorithm selects the neighbor with the smallest area.

      It performs a unary_union to merge the sliver into that neighbor.

  - Logic: Merging with the smallest neighbor helps balance the overall area distribution, preventing any single zone from becoming             disproportionately large.

  - Convergence: The process repeats in a while loop until no zones remain below the threshold or no valid neighbors are left to merge with.
