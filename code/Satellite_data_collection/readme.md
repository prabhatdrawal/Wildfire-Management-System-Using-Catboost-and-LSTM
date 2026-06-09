# 1. FIRMS : 
link : ee.ImageCollection("FIRMS")
bands : t21, confidence( replaceable with (MOD14A1)- firemask)

# 2. mod14a1 - done
link: ee.ImageCollection("MODIS/061/MOD14A1")
bands : fire mask ( 7: Fire (low confidence, land or water)
                     8: Fire (nominal confidence, land or water)
                     9: Fire (high confidence, land or water))

# 3. Sential ( La2) - done 
link : ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
bands : (B2, B3, B4, B8, B11, B12)   
masking : SCL / MSK_CLDPRB     
can calculate :  
                - NDVI (Normalized Difference Vegetation Index)
                    (B8 - B4) / (B8 + B4)
                - GNDVI (Green NDVI)
                    (B8 - B3) / (B8 + B3)
                - NBR (Normalized Burn Ratio)
                    (B8 - B11) / (B8 + B11)
                - NDWI (Normalized Difference Water Index)
                    (B8 - B11) / (B8 + B11)
                - NDSI (Normalized Difference SWIR Index)
                    NDSI = (B11 - B12) / (B11 + B12)
                - EVI (Enhanced Vegetation Index)
                    EVI = 2.5 * (B8 - B4) / (B8 + 6*B4 - 7.5*B2 + 1)
                - SAVI (Soil-Adjusted Vegetation Index)
                    SAVI = ((B8 - B4) / (B8 + B4 + L)) * (1 + L)
                    L = 0.5 
            
# 4. USGS Landsat - done
link : ee.ImageCollection("LANDSAT/LC08/C02/T1") - for historical
link 2 : ee.ImageCollection("LANDSAT/LC09/C02/T1") - for REAL-TIME
bands : bands : (B2, B3, B4, B8, B11, B12) 
===> reason: same as S2, but it has 13 years of data (useful for ML training) 
                (Sentinel2 only has 8 years of data)


# 5. MODIS - Terra+Aqua LST (Land Surface Temperature) - done
link 1 : ee.ImageCollection("MODIS/061/MOD11A1") - Terra
link 2 : ee.ImageCollection("MODIS/061/MYD11A1") - Aqua
bands : LST_Day_1km, LST_Night_1km (from both)
masking : QC_Day (Bits 0-1), QC_Night (Bits 0-1), Clear_day_cov, Clear_night_cov

# 6. SRTM - done 
link : ee.Image("CSP/ERGo/1_0/Global/SRTM_mTPI") - shlok recommended
       ee.Image("USGS/SRTMGL1_003") - chatgpt recommmended
bands : elevation

# 7. Era - 5 land - processing
link : ("ECMWF/ERA5_LAND/DAILY_AGGR")
bands : temperature_2m,dewpoint_temperature_2m,skin_temperature,      soil_temperature_level_1,volumetric_soil_water_layer_1,total_precipitation_sum,u_component_of_wind_10m,v_component_of_wind_10m,surface_pressure)

