"""
=============================================================================
FFSFM (Forest Fire Spread Forecast Model) - Dataset Builder  v3
=============================================================================
Matches EXACT column values confirmed from parquet diagnostic:
  district : 'District_0' … 'District_4'   (category dtype)
  zone     : 1 … 19                         (int64)
  date     : datetime64[ns]                 (already parsed)

District → Name mapping (from your classification output):
  District_0 = Banke   (14 zones)
  District_1 = Bardiya (15 zones)
  District_2 = Surkhet (19 zones)
  District_3 = Dang    (17 zones)
  District_4 = Salyan  (14 zones)

ConvBiLSTM targets:
  Input  : (N, T=14, Z, F)   — 14-day lookback
  Output : (N, 7,    Z)      — 7-day binary fire spread per zone

Output: /Users/prabhatrawal/Minor_project_code/ffsfm_data/
=============================================================================
"""

import json
import math
import warnings
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE        = Path("/Users/prabhatrawal/Minor_project_code")
DATA_IN     = BASE / "data" / "integrated_data"
POLYGON_DIR = BASE / "polygon_file"
OUT_DIR     = BASE / "ffsfm_data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FFOPM_PARQUET = DATA_IN / "Master_FFOPM_Table.parquet"
FFOPM_CSV     = DATA_IN / "Master_FFOPM_Table.csv"
SHP_FILE      = POLYGON_DIR / "actual_timezone_designated_district_using_EPSG_32644.shp"

# ─────────────────────────────────────────────────────────────────────────────
# District catalogue
# Keys = EXACT values stored in the parquet 'district' column
# ─────────────────────────────────────────────────────────────────────────────
DISTRICT_INFO = {
    "District_0": {"name": "Banke",   "n_zones": 14, "district_id": 0},
    "District_1": {"name": "Bardiya", "n_zones": 15, "district_id": 1},
    "District_2": {"name": "Surkhet", "n_zones": 19, "district_id": 2},
    "District_3": {"name": "Dang",    "n_zones": 17, "district_id": 3},
    "District_4": {"name": "Salyan",  "n_zones": 14, "district_id": 4},
}

# Reverse lookup: District_N → NAME_3 from shapefile (for zone reconstruction)
DISTRICT_TO_NAME3 = {
    "District_0": "Banke",
    "District_1": "Bardiya",
    "District_2": "Surkhet",
    "District_3": "Dang",
    "District_4": "Salyan",
}

# ─────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
LOOKBACK   = 14
HORIZON    = 7
DATE_START = "2000-01-01"
DATE_END   = "2025-12-31"
TARGET_COLS = [f"fire_label_t{k}" for k in range(1, HORIZON + 1)]

# ─────────────────────────────────────────────────────────────────────────────
# Feature groups
# ─────────────────────────────────────────────────────────────────────────────
STATIC_FEATURES = [
    "elevation_mean_m", "elevation_stddev_m", "elevation_range_m",
    "slope_mean_deg",   "slope_stddev_deg",
    "aspect_mean_deg",  "aspect_stddev_deg",
    "mtpi_mean",        "mtpi_stddev",
]

DYNAMIC_FEATURES = [
    # Fire pixels
    "total_fire_pixels", "fire_percentage",
    "low_confidence_fire_pixels", "nominal_confidence_fire_pixels",
    "high_confidence_fire_pixels",
    # Vegetation
    "ndvi_composite", "s2_ndvi", "s2_nbr", "s2_evi", "s2_savi",
    "landsat_ndvi",   "landsat_nbr",
    # Moisture
    "s2_ndwi", "s2_ndsi", "landsat_ndwi",
    # LST
    "lst_day_c",
    # Meteorology
    "temperature_2m_celsius", "skin_temperature_celsius",
    "dewpoint_2m_celsius",    "soil_temperature_celsius",
    "soil_moisture_m3m3",     "precipitation_mm",
    "wind_speed_ms",          "wind_direction_deg",
    "u_wind_component_ms",    "v_wind_component_ms",
    "surface_pressure_hpa",   "relative_humidity_pct",
    "vapor_pressure_deficit_kpa",
    # Lagged
    "precipitation_mm_lag1",   "precipitation_mm_lag5",
    "precipitation_mm_lag10",  "precipitation_mm_lag30",
    "vapor_pressure_deficit_kpa_lag1",
    "vapor_pressure_deficit_kpa_lag3",
    "vapor_pressure_deficit_kpa_lag7",
    "soil_moisture_m3m3_lag1", "soil_moisture_m3m3_lag3",
    "soil_moisture_m3m3_lag7",
    "temperature_2m_celsius_lag1", "temperature_2m_celsius_lag3",
    "relative_humidity_pct_lag1",  "relative_humidity_pct_lag3",
    # Rolling
    "precipitation_mm_roll7_sum", "precipitation_mm_roll14_sum",
    "precipitation_mm_roll30_sum",
    "temperature_2m_celsius_roll7_mean",
    "temperature_2m_celsius_roll14_mean",
    "vapor_pressure_deficit_kpa_roll7_mean",
]
DYNAMIC_FEATURES = list(dict.fromkeys(DYNAMIC_FEATURES))


# =============================================================================
# STEP 1  Reconstruct zones from shapefile
#         Mirrors EXACT logic from your actual zone division script
# =============================================================================
def reconstruct_zones_from_shapefile(target_n: int = 10,
                                      merge_threshold: float = 0.25) -> dict:
    """
    Returns {district_code: GeoDataFrame}
    where district_code is 'District_0' … 'District_4'
    and zone numbers are 1-based integers matching the parquet 'zone' column.
    """
    log.info("Reconstructing zones from shapefile...")
    gdf = gpd.read_file(SHP_FILE)

    # ── Build global grid (exact copy from your zone division script) ──────
    union_poly = gdf.geometry.unary_union
    minx, miny, maxx, maxy = union_poly.bounds
    width      = maxx - minx
    height     = maxy - miny
    total_area = gdf.geometry.area.sum()

    n_districts    = len(gdf)
    target_n_total = target_n * n_districts
    cell_size      = math.sqrt(total_area / target_n_total)

    n_cols = math.ceil(width  / cell_size)
    n_rows = math.ceil(height / cell_size)
    dx = width  / n_cols
    dy = height / n_rows

    grid_boxes = []
    for i in range(n_cols):
        for j in range(n_rows):
            xmin = minx + i * dx
            xmax = xmin + dx
            ymin = miny + j * dy
            ymax = ymin + dy
            grid_boxes.append(box(xmin, ymin, xmax, ymax))

    log.info(f"  Grid: {n_cols}x{n_rows} = {len(grid_boxes)} cells")

    # ── Per-district zone assignment ───────────────────────────────────────
    # Map NAME_3 values back to District_N codes
    name3_to_code = {v: k for k, v in DISTRICT_TO_NAME3.items()}

    district_zones = {}
    for idx, row in gdf.iterrows():
        # Your zone division script uses NAME_3 from the shapefile
        name3 = row.get("NAME_3", row.get("NAME", f"Unknown_{idx}"))
        dist_code = name3_to_code.get(name3)
        if dist_code is None:
            log.warning(f"  Skipping shapefile row '{name3}' (not in DISTRICT_INFO)")
            continue

        zones_gdf = _assign_and_merge_zones(
            row["geometry"], grid_boxes, gdf.crs,
            target_n=target_n, merge_threshold=merge_threshold
        )
        if zones_gdf is not None:
            district_zones[dist_code] = zones_gdf
            log.info(f"  {dist_code} ({name3}): {len(zones_gdf)} zones")

    return district_zones


def _assign_and_merge_zones(district_poly, grid_boxes, crs,
                              target_n=10, merge_threshold=0.25):
    """Exact copy of assign_and_merge_zones from your zone division script."""
    zones = []
    for gb in grid_boxes:
        inter = gb.intersection(district_poly)
        if not inter.is_empty:
            zones.append({"geometry": inter})

    if not zones:
        return None

    zones_gdf = gpd.GeoDataFrame(zones, crs=crs)
    zones_gdf["area"] = zones_gdf.geometry.area

    district_area  = district_poly.area
    target_area    = district_area / target_n
    threshold_area = merge_threshold * target_area

    merged = True
    while merged:
        merged = False
        small_zones = zones_gdf[zones_gdf["area"] < threshold_area].copy()
        if small_zones.empty:
            break
        for idx, small_row in small_zones.iterrows():
            if idx not in zones_gdf.index:
                continue
            touches   = zones_gdf.geometry.touches(small_row.geometry)
            neighbors = zones_gdf[touches & (zones_gdf.index != idx)]
            if neighbors.empty:
                continue
            smallest_idx = neighbors["area"].idxmin()
            merged_geom  = unary_union([small_row.geometry,
                                        zones_gdf.at[smallest_idx, "geometry"]])
            zones_gdf.at[smallest_idx, "geometry"] = merged_geom
            zones_gdf.at[smallest_idx, "area"]     = merged_geom.area
            zones_gdf = zones_gdf.drop(idx)
            merged = True
            break

    zones_gdf         = zones_gdf.reset_index(drop=True)
    zones_gdf["zone"] = zones_gdf.index + 1   # 1-based → matches parquet int zone
    return zones_gdf


# =============================================================================
# STEP 2  Adjacency from reconstructed zone geometries
# =============================================================================
def build_adjacency(district_zones: dict) -> dict:
    """
    Returns {dist_code: {zone_int: [neighbour_zone_int, ...]}}
    e.g. {'District_0': {1: [2, 3], 2: [1, 4], ...}}
    """
    log.info("Building zone adjacency graph...")
    adjacency = {}
    for dist_code, zones_gdf in district_zones.items():
        zone_nums = zones_gdf["zone"].tolist()   # [1, 2, 3, ...]
        adj       = {z: [] for z in zone_nums}
        for i in range(len(zones_gdf)):
            for j in range(i + 1, len(zones_gdf)):
                gi = zones_gdf.iloc[i].geometry
                gj = zones_gdf.iloc[j].geometry
                if gi.touches(gj) or gi.intersects(gj):
                    zi = int(zones_gdf.iloc[i]["zone"])
                    zj = int(zones_gdf.iloc[j]["zone"])
                    adj[zi].append(zj)
                    adj[zj].append(zi)
        adjacency[dist_code] = adj
        avg_nb = np.mean([len(v) for v in adj.values()]) if adj else 0
        name   = DISTRICT_INFO[dist_code]["name"]
        log.info(f"  {dist_code} ({name}): {len(adj)} zones, avg_neighbours={avg_nb:.1f}")
    return adjacency


# =============================================================================
# STEP 3  Load FFOPM
# =============================================================================
def load_ffopm() -> pd.DataFrame:
    log.info("Loading FFOPM master table...")
    if FFOPM_PARQUET.exists():
        try:
            df = pd.read_parquet(FFOPM_PARQUET)
            log.info(f"  Loaded parquet  {df.shape}")
            return df
        except Exception as e:
            log.warning(f"  Parquet failed ({e}), trying CSV...")
    df = pd.read_csv(FFOPM_CSV, low_memory=False)
    log.info(f"  Loaded CSV  {df.shape}")
    return df


# =============================================================================
# STEP 4  Clean  (NO name conversion — keep District_N and int zones as-is)
# =============================================================================
def clean_ffopm(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Cleaning FFOPM data...")

    # date is already datetime64[ns] but do it safely anyway
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= DATE_START) & (df["date"] <= DATE_END)].copy()

    # district: keep as string (convert from category if needed)
    df["district"] = df["district"].astype(str).str.strip()

    # zone: keep as int
    df["zone"] = df["zone"].astype(int)

    # Filter to known districts only
    known = list(DISTRICT_INFO.keys())   # ['District_0', ..., 'District_4']
    df    = df[df["district"].isin(known)].copy()

    # Binary fire label
    if "fire_label" in df.columns:
        df["fire_label"] = df["fire_label"].fillna(0).astype(int).clip(0, 1)
    else:
        df["fire_label"] = (df["total_fire_pixels"].fillna(0) > 0).astype(int)

    df = df.sort_values(["district", "zone", "date"]).reset_index(drop=True)

    log.info(f"  Shape     : {df.shape}")
    log.info(f"  Date range: {df['date'].min()} → {df['date'].max()}")
    log.info(f"  Districts : {sorted(df['district'].unique())}")
    log.info(f"  Zones     : {sorted(df['zone'].unique())}")
    log.info(f"  fire=1    : {df['fire_label'].sum():,}  "
             f"({100*df['fire_label'].mean():.2f}%)")
    return df


# =============================================================================
# STEP 5  Spread feature engineering
# =============================================================================
def engineer_spread_features(df: pd.DataFrame, adjacency: dict) -> pd.DataFrame:
    log.info("Engineering spread features...")
    records = []

    for dist_code, grp_d in df.groupby("district"):
        adj       = adjacency.get(dist_code, {})  # {zone_int: [neighbour_ints]}
        all_zones = sorted(grp_d["zone"].unique())

        # Pivot: date × zone_int → fire_label  (fast lookup)
        fire_pivot = grp_d.pivot_table(
            index="date", columns="zone", values="fire_label", fill_value=0
        )

        for zone in all_zones:
            neighbours = adj.get(zone, [])         # list of int zone numbers
            zone_df    = grp_d[grp_d["zone"] == zone].copy().reset_index(drop=True)

            # ── Neighbour fire features ───────────────────────────────────
            valid_nb = [n for n in neighbours if n in fire_pivot.columns]
            if valid_nb:
                nb_vals = fire_pivot[valid_nb].reindex(zone_df["date"]).fillna(0)
                zone_df["neighbour_fire_count"] = nb_vals.values.sum(axis=1)
                zone_df["neighbour_fire_frac"]  = (
                    zone_df["neighbour_fire_count"] / len(valid_nb)
                )
                zone_df["any_neighbour_fire"]   = (
                    zone_df["neighbour_fire_count"] > 0
                ).astype(int)
            else:
                zone_df["neighbour_fire_count"] = 0
                zone_df["neighbour_fire_frac"]  = 0.0
                zone_df["any_neighbour_fire"]   = 0

            # ── Wind circular encoding ────────────────────────────────────
            wd_rad = np.deg2rad(zone_df["wind_direction_deg"].fillna(0))
            zone_df["wind_sin"] = np.sin(wd_rad)
            zone_df["wind_cos"] = np.cos(wd_rad)

            # ── Fire Weather Index ────────────────────────────────────────
            zone_df["fire_weather_index"] = (
                zone_df["vapor_pressure_deficit_kpa"].fillna(0).clip(0)
                * zone_df["wind_speed_ms"].fillna(0).clip(0)
                * (1 - zone_df["soil_moisture_m3m3"].fillna(0.3).clip(0, 1))
            )

            # ── Dry spell duration (consecutive days precip < 1 mm) ───────
            precip = zone_df["precipitation_mm"].fillna(0)
            dry    = (precip < 1.0).astype(int)
            cumsum = (dry != dry.shift()).cumsum()
            zone_df["dry_spell_days"] = (
                dry.groupby(cumsum).cumcount() + 1
            ) * dry

            # ── 7-day future fire labels ──────────────────────────────────
            if zone in fire_pivot.columns:
                fire_arr = fire_pivot[zone].reindex(zone_df["date"]).values
            else:
                fire_arr = np.zeros(len(zone_df), dtype=int)

            fire_series = pd.Series(fire_arr)
            for k in range(1, HORIZON + 1):
                zone_df[f"fire_label_t{k}"] = fire_series.shift(-k).values

            records.append(zone_df)

    out = pd.concat(records, ignore_index=True)
    out = out.dropna(subset=TARGET_COLS)
    for c in TARGET_COLS:
        out[c] = out[c].astype(int)

    log.info(f"  After engineering: {out.shape}")
    return out


# =============================================================================
# STEP 6  Zone integer index  (0-based for tensor axis)
#         zone int 1 → tensor index 0,  zone 2 → 1,  etc.
# =============================================================================
def build_zone_index(district_zones: dict) -> dict:
    """
    Returns {dist_code: {zone_int: tensor_index}}
    e.g. {'District_0': {1: 0, 2: 1, 3: 2, ...}}
    """
    index = {}
    for dist_code, zones_gdf in district_zones.items():
        sorted_zone_nums = sorted(zones_gdf["zone"].tolist())
        index[dist_code] = {int(z): i for i, z in enumerate(sorted_zone_nums)}
    return index


# =============================================================================
# STEP 7  Z-score normalisation
# =============================================================================
def normalise(df: pd.DataFrame, feat_cols: list) -> tuple:
    log.info("Normalising features...")
    params = {}
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0.0
        mu  = float(df[col].mean())
        std = float(df[col].std()) or 1.0
        df[col] = (df[col].fillna(mu) - mu) / std
        params[col] = {"mean": mu, "std": std}
    return df, params


# =============================================================================
# STEP 8  Build ConvBiLSTM tensors
#
#   Per district:
#     X  (N, LOOKBACK, n_zones, n_features)
#     y  (N, HORIZON,  n_zones)
#
#   Chronological sliding window, step = 1 day
# =============================================================================
def build_tensors(df: pd.DataFrame,
                  feat_cols: list,
                  zone_index: dict) -> tuple:
    log.info("Building ConvBiLSTM tensors...")
    X_dict    = {}
    y_dict    = {}
    meta_rows = []
    sample_id = 0
    F         = len(feat_cols)

    for dist_code, grp_d in df.groupby("district"):
        if dist_code not in zone_index:
            log.warning(f"  {dist_code}: no zone index, skipping")
            continue

        zi      = zone_index[dist_code]          # {zone_int: tensor_idx}
        n_zones = len(zi)
        dates   = sorted(grp_d["date"].unique())
        n_days  = len(dates)

        if n_days < LOOKBACK + HORIZON:
            log.warning(f"  {dist_code}: only {n_days} days, skipping")
            continue

        date_to_idx = {d: i for i, d in enumerate(dates)}

        # Pre-allocate
        feat_arr  = np.zeros((n_days, n_zones, F),       dtype=np.float32)
        label_arr = np.zeros((n_days, n_zones, HORIZON),  dtype=np.float32)

        for _, row in grp_d.iterrows():
            d_idx = date_to_idx.get(row["date"])
            z_idx = zi.get(int(row["zone"]))        # zone is int in parquet
            if d_idx is None or z_idx is None:
                continue
            for f_idx, col in enumerate(feat_cols):
                v = row.get(col, 0.0)
                feat_arr[d_idx, z_idx, f_idx] = float(v) if pd.notna(v) else 0.0
            for k in range(HORIZON):
                v = row.get(f"fire_label_t{k+1}", 0)
                label_arr[d_idx, z_idx, k] = int(v) if pd.notna(v) else 0

        # Sliding window
        X_list, y_list = [], []
        for t in range(LOOKBACK, n_days - HORIZON + 1):
            X_list.append(feat_arr[t - LOOKBACK: t])   # (LOOKBACK, Z, F)
            y_list.append(label_arr[t].T)               # (HORIZON, Z)
            meta_rows.append({
                "sample_id"    : sample_id,
                "district_code": dist_code,
                "district_name": DISTRICT_INFO[dist_code]["name"],
                "date"         : dates[t].strftime("%Y-%m-%d"),
            })
            sample_id += 1

        if X_list:
            X_dict[dist_code] = np.array(X_list, dtype=np.float32)
            y_dict[dist_code] = np.array(y_list, dtype=np.float32)
            name = DISTRICT_INFO[dist_code]["name"]
            log.info(
                f"  {dist_code} ({name}):  "
                f"X={X_dict[dist_code].shape}  "
                f"y={y_dict[dist_code].shape}"
            )

    return X_dict, y_dict, meta_rows


# =============================================================================
# STEP 9  Save outputs
# =============================================================================
def save_outputs(df_spread, X_dict, y_dict, meta_rows,
                 adjacency, zone_index, feat_cols,
                 scaler_params, district_zones):
    log.info(f"\nSaving to {OUT_DIR} ...")

    # 1. master_table_spread
    df_spread.to_csv(OUT_DIR / "master_table_spread.csv", index=False)
    log.info(f"  ✓ master_table_spread.csv  ({len(df_spread):,} rows)")
    try:
        df_spread.to_parquet(OUT_DIR / "master_table_spread.parquet", index=False)
        log.info("  ✓ master_table_spread.parquet")
    except Exception:
        log.warning("  ✗ parquet skipped (install pyarrow if needed)")

    # 2. Zone adjacency  — keys are dist_code, sub-keys are int zone numbers
    with open(OUT_DIR / "zone_adjacency.json", "w") as f:
        json.dump(adjacency, f, indent=2)
    log.info("  ✓ zone_adjacency.json")

    # 3. Zone integer index
    with open(OUT_DIR / "district_zone_index.json", "w") as f:
        json.dump(zone_index, f, indent=2)
    log.info("  ✓ district_zone_index.json")

    # 4. Feature metadata
    feat_meta = {
        "lookback"        : LOOKBACK,
        "horizon"         : HORIZON,
        "n_features"      : len(feat_cols),
        "feature_cols"    : feat_cols,
        "static_features" : [c for c in STATIC_FEATURES  if c in feat_cols],
        "dynamic_features": [c for c in DYNAMIC_FEATURES if c in feat_cols],
        "spread_features" : [
            "neighbour_fire_count", "neighbour_fire_frac",
            "any_neighbour_fire", "wind_sin", "wind_cos",
            "fire_weather_index", "dry_spell_days",
        ],
        "target_cols"     : TARGET_COLS,
        "district_info"   : DISTRICT_INFO,
        "notes": {
            "district_col": "District_0 … District_4 (matches parquet)",
            "zone_col"    : "integer 1 … N (matches parquet)",
            "zone_index"  : "0-based tensor index: zone 1 → index 0",
        },
    }
    with open(OUT_DIR / "feature_meta.json", "w") as f:
        json.dump(feat_meta, f, indent=2)
    log.info("  ✓ feature_meta.json")

    # 5. Zone geometry details
    zone_rows = []
    for dist_code, zones_gdf in district_zones.items():
        name = DISTRICT_INFO[dist_code]["name"]
        for _, zrow in zones_gdf.iterrows():
            zn = int(zrow["zone"])
            zone_rows.append({
                "district_code": dist_code,
                "district_name": name,
                "zone"         : zn,
                "tensor_index" : zone_index.get(dist_code, {}).get(zn, -1),
                "area_sqm"     : float(zrow["area"]),
                "n_neighbours" : len(adjacency.get(dist_code, {}).get(zn, [])),
            })
    pd.DataFrame(zone_rows).to_csv(
        OUT_DIR / "zone_geometry_details.csv", index=False
    )
    log.info("  ✓ zone_geometry_details.csv")

    # 6. Per-district tensors  (filename uses human name for clarity)
    for dist_code, X in X_dict.items():
        safe = DISTRICT_INFO[dist_code]["name"].lower()   # e.g. "banke"
        np.save(OUT_DIR / f"X_{safe}.npy", X)
        np.save(OUT_DIR / f"y_{safe}.npy", y_dict[dist_code])
        log.info(
            f"  ✓ X_{safe}.npy  {X.shape}   "
            f"y_{safe}.npy  {y_dict[dist_code].shape}"
        )

    # 7. Sample metadata
    pd.DataFrame(meta_rows).to_csv(OUT_DIR / "sample_meta.csv", index=False)
    log.info(f"  ✓ sample_meta.csv  ({len(meta_rows):,} samples)")

    # 8. Scaler params
    with open(OUT_DIR / "scaler_params.json", "w") as f:
        json.dump(scaler_params, f, indent=2)
    log.info("  ✓ scaler_params.json")

    # 9. Dataset report
    _write_report(df_spread, X_dict, y_dict, meta_rows, adjacency, feat_cols)


def _write_report(df, X_dict, y_dict, meta_rows, adjacency, feat_cols):
    lines = []
    a = lines.append
    sep = "=" * 70
    a(sep);  a("FFSFM DATASET REPORT");  a(sep)
    a(f"Generated      : {pd.Timestamp.now():%Y-%m-%d %H:%M:%S}")
    a(f"Data window    : {DATE_START}  →  {DATE_END}")
    a(f"Lookback       : {LOOKBACK} days   |   Horizon: {HORIZON} days")
    a(f"Features (F)   : {len(feat_cols)}")
    a(f"Total rows     : {len(df):,}")
    a("")
    hdr = f"{'Code':<12} {'Name':<10} {'Zones':>6} {'Rows':>10} {'Fire=1':>8} {'Fire%':>7}"
    a(hdr);  a("-" * len(hdr))
    for code, info in DISTRICT_INFO.items():
        sub = df[df["district"] == code]
        nf  = int(sub["fire_label"].sum()) if "fire_label" in sub.columns else 0
        pct = 100 * nf / max(len(sub), 1)
        adj = adjacency.get(code, {})
        a(f"{code:<12} {info['name']:<10} {len(adj):>6} {len(sub):>10,} {nf:>8,} {pct:>6.2f}%")
    a("")
    a(f"{'Code':<12} {'Name':<10}  {'X shape':<30}  {'y shape'}")
    a("-" * 70)
    for code, X in X_dict.items():
        name = DISTRICT_INFO[code]["name"]
        a(f"{code:<12} {name:<10}  {str(X.shape):<30}  {str(y_dict[code].shape)}")
    a(f"\nTotal samples  : {len(meta_rows):,}")
    a("")
    a("Adjacency (avg neighbours per zone)")
    a("-" * 40)
    for code, adj in adjacency.items():
        avg = np.mean([len(v) for v in adj.values()]) if adj else 0
        a(f"  {code} ({DISTRICT_INFO[code]['name']:<10}): {avg:.2f} avg neighbours")
    a(sep)

    with open(OUT_DIR / "dataset_report.txt", "w") as f:
        f.write("\n".join(lines))
    log.info("  ✓ dataset_report.txt")


# =============================================================================
# Main
# =============================================================================
def main():
    log.info("=" * 60)
    log.info("FFSFM Dataset Builder  v3  —  start")
    log.info("=" * 60)

    # 1. Reconstruct zones (same grid logic as zone division script)
    district_zones = reconstruct_zones_from_shapefile()

    # 2. Adjacency from zone geometries
    adjacency = build_adjacency(district_zones)

    # 3. Zone integer index (zone int → 0-based tensor axis)
    zone_index = build_zone_index(district_zones)

    # 4. Load FFOPM
    df = load_ffopm()

    # 5. Clean  (district stays 'District_N', zone stays int)
    df = clean_ffopm(df)

    # 6. Spread feature engineering
    df = engineer_spread_features(df, adjacency)

    # 7. Resolve feature columns (only those present in the dataframe)
    spread_extras = [
        "neighbour_fire_count", "neighbour_fire_frac", "any_neighbour_fire",
        "wind_sin", "wind_cos", "fire_weather_index", "dry_spell_days",
    ]
    all_feats = STATIC_FEATURES + DYNAMIC_FEATURES + spread_extras
    feat_cols = [c for c in dict.fromkeys(all_feats) if c in df.columns]
    log.info(f"Feature columns resolved: {len(feat_cols)}")

    # 8. Z-score normalise
    df, scaler_params = normalise(df, feat_cols)

    # 9. Build tensors
    X_dict, y_dict, meta_rows = build_tensors(df, feat_cols, zone_index)

    # 10. Save everything
    save_outputs(df, X_dict, y_dict, meta_rows,
                 adjacency, zone_index, feat_cols,
                 scaler_params, district_zones)

    log.info("=" * 60)
    log.info("FFSFM Dataset Builder  v3  —  DONE ✓")
    log.info(f"Output: {OUT_DIR}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()