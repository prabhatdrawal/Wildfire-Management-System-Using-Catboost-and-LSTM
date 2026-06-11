"""
=============================================================================
FFSFM  —  Fire Spread Forecast  |  Streamlit App
=============================================================================
Features:
  • Manual input of today's weather / vegetation / fire data
  • Animated 7-day fire spread on the actual polygon map
  • Different colour per forecast day (Day 1 → Day 7)
  • Zone selector — click any zone to inspect its 7-day probability chart
  • District switcher (all 5 districts)
  • Works with trained .pt checkpoints OR in demo mode (random predictions)

Run:
  cd /Users/prabhatrawal/Minor_project_code/ffsfm/ffsfm_code
  streamlit run ffsfm_app.py
=============================================================================
"""

# ── Standard library ───────────────────────────────────────────────────────
import json
import math
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Third-party ────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.collections import PatchCollection
from shapely.geometry import box
from shapely.ops import unary_union
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap

# ── Paths ──────────────────────────────────────────────────────────────────
BASE        = Path("/Users/prabhatrawal/Minor_project_code")
DATA_DIR    = BASE / "ffsfm_data"
MODEL_DIR   = DATA_DIR / "models"
SHP_FILE    = BASE / "polygon_file" / "actual_timezone_designated_district_using_EPSG_32644.shp"
FEAT_META   = DATA_DIR / "feature_meta.json"
SCALER_FILE = DATA_DIR / "scaler_params.json"
ADJ_FILE    = DATA_DIR / "zone_adjacency.json"

# ── District config ────────────────────────────────────────────────────────
DISTRICTS = {
    "Banke"  : {"code": "District_0", "n_zones": 14, "name3": "Banke"},
    "Bardiya": {"code": "District_1", "n_zones": 15, "name3": "Bardiya"},
    "Surkhet": {"code": "District_2", "n_zones": 19, "name3": "Surkhet"},
    "Dang"   : {"code": "District_3", "n_zones": 17, "name3": "Dang"},
    "Salyan" : {"code": "District_4", "n_zones": 14, "name3": "Salyan"},
}

# 7 colours — one per forecast day (cool→hot progression)
DAY_COLORS = [
    "#FFE066",   # Day 1 — pale yellow
    "#FFB347",   # Day 2 — light orange
    "#FF7F2A",   # Day 3 — orange
    "#FF4500",   # Day 4 — orange-red
    "#E8000D",   # Day 5 — red
    "#B50000",   # Day 6 — deep red
    "#7B0000",   # Day 7 — dark maroon
]

HORIZON = 7
LOOKBACK = 14

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="FFSFM — Fire Spread Forecast",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
  .main { background-color: #0e1117; }
  .stSidebar { background-color: #161b22; }
  .block-container { padding-top: 1rem; }
  h1 { color: #FF6B35; font-weight: 800; }
  h2, h3 { color: #FFA07A; }
  .metric-card {
      background: #1e2a38;
      border-radius: 8px;
      padding: 12px 16px;
      border-left: 4px solid #FF6B35;
      margin: 4px 0;
  }
  .day-badge {
      display: inline-block;
      padding: 2px 10px;
      border-radius: 12px;
      font-size: 0.8em;
      font-weight: 700;
      margin: 2px;
  }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Cached loaders
# =============================================================================
@st.cache_data(show_spinner=False)
def load_shapefile():
    if not SHP_FILE.exists():
        return None
    return gpd.read_file(SHP_FILE)

@st.cache_data(show_spinner=False)
def load_feature_meta():
    if not FEAT_META.exists():
        return None
    with open(FEAT_META) as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_scaler():
    if not SCALER_FILE.exists():
        return None
    with open(SCALER_FILE) as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_adjacency():
    if not ADJ_FILE.exists():
        return {}
    with open(ADJ_FILE) as f:
        raw = json.load(f)
    # Convert string keys to int
    out = {}
    for dist_code, adj in raw.items():
        out[dist_code] = {int(k): [int(v) for v in vs]
                          for k, vs in adj.items()}
    return out

@st.cache_resource(show_spinner=False)
def load_model(district_name: str, n_zones: int):
    """Load PyTorch model. Returns None if not found."""
    ckpt_path = MODEL_DIR / f"best_{district_name.lower()}.pt"
    if not ckpt_path.exists():
        return None
    try:
        import torch
        from ffsfm_model import build_model
        device = torch.device("cpu")
        model  = build_model(n_zones=n_zones, device=device)
        ckpt   = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model
    except Exception as e:
        st.warning(f"Could not load model for {district_name}: {e}")
        return None


# =============================================================================
# Zone reconstruction (mirrors extraction script)
# =============================================================================
@st.cache_data(show_spinner=False)
def get_zone_geodataframes():
    """Returns {district_name: GeoDataFrame with zone polygons}"""
    gdf_all = load_shapefile()
    if gdf_all is None:
        return {}

    union_poly = gdf_all.geometry.unary_union
    minx, miny, maxx, maxy = union_poly.bounds
    width  = maxx - minx
    height = maxy - miny
    total_area = gdf_all.geometry.area.sum()
    n_districts    = len(gdf_all)
    cell_size      = math.sqrt(total_area / (10 * n_districts))
    n_cols = math.ceil(width  / cell_size)
    n_rows = math.ceil(height / cell_size)
    dx = width  / n_cols
    dy = height / n_rows

    grid_boxes = []
    for i in range(n_cols):
        for j in range(n_rows):
            xmin = minx + i * dx
            grid_boxes.append(box(xmin, miny + j * dy,
                                  xmin + dx, miny + (j+1) * dy))

    result = {}
    name3_map = {v["name3"]: k for k, v in DISTRICTS.items()}

    for _, row in gdf_all.iterrows():
        name3 = row.get("NAME_3", row.get("NAME", "Unknown"))
        dist_name = name3_map.get(name3)
        if dist_name is None:
            continue

        # Intersect grid with district polygon
        zones = []
        for gb in grid_boxes:
            inter = gb.intersection(row["geometry"])
            if not inter.is_empty:
                zones.append({"geometry": inter})
        if not zones:
            continue

        zdf = gpd.GeoDataFrame(zones, crs=gdf_all.crs)
        zdf["area"] = zdf.geometry.area

        # Merge small zones (same logic as extraction script)
        n_target   = 10
        t_area     = row["geometry"].area / n_target
        thresh     = 0.25 * t_area
        merged = True
        while merged:
            merged = False
            small = zdf[zdf["area"] < thresh].copy()
            if small.empty:
                break
            for idx, sr in small.iterrows():
                if idx not in zdf.index:
                    continue
                nb = zdf[zdf.geometry.touches(sr.geometry) & (zdf.index != idx)]
                if nb.empty:
                    continue
                si = nb["area"].idxmin()
                mg = unary_union([sr.geometry, zdf.at[si, "geometry"]])
                zdf.at[si, "geometry"] = mg
                zdf.at[si, "area"]     = mg.area
                zdf = zdf.drop(idx)
                merged = True
                break

        zdf = zdf.reset_index(drop=True)
        zdf["zone"]     = zdf.index + 1
        zdf["district"] = dist_name
        result[dist_name] = zdf.to_crs(epsg=4326)   # convert to lat/lon for display

    return result


# =============================================================================
# Prediction
# =============================================================================
def run_model_prediction(input_features: dict,
                          district_name: str,
                          n_zones: int,
                          scaler: dict,
                          feat_cols: list,
                          selected_zone: int) -> np.ndarray:
    """
    Build a synthetic (14, Z, F) input tensor from the user's manual inputs
    and run the model (or demo mode).
    Returns spread_probs: (7, Z)  — probabilities per day per zone.
    """
    model = load_model(district_name, n_zones)
    F     = len(feat_cols)

    # Build normalised feature vector for the selected zone
    feat_vec = np.zeros(F, dtype=np.float32)
    for i, col in enumerate(feat_cols):
        raw_val = input_features.get(col, 0.0)
        if scaler and col in scaler:
            mu  = scaler[col]["mean"]
            std = scaler[col]["std"]
            feat_vec[i] = (raw_val - mu) / std
        else:
            feat_vec[i] = float(raw_val)

    # Tile across all zones (same input for all zones — user inputs one zone)
    # For the selected zone use full values; for others use 0 (no fire)
    X = np.zeros((1, LOOKBACK, n_zones, F), dtype=np.float32)
    z_idx = selected_zone - 1   # 0-based
    for t in range(LOOKBACK):
        X[0, t, z_idx, :] = feat_vec
        # Decay over lookback: earlier days have less fire signal
        decay = (t + 1) / LOOKBACK
        X[0, t, z_idx, 0] *= decay   # total_fire_pixels decays back in time

    if model is not None:
        try:
            import torch
            with torch.no_grad():
                logits = model(torch.tensor(X))
                probs  = torch.sigmoid(logits).numpy()[0]   # (7, Z)
            return probs
        except Exception as e:
            st.warning(f"Model inference failed ({e}). Using demo mode.")

    # ── Demo mode: physics-inspired spread simulation ──────────────────────
    return _demo_spread(
        selected_zone, n_zones, input_features,
        load_adjacency().get(DISTRICTS[district_name]["code"], {})
    )


def _demo_spread(origin_zone: int, n_zones: int,
                 inputs: dict, adjacency: dict) -> np.ndarray:
    """
    Simulate fire spread without a model.
    - Starts at origin_zone with high probability
    - Spreads to neighbours each day, modulated by wind & VPD
    """
    probs = np.zeros((HORIZON, n_zones), dtype=np.float32)
    wind_speed  = inputs.get("wind_speed_ms", 2.0)
    vpd         = inputs.get("vapor_pressure_deficit_kpa", 1.0)
    soil_moist  = inputs.get("soil_moisture_m3m3", 0.2)
    spread_rate = min(0.95, 0.3 + 0.1 * wind_speed + 0.15 * vpd
                      - 0.2 * soil_moist)

    # Day 0 (t+1): origin zone
    burning = {origin_zone - 1}   # 0-based
    probs[0, origin_zone - 1] = 0.90

    for day in range(1, HORIZON):
        new_burning = set()
        for z in burning:
            zone_int = z + 1   # 1-based for adjacency lookup
            probs[day, z] = min(1.0, probs[day - 1, z] + 0.05)  # persistence
            for nb in adjacency.get(zone_int, []):
                nb_idx = nb - 1
                if nb_idx < n_zones:
                    spread_p = spread_rate * (0.7 ** day)
                    probs[day, nb_idx] = max(probs[day, nb_idx], spread_p)
                    if spread_p > 0.3:
                        new_burning.add(nb_idx)
        burning |= new_burning

    # Add small background risk for all zones
    noise = np.random.uniform(0, 0.05, probs.shape)
    probs = np.clip(probs + noise, 0, 1)
    return probs


# =============================================================================
# Map drawing
# =============================================================================
def draw_spread_map(zone_gdf: gpd.GeoDataFrame,
                    probs: np.ndarray,
                    day: int,
                    selected_zone: int,
                    threshold: float = 0.3,
                    title: str = "") -> plt.Figure:
    """
    Draw one frame of the spread map for the given day.
    probs : (7, Z)
    day   : 0-indexed forecast day (0 = Day 1)
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1a2332")

    day_probs = probs[day]   # (Z,)
    n_zones   = len(zone_gdf)

    for _, row in zone_gdf.iterrows():
        z_idx = int(row["zone"]) - 1
        if z_idx >= len(day_probs):
            continue
        p = float(day_probs[z_idx])

        # Fill colour based on probability
        if p >= threshold:
            alpha = 0.4 + 0.6 * p
            facecolor = mcolors.to_rgba(DAY_COLORS[day], alpha=alpha)
        else:
            facecolor = (0.15, 0.20, 0.28, 0.8)   # dark base colour

        # Border
        is_selected = int(row["zone"]) == selected_zone
        edge_color  = "#FFD700" if is_selected else "#4a6080"
        edge_width  = 2.5 if is_selected else 0.8

        try:
            geom = row["geometry"]
            if geom.geom_type == "Polygon":
                patch = plt.Polygon(
                    list(geom.exterior.coords),
                    closed=True,
                    facecolor=facecolor,
                    edgecolor=edge_color,
                    linewidth=edge_width,
                )
                ax.add_patch(patch)
            elif geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    patch = plt.Polygon(
                        list(poly.exterior.coords),
                        closed=True,
                        facecolor=facecolor,
                        edgecolor=edge_color,
                        linewidth=edge_width,
                    )
                    ax.add_patch(patch)
        except Exception:
            pass

        # Zone label
        try:
            cx = geom.centroid.x
            cy = geom.centroid.y
            label_color = "#FFD700" if is_selected else (
                "white" if p >= threshold else "#8899aa"
            )
            fontsize = 7.5
            weight   = "bold" if p >= threshold or is_selected else "normal"
            ax.text(cx, cy, str(int(row["zone"])),
                    ha="center", va="center",
                    fontsize=fontsize, color=label_color,
                    fontweight=weight,
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="#00000066",
                              edgecolor="none"))
        except Exception:
            pass

    # Auto-extent
    bounds = zone_gdf.total_bounds   # minx, miny, maxx, maxy
    pad = (bounds[2] - bounds[0]) * 0.05
    ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
    ax.set_ylim(bounds[1] - pad, bounds[3] + pad)

    # Title
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Longitude", color="#8899aa", fontsize=8)
    ax.set_ylabel("Latitude",  color="#8899aa", fontsize=8)
    ax.tick_params(colors="#8899aa", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a4a5a")

    # Colour bar for probability
    sm = plt.cm.ScalarMappable(
        cmap=LinearSegmentedColormap.from_list(
            "fire", ["#1a2332", "#FFE066", "#FF4500", "#7B0000"]
        ),
        norm=plt.Normalize(0, 1)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
    cbar.set_label("Fire Spread Probability", color="white", fontsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Threshold line annotation
    cbar.ax.axhline(y=threshold, color="#FFD700", linewidth=1.5, linestyle="--")

    fig.tight_layout()
    return fig


def draw_zone_timeseries(probs: np.ndarray,
                          selected_zone: int,
                          district_name: str) -> plt.Figure:
    """
    Bar chart of 7-day fire probability for the selected zone.
    """
    z_idx = selected_zone - 1
    if z_idx >= probs.shape[1]:
        z_idx = 0
    zone_probs = probs[:, z_idx]

    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1a2332")

    days  = [f"Day {d+1}" for d in range(HORIZON)]
    bars  = ax.bar(days, zone_probs, color=DAY_COLORS, edgecolor="#2a3a4a",
                   linewidth=0.8, width=0.65)

    # Value labels on bars
    for bar, p in zip(bars, zone_probs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{p:.2f}",
                ha="center", va="bottom",
                color="white", fontsize=9, fontweight="bold")

    ax.axhline(y=0.3, color="#FFD700", linewidth=1.2, linestyle="--", alpha=0.7,
               label="Threshold (0.3)")
    ax.axhline(y=0.5, color="#FF4500", linewidth=1.0, linestyle=":",  alpha=0.6,
               label="High risk (0.5)")

    ax.set_ylim(0, 1.15)
    ax.set_title(
        f"Zone {selected_zone}  —  7-Day Fire Spread Probability  ({district_name})",
        color="white", fontsize=11, fontweight="bold"
    )
    ax.set_ylabel("Probability", color="#8899aa", fontsize=9)
    ax.tick_params(colors="#8899aa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#3a4a5a")
    ax.legend(facecolor="#1a2332", edgecolor="#3a4a5a",
              labelcolor="white", fontsize=8)
    fig.tight_layout()
    return fig


def draw_all_zones_heatmap(probs: np.ndarray, district_name: str) -> plt.Figure:
    """
    Heatmap: rows = zones, cols = forecast days.
    """
    n_zones = probs.shape[1]
    fig, ax = plt.subplots(figsize=(9, max(4, n_zones * 0.38)))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    cmap = LinearSegmentedColormap.from_list(
        "fire_heat", ["#1a2332", "#FFE066", "#FF4500", "#7B0000"]
    )
    im = ax.imshow(probs.T, aspect="auto", cmap=cmap,
                   vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(range(HORIZON))
    ax.set_xticklabels([f"Day {d+1}" for d in range(HORIZON)],
                       color="white", fontsize=9)
    ax.set_yticks(range(n_zones))
    ax.set_yticklabels([f"Zone {z+1}" for z in range(n_zones)],
                       color="white", fontsize=8)

    # Cell values
    for h in range(HORIZON):
        for z in range(n_zones):
            p = probs[h, z]
            txt_color = "black" if p > 0.6 else "white"
            ax.text(h, z, f"{p:.2f}", ha="center", va="center",
                    fontsize=6.5, color=txt_color)

    ax.set_title(f"{district_name}  —  Fire Spread Probability Heatmap (All Zones)",
                 color="white", fontsize=11, fontweight="bold", pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=7)
    cbar.set_label("Probability", color="white", fontsize=8)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    fig.tight_layout()
    return fig


# =============================================================================
# Sidebar — input form
# =============================================================================
def render_sidebar(feat_cols: list) -> tuple:
    """Returns (selected_district, selected_zone, input_values, threshold, speed)"""
    st.sidebar.markdown("## 🔥 FFSFM")
    st.sidebar.markdown("**Fire Spread Forecast Model**")
    st.sidebar.divider()

    # District
    st.sidebar.markdown("### 📍 District")
    district_name = st.sidebar.selectbox(
        "Select district", list(DISTRICTS.keys()), index=0,
        label_visibility="collapsed"
    )
    n_zones = DISTRICTS[district_name]["n_zones"]

    # Zone
    st.sidebar.markdown("### 🎯 Origin Zone")
    selected_zone = st.sidebar.slider(
        "Zone number (fire origin)", 1, n_zones, 1,
        help="The zone where fire is observed today"
    )

    st.sidebar.divider()
    st.sidebar.markdown("### 🌤️ Today's Conditions")

    inputs = {}

    # ── Weather inputs ────────────────────────────────────────────────────
    with st.sidebar.expander("🌡️ Temperature & Humidity", expanded=True):
        inputs["temperature_2m_celsius"] = st.slider(
            "Temperature (°C)", -5.0, 45.0, 28.0, 0.5
        )
        inputs["relative_humidity_pct"] = st.slider(
            "Relative Humidity (%)", 5.0, 100.0, 35.0, 1.0
        )
        inputs["dewpoint_2m_celsius"] = st.slider(
            "Dew Point (°C)", -10.0, 30.0, 10.0, 0.5
        )
        inputs["skin_temperature_celsius"] = st.slider(
            "Skin Temperature (°C)", -5.0, 55.0, 32.0, 0.5
        )

    with st.sidebar.expander("💨 Wind", expanded=True):
        inputs["wind_speed_ms"] = st.slider(
            "Wind Speed (m/s)", 0.0, 20.0, 3.0, 0.1
        )
        inputs["wind_direction_deg"] = st.slider(
            "Wind Direction (°)", 0.0, 360.0, 180.0, 5.0
        )
        inputs["u_wind_component_ms"] = inputs["wind_speed_ms"] * np.sin(
            np.deg2rad(inputs["wind_direction_deg"])
        )
        inputs["v_wind_component_ms"] = inputs["wind_speed_ms"] * np.cos(
            np.deg2rad(inputs["wind_direction_deg"])
        )

    with st.sidebar.expander("🌱 Soil & Precipitation"):
        inputs["soil_moisture_m3m3"] = st.slider(
            "Soil Moisture (m³/m³)", 0.0, 0.6, 0.15, 0.01
        )
        inputs["soil_temperature_celsius"] = st.slider(
            "Soil Temperature (°C)", 0.0, 50.0, 25.0, 0.5
        )
        inputs["precipitation_mm"] = st.slider(
            "Precipitation (mm)", 0.0, 50.0, 0.0, 0.1
        )

    with st.sidebar.expander("🌿 Vegetation"):
        inputs["ndvi_composite"] = st.slider(
            "NDVI", -0.2, 1.0, 0.35, 0.01
        )
        inputs["s2_ndvi"]   = inputs["ndvi_composite"]
        inputs["s2_ndwi"]   = st.slider("NDWI (moisture)", -0.5, 0.5, -0.1, 0.01)
        inputs["s2_nbr"]    = st.slider("NBR (burn ratio)", -1.0, 1.0, 0.2, 0.01)
        inputs["landsat_ndvi"] = inputs["ndvi_composite"] * 0.98

    with st.sidebar.expander("🔥 Fire Observation"):
        inputs["total_fire_pixels"] = st.slider(
            "Fire Pixels (today)", 0, 100, 0, 1
        )
        inputs["fire_percentage"] = st.slider(
            "Fire Coverage (%)", 0.0, 100.0, 0.0, 0.1
        )
        inputs["lst_day_c"] = st.slider(
            "Land Surface Temp (°C)", 20.0, 70.0, 35.0, 0.5
        )

    # VPD derived
    T  = inputs["temperature_2m_celsius"]
    Td = inputs["dewpoint_2m_celsius"]
    es = 6.112 * np.exp(17.67 * T  / (T  + 243.5))
    ea = 6.112 * np.exp(17.67 * Td / (Td + 243.5))
    inputs["vapor_pressure_deficit_kpa"] = max(0.0, (es - ea) / 10.0)

    # Fill remaining features with sensible defaults
    for col in (feat_cols or []):
        if col not in inputs:
            inputs[col] = 0.0

    st.sidebar.divider()
    st.sidebar.markdown("### ⚙️ Display Settings")
    threshold = st.sidebar.slider(
        "Fire threshold", 0.1, 0.9, 0.3, 0.05,
        help="Probability above which a zone is shown as 'on fire'"
    )
    speed = st.sidebar.select_slider(
        "Animation speed",
        options=["Slow", "Normal", "Fast"],
        value="Normal"
    )

    return district_name, selected_zone, inputs, threshold, speed


# =============================================================================
# Main app
# =============================================================================
def main():
    # ── Load resources ─────────────────────────────────────────────────────
    feat_meta  = load_feature_meta()
    scaler     = load_scaler()
    zone_gdfs  = get_zone_geodataframes()

    feat_cols = feat_meta["feature_cols"] if feat_meta else []

    # ── Sidebar ────────────────────────────────────────────────────────────
    district_name, selected_zone, inputs, threshold, speed = render_sidebar(feat_cols)
    n_zones   = DISTRICTS[district_name]["n_zones"]
    zone_gdf  = zone_gdfs.get(district_name)

    speed_map = {"Slow": 1.5, "Normal": 0.7, "Fast": 0.2}
    frame_delay = speed_map[speed]

    # ── Header ─────────────────────────────────────────────────────────────
    col_title, col_meta = st.columns([3, 1])
    with col_title:
        st.markdown(f"## 🔥 Fire Spread Forecast — {district_name}")
        st.markdown(
            f"**Origin Zone:** `Zone {selected_zone}`  |  "
            f"**District:** `{DISTRICTS[district_name]['code']}`  |  "
            f"**Zones:** `{n_zones}`  |  "
            f"**Horizon:** `7 days`"
        )

    with col_meta:
        model_loaded = (MODEL_DIR / f"best_{district_name.lower()}.pt").exists()
        st.markdown(
            f"**Model:** {'✅ Loaded' if model_loaded else '⚠️ Demo mode'}"
        )
        vpd = inputs.get("vapor_pressure_deficit_kpa", 0)
        ws  = inputs.get("wind_speed_ms", 0)
        sm  = inputs.get("soil_moisture_m3m3", 0)
        fire_risk = "🔴 HIGH" if vpd > 1.5 and ws > 5 else (
                    "🟡 MODERATE" if vpd > 0.8 else "🟢 LOW")
        st.markdown(f"**Fire Risk:** {fire_risk}")
        st.markdown(f"**VPD:** `{vpd:.2f} kPa`")

    st.divider()

    # ── Run prediction ──────────────────────────────────────────────────────
    if st.button("🚀 Run 7-Day Spread Forecast", type="primary", use_container_width=True):
        with st.spinner("Running ConvBiLSTM forecast..."):
            probs = run_model_prediction(
                inputs, district_name, n_zones,
                scaler, feat_cols, selected_zone
            )
        st.session_state["probs"]         = probs
        st.session_state["district_name"] = district_name
        st.session_state["selected_zone"] = selected_zone
        st.session_state["zone_gdf"]      = zone_gdf
        st.session_state["threshold"]     = threshold
        st.session_state["n_zones"]       = n_zones

    # ── Display results ─────────────────────────────────────────────────────
    if "probs" in st.session_state and st.session_state.get("district_name") == district_name:
        probs         = st.session_state["probs"]
        cached_zone   = st.session_state.get("selected_zone", selected_zone)
        cached_gdf    = st.session_state.get("zone_gdf", zone_gdf)
        cached_thresh = st.session_state.get("threshold", threshold)

        # ── Tab layout ──────────────────────────────────────────────────
        tab_anim, tab_zone, tab_heat, tab_stats = st.tabs([
            "🗺️ Animated Spread Map",
            "📍 Zone Detail",
            "🌡️ Heatmap (All Zones)",
            "📊 Stats & Summary",
        ])

        # ── Tab 1: Animated map ─────────────────────────────────────────
        with tab_anim:
            st.markdown("#### 7-Day Fire Spread Animation")

            # Day colour legend
            legend_html = " ".join([
                f'<span class="day-badge" style="background:{DAY_COLORS[d]};'
                f'color:{"black" if d < 2 else "white"}">Day {d+1}</span>'
                for d in range(HORIZON)
            ])
            st.markdown(legend_html, unsafe_allow_html=True)
            st.markdown("")

            map_placeholder = st.empty()
            prog_bar        = st.progress(0)
            day_label       = st.empty()

            # Auto-play animation
            for day in range(HORIZON):
                prog_bar.progress((day + 1) / HORIZON)
                day_label.markdown(
                    f"<h4 style='color:{DAY_COLORS[day]};text-align:center;'>"
                    f"📅 Forecast Day {day+1}  — "
                    f"{pd.Timestamp.now() + pd.Timedelta(days=day+1):%B %d, %Y}"
                    f"</h4>",
                    unsafe_allow_html=True
                )

                if cached_gdf is not None:
                    n_fire = int((probs[day] >= cached_thresh).sum())
                    fig = draw_spread_map(
                        cached_gdf, probs, day,
                        selected_zone=cached_zone,
                        threshold=cached_thresh,
                        title=(f"Day {day+1}  |  {district_name}  |  "
                               f"Zones at risk: {n_fire}/{n_zones}  "
                               f"(threshold={cached_thresh:.2f})")
                    )
                    map_placeholder.pyplot(fig, use_container_width=True)
                    plt.close(fig)
                else:
                    # Fallback text display when shapefile unavailable
                    at_risk = [z+1 for z in range(n_zones)
                               if probs[day, z] >= cached_thresh]
                    map_placeholder.warning(
                        f"Shapefile not found. Zones at risk on Day {day+1}: "
                        f"{at_risk if at_risk else 'None'}"
                    )

                time.sleep(frame_delay)

            # Static frame selector after animation
            st.markdown("---")
            st.markdown("#### 🔍 Inspect a Specific Day")
            view_day = st.select_slider(
                "Select forecast day",
                options=[f"Day {d+1}" for d in range(HORIZON)],
                value="Day 1",
            )
            d_idx = int(view_day.split()[1]) - 1
            if cached_gdf is not None:
                n_fire = int((probs[d_idx] >= cached_thresh).sum())
                fig2 = draw_spread_map(
                    cached_gdf, probs, d_idx,
                    selected_zone=cached_zone,
                    threshold=cached_thresh,
                    title=(f"Day {d_idx+1}  |  {district_name}  |  "
                           f"Zones at risk: {n_fire}/{n_zones}")
                )
                st.pyplot(fig2, use_container_width=True)
                plt.close(fig2)

        # ── Tab 2: Zone detail ──────────────────────────────────────────
        with tab_zone:
            st.markdown(f"#### Zone {cached_zone} — 7-Day Fire Probability")

            # Metric cards for each day
            cols = st.columns(7)
            for d in range(HORIZON):
                z_idx = min(cached_zone - 1, probs.shape[1] - 1)
                p = probs[d, z_idx]
                risk = "🔴" if p >= 0.5 else ("🟡" if p >= 0.3 else "🟢")
                cols[d].markdown(
                    f"<div class='metric-card' style='border-left-color:{DAY_COLORS[d]}'>"
                    f"<b style='color:{DAY_COLORS[d]}'>Day {d+1}</b><br>"
                    f"<span style='font-size:1.4em;font-weight:800;color:white'>"
                    f"{p:.0%}</span><br>{risk}</div>",
                    unsafe_allow_html=True
                )

            st.markdown("")
            fig_ts = draw_zone_timeseries(probs, cached_zone, district_name)
            st.pyplot(fig_ts, use_container_width=True)
            plt.close(fig_ts)

            # Zone picker to compare
            st.markdown("---")
            st.markdown("#### Compare Another Zone")
            compare_zone = st.selectbox(
                "Select zone to compare",
                options=list(range(1, n_zones + 1)),
                index=min(cached_zone, n_zones - 1),
            )
            if compare_zone != cached_zone:
                fig_cmp = draw_zone_timeseries(probs, compare_zone, district_name)
                st.pyplot(fig_cmp, use_container_width=True)
                plt.close(fig_cmp)

        # ── Tab 3: Heatmap ──────────────────────────────────────────────
        with tab_heat:
            st.markdown("#### All Zones × All Days — Fire Spread Heatmap")
            fig_hm = draw_all_zones_heatmap(probs, district_name)
            st.pyplot(fig_hm, use_container_width=True)
            plt.close(fig_hm)

        # ── Tab 4: Stats ────────────────────────────────────────────────
        with tab_stats:
            st.markdown("#### Forecast Summary Statistics")

            col_a, col_b = st.columns(2)
            with col_a:
                df_stats = pd.DataFrame({
                    "Day": [f"Day {d+1}" for d in range(HORIZON)],
                    "Zones at Risk": [
                        int((probs[d] >= cached_thresh).sum())
                        for d in range(HORIZON)
                    ],
                    "Max Probability": [
                        f"{probs[d].max():.3f}" for d in range(HORIZON)
                    ],
                    "Mean Probability": [
                        f"{probs[d].mean():.3f}" for d in range(HORIZON)
                    ],
                    "High Risk Zones (≥0.5)": [
                        int((probs[d] >= 0.5).sum()) for d in range(HORIZON)
                    ],
                })
                st.dataframe(df_stats, use_container_width=True, hide_index=True)

            with col_b:
                # Peak risk day
                peak_day  = int(probs.max(axis=1).argmax())
                peak_zone = int(probs[peak_day].argmax()) + 1
                peak_prob = float(probs[peak_day].max())
                total_at_risk = int((probs >= cached_thresh).any(axis=0).sum())

                st.markdown(f"""
                <div class='metric-card'>
                  <b style='color:#FFB347'>Peak Risk Day</b><br>
                  <span style='font-size:1.3em;color:white'>Day {peak_day+1}</span>
                </div>
                <div class='metric-card'>
                  <b style='color:#FFB347'>Highest Risk Zone</b><br>
                  <span style='font-size:1.3em;color:white'>Zone {peak_zone}  ({peak_prob:.0%})</span>
                </div>
                <div class='metric-card'>
                  <b style='color:#FFB347'>Total Zones at Risk (7 days)</b><br>
                  <span style='font-size:1.3em;color:white'>{total_at_risk} / {n_zones}</span>
                </div>
                """, unsafe_allow_html=True)

            # Input summary
            st.markdown("---")
            st.markdown("#### Today's Input Summary")
            display_inputs = {
                "Temperature (°C)"     : inputs.get("temperature_2m_celsius"),
                "Humidity (%)"         : inputs.get("relative_humidity_pct"),
                "Wind Speed (m/s)"     : inputs.get("wind_speed_ms"),
                "Wind Direction (°)"   : inputs.get("wind_direction_deg"),
                "VPD (kPa)"            : round(inputs.get("vapor_pressure_deficit_kpa", 0), 3),
                "Soil Moisture"        : inputs.get("soil_moisture_m3m3"),
                "Precipitation (mm)"   : inputs.get("precipitation_mm"),
                "NDVI"                 : inputs.get("ndvi_composite"),
                "Fire Pixels"          : inputs.get("total_fire_pixels"),
                "LST (°C)"             : inputs.get("lst_day_c"),
            }
            df_inputs = pd.DataFrame(
                list(display_inputs.items()),
                columns=["Parameter", "Value"]
            )
            st.dataframe(df_inputs, use_container_width=True, hide_index=True)

    else:
        # Before first run — show instructions
        if zone_gdf is not None:
            st.markdown("#### District Map Preview")
            fig_preview, ax = plt.subplots(figsize=(9, 6))
            fig_preview.patch.set_facecolor("#0e1117")
            ax.set_facecolor("#1a2332")
            for _, row in zone_gdf.iterrows():
                try:
                    geom = row["geometry"]
                    patches = []
                    if geom.geom_type == "Polygon":
                        patches = [plt.Polygon(list(geom.exterior.coords),
                                               closed=True)]
                    elif geom.geom_type == "MultiPolygon":
                        patches = [plt.Polygon(list(p.exterior.coords), closed=True)
                                   for p in geom.geoms]
                    for p in patches:
                        p.set_facecolor("#1e3050")
                        p.set_edgecolor("#4a7aa0")
                        p.set_linewidth(0.8)
                        ax.add_patch(p)
                    cx, cy = geom.centroid.x, geom.centroid.y
                    ax.text(cx, cy, str(int(row["zone"])),
                            ha="center", va="center",
                            fontsize=7.5, color="#8ab4d4")
                except Exception:
                    pass
            bounds = zone_gdf.total_bounds
            pad = (bounds[2] - bounds[0]) * 0.05
            ax.set_xlim(bounds[0]-pad, bounds[2]+pad)
            ax.set_ylim(bounds[1]-pad, bounds[3]+pad)
            ax.set_title(
                f"{district_name} — {n_zones} zones  "
                f"(Select inputs and click 'Run Forecast')",
                color="white", fontsize=12, fontweight="bold"
            )
            ax.tick_params(colors="#8899aa", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#3a4a5a")
            fig_preview.tight_layout()
            st.pyplot(fig_preview, use_container_width=True)
            plt.close(fig_preview)
        else:
            st.info(
                "ℹ️  Shapefile not found at expected path.\n"
                "Set your inputs and click **Run Forecast** to see predictions."
            )

        st.markdown("""
        ---
        ### How to use
        1. **Select a district** from the sidebar
        2. **Set the origin zone** (where fire is observed today)
        3. **Adjust today's conditions** (temperature, wind, humidity, etc.)
        4. Click **🚀 Run 7-Day Spread Forecast**
        5. Watch the **animated spread map** — each colour = one forecast day
        6. Switch to **Zone Detail** tab to inspect any specific zone's probability

        > ⚠️ If no trained model `.pt` file is found for the selected district,
        > the app runs in **Demo Mode** using a physics-inspired spread simulation.
        """)


if __name__ == "__main__":
    main()