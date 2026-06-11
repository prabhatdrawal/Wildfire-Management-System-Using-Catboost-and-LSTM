"""
=============================================================================
ffsfm_baseline.py  —  Two Simple Baselines
=============================================================================
Baseline 1 — Persistence:
  Predict tomorrow's fire = today's fire label (last known observation).
  "If it's burning today, it will still burn tomorrow."

Baseline 2 — Neighbour Spread:
  Predict fire = 1  if ANY neighbouring zone had fire in last 3 days.
  "Fire will spread to adjacent zones."

Both are evaluated on the TEST split with the same metric table as FFSFM.

Saves:
  /ffsfm_data/results/baseline_metrics.csv
  /ffsfm_data/results/baseline_report.txt

Usage:
  python ffsfm_baseline.py
=============================================================================
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    recall_score, precision_score, f1_score,
    confusion_matrix,
)

# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("/Users/prabhatrawal/Minor_project_code")
DATA_DIR    = BASE_DIR / "ffsfm_data"
RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TRAIN_END = "2018-12-31"
VAL_END   = "2021-12-31"
HORIZON   = 7

DISTRICTS = {
    "banke"  : {"code": "District_0", "n_zones": 14},
    "bardiya": {"code": "District_1", "n_zones": 15},
    "surkhet": {"code": "District_2", "n_zones": 19},
    "dang"   : {"code": "District_3", "n_zones": 17},
    "salyan" : {"code": "District_4", "n_zones": 14},
}


# =============================================================================
# Metric helper
# =============================================================================
def compute_metrics(y_true, y_prob, y_pred,
                    model_name, split_name, district) -> dict:
    roc_auc = roc_auc_score(y_true, y_prob)           if y_true.sum() > 0 else 0.0
    pr_auc  = average_precision_score(y_true, y_prob)  if y_true.sum() > 0 else 0.0
    recall  = recall_score(y_true, y_pred, zero_division=0)
    prec    = precision_score(y_true, y_pred, zero_division=0)
    f1      = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    far     = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    spec    = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "district"        : district,
        "Model"           : model_name,
        "Split"           : split_name,
        "ROC_AUC"         : round(roc_auc, 4),
        "PR_AUC"          : round(pr_auc,  4),
        "Recall_Fire"     : round(recall,  4),
        "Precision_Fire"  : round(prec,    4),
        "F1_Score"        : round(f1,      4),
        "False_Alarm_Rate": round(far,     4),
        "Specificity"     : round(spec,    4),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


# =============================================================================
# Baseline 1 — Persistence
# =============================================================================
def persistence_baseline(X: np.ndarray, y: np.ndarray) -> tuple:
    """
    X shape: (N, 14, Z, F)
    The last timestep's fire_label is stored as the first feature (index 0)
    after normalisation, BUT we use the raw y target from the lookback window.

    Strategy: the last day in the input window is timestep index 13.
    We take X[:, -1, :, 0] as the 'current fire' proxy (total_fire_pixels
    is the first dynamic feature after normalisation, but actual fire_label
    is binarised separately).

    Actually cleaner: use X[:, -1, :, feature_idx] where feature_idx
    corresponds to 'total_fire_pixels' (index in feat_cols).
    We'll use a simple sign-based rule: if total_fire_pixels > 0 (after
    z-score, that means > mean → use > 0 as proxy).

    Persist that binary for all 7 horizon days.
    Returns (y_prob, y_pred) both shape (N, H, Z).
    """
    # Use the fire-pixel feature from last timestep as persistence signal
    # Feature index 0 = total_fire_pixels (first dynamic feature)
    # After z-score: positive value → above mean → likely fire
    last_fire = (X[:, -1, :, 0] > 0).astype(float)     # (N, Z)

    # Persist to all horizon days
    y_prob = np.stack([last_fire] * HORIZON, axis=1)    # (N, H, Z)
    y_pred = y_prob.astype(int)
    return y_prob, y_pred


# =============================================================================
# Baseline 2 — Neighbour Spread
# =============================================================================
def neighbour_baseline(X: np.ndarray, y: np.ndarray,
                        adjacency: dict, district_code: str) -> tuple:
    """
    Predict fire in zone z at t+k if ANY neighbour of z had fire
    in the last 3 days of the lookback window.

    X[:, -3:, :, 0]  →  last 3 days, all zones, fire-pixel feature

    Returns (y_prob, y_pred).
    """
    adj = adjacency.get(district_code, {})   # {zone_int: [neighbour_ints]}
    n_zones = X.shape[2]

    # last 3 days fire activity per zone: (N, Z)
    recent_fire = (X[:, -3:, :, 0] > 0).any(axis=1).astype(float)  # (N, Z)

    # For each zone, check if any neighbour had recent fire
    neighbour_fire = np.zeros_like(recent_fire)   # (N, Z)
    for zone_int, neighbours in adj.items():
        z_idx      = int(zone_int) - 1            # 0-based tensor index
        nb_indices = [int(nb) - 1 for nb in neighbours
                      if 0 <= int(nb) - 1 < n_zones]
        if nb_indices:
            nb_fire = recent_fire[:, nb_indices].any(axis=1)  # (N,)
            neighbour_fire[:, z_idx] = nb_fire.astype(float)

    # Predict fire = 1 if zone itself OR a neighbour had recent fire
    y_prob = np.maximum(recent_fire, neighbour_fire)     # (N, Z)
    y_prob = np.stack([y_prob] * HORIZON, axis=1)        # (N, H, Z)
    y_pred = y_prob.astype(int)
    return y_prob, y_pred


# =============================================================================
# Run baselines for one district
# =============================================================================
def run_baselines_district(district_name: str,
                            adjacency: dict) -> list:
    info    = DISTRICTS[district_name]
    n_zones = info["n_zones"]

    X    = np.load(DATA_DIR / f"X_{district_name}.npy")
    y    = np.load(DATA_DIR / f"y_{district_name}.npy")
    meta = pd.read_csv(DATA_DIR / "sample_meta.csv")
    meta_d = meta[meta["district_code"] == info["code"]].reset_index(drop=True)

    # Chronological test split
    if len(meta_d) == len(X):
        dates     = pd.to_datetime(meta_d["date"])
        test_mask = dates > VAL_END
        X_test    = X[test_mask.values]
        y_test    = y[test_mask.values]
    else:
        n       = len(X)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.15)
        X_test  = X[n_train + n_val:]
        y_test  = y[n_train + n_val:]

    y_true = y_test.ravel().astype(int)
    rows   = []

    for model_name, (y_prob, y_pred) in [
        ("Persistence",  persistence_baseline(X_test, y_test)),
        ("NeighbourSpread", neighbour_baseline(X_test, y_test,
                                               adjacency, info["code"])),
    ]:
        prob_flat = y_prob.ravel()
        pred_flat = y_pred.ravel().astype(int)
        m = compute_metrics(y_true, prob_flat, pred_flat,
                            model_name, "Test", district_name)
        rows.append(m)
        log.info(
            f"  {model_name:20s}  "
            f"ROC={m['ROC_AUC']:.4f}  PR={m['PR_AUC']:.4f}  "
            f"Recall={m['Recall_Fire']:.4f}  F1={m['F1_Score']:.4f}"
        )
    return rows


# =============================================================================
# Main
# =============================================================================
def main():
    # Load adjacency
    adj_path = DATA_DIR / "zone_adjacency.json"
    with open(adj_path) as f:
        adjacency_raw = json.load(f)

    # Convert keys to int (JSON serialises int keys as strings)
    adjacency = {}
    for dist_code, adj in adjacency_raw.items():
        adjacency[dist_code] = {int(k): [int(v) for v in vs]
                                for k, vs in adj.items()}

    all_rows = []
    for district_name in DISTRICTS:
        log.info("=" * 60)
        log.info(f"Baselines: {district_name.upper()}")
        log.info("=" * 60)
        rows = run_baselines_district(district_name, adjacency)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    col_order = [
        "district", "Model", "Split",
        "ROC_AUC", "PR_AUC",
        "Recall_Fire", "Precision_Fire", "F1_Score",
        "False_Alarm_Rate", "Specificity",
        "TP", "FP", "FN", "TN",
    ]
    df = df[col_order]
    out = RESULTS_DIR / "baseline_metrics.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 100)
    print("BASELINE RESULTS")
    print("=" * 100)
    print(df[[
        "district", "Model", "ROC_AUC", "PR_AUC",
        "Recall_Fire", "Precision_Fire", "F1_Score",
        "False_Alarm_Rate", "Specificity",
    ]].to_string(index=False))
    print("=" * 100)
    log.info(f"Saved → {out}")


if __name__ == "__main__":
    main()