"""
=============================================================================
ffsfm_evaluate.py  —  Full Evaluation with All Metrics
=============================================================================
Produces the EXACT same metric table as your FFOPM evaluation:
  Model | Split | ROC_AUC | PR_AUC | Recall_Fire | Precision_Fire
        | F1_Score | False_Alarm_Rate | Specificity

Threshold selection: maximises F1 on validation set, then applies to test.

Saves:
  /ffsfm_data/results/metrics_table.csv      — full metrics per district
  /ffsfm_data/results/threshold_table.csv    — optimal threshold per district
  /ffsfm_data/results/predictions_<dist>.csv — per-sample predictions

Usage:
  python ffsfm_evaluate.py
  python ffsfm_evaluate.py --district banke
=============================================================================
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    recall_score, precision_score, f1_score,
    confusion_matrix,
)

from ffsfm_model import build_model

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR    = Path("/Users/prabhatrawal/Minor_project_code")
DATA_DIR    = BASE_DIR / "ffsfm_data"
MODEL_DIR   = DATA_DIR / "models"
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
BATCH_SIZE = 256

DISTRICTS = {
    "banke"  : {"code": "District_0", "n_zones": 14},
    "bardiya": {"code": "District_1", "n_zones": 15},
    "surkhet": {"code": "District_2", "n_zones": 19},
    "dang"   : {"code": "District_3", "n_zones": 17},
    "salyan" : {"code": "District_4", "n_zones": 14},
}


# =============================================================================
# Metric helpers
# =============================================================================
def compute_metrics(y_true: np.ndarray,
                    y_prob: np.ndarray,
                    threshold: float,
                    split_name: str,
                    model_name: str = "ConvBiLSTM") -> dict:
    """
    Compute all metrics matching your FFOPM evaluation table.
    y_true / y_prob : flattened 1-D arrays
    """
    y_pred = (y_prob >= threshold).astype(int)

    # Guard: no positive samples
    if y_true.sum() == 0:
        log.warning(f"  {split_name}: no positive samples — metrics may be 0")

    roc_auc  = roc_auc_score(y_true, y_prob)         if y_true.sum() > 0 else 0.0
    pr_auc   = average_precision_score(y_true, y_prob) if y_true.sum() > 0 else 0.0
    recall   = recall_score(y_true, y_pred, zero_division=0)
    prec     = precision_score(y_true, y_pred, zero_division=0)
    f1       = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    far          = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # False Alarm Rate
    specificity  = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "Model"            : model_name,
        "Split"            : split_name,
        "ROC_AUC"          : round(roc_auc,   4),
        "PR_AUC"           : round(pr_auc,    4),
        "Recall_Fire"      : round(recall,    4),
        "Precision_Fire"   : round(prec,      4),
        "F1_Score"         : round(f1,        4),
        "False_Alarm_Rate" : round(far,       4),
        "Specificity"      : round(specificity, 4),
        "Threshold"        : round(threshold, 4),
        "TP"               : int(tp),
        "FP"               : int(fp),
        "FN"               : int(fn),
        "TN"               : int(tn),
    }


def find_best_threshold(y_true: np.ndarray,
                         y_prob: np.ndarray,
                         thresholds: np.ndarray = None) -> float:
    """
    Sweep thresholds [0.05 … 0.95] and return the one that maximises F1.
    Applied on VAL set, then used for TEST evaluation.
    """
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)
    best_t  = 0.5
    best_f1 = 0.0
    for t in thresholds:
        f1 = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t
    log.info(f"  Best threshold on val: {best_t:.2f}  (F1={best_f1:.4f})")
    return float(best_t)


# =============================================================================
# Get predictions from model
# =============================================================================
def predict(model, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=BATCH_SIZE, shuffle=False
    )
    probs = []
    with torch.no_grad():
        for (Xb,) in loader:
            logits = model(Xb.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(probs, axis=0)   # (N, H, Z)


# =============================================================================
# Chronological split (same as train)
# =============================================================================
def chronological_split(X, y, meta):
    dates = pd.to_datetime(meta["date"])
    splits = {}
    for name, mask in [
        ("train", dates <= TRAIN_END),
        ("val",   (dates > TRAIN_END) & (dates <= VAL_END)),
        ("test",  dates > VAL_END),
    ]:
        idx = np.where(mask)[0]
        splits[name] = (X[idx], y[idx], idx)
    return splits


# =============================================================================
# Evaluate one district
# =============================================================================
def evaluate_district(district_name: str) -> list:
    info    = DISTRICTS[district_name]
    n_zones = info["n_zones"]

    ckpt_path = MODEL_DIR / f"best_{district_name}.pt"
    if not ckpt_path.exists():
        log.warning(f"  No checkpoint for {district_name} — train first.")
        return []

    device = torch.device("cpu")   # evaluation always on CPU for reproducibility
    model  = build_model(n_zones=n_zones, device=device)
    ckpt   = torch.load(ckpt_path, map_location=device,weights_only = False)
    model.load_state_dict(ckpt["model_state"])
    log.info(f"Loaded {district_name} checkpoint (epoch {ckpt['epoch']}, "
             f"val PR-AUC={ckpt['val_pr_auc']:.4f})")

    # Load data
    X    = np.load(DATA_DIR / f"X_{district_name}.npy")
    y    = np.load(DATA_DIR / f"y_{district_name}.npy")
    meta = pd.read_csv(DATA_DIR / "sample_meta.csv")
    meta_d = meta[meta["district_code"] == info["code"]].reset_index(drop=True)

    if len(meta_d) != len(X):
        n = len(X)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.15)
        splits_raw = {
            "train": (X[:n_train],              y[:n_train],              np.arange(n_train)),
            "val"  : (X[n_train:n_train+n_val], y[n_train:n_train+n_val], np.arange(n_train, n_train+n_val)),
            "test" : (X[n_train+n_val:],        y[n_train+n_val:],        np.arange(n_train+n_val, n)),
        }
    else:
        splits_raw = chronological_split(X, y, meta_d)

    rows = []
    threshold = 0.5   # default before val calibration

    for split_name in ["train", "val", "test"]:
        Xs, ys, _ = splits_raw[split_name]
        y_prob = predict(model, Xs, device).ravel()    # (N*H*Z,)
        y_true = ys.ravel().astype(int)

        # Calibrate threshold on validation set
        if split_name == "val":
            threshold = find_best_threshold(y_true, y_prob)

        metrics = compute_metrics(
            y_true, y_prob, threshold,
            split_name=split_name.capitalize(),
            model_name=f"ConvBiLSTM_{district_name.capitalize()}"
        )
        metrics["district"] = district_name
        rows.append(metrics)
        log.info(
            f"  {split_name.upper():5s}  "
            f"ROC={metrics['ROC_AUC']:.4f}  "
            f"PR={metrics['PR_AUC']:.4f}  "
            f"Recall={metrics['Recall_Fire']:.4f}  "
            f"F1={metrics['F1_Score']:.4f}  "
            f"FAR={metrics['False_Alarm_Rate']:.4f}  "
            f"Spec={metrics['Specificity']:.4f}"
        )

    # Save per-district predictions (test split only for size)
    Xt, yt, idxt = splits_raw["test"]
    probs_test = predict(model, Xt, device)   # (N, H, Z)
    pred_rows  = []
    for i, sample_idx in enumerate(idxt):
        for h in range(probs_test.shape[1]):
            for z in range(probs_test.shape[2]):
                pred_rows.append({
                    "sample_idx": int(sample_idx),
                    "horizon_day": h + 1,
                    "zone"       : z + 1,
                    "y_true"     : int(yt[i, h, z]),
                    "y_prob"     : float(probs_test[i, h, z]),
                    "y_pred"     : int(probs_test[i, h, z] >= threshold),
                })
    pd.DataFrame(pred_rows).to_csv(
        RESULTS_DIR / f"predictions_{district_name}.csv", index=False
    )
    log.info(f"  Predictions saved → predictions_{district_name}.csv")

    return rows


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--district", type=str, default="all")
    args = parser.parse_args()

    targets = (list(DISTRICTS.keys())
               if args.district == "all"
               else [args.district.lower()])

    all_rows = []
    for d in targets:
        log.info("=" * 60)
        log.info(f"Evaluating: {d.upper()}")
        log.info("=" * 60)
        rows = evaluate_district(d)
        all_rows.extend(rows)

    if not all_rows:
        log.error("No results. Train models first.")
        return

    # ── Build final metrics table ──────────────────────────────────────────
    df = pd.DataFrame(all_rows)
    col_order = [
        "district", "Model", "Split",
        "ROC_AUC", "PR_AUC",
        "Recall_Fire", "Precision_Fire", "F1_Score",
        "False_Alarm_Rate", "Specificity",
        "Threshold", "TP", "FP", "FN", "TN",
    ]
    df = df[col_order]

    out_path = RESULTS_DIR / "metrics_table.csv"
    df.to_csv(out_path, index=False)
    log.info(f"\nMetrics table saved → {out_path}")

    # ── Pretty print ──────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("FFSFM  —  EVALUATION RESULTS")
    print("=" * 100)
    cols_display = [
        "district", "Split", "ROC_AUC", "PR_AUC",
        "Recall_Fire", "Precision_Fire", "F1_Score",
        "False_Alarm_Rate", "Specificity",
    ]
    print(df[cols_display].to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    main()