"""
=============================================================================
ffsfm_train.py  —  Train ConvBiLSTM per district
=============================================================================
Chronological split (NO random shuffle on split):
  Train : 2000-02-24 → 2018-12-31
  Val   : 2019-01-01 → 2021-12-31
  Test  : 2022-01-01 → 2025-01-18

Class imbalance handled via pos_weight in BCEWithLogitsLoss.

Saves per district:
  /ffsfm_data/models/best_<district>.pt   — best val PR-AUC checkpoint
  /ffsfm_data/models/history_<district>.csv

Usage:
  # Train all districts
  python ffsfm_train.py

  # Train one district only
  python ffsfm_train.py --district banke
=============================================================================
"""

import argparse
import json
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score

from ffsfm_model import build_model

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path("/Users/prabhatrawal/Minor_project_code")
DATA_DIR   = BASE_DIR / "ffsfm_data"
MODEL_DIR  = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Chronological split boundaries
TRAIN_END = "2018-12-31"
VAL_END   = "2021-12-31"

DISTRICTS = {
    "banke"  : {"code": "District_0", "n_zones": 14},
    "bardiya": {"code": "District_1", "n_zones": 15},
    "surkhet": {"code": "District_2", "n_zones": 19},
    "dang"   : {"code": "District_3", "n_zones": 17},
    "salyan" : {"code": "District_4", "n_zones": 14},
}

# Training hyper-parameters
EPOCHS      = 60
BATCH_SIZE  = 64
LR          = 1e-3
LR_PATIENCE = 6       # ReduceLROnPlateau patience
ES_PATIENCE = 12      # EarlyStopping patience
WEIGHT_DECAY = 1e-4


# =============================================================================
# Helpers
# =============================================================================
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")   # Apple Silicon
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def chronological_split(X: np.ndarray,
                         y: np.ndarray,
                         meta: pd.DataFrame) -> dict:
    """
    Split samples by the date in sample_meta.csv.
    Returns dict with keys: train, val, test — each a (X, y) tuple.
    """
    dates = pd.to_datetime(meta["date"])

    train_mask = dates <= TRAIN_END
    val_mask   = (dates > TRAIN_END) & (dates <= VAL_END)
    test_mask  = dates > VAL_END

    splits = {}
    for name, mask in [("train", train_mask),
                        ("val",   val_mask),
                        ("test",  test_mask)]:
        idx = np.where(mask)[0]
        splits[name] = (X[idx], y[idx])
        log.info(f"  {name:5s}: {len(idx):,} samples  "
                 f"({dates[mask].min().date()} → {dates[mask].max().date()})")
    return splits


def compute_pos_weight(y_train: np.ndarray) -> float:
    """pos_weight = n_neg / n_pos  (applied to BCEWithLogitsLoss)."""
    pos = float(y_train.sum())
    neg = float(y_train.size - pos)
    if pos == 0:
        return 1.0
    return neg / pos


def make_loader(X: np.ndarray, y: np.ndarray,
                batch_size: int, shuffle: bool) -> DataLoader:
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    return DataLoader(TensorDataset(Xt, yt),
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=0)


# =============================================================================
# Train one epoch
# =============================================================================
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)                  # (B, H, Z)
        loss   = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(Xb)
    return total_loss / len(loader.dataset)


# =============================================================================
# Evaluate one split
# =============================================================================
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    all_probs, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            logits = model(Xb)
            loss   = criterion(logits, yb)
            total_loss += loss.item() * len(Xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(yb.cpu().numpy())

    probs  = np.concatenate(all_probs,  axis=0).ravel()
    labels = np.concatenate(all_labels, axis=0).ravel().astype(int)

    roc_auc = roc_auc_score(labels, probs) if labels.sum() > 0 else 0.0
    pr_auc  = average_precision_score(labels, probs) if labels.sum() > 0 else 0.0

    return {
        "loss"   : total_loss / len(loader.dataset),
        "roc_auc": roc_auc,
        "pr_auc" : pr_auc,
    }


# =============================================================================
# Train one district
# =============================================================================
def train_district(district_name: str) -> None:
    info   = DISTRICTS[district_name]
    n_zones = info["n_zones"]
    device  = get_device()

    log.info("=" * 60)
    log.info(f"Training: {district_name.upper()}  ({info['code']}, {n_zones} zones)")
    log.info(f"Device  : {device}")
    log.info("=" * 60)

    # ── Load tensors ───────────────────────────────────────────────────────
    X = np.load(DATA_DIR / f"X_{district_name}.npy")   # (N, 14, Z, 65)
    y = np.load(DATA_DIR / f"y_{district_name}.npy")   # (N, 7,  Z)

    log.info(f"Loaded   X={X.shape}  y={y.shape}")

    # ── Load sample meta for chronological split ───────────────────────────
    meta = pd.read_csv(DATA_DIR / "sample_meta.csv")
    # Filter to this district
    meta_d = meta[meta["district_code"] == info["code"]].reset_index(drop=True)

    if len(meta_d) != len(X):
        # Fallback: use positional split if meta lengths don't align
        log.warning(f"  Meta rows ({len(meta_d)}) != X rows ({len(X)}). "
                    f"Using positional split.")
        n = len(X)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.15)
        splits = {
            "train": (X[:n_train],           y[:n_train]),
            "val"  : (X[n_train:n_train+n_val], y[n_train:n_train+n_val]),
            "test" : (X[n_train+n_val:],     y[n_train+n_val:]),
        }
    else:
        splits = chronological_split(X, y, meta_d)

    X_train, y_train = splits["train"]
    X_val,   y_val   = splits["val"]

    # ── Class weight ───────────────────────────────────────────────────────
    pw = compute_pos_weight(y_train)
    log.info(f"pos_weight = {pw:.2f}  "
             f"(fire rate: {100*y_train.mean():.3f}%)")

    # ── Loaders ────────────────────────────────────────────────────────────
    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   BATCH_SIZE, shuffle=False)

    # ── Model + optimizer + loss ───────────────────────────────────────────
    model = build_model(n_zones=n_zones, device=device)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5,
        patience=LR_PATIENCE
    )
    pw_tensor = torch.tensor(pw, dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)

    # ── Training loop ──────────────────────────────────────────────────────
    best_pr_auc  = 0.0
    best_epoch   = 0
    patience_cnt = 0
    history      = []

    log.info(f"Starting training for {EPOCHS} epochs ...")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        val_pr_auc  = val_metrics["pr_auc"]
        val_roc_auc = val_metrics["roc_auc"]
        val_loss    = val_metrics["loss"]

        scheduler.step(val_pr_auc)

        row = {
            "epoch"      : epoch,
            "train_loss" : round(train_loss, 5),
            "val_loss"   : round(val_loss,   5),
            "val_roc_auc": round(val_roc_auc, 5),
            "val_pr_auc" : round(val_pr_auc,  5),
            "lr"         : optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        log.info(
            f"  Epoch {epoch:3d}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"ROC-AUC={val_roc_auc:.4f}  "
            f"PR-AUC={val_pr_auc:.4f}"
        )

        # Save best checkpoint (monitor: PR-AUC — better for imbalanced data)
        if val_pr_auc > best_pr_auc:
            best_pr_auc  = val_pr_auc
            best_epoch   = epoch
            patience_cnt = 0
            torch.save({
                "epoch"       : epoch,
                "model_state" : model.state_dict(),
                "optimizer"   : optimizer.state_dict(),
                "val_pr_auc"  : val_pr_auc,
                "val_roc_auc" : val_roc_auc,
                "n_zones"     : n_zones,
                "district"    : district_name,
            }, MODEL_DIR / f"best_{district_name}.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= ES_PATIENCE:
                log.info(f"  Early stopping at epoch {epoch} "
                         f"(best epoch={best_epoch}, best PR-AUC={best_pr_auc:.4f})")
                break

    elapsed = time.time() - t0
    log.info(f"Training done in {elapsed/60:.1f} min  |  "
             f"Best epoch={best_epoch}  Best val PR-AUC={best_pr_auc:.4f}")

    # ── Save history ───────────────────────────────────────────────────────
    hist_path = MODEL_DIR / f"history_{district_name}.csv"
    pd.DataFrame(history).to_csv(hist_path, index=False)
    log.info(f"History saved → {hist_path}")

    # ── Quick test-set evaluation with best model ──────────────────────────
    X_test, y_test = splits["test"]
    test_loader    = make_loader(X_test, y_test, BATCH_SIZE, shuffle=False)

    ckpt = torch.load(MODEL_DIR / f"best_{district_name}.pt",
                      map_location=device,weights_only = False)
    model.load_state_dict(ckpt["model_state"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    log.info(
        f"  TEST  ROC-AUC={test_metrics['roc_auc']:.4f}  "
        f"PR-AUC={test_metrics['pr_auc']:.4f}"
    )


# =============================================================================
# CLI
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--district", type=str, default="all",
        help="District name to train (banke/bardiya/surkhet/dang/salyan/all)"
    )
    args = parser.parse_args()

    targets = (list(DISTRICTS.keys())
               if args.district == "all"
               else [args.district.lower()])

    for d in targets:
        if d not in DISTRICTS:
            log.error(f"Unknown district: {d}. "
                      f"Choose from: {list(DISTRICTS.keys())}")
            continue
        train_district(d)

    log.info("All training complete ✓")


if __name__ == "__main__":
    main()