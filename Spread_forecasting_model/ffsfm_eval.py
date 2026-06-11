"""
=============================================================================
ffsfm_final_eval.py  —  Overall Model Evaluation (PR, ROC, CM)
=============================================================================
Reads directly from predictions_*.csv files (y_true, y_prob columns).
Combines all 5 districts × all horizon days × all zones into ONE overall
evaluation → 3 clean publication-ready figures on white background.

Outputs saved to:
  /Users/prabhatrawal/Minor_project_code/ffsfm/ffsfm_data/results/

  • ffsfm_pr_curve.png
  • ffsfm_roc_curve.png
  • ffsfm_confusion_matrix.png
  • ffsfm_eval_metrics.csv

Usage:
  cd /Users/prabhatrawal/Minor_project_code/ffsfm/ffsfm_code
  python ffsfm_final_eval.py
=============================================================================
"""

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score,
    confusion_matrix,
)

# ── Paths ──────────────────────────────────────────────────────────────────
RESULTS_DIR = Path(
    "/Users/prabhatrawal/Minor_project_code/ffsfm_data/results"
)
PRED_FILES = [
    RESULTS_DIR / "predictions_banke.csv",
    RESULTS_DIR / "predictions_bardiya.csv",
    RESULTS_DIR / "predictions_dang.csv",
    RESULTS_DIR / "predictions_salyan.csv",
    RESULTS_DIR / "predictions_surkhet.csv",
]

# ── White presentation style ───────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#CCCCCC",
    "axes.labelcolor":   "#222222",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.color":       "#444444",
    "ytick.color":       "#444444",
    "text.color":        "#222222",
    "grid.color":        "#EEEEEE",
    "grid.linestyle":    "-",
    "grid.alpha":        1.0,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
})

RED  = "#D62728"   # PR curve
BLUE = "#1F77B4"   # ROC curve
GRAY = "#AAAAAA"   # baselines


# =============================================================================
# Load & combine all prediction files  (Test split only)
# =============================================================================

def load_predictions(split: str = "Test") -> tuple:
    """
    Reads predictions_*.csv, filters to the requested split,
    returns (y_true, y_prob) as flat numpy arrays.
    """
    frames = []
    for p in PRED_FILES:
        if not p.exists():
            print(f"  ⚠  Not found: {p.name}")
            continue
        df = pd.read_csv(p)
        # Keep only the requested split if a Split column exists
        if "Split" in df.columns:
            df = df[df["Split"] == split]
        frames.append(df[["y_true", "y_prob"]])
        print(f"  ✓  {p.name}  ({len(df):,} rows)")

    if not frames:
        raise FileNotFoundError("No prediction files found.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Total rows combined : {len(combined):,}")
    print(f"  Positive rate       : {combined['y_true'].mean():.4f}")

    y_true = combined["y_true"].to_numpy().astype(int)
    y_prob = combined["y_prob"].to_numpy().astype(float)
    return y_true, y_prob


# =============================================================================
# Plot 1 — Precision-Recall Curve
# =============================================================================

def plot_pr_curve(y_true, y_prob, save_path):
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    ap   = average_precision_score(y_true, y_prob)
    base = float(y_true.mean())

    # Best F1 operating point
    with np.errstate(invalid="ignore"):
        f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    best_idx = int(np.argmax(f1s))
    best_thr = float(thresholds[best_idx])
    best_f1  = float(f1s[best_idx])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    ax.fill_between(rec, prec, alpha=0.10, color=RED)
    ax.plot(rec, prec, color=RED, linewidth=2.5,
            label=f"ConvBiLSTM  (AP = {ap:.3f})")
    ax.axhline(base, color=GRAY, linewidth=1.4, linestyle="--",
               label=f"Random baseline  ({base:.3f})")

    ax.scatter(rec[best_idx], prec[best_idx],
               color=BLUE, s=90, zorder=5,
               label=f"Best F1 = {best_f1:.3f}  (thr = {best_thr:.2f})")
    ax.annotate(
        f" thr = {best_thr:.2f}\n F1   = {best_f1:.3f}",
        xy=(rec[best_idx], prec[best_idx]),
        fontsize=8.5, color=BLUE,
        xytext=(rec[best_idx] + 0.04, prec[best_idx] - 0.10),
    )

    ax.set_xlabel("Recall",    fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision–Recall Curve  (All Districts · Test Split)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(-0.01, 1.02)
    ax.set_ylim(0.0,   1.05)
    ax.grid(True)
    ax.legend(fontsize=9, loc="upper right",
              framealpha=0.9, edgecolor="#CCCCCC")

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n✓  PR curve saved  → {save_path.name}")
    return ap, best_thr, best_f1


# =============================================================================
# Plot 2 — ROC Curve
# =============================================================================

def plot_roc_curve(y_true, y_prob, save_path):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    # Youden's J — optimal operating point
    j_idx    = int(np.argmax(tpr - fpr))
    best_thr = float(thresholds[j_idx])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    ax.fill_between(fpr, tpr, alpha=0.10, color=BLUE)
    ax.plot(fpr, tpr, color=BLUE, linewidth=2.5,
            label=f"ConvBiLSTM  (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=GRAY, linewidth=1.4,
            linestyle="--", label="Random  (AUC = 0.500)")

    ax.scatter(fpr[j_idx], tpr[j_idx],
               color=RED, s=90, zorder=5,
               label=f"Optimal threshold = {best_thr:.2f}")
    ax.annotate(
        f" thr = {best_thr:.2f}\n TPR = {tpr[j_idx]:.3f}",
        xy=(fpr[j_idx], tpr[j_idx]),
        fontsize=8.5, color=RED,
        xytext=(fpr[j_idx] + 0.04, tpr[j_idx] - 0.10),
    )

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("ROC Curve  (All Districts · Test Split)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(-0.01, 1.02)
    ax.set_ylim(0.0,   1.05)
    ax.grid(True)
    ax.legend(fontsize=9, loc="lower right",
              framealpha=0.9, edgecolor="#CCCCCC")

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓  ROC curve saved → {save_path.name}")
    return auc, best_thr


# =============================================================================
# Plot 3 — Confusion Matrix
# =============================================================================

def plot_confusion_matrix(y_true, y_prob, threshold, save_path):
    y_pred         = (y_prob >= threshold).astype(int)
    cm             = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Row-normalised for colour intensity
    row_sums        = cm.sum(axis=1, keepdims=True).astype(float)
    row_sums[row_sums == 0] = 1
    cm_norm         = cm.astype(float) / row_sums

    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    accuracy  = (tp + tn) / cm.sum()
    specificity = tn / (tn + fp + 1e-9)

    fig, ax = plt.subplots(figsize=(5.8, 5.2))

    cmap = LinearSegmentedColormap.from_list(
        "cm_wr", ["#FFFFFF", "#FADADD", "#D62728"]
    )
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1, aspect="auto")

    raw_vals = [[tn, fp], [fn, tp]]
    for i in range(2):
        for j in range(2):
            nv  = cm_norm[i, j]
            raw = raw_vals[i][j]
            tc  = "white" if nv > 0.55 else "#222222"
            ax.text(j, i,
                    f"{raw:,}",
                    ha="center", va="center",
                    fontsize=18, fontweight="bold", color=tc)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted: No Fire", "Predicted: Fire"], fontsize=10)
    ax.set_yticklabels(["Actual: No Fire",    "Actual: Fire"],    fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=11, labelpad=8)
    ax.set_ylabel("True Label",      fontsize=11, labelpad=8)
    ax.set_title("Confusion Matrix  (All Districts · Test Split)",
                 fontsize=13, fontweight="bold", pad=12)

    footer = (
        f"Accuracy = {accuracy:.3f}   |   Precision = {precision:.3f}   |   "
        f"Recall = {recall:.3f}   |   F1 = {f1:.3f}   |   "
        f"Specificity = {specificity:.3f}   |   threshold = {threshold:.2f}"
    )
    fig.text(0.5, -0.03, footer, ha="center", fontsize=8,
             color="#555555", style="italic")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalised proportion", fontsize=8, color="#444444")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#444444", fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✓  Confusion matrix → {save_path.name}")

    return {
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
        "Accuracy":    round(float(accuracy),    4),
        "Precision":   round(float(precision),   4),
        "Recall":      round(float(recall),      4),
        "F1":          round(float(f1),          4),
        "Specificity": round(float(specificity), 4),
        "Threshold":   round(float(threshold),   3),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 55)
    print("  FFSFM — Overall Model Evaluation (Test Split)")
    print("=" * 55)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load all predictions (Test split only)
    y_true, y_prob = load_predictions(split="Test")

    # 1. PR Curve
    print("\nGenerating PR curve …")
    ap, best_thr_pr, best_f1 = plot_pr_curve(
        y_true, y_prob,
        save_path=RESULTS_DIR / "ffsfm_pr_curve.png",
    )

    # 2. ROC Curve
    print("Generating ROC curve …")
    auc, _ = plot_roc_curve(
        y_true, y_prob,
        save_path=RESULTS_DIR / "ffsfm_roc_curve.png",
    )

    # 3. Confusion Matrix  (use best-F1 threshold from PR curve)
    print("Generating Confusion Matrix …")
    cm_metrics = plot_confusion_matrix(
        y_true, y_prob,
        threshold=best_thr_pr,
        save_path=RESULTS_DIR / "ffsfm_confusion_matrix.png",
    )

    # 4. CSV summary
    summary = {
        "AUC-ROC":            round(auc, 4),
        "Avg-Precision (AP)": round(ap,  4),
        **cm_metrics,
    }
    pd.DataFrame([summary]).to_csv(
        RESULTS_DIR / "ffsfm_eval_metrics.csv", index=False
    )
    print(f"✓  CSV saved → ffsfm_eval_metrics.csv")

    print("\n" + "=" * 55)
    print("  RESULTS SUMMARY")
    print("=" * 55)
    for k, v in summary.items():
        print(f"  {k:<26}: {v}")
    print("=" * 55)
    print("\n✅  All outputs saved to:", RESULTS_DIR)
    print("   • ffsfm_pr_curve.png")
    print("   • ffsfm_roc_curve.png")
    print("   • ffsfm_confusion_matrix.png")
    print("   • ffsfm_eval_metrics.csv")


if __name__ == "__main__":
    main()