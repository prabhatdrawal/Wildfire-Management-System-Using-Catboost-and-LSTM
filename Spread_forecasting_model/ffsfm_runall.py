"""
=============================================================================
ffsfm_run_all.py  —  Run Complete Pipeline End-to-End
=============================================================================
Step 1: Baselines
Step 2: Train all 5 districts
Step 3: Evaluate all 5 districts
Step 4: Merge baseline + model results into one comparison table

Usage:
  python ffsfm_run_all.py               # full pipeline
  python ffsfm_run_all.py --skip_train  # skip training (just eval + merge)
=============================================================================
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

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

SCRIPT_DIR = Path(__file__).parent   # same folder as this file


def run(script: str, args: list = None):
    cmd = [sys.executable, str(SCRIPT_DIR / script)]
    if args:
        cmd.extend(args)
    log.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result


def merge_results():
    """Combine baseline + model metrics into one table for comparison."""
    baseline_path = RESULTS_DIR / "baseline_metrics.csv"
    model_path    = RESULTS_DIR / "metrics_table.csv"

    dfs = []
    if baseline_path.exists():
        dfs.append(pd.read_csv(baseline_path))
    if model_path.exists():
        dfs.append(pd.read_csv(model_path))

    if not dfs:
        log.warning("No result files found. Run training and evaluation first.")
        return

    combined = pd.concat(dfs, ignore_index=True)

    # Reorder columns to match FFOPM table format
    cols = [
        "district", "Model", "Split",
        "ROC_AUC", "PR_AUC",
        "Recall_Fire", "Precision_Fire", "F1_Score",
        "False_Alarm_Rate", "Specificity",
    ]
    existing_cols = [c for c in cols if c in combined.columns]
    combined_display = combined[existing_cols]

    out = RESULTS_DIR / "final_comparison_table.csv"
    combined.to_csv(out, index=False)

    print("\n" + "=" * 110)
    print("FFSFM  FINAL COMPARISON  (Baselines vs ConvBiLSTM)")
    print("=" * 110)
    print(combined_display.to_string(index=False))
    print("=" * 110)
    log.info(f"Final comparison saved → {out}")

    # Per-district summary: best model per metric on Test split
    test_only = combined[combined["Split"] == "Test"].copy()
    if not test_only.empty:
        print("\n── Best model per district (Test split, by PR-AUC) ──")
        best = (test_only.sort_values("PR_AUC", ascending=False)
                         .groupby("district")
                         .first()
                         .reset_index()
                         [["district", "Model", "ROC_AUC", "PR_AUC",
                           "Recall_Fire", "F1_Score"]])
        print(best.to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training, only evaluate and merge.")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("FFSFM Full Pipeline")
    log.info("=" * 60)

    # Step 1 — Baselines
    log.info("\n── STEP 1: Baselines ──")
    run("ffsfm_baseline.py")

    # Step 2 — Train
    if not args.skip_train:
        log.info("\n── STEP 2: Training ConvBiLSTM ──")
        run("ffsfm_train.py")
    else:
        log.info("\n── STEP 2: Skipping training ──")

    # Step 3 — Evaluate
    log.info("\n── STEP 3: Evaluating ConvBiLSTM ──")
    run("ffsfm_evaluate.py")

    # Step 4 — Merge
    log.info("\n── STEP 4: Merging results ──")
    merge_results()

    log.info("\n✓ FFSFM Pipeline complete.")
    log.info(f"  Results → {RESULTS_DIR}")


if __name__ == "__main__":
    main()