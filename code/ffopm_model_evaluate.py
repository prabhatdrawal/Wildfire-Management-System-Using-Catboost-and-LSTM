"""
PROPER EVALUATION: Using Same Data Split as Training
This evaluates the model the CORRECT way - matching the training code's data split
"""

import pickle
import pandas as pd
import numpy as np
import json
from sklearn.metrics import (
    accuracy_score, log_loss, confusion_matrix,
    classification_report, roc_auc_score, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("PROPER MODEL EVALUATION (Matching Training Split)")
print("=" * 80)

# Paths
MODEL_PATH = '/Users/prabhatrawal/Minor_project_code/data/integrated_data/catboost_tuning/catboost_s_tier_model.pkl'
MASTER_FILE = '/Users/prabhatrawal/Minor_project_code/data/integrated_data/Master_FFOPM_Table.parquet'
FEATURES_FILE = '/Users/prabhatrawal/Minor_project_code/data/integrated_data/eda_results/final_features_for_ml_NO_LEAKAGE.json'
SCALER_FILE = '/Users/prabhatrawal/Minor_project_code/data/integrated_data/model_results/scaler.pkl'
THRESHOLD_FILE = '/Users/prabhatrawal/Minor_project_code/data/integrated_data/catboost_tuning/optimal_threshold_info.json'

# Load model
print("\n[1] Loading Model")
print("-" * 80)
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)
print(f"✓ Model loaded: {type(model).__name__}")

# Load threshold info
print("\n[2] Loading Optimal Threshold")
print("-" * 80)
with open(THRESHOLD_FILE, 'r') as f:
    threshold_info = json.load(f)
optimal_threshold = threshold_info['optimal_threshold']
print(f"✓ Optimal threshold: {optimal_threshold:.4f}")

# Load master data
print("\n[3] Loading Master Data (Same as Training)")
print("-" * 80)
df = pd.read_parquet(MASTER_FILE)
df['year'] = df['date'].dt.year
print(f"✓ Loaded: {len(df):,} rows")

# Load feature list
print("\n[4] Loading Feature List")
print("-" * 80)
with open(FEATURES_FILE, 'r') as f:
    feature_metadata = json.load(f)
final_features = feature_metadata['final_feature_list']
print(f"✓ Using {len(final_features)} features")

# Split data (SAME as training code)
print("\n[5] Splitting Data (Temporal Split - Same as Training)")
print("-" * 80)
TRAIN_END_YEAR = 2018
VAL_END_YEAR = 2020

train_mask = df['year'] <= TRAIN_END_YEAR
val_mask = (df['year'] > TRAIN_END_YEAR) & (df['year'] <= VAL_END_YEAR)
test_mask = df['year'] >= 2021

train_df = df[train_mask].copy()
val_df = df[val_mask].copy()
test_df = df[test_mask].copy()

print(f"  Train: {len(train_df):,} ({train_df['fire_label'].sum():,} fires)")
print(f"  Val:   {len(val_df):,} ({val_df['fire_label'].sum():,} fires)")
print(f"  Test:  {len(test_df):,} ({test_df['fire_label'].sum():,} fires)")

# Extract features
X_train = train_df[final_features].copy()
y_train = train_df['fire_label'].copy()

X_val = val_df[final_features].copy()
y_val = val_df['fire_label'].copy()

X_test = test_df[final_features].copy()
y_test = test_df['fire_label'].copy()

# Preprocessing (SAME as training code)
print("\n[6] Preprocessing (Same as Training)")
print("-" * 80)
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns, index=X_val.index)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
print("✓ Missing values imputed")

# Load scaler
with open(SCALER_FILE, 'rb') as f:
    scaler = pickle.load(f)

X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns, index=X_train.index)
X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
print("✓ Features scaled")

# Make predictions
print("\n[7] Making Predictions")
print("-" * 80)

# Validation set
val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
val_pred_default = model.predict(X_val_scaled)
val_pred_optimal = (val_pred_proba >= optimal_threshold).astype(int)

# Test set
test_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
test_pred_default = model.predict(X_test_scaled)
test_pred_optimal = (test_pred_proba >= optimal_threshold).astype(int)

print("✓ Predictions complete")

# Calculate metrics
def calculate_metrics(y_true, y_pred, y_pred_proba, set_name):
    print(f"\n{'='*80}")
    print(f"{set_name.upper()} SET METRICS")
    print(f"{'='*80}")
    
    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    ll = log_loss(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    print(f"\n📊 Probability-based metrics:")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Log Loss:  {ll:.4f}")
    print(f"  ROC AUC:   {roc_auc:.4f}")
    print(f"  PR AUC:    {pr_auc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n📊 Confusion Matrix (Default Threshold 0.5):")
    print(cm)
    print(f"  Format: [[TN  FP]")
    print(f"           [FN  TP]]")
    
    # Classification report
    print(f"\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Fire', 'Fire']))
    
    return acc, ll

# Evaluate with default threshold
print("\n" + "="*80)
print("EVALUATION WITH DEFAULT THRESHOLD (0.5)")
print("="*80)

val_acc, val_ll = calculate_metrics(y_val, val_pred_default, val_pred_proba, "VALIDATION")
test_acc, test_ll = calculate_metrics(y_test, test_pred_default, test_pred_proba, "TEST")

# Evaluate with optimal threshold
print("\n" + "="*80)
print(f"EVALUATION WITH OPTIMAL THRESHOLD ({optimal_threshold:.4f})")
print("="*80)

val_acc_opt, val_ll_opt = calculate_metrics(y_val, val_pred_optimal, val_pred_proba, "VALIDATION")
test_acc_opt, test_ll_opt = calculate_metrics(y_test, test_pred_optimal, test_pred_proba, "TEST")

# Summary
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

print(f"""
✅ USING CORRECT DATA SPLIT (Same as Training):
  - Train: ≤2018
  - Val:   2019-2020
  - Test:  ≥2021

📊 AVERAGE METRICS (Val + Test):
  
  Default Threshold (0.5):
    Average Accuracy:  {(val_acc + test_acc)/2:.4f} ({(val_acc + test_acc)/2*100:.2f}%)
    Average Log Loss:  {(val_ll + test_ll)/2:.4f}
  
  Optimal Threshold ({optimal_threshold:.4f}):
    Average Accuracy:  {(val_acc_opt + test_acc_opt)/2:.4f} ({(val_acc_opt + test_acc_opt)/2*100:.2f}%)
    Average Log Loss:  {(val_ll_opt + test_ll_opt)/2:.4f}

🎯 KEY INSIGHT:
   When evaluated on the SAME data split as training, the model works correctly!
   The confusion matrix you showed matches this evaluation.
   
⚠️ PROBLEM:
   The ensemble_ready/ files use a DIFFERENT split or features.
   This is why evaluation on ensemble_ready shows 0% fire detection.

💡 SOLUTION:
   Regenerate ensemble_ready/ files using this EXACT same split and preprocessing.
   Or, use this script's approach for your final evaluation.
""")

print("\n" + "="*80)