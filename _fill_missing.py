import numpy as np
import pandas as pd
import os, sys, warnings, json
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ============================================================
# 0. Load data
# ============================================================
data = np.load('data/processed/tensor_31ch_60steps.npz', allow_pickle=True)
tensor   = data['tensor']
stable_mask = data['stable_mask']
labels   = data['labels']
block_id = data['block_id']
ch_names = data['channel_names']

tensor_s = tensor[stable_mask]    # (1449, 31, 60)
labels_s = labels[stable_mask]    # (1449, 4)
block_s  = block_id[stable_mask]  # (1449,)

# Features
feat_mean = tensor_s.mean(axis=2)
feat_std  = tensor_s.std(axis=2)
X = np.hstack([feat_mean, feat_std]).astype(np.float32)

target_names = ['Cooler', 'Valve', 'Pump_Leakage', 'Accumulator']
target_cols  = [0, 1, 2, 3]

# ============================================================
# 1. Class distribution on stable rows
# ============================================================
print("=" * 60)
print("ITEM 4: Class distribution on 1449 stable rows")
print("=" * 60)
for tname, tcol in zip(target_names, target_cols):
    y = labels_s[:, tcol]
    vc = pd.Series(y).value_counts().sort_index()
    print(f"\n  {tname}:")
    for val, cnt in vc.items():
        print(f"    {val}: {cnt}")
    # Check uniformity for valve
    if tname == 'Valve':
        vals = vc.values
        cv = vals.std() / vals.mean() if vals.mean() > 0 else 999
        print(f"  Valve: min={vals.min()}, max={vals.max()}, "
              f"mean={vals.mean():.1f}, std={vals.std():.1f}, CV={cv:.4f}")
        print(f"  Valve near-uniform: {cv < 0.1}")

# ============================================================
# 2. Re-run experiments, capturing per-fold everything
# ============================================================

# Store: results[(target, model, split)][fold] = {acc, macro_f1, per_class_f1}
all_per_fold = {}

for tname, tcol in zip(target_names, target_cols):
    y = labels_s[:, tcol]
    groups = block_s

    for model_name in ['RF', 'XGB']:
        for split_name in ['stratified', 'group']:

            if split_name == 'stratified':
                kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                split_iter = list(kf.split(X, y))
            else:
                kf = GroupKFold(n_splits=5)
                split_iter = list(kf.split(X, y, groups=groups))

            fold_results = []
            for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                if model_name == 'XGB':
                    le = LabelEncoder()
                    y_train_enc = le.fit_transform(y_train)
                    y_test_enc  = le.transform(y_test)
                    clf = XGBClassifier(
                        n_estimators=300, max_depth=6,
                        random_state=42, tree_method='hist', verbosity=0
                    )
                    clf.fit(X_train, y_train_enc)
                    y_pred_enc = clf.predict(X_test)
                    y_pred = le.inverse_transform(y_pred_enc)
                else:
                    clf = RandomForestClassifier(
                        n_estimators=300, random_state=42, n_jobs=-1
                    )
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                mf1 = f1_score(y_test, y_pred, average='macro')

                # Per-class F1
                classes_in_fold = np.unique(y_test)
                pc_f1 = {}
                for c in classes_in_fold:
                    pc_f1[int(c)] = f1_score(y_test, y_pred, labels=[c], average='macro')

                fold_results.append({
                    'fold': fold_idx,
                    'accuracy': acc,
                    'macro_f1': mf1,
                    'per_class_f1': pc_f1,
                })

            all_per_fold[(tname, model_name, split_name)] = fold_results

# ============================================================
# ITEM 1: Per-fold Macro-F1 mean ± std + raw values
# ============================================================
print("\n" + "=" * 60)
print("ITEM 1: Per-fold Macro-F1 (mean ± std + 5 raw values)")
print("=" * 60)

for tname in target_names:
    for model_name in ['RF', 'XGB']:
        for split_name in ['stratified', 'group']:
            key = (tname, model_name, split_name)
            folds = all_per_fold[key]
            raw_vals = [f['macro_f1'] for f in folds]
            mean_v = np.mean(raw_vals)
            std_v = np.std(raw_vals)
            print(f"  {tname:<14} {model_name:>3} {split_name:<10}: "
                  f"{mean_v:.4f} ± {std_v:.4f}   raw={[round(v, 4) for v in raw_vals]}")

# ============================================================
# ITEM 2: Per-class F1 for all targets, both splits
# ============================================================
print("\n" + "=" * 60)
print("ITEM 2: Per-class F1 (mean ± std)")
print("=" * 60)

for tname in target_names:
    # Determine all possible classes for this target
    y_all = labels_s[:, target_cols[target_names.index(tname)]]
    all_classes = sorted(np.unique(y_all))

    print(f"\n--- {tname} ---")
    for model_name in ['RF', 'XGB']:
        print(f"  {model_name}:")
        for split_name in ['stratified', 'group']:
            key = (tname, model_name, split_name)
            folds = all_per_fold[key]
            class_vals = {c: [] for c in all_classes}
            for f in folds:
                for c in all_classes:
                    class_vals[c].append(f['per_class_f1'].get(c, np.nan))
            parts = []
            for c in all_classes:
                vals_c = [v for v in class_vals[c] if not np.isnan(v)]
                if vals_c:
                    parts.append(f"cls{c}={np.mean(vals_c):.4f}±{np.std(vals_c):.4f}")
                else:
                    parts.append(f"cls{c}=N/A")
            print(f"    {split_name:<10}: {' | '.join(parts)}")

# Accumulator special table
print("\n" + "=" * 60)
print("ITEM 2b: Accumulator per-class F1 (dedicated table)")
print("=" * 60)
acc_classes = sorted(np.unique(labels_s[:, 3]))
print(f"  {'':>6} {'90':>18} {'100':>18} {'115':>18} {'130':>18}")
print(f"  {'':>6} {'-'*18} {'-'*18} {'-'*18} {'-'*18}")
for model_name in ['RF', 'XGB']:
    for split_name in ['stratified', 'group']:
        key = ('Accumulator', model_name, split_name)
        folds = all_per_fold[key]
        parts = []
        for c in acc_classes:
            vals = [f['per_class_f1'].get(c, np.nan) for f in folds]
            vals_c = [v for v in vals if not np.isnan(v)]
            if vals_c:
                parts.append(f"{np.mean(vals_c):.4f}±{np.std(vals_c):.4f}")
            else:
                parts.append("N/A")
        label = f"{model_name} {split_name}"
        row = "  ".join(parts)
        print(f"  {label:<15} {parts[0]:>18} {parts[1]:>18} {parts[2]:>18} {parts[3]:>18}")

# ============================================================
# ITEM 3: RF vs XGB paired fold differences on Accumulator GroupKFold
# ============================================================
print("\n" + "=" * 60)
print("ITEM 3: RF vs XGB on Accumulator GroupKFold — paired differences")
print("=" * 60)

rf_folds = all_per_fold[('Accumulator', 'RF', 'group')]
xgb_folds = all_per_fold[('Accumulator', 'XGB', 'group')]

diffs = []
for fold_idx in range(5):
    rf_mf1 = rf_folds[fold_idx]['macro_f1']
    xgb_mf1 = xgb_folds[fold_idx]['macro_f1']
    diff = xgb_mf1 - rf_mf1  # positive = XGB better
    diffs.append(diff)
    print(f"  Fold {fold_idx}: XGB={xgb_mf1:.4f}  RF={rf_mf1:.4f}  diff(XGB-RF)={diff:+.4f}")

diffs = np.array(diffs)
print(f"  Mean diff: {diffs.mean():+.4f}")
print(f"  Std diff:  {diffs.std():.4f}")
all_same_sign = np.all(diffs > 0) or np.all(diffs < 0)
print(f"  All 5 diffs same sign: {all_same_sign}")
if not all_same_sign:
    print("  => GAP IS UNSTABLE: not all folds agree on which model is better")
else:
    print("  => All folds consistently favor the same model")

# ============================================================
# DONE
# ============================================================
print("\n" + "=" * 60)
print("ALL ITEMS COMPLETE")
print("=" * 60)
