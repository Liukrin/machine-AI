import numpy as np
import pandas as pd
import os, sys, warnings, json
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_validate
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ============================================================
# 0. Load data
# ============================================================
data = np.load('data/processed/tensor_31ch_60steps.npz', allow_pickle=True)
tensor   = data['tensor']         # (2205, 31, 60)
stable_mask = data['stable_mask'] # (2205,) bool
labels   = data['labels']         # (2205, 4) int
combo_id = data['combo_id']       # (2205,) int
block_id = data['block_id']       # (2205,) int
ch_names = data['channel_names']  # (31,) str

# Filter to stable
tensor_s = tensor[stable_mask]    # (1449, 31, 60)
labels_s = labels[stable_mask]    # (1449, 4)
combo_s  = combo_id[stable_mask]  # (1449,)
block_s  = block_id[stable_mask]  # (1449,) int

print(f"Stable tensor: {tensor_s.shape}, dtype={tensor_s.dtype}")
print(f"Stable labels: {labels_s.shape}")
print(f"Stable combo_id unique: {len(np.unique(combo_s))}")
print(f"Stable block_id unique: {len(np.unique(block_s))}")

# ============================================================
# 1. Build 62-dim cycle-level features
# ============================================================
feat_mean = tensor_s.mean(axis=2)  # (1449, 31)
feat_std  = tensor_s.std(axis=2)   # (1449, 31)
X = np.hstack([feat_mean, feat_std]).astype(np.float32)  # (1449, 62)

# Feature names
fnames = []
for ch in ch_names:
    fnames.append(f'{ch}_tmean')
for ch in ch_names:
    fnames.append(f'{ch}_tstd')
print(f"Feature matrix: {X.shape}")
print(f"First 5 feature names: {fnames[:5]}")

# ============================================================
# 2. Pre-check: block_id vs combo_id 1:1?
# ============================================================
# Map block_id -> set of combo_ids in that block
block_to_combos = {}
for b in np.unique(block_s):
    mask_b = block_s == b
    combos_in_block = set(combo_s[mask_b])
    block_to_combos[b] = combos_in_block

# Map combo_id -> set of block_ids
combo_to_blocks = {}
for c in np.unique(combo_s):
    mask_c = combo_s == c
    blocks_in_combo = set(block_s[mask_c])
    combo_to_blocks[c] = blocks_in_combo

n_blocks = len(block_to_combos)
n_combos = len(combo_to_blocks)
print(f"\n=== PRECHECK ===")
print(f"Unique block_ids: {n_blocks}")
print(f"Unique combo_ids: {n_combos}")

# Check 1:1
all_block_one_combo = all(len(v) == 1 for v in block_to_combos.values())
all_combo_one_block = all(len(v) == 1 for v in combo_to_blocks.values())
is_one_to_one = all_block_one_combo and all_combo_one_block
print(f"Each block maps to exactly 1 combo: {all_block_one_combo}")
print(f"Each combo maps to exactly 1 block: {all_combo_one_block}")
print(f"1:1 mapping: {is_one_to_one}")

if not is_one_to_one:
    print("WARNING: NOT 1:1! Details:")
    for b, combos in block_to_combos.items():
        if len(combos) != 1:
            print(f"  block {b} has combos: {combos}")
    for c, blocks in combo_to_blocks.items():
        if len(blocks) != 1:
            print(f"  combo {c} has blocks: {blocks}")

if is_one_to_one:
    print("\nGroupKFold(groups=block_id) 等价于按工况配置留出，")
    print("测试集中的标签组合在训练集中完全不出现。")
else:
    print("\n不是 1:1，实际见上。")

# ============================================================
# 3. Experiment setup
# ============================================================
target_names = ['Cooler', 'Valve', 'Pump_Leakage', 'Accumulator']
target_cols  = [0, 1, 2, 3]

# Custom scorers
def macro_f1_scorer(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')

def per_class_f1_scorer(y_true, y_pred):
    """Return dict of per-class F1"""
    classes = np.unique(y_true)
    out = {}
    for c in classes:
        out[f'f1_class_{c}'] = f1_score(y_true, y_pred, labels=[c], average='macro')
    return out

scoring = {
    'accuracy': 'accuracy',
    'macro_f1': make_scorer(macro_f1_scorer),
}

# ============================================================
# 4. Run all experiments
# ============================================================

def run_experiment(X, y, groups, split_method, model, model_name, target_name):
    """
    Run 5-fold CV with given split.
    Returns dict of per-fold metrics.
    """
    results_per_fold = []

    if split_method == 'stratified':
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        split_iter = list(kf.split(X, y))
    else:  # 'group'
        kf = GroupKFold(n_splits=5)
        split_iter = list(kf.split(X, y, groups=groups))

    for fold_idx, (train_idx, test_idx) in enumerate(split_iter):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if model_name == 'XGB':
            # LabelEncode y for XGB
            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train)
            y_test_enc  = le.transform(y_test)

            clf = XGBClassifier(
                n_estimators=300, max_depth=6,
                random_state=42, tree_method='hist',
                verbosity=0
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
        macro_f1 = f1_score(y_test, y_pred, average='macro')

        # Per-class F1
        classes_in_test = np.unique(y_test)
        per_class_f1 = {}
        for c in classes_in_test:
            per_class_f1[int(c)] = f1_score(y_test, y_pred, labels=[c], average='macro')

        fold_result = {
            'fold': fold_idx,
            'accuracy': acc,
            'macro_f1': macro_f1,
            'per_class_f1': per_class_f1,
            'test_size': len(test_idx),
            'test_classes': {int(c): int((y_test == c).sum()) for c in classes_in_test},
        }

        # For GroupKFold, also record block info
        if split_method == 'group':
            fold_result['test_blocks'] = len(np.unique(groups[test_idx]))

        results_per_fold.append(fold_result)

    return results_per_fold

# Storage
all_results = {}  # key: (target, model, split_method)

# Run everything
for tname, tcol in zip(target_names, target_cols):
    y = labels_s[:, tcol]
    groups = block_s

    print(f"\n{'='*60}")
    print(f"Target: {tname}")
    print(f"{'='*60}")

    for model_cls, model_name in [(RandomForestClassifier, 'RF'), (XGBClassifier, 'XGB')]:
        for split_method in ['stratified', 'group']:
            key = (tname, model_name, split_method)
            print(f"  Running {model_name} + {split_method}...", end=' ', flush=True)
            results = run_experiment(X, y, groups, split_method, model_cls, model_name, tname)
            all_results[key] = results
            n_done = len(all_results)
            print(f"done ({n_done}/16)")

# ============================================================
# 5. Print intermediate values for GroupKFold
# ============================================================
print("\n" + "="*70)
print("GROUP KFOLD INTERMEDIATE VALUES")
print("="*70)

for tname in target_names:
    print(f"\n--- {tname} (XGB GroupKFold) ---")
    key = (tname, 'XGB', 'group')
    for fold_r in all_results[key]:
        fidx = fold_r['fold']
        print(f"  Fold {fidx}: test_size={fold_r['test_size']}, "
              f"test_blocks={fold_r['test_blocks']}")
        print(f"    Class distribution: {fold_r['test_classes']}")
        # Check for zero-count classes
        all_possible = sorted(set(labels_s[:, target_cols[target_names.index(tname)]]))
        for c in all_possible:
            if c not in fold_r['test_classes'] or fold_r['test_classes'][c] == 0:
                print(f"    *** WARNING: class {c} count is ZERO in fold {fidx}! ***")

# ============================================================
# 6. Build summary tables
# ============================================================

def summarize(results_list):
    """Aggregate 5-fold results into mean ± std dict."""
    accs = [r['accuracy'] for r in results_list]
    mfs  = [r['macro_f1'] for r in results_list]

    # Collect all classes seen across folds
    all_classes = set()
    for r in results_list:
        all_classes.update(r['per_class_f1'].keys())
    all_classes = sorted(all_classes)

    per_class_stats = {}
    for c in all_classes:
        vals = [r['per_class_f1'].get(c, np.nan) for r in results_list]
        # Filter out nan (class not in fold)
        vals_clean = [v for v in vals if not np.isnan(v)]
        if vals_clean:
            per_class_stats[c] = (np.mean(vals_clean), np.std(vals_clean))
        else:
            per_class_stats[c] = (np.nan, np.nan)

    return {
        'accuracy': (np.mean(accs), np.std(accs)),
        'macro_f1': (np.mean(mfs), np.std(mfs)),
        'per_class': per_class_stats,
    }

# Build all summaries
summaries = {}
for key, results in all_results.items():
    summaries[key] = summarize(results)

# ============================================================
# 7. Print main tables and build CSV/MD
# ============================================================

csv_rows = []
md_lines = []
md_lines.append("# Split Comparison: StratifiedKFold vs GroupKFold\n")
md_lines.append("## Model: RandomForestClassifier(n_estimators=300, random_state=42)\n")

for model_name in ['RF', 'XGB']:
    if model_name == 'XGB':
        md_lines.append("\n## Model: XGBClassifier(n_estimators=300, max_depth=6, tree_method='hist')\n")

    for tname in target_names:
        print(f"\n{'='*70}")
        print(f"Target: {tname} | Model: {model_name}")
        print(f"{'='*70}")

        s_strat = summaries[(tname, model_name, 'stratified')]
        s_group = summaries[(tname, model_name, 'group')]

        # Row 1: Stratified
        acc_s_mean, acc_s_std = s_strat['accuracy']
        mf_s_mean, mf_s_std = s_strat['macro_f1']
        row1 = f"Stratified  | Acc={acc_s_mean:.4f}±{acc_s_std:.4f} | Macro-F1={mf_s_mean:.4f}±{mf_s_std:.4f}"

        # Per-class for stratified
        pcf_s_parts = []
        for c in sorted(s_strat['per_class'].keys()):
            m, s = s_strat['per_class'][c]
            pcf_s_parts.append(f"cls{c}={m:.4f}±{s:.4f}")
        row1_pc = " | ".join(pcf_s_parts)

        # Row 2: Group
        acc_g_mean, acc_g_std = s_group['accuracy']
        mf_g_mean, mf_g_std = s_group['macro_f1']
        row2 = f"GroupKFold  | Acc={acc_g_mean:.4f}±{acc_g_std:.4f} | Macro-F1={mf_g_mean:.4f}±{mf_g_std:.4f}"

        pcf_g_parts = []
        for c in sorted(s_group['per_class'].keys()):
            m, s = s_group['per_class'][c]
            pcf_g_parts.append(f"cls{c}={m:.4f}±{s:.4f}")
        row2_pc = " | ".join(pcf_g_parts)

        # Row 3: Delta
        d_acc = acc_s_mean - acc_g_mean
        d_mf  = mf_s_mean - mf_g_mean
        row3 = f"Delta       | Acc={d_acc:.4f} | Macro-F1={d_mf:.4f}"

        pcf_d_parts = []
        for c in sorted(s_strat['per_class'].keys()):
            ms, _ = s_strat['per_class'][c]
            mg, _ = s_group['per_class'][c]
            pcf_d_parts.append(f"cls{c}={ms-mg:.4f}")
        row3_pc = " | ".join(pcf_d_parts)

        # Row 4: Relative drop
        rd_acc = (d_acc / acc_s_mean * 100) if acc_s_mean > 0 else 0
        rd_mf  = (d_mf / mf_s_mean * 100) if mf_s_mean > 0 else 0
        row4 = f"Rel Drop %  | Acc={rd_acc:.2f}% | Macro-F1={rd_mf:.2f}%"

        pcf_r_parts = []
        for c in sorted(s_strat['per_class'].keys()):
            ms, _ = s_strat['per_class'][c]
            mg, _ = s_group['per_class'][c]
            rd = ((ms - mg) / ms * 100) if ms > 0 else 0
            pcf_r_parts.append(f"cls{c}={rd:.2f}%")
        row4_pc = " | ".join(pcf_r_parts)

        print(row1)
        print(f"  Per-class: {row1_pc}")
        print(row2)
        print(f"  Per-class: {row2_pc}")
        print(row3)
        print(f"  Per-class: {row3_pc}")
        print(row4)
        print(f"  Per-class: {row4_pc}")

        # CSV rows
        csv_rows.append({
            'model': model_name, 'target': tname,
            'row': 'Stratified', 'accuracy_mean': acc_s_mean, 'accuracy_std': acc_s_std,
            'macro_f1_mean': mf_s_mean, 'macro_f1_std': mf_s_std,
            **{f'f1_cls{c}_mean': m for c, (m, s) in s_strat['per_class'].items()},
            **{f'f1_cls{c}_std': s for c, (m, s) in s_strat['per_class'].items()},
        })
        csv_rows.append({
            'model': model_name, 'target': tname,
            'row': 'GroupKFold', 'accuracy_mean': acc_g_mean, 'accuracy_std': acc_g_std,
            'macro_f1_mean': mf_g_mean, 'macro_f1_std': mf_g_std,
            **{f'f1_cls{c}_mean': m for c, (m, s) in s_group['per_class'].items()},
            **{f'f1_cls{c}_std': s for c, (m, s) in s_group['per_class'].items()},
        })
        csv_rows.append({
            'model': model_name, 'target': tname,
            'row': 'Delta', 'accuracy_mean': d_acc, 'accuracy_std': 0,
            'macro_f1_mean': d_mf, 'macro_f1_std': 0,
            **{f'f1_cls{c}_mean':
               s_strat['per_class'][c][0] - s_group['per_class'][c][0]
               for c in sorted(s_strat['per_class'].keys())},
        })
        csv_rows.append({
            'model': model_name, 'target': tname,
            'row': 'RelDrop%', 'accuracy_mean': rd_acc, 'accuracy_std': 0,
            'macro_f1_mean': rd_mf, 'macro_f1_std': 0,
            **{f'f1_cls{c}_mean':
               ((s_strat['per_class'][c][0] - s_group['per_class'][c][0]) / s_strat['per_class'][c][0] * 100)
               if s_strat['per_class'][c][0] > 0 else 0
               for c in sorted(s_strat['per_class'].keys())},
        })

        # MD lines
        md_lines.append(f"\n### {tname}\n")
        md_lines.append(f"| Split | Accuracy | Macro-F1 | " +
                        " | ".join(f"F1 cls={c}" for c in sorted(s_strat['per_class'].keys())) + " |")
        md_lines.append("|-------|----------|----------|" +
                        "|".join(["----------" for _ in s_strat['per_class']]) + "|")

        # Stratified row
        vals_s = [f"{acc_s_mean:.4f}±{acc_s_std:.4f}", f"{mf_s_mean:.4f}±{mf_s_std:.4f}"]
        for c in sorted(s_strat['per_class'].keys()):
            m, s = s_strat['per_class'][c]
            vals_s.append(f"{m:.4f}±{s:.4f}")
        md_lines.append("| StratifiedKFold | " + " | ".join(vals_s) + " |")

        # Group row
        vals_g = [f"{acc_g_mean:.4f}±{acc_g_std:.4f}", f"{mf_g_mean:.4f}±{mf_g_std:.4f}"]
        for c in sorted(s_group['per_class'].keys()):
            m, s = s_group['per_class'][c]
            vals_g.append(f"{m:.4f}±{s:.4f}")
        md_lines.append("| GroupKFold | " + " | ".join(vals_g) + " |")

        # Delta row
        vals_d = [f"{d_acc:.4f}", f"{d_mf:.4f}"]
        for c in sorted(s_strat['per_class'].keys()):
            ms, _ = s_strat['per_class'][c]
            mg, _ = s_group['per_class'][c]
            vals_d.append(f"{ms-mg:.4f}")
        md_lines.append("| **Delta (Strat-Group)** | " + " | ".join(vals_d) + " |")

        # Rel drop row
        vals_r = [f"{rd_acc:.2f}%", f"{rd_mf:.2f}%"]
        for c in sorted(s_strat['per_class'].keys()):
            ms, _ = s_strat['per_class'][c]
            mg, _ = s_group['per_class'][c]
            rd = ((ms - mg) / ms * 100) if ms > 0 else 0
            vals_r.append(f"{rd:.2f}%")
        md_lines.append("| **Rel Drop %** | " + " | ".join(vals_r) + " |")

# Save CSV
os.makedirs('reports', exist_ok=True)
csv_df = pd.DataFrame(csv_rows)
csv_df.to_csv('reports/split_comparison.csv', index=False)
print(f"\nSaved reports/split_comparison.csv")

# Save MD
with open('reports/split_comparison.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))
print(f"Saved reports/split_comparison.md")

# ============================================================
# 8. Answer four questions
# ============================================================
print("\n" + "="*70)
print("FOUR QUESTIONS")
print("="*70)

# Find max/min drop targets by macro-F1 relative drop
drops = {}
for tname in target_names:
    for model_name in ['RF', 'XGB']:
        s_s = summaries[(tname, model_name, 'stratified')]
        s_g = summaries[(tname, model_name, 'group')]
        mf_s = s_s['macro_f1'][0]
        mf_g = s_g['macro_f1'][0]
        rd = (mf_s - mf_g) / mf_s * 100 if mf_s > 0 else 0
        drops[(tname, model_name)] = rd

# Q1
print("\nQ1: Macro-F1 relative drops by target:")
for tname in target_names:
    rf_d = drops[(tname, 'RF')]
    xgb_d = drops[(tname, 'XGB')]
    print(f"  {tname}: RF={rf_d:.2f}%, XGB={xgb_d:.2f}%")

# Find max/min averaged over both models
avg_drops = {}
for tname in target_names:
    avg_drops[tname] = (drops[(tname, 'RF')] + drops[(tname, 'XGB')]) / 2
max_target = max(avg_drops, key=avg_drops.get)
min_target = min(avg_drops, key=avg_drops.get)
print(f"  => Max drop: {max_target} ({avg_drops[max_target]:.2f}% avg)")
print(f"  => Min drop: {min_target} ({avg_drops[min_target]:.2f}% avg)")

# Q2: Cooler GroupKFold
print("\nQ2: Cooler GroupKFold scores:")
for model_name in ['RF', 'XGB']:
    s_g = summaries[('Cooler', model_name, 'group')]
    print(f"  {model_name}: Acc={s_g['accuracy'][0]:.4f}, Macro-F1={s_g['macro_f1'][0]:.4f}")

# Q3: Accumulator GroupKFold Macro-F1
print("\nQ3: Accumulator GroupKFold Macro-F1:")
for model_name in ['RF', 'XGB']:
    s_g = summaries[('Accumulator', model_name, 'group')]
    print(f"  {model_name}: Macro-F1={s_g['macro_f1'][0]:.4f}")

# Q4: RF vs XGBoost consistency
print("\nQ4: RF vs XGBoost direction consistency:")
consistent = True
for tname in target_names:
    rf_drop = drops[(tname, 'RF')]
    xgb_drop = drops[(tname, 'XGB')]
    same_direction = (rf_drop > 0) == (xgb_drop > 0)
    rank_rf = sorted(avg_drops.keys(), key=lambda k: drops[(k, 'RF')])
    rank_xgb = sorted(avg_drops.keys(), key=lambda k: drops[(k, 'XGB')])
    print(f"  {tname}: RF drop={rf_drop:.2f}%, XGB drop={xgb_drop:.2f}%, same_dir={same_direction}")
print(f"  RF severity order:   {rank_rf}")
print(f"  XGB severity order:  {rank_xgb}")
print(f"  Same ordering: {rank_rf == rank_xgb}")

print("\n" + "="*70)
print("DONE")
print("="*70)
