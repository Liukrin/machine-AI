import numpy as np
import pandas as pd
import os, sys, warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

# ============================================================
# 0. Load
# ============================================================
print("[0] Loading...")
data = np.load('data/processed/tensor_31ch_60steps.npz', allow_pickle=True)
tensor   = data['tensor']
stable_mask = data['stable_mask']
labels   = data['labels']
block_id = data['block_id']
ch_names = list(data['channel_names'])

tensor_s = tensor[stable_mask]
labels_s = labels[stable_mask]
block_s  = block_id[stable_mask]

# dp12 = ps1_mean - ps2_mean
idx_ps1 = ch_names.index('ps1_mean')
idx_ps2 = ch_names.index('ps2_mean')
dp12 = tensor_s[:, idx_ps1, :] - tensor_s[:, idx_ps2, :]  # (1449, 60)
# Cycle-mean to scalar
dp_scalar = dp12.mean(axis=1)  # (1449,)  bar

print(f"    dp12 shape: {dp12.shape}")
print(f"    dp cycle-mean range: [{dp_scalar.min():.3f}, {dp_scalar.max():.3f}] bar")

# Block info
unique_blocks = np.unique(block_s)
block_info = {}
for b in unique_blocks:
    mask = block_s == b
    indices = np.where(mask)[0]
    block_info[b] = {
        'start': indices[0], 'end': indices[-1], 'n': len(indices),
        'cooler': int(labels_s[indices[0], 0]),
        'valve': int(labels_s[indices[0], 1]),
        'pump': int(labels_s[indices[0], 2]),
        'accum': int(labels_s[indices[0], 3]),
        'indices': indices,
        'dp_median': np.median(dp_scalar[indices]),
        'dp_values': dp_scalar[indices],
    }

# X triples
cooler_vals = [3, 20, 100]
pump_vals   = [0, 1, 2]
accum_vals  = [90, 100, 115, 130]
all_X = [(c, p, a) for c in cooler_vals for p in pump_vals for a in accum_vals]

# Order X by stable_idx of valve=100 block
X_with_start = []
for X in all_X:
    c, p, a = X
    for b, info in block_info.items():
        if info['cooler'] == c and info['pump'] == p and info['accum'] == a and info['valve'] == 100:
            X_with_start.append((X, info['start']))
            break
X_with_start.sort(key=lambda x: x[1])
X_ordered = [x[0] for x in X_with_start]

X_calib = set(X_ordered[:18])
X_test  = set(X_ordered[18:])
print(f"    Calibration X: {len(X_calib)}, Test X: {len(X_test)}, Overlap: {len(X_calib & X_test)}")

# Healthy block map
healthy_map = {}
for X in all_X:
    c, p, a = X
    for b, info in block_info.items():
        if info['cooler'] == c and info['pump'] == p and info['accum'] == a and info['valve'] == 100:
            healthy_map[X] = b
            break

# Fault block map
def get_fault_block(X, v):
    c, p, a = X
    for b, info in block_info.items():
        if info['cooler'] == c and info['pump'] == p and info['accum'] == a and info['valve'] == v:
            return b
    return None

# ============================================================
# STEP 1: Absolute scale examination (calibration only)
# ============================================================
print("\n" + "=" * 70)
print("STEP 1: Absolute scale examination (calibration only)")
print("=" * 70)

# 1. Healthy block dp medians
healthy_dp = []
for X in X_calib:
    b = healthy_map.get(X)
    if b: healthy_dp.append(block_info[b]['dp_median'])
healthy_dp = np.array(healthy_dp)
print(f"\n  1. Healthy block dp median (bar):")
print(f"     min={healthy_dp.min():.4f}, median={np.median(healthy_dp):.4f}, max={healthy_dp.max():.4f}")

# 2. Fault-healthy differences
print(f"\n  2. Fault-healthy dp median difference |Δdp| (bar):")
for v in [73, 80, 90]:
    diffs = []
    for X in X_calib:
        b_h = healthy_map.get(X)
        b_f = get_fault_block(X, v)
        if b_h and b_f:
            diffs.append(abs(block_info[b_f]['dp_median'] - block_info[b_h]['dp_median']))
    diffs = np.array(diffs)
    print(f"     valve={v}: min={diffs.min():.4f}, median={np.median(diffs):.4f}, max={diffs.max():.4f} bar")

# 3. Within-block std of healthy dp
print(f"\n  3. std(healthy dp) within block (bar):")
h_stds = []
for X in X_calib:
    b = healthy_map.get(X)
    if b: h_stds.append(np.std(block_info[b]['dp_values']))
h_stds = np.array(h_stds)
print(f"     min={h_stds.min():.4f}, median={np.median(h_stds):.4f}, max={h_stds.max():.4f}")

# Normalized: sep with 3σ band
print(f"\n    Normalized (Δdp / σ_healthy):")
for v in [73, 80, 90]:
    ratios = []
    for X in X_calib:
        b_h = healthy_map.get(X)
        b_f = get_fault_block(X, v)
        if b_h and b_f:
            diff = abs(block_info[b_f]['dp_median'] - block_info[b_h]['dp_median'])
            std_h = np.std(block_info[b_h]['dp_values'])
            ratios.append(diff / (std_h + 1e-9))
    ratios = np.array(ratios)
    print(f"     valve={v}: min={ratios.min():.1f}, median={np.median(ratios):.1f}, max={ratios.max():.1f}")

# 4. Nearest healthy drift
print(f"\n  4. Nearest healthy block drift |Δdp| (bar):")
drift_vals = []
for X_i in X_calib:
    b_i = healthy_map.get(X_i)
    if b_i is None: continue
    h_start = block_info[b_i]['start']
    best_dist = 999999; best_X_j = None
    for X_j in all_X:
        if X_j == X_i: continue
        b_j = healthy_map.get(X_j)
        if b_j is None: continue
        dist = abs(block_info[b_j]['start'] - h_start)
        if dist < best_dist:
            best_dist = dist; best_X_j = X_j
    if best_X_j:
        d = abs(block_info[healthy_map[best_X_j]]['dp_median'] - block_info[b_i]['dp_median'])
        drift_vals.append(d)
drift_vals = np.array(drift_vals)
print(f"     min={drift_vals.min():.4f}, median={np.median(drift_vals):.4f}, max={drift_vals.max():.4f} bar")

# Explicit comparison
print(f"\n  Fault offset vs drift (absolute, bar):")
for v in [73, 80, 90]:
    diffs_v = []
    for X in X_calib:
        b_h = healthy_map.get(X); b_f = get_fault_block(X, v)
        if b_h and b_f: diffs_v.append(abs(block_info[b_f]['dp_median'] - block_info[b_h]['dp_median']))
    diffs_v = np.array(diffs_v)
    bigger = (np.median(diffs_v) > np.median(drift_vals))
    print(f"    valve={v}: fault |Δdp| median={np.median(diffs_v):.4f} vs drift median={np.median(drift_vals):.4f} "
          f"-> fault > drift: {bigger}")

# ============================================================
# STEP 2: Threshold calibration (healthy-healthy pairs, calib only)
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Threshold calibration")
print("=" * 70)

d_drift_list = []
d_drift_details = []
for X_i in X_calib:
    b_i = healthy_map.get(X_i)
    if b_i is None: continue
    h_start = block_info[b_i]['start']
    best_dist = 999999; best_X_j = None
    for X_j in all_X:
        if X_j == X_i: continue
        b_j = healthy_map.get(X_j)
        if b_j is None: continue
        dist = abs(block_info[b_j]['start'] - h_start)
        if dist < best_dist:
            best_dist = dist; best_X_j = X_j
    if best_X_j:
        b_j = healthy_map[best_X_j]
        d = abs(block_info[b_j]['dp_median'] - block_info[b_i]['dp_median'])
        d_drift_list.append(d)

        # What labels differ?
        diffs = []
        if X_i[0] != best_X_j[0]: diffs.append(f'cooler')
        if X_i[1] != best_X_j[1]: diffs.append(f'pump')
        if X_i[2] != best_X_j[2]: diffs.append(f'accum')
        dpump = abs(X_i[1] - best_X_j[1]) if X_i[1] != best_X_j[1] else 0
        d_drift_details.append({'d_drift': d, 'dpump': dpump, 'diffs': diffs})

d_drift_arr = np.array(sorted(d_drift_list))
print(f"  Sorted d_drift (18 values, bar):")
print(f"  {np.array2string(d_drift_arr, precision=4, max_line_width=120)}")

p50 = np.percentile(d_drift_arr, 50)
p90 = np.percentile(d_drift_arr, 90)
p95 = np.percentile(d_drift_arr, 95)
tau_dp = p95
print(f"\n  P50={p50:.4f}, P90={p90:.4f}, P95={p95:.4f}")
print(f"  tau_dp = P95 = {tau_dp:.4f} bar")
spread = (p95 - p50) / (p50 + 1e-9)
print(f"  Spread (P95-P50)/P50 = {spread:.2f}")
if spread > 0.5:
    print(f"  => P95 far from P50/P90. Distribution has a heavy tail — drift is dominated by outlier pairs.")
else:
    print(f"  => P95 reasonably close to P50.")

# ============================================================
# STEP 3: Evaluation (test only)
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Evaluation (test 18 X)")
print("=" * 70)

# A: fault detection
print(f"\n  A (Fault) detection:")
for v in [73, 80, 90]:
    n_tot = 0; n_det = 0
    for X in X_test:
        b_h = healthy_map.get(X)
        b_f = get_fault_block(X, v)
        if b_h is None or b_f is None: continue
        h_med = block_info[b_h]['dp_median']
        f_med = block_info[b_f]['dp_median']
        if abs(f_med - h_med) > tau_dp:
            n_det += 1
        n_tot += 1
    print(f"    valve={v}: {n_det}/{n_tot} = {n_det/n_tot:.4f}" if n_tot > 0 else f"    valve={v}: NO DATA")

# B: near healthy (within-block, last 5 vs first 5)
print(f"\n  B (Near healthy) false alarm:")
n_B_tot = 0; n_B_fa = 0
for X in X_test:
    b_h = healthy_map.get(X)
    if b_h is None: continue
    indices = block_info[b_h]['indices']
    if len(indices) < 10: continue
    first5 = np.median(dp_scalar[indices[:5]])
    last5  = np.median(dp_scalar[indices[5:10]])
    if abs(last5 - first5) > tau_dp:
        n_B_fa += 1
    n_B_tot += 1
print(f"    {n_B_fa}/{n_B_tot} = {n_B_fa/n_B_tot:.4f}" if n_B_tot > 0 else "    NO DATA")

# C: nearest healthy
print(f"\n  C (Nearest healthy) false alarm:")
n_C_tot = 0; n_C_fa = 0
for X_i in X_test:
    b_i = healthy_map.get(X_i)
    if b_i is None: continue
    h_start = block_info[b_i]['start']
    h_med = block_info[b_i]['dp_median']
    best_dist = 999999; best_X_j = None
    for X_j in all_X:
        if X_j == X_i: continue
        b_j = healthy_map.get(X_j)
        if b_j is None: continue
        dist = abs(block_info[b_j]['start'] - h_start)
        if dist < best_dist:
            best_dist = dist; best_X_j = X_j
    if best_X_j:
        near_med = block_info[healthy_map[best_X_j]]['dp_median']
        if abs(near_med - h_med) > tau_dp:
            n_C_fa += 1
        n_C_tot += 1
print(f"    {n_C_fa}/{n_C_tot} = {n_C_fa/n_C_tot:.4f}" if n_C_tot > 0 else "    NO DATA")
print(f"    Near 5%? {'yes' if abs(n_C_fa/n_C_tot - 0.05) < 0.05 else 'NO, deviation >5pp'}")

# ============================================================
# STEP 4: Comparison with Stage 1 GroupKFold F1
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Comparison with Stage 1 GroupKFold")
print("=" * 70)

# Recompute detection rate for combined A
n_A_tot = 0; n_A_det = 0
det_by_v = {}
for v in [73, 80, 90]:
    n_vt = 0; n_vd = 0
    for X in X_test:
        b_h = healthy_map.get(X); b_f = get_fault_block(X, v)
        if b_h and b_f:
            if abs(block_info[b_f]['dp_median'] - block_info[b_h]['dp_median']) > tau_dp:
                n_vd += 1
            n_vt += 1
    det_by_v[v] = n_vd / n_vt if n_vt > 0 else 0
    n_A_tot += n_vt; n_A_det += n_vd

stage1_f1 = {73: 0.98, 80: 0.91, 90: 0.89}

print(f"  {'Valve':<10} {'ΔP Det Rate':>14} {'Stage1 GKF F1':>16}")
print(f"  {'-'*42}")
for v in [73, 80, 90]:
    print(f"  {v:<10} {det_by_v[v]:>14.4f} {stage1_f1[v]:>16.2f}")
order_dp = sorted([73, 80, 90], key=lambda v: det_by_v[v], reverse=True)
order_s1 = sorted([73, 80, 90], key=lambda v: stage1_f1[v], reverse=True)
print(f"  ΔP difficulty order: {order_dp} (hardest last)")
print(f"  Stage1 difficulty order: {order_s1} (hardest last)")
print(f"  Same ordering: {order_dp == order_s1}")

# ============================================================
# STEP 5: Confounding quantification (pump effect)
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Pump confounding")
print("=" * 70)

# Use calib d_drift_details
dpump1 = [dd['d_drift'] for dd in d_drift_details if dd['dpump'] == 1]
dpump2 = [dd['d_drift'] for dd in d_drift_details if dd['dpump'] == 2]

print(f"  |Δpump|=1: n={len(dpump1)}, median d_drift={np.median(dpump1):.4f} bar")
print(f"  |Δpump|=2: n={len(dpump2)}, median d_drift={np.median(dpump2):.4f} bar")
if dpump1 and dpump2:
    ratio_pump = np.median(dpump2) / (np.median(dpump1) + 1e-9)
    print(f"  Ratio (|Δpump|=2 / |Δpump|=1): {ratio_pump:.2f}")
    print(f"  Pump change IS a major source of drift: {ratio_pump > 1.5}")

# ============================================================
# 6. Plot
# ============================================================
print("\n[6] Plot...")
os.makedirs('reports/figures', exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6))
bins = np.linspace(dp_scalar.min(), dp_scalar.max(), 60)

# Collect test-set dp values by valve
dp_by_valve = {100: [], 73: [], 80: [], 90: []}
for X in X_test:
    for v in [100, 73, 80, 90]:
        if v == 100:
            b = healthy_map.get(X)
        else:
            b = get_fault_block(X, v)
        if b:
            dp_by_valve[v].extend(block_info[b]['dp_values'])

colors = {100: '#4575b4', 73: '#d73027', 80: '#fc8d59', 90: '#fdae61'}
labels = {100: 'valve=100 (healthy)', 73: 'valve=73', 80: 'valve=80', 90: 'valve=90'}

for v in [100, 73, 80, 90]:
    vals = dp_by_valve[v]
    if vals:
        ax.hist(vals, bins=bins, alpha=0.5, color=colors[v], label=f'{labels[v]} (n={len(vals)})',
                edgecolor='white', linewidth=0.3)

# Threshold lines using healthy medians
h_meds_test = []
for X in X_test:
    b = healthy_map.get(X)
    if b: h_meds_test.append(block_info[b]['dp_median'])
if h_meds_test:
    ref_med = np.median(h_meds_test)
    ax.axvline(x=ref_med - tau_dp, color='black', linestyle='--', linewidth=1.2, alpha=0.7,
               label=f'tau_dp={tau_dp:.3f} bar')
    ax.axvline(x=ref_med + tau_dp, color='black', linestyle='--', linewidth=1.2, alpha=0.7)

ax.set_xlabel('dp12 cycle-mean (bar)', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
n_test = len(X_test)
ax.set_title(f'ΔP = PS1_mean − PS2_mean  Distribution\n'
             f'Test set ({n_test} X triples), τ_dp = P95 = {tau_dp:.4f} bar', fontsize=13)
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig('reports/figures/valve_dp12_distribution.png', dpi=150)
plt.close(fig)
print("    Saved reports/figures/valve_dp12_distribution.png")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
