import numpy as np
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

idx_ps1 = ch_names.index('ps1_mean')
idx_ps2 = ch_names.index('ps2_mean')
dp12 = tensor_s[:, idx_ps1, :] - tensor_s[:, idx_ps2, :]
dp_scalar = dp12.mean(axis=1)

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

# X triples & ordering
cooler_vals = [3, 20, 100]
pump_vals   = [0, 1, 2]
accum_vals  = [90, 100, 115, 130]
all_X = [(c, p, a) for c in cooler_vals for p in pump_vals for a in accum_vals]

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

def get_fault_block(X, v):
    c, p, a = X
    for b, info in block_info.items():
        if info['cooler'] == c and info['pump'] == p and info['accum'] == a and info['valve'] == v:
            return b
    return None

# ============================================================
# STEP 1: Confounding attribution (previous round pairings)
# ============================================================
print("\n" + "=" * 70)
print("STEP 1: Confounding attribution")
print("=" * 70)

# Reconstruct previous round's "nearest healthy" pairings for calib
prev_pairs = []
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
        prev_pairs.append({
            'X_i': X_i, 'X_j': best_X_j,
            'pump_i': X_i[1], 'pump_j': best_X_j[1],
            'd_drift': d, 'dist': best_dist,
        })

# Group by ordered pump change
from collections import defaultdict
pump_groups = defaultdict(list)
for pp in prev_pairs:
    key = (pp['pump_i'], pp['pump_j'])
    pump_groups[key].append(pp['d_drift'])

print(f"  Ordered pump change (pump_i -> pump_j):")
all_keys = sorted(pump_groups.keys())
for key in all_keys:
    vals = pump_groups[key]
    print(f"    {key[0]}->{key[1]}: n={len(vals)}, median d_drift={np.median(vals):.4f} bar, "
          f"values={[f'{v:.4f}' for v in sorted(vals)]}")

# Monotonic check
for delta_pump in [1, 2]:
    keys_d = [k for k in all_keys if abs(k[0]-k[1]) == delta_pump]
    if len(keys_d) >= 2:
        m0 = np.median(pump_groups[keys_d[0]]) if pump_groups[keys_d[0]] else 0
        m1 = np.median(pump_groups[keys_d[1]]) if pump_groups[keys_d[1]] else 0
        print(f"    |Δpump|={delta_pump}: {keys_d[0]} median={m0:.4f}, {keys_d[1]} median={m1:.4f}")

# Check: |Δpump|=1 pairs that landed in high-value region
# High-value region = > 0.5 bar
for key in all_keys:
    if abs(key[0] - key[1]) == 1:
        vals = pump_groups[key]
        high = [v for v in vals if v > 0.5]
        low = [v for v in vals if v <= 0.5]
        print(f"    {key[0]}->{key[1]}: {len(low)} low (<=0.5), {len(high)} high (>0.5)")
        if high:
            print(f"      High-value pairs belong to: {key[0]}->{key[1]}")

# ============================================================
# STEP 2: Construct matched control pairs (calib)
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Matched control pairs")
print("=" * 70)

# New rule: same cooler, same pump_leakage, different accumulator
matched_pairs_calib = []
for i, X_i in enumerate(X_calib):
    for j, X_j in enumerate(X_calib):
        if j <= i: continue  # dedup: only i<j
        # Must have same cooler and pump, different accumulator
        if X_i[0] != X_j[0]: continue  # cooler must match
        if X_i[1] != X_j[1]: continue  # pump must match
        if X_i[2] == X_j[2]: continue  # accum must differ
        b_i = healthy_map.get(X_i); b_j = healthy_map.get(X_j)
        if b_i is None or b_j is None: continue
        dist = abs(block_info[b_i]['start'] - block_info[b_j]['start'])
        d = abs(block_info[b_j]['dp_median'] - block_info[b_i]['dp_median'])
        matched_pairs_calib.append({
            'X_i': X_i, 'X_j': X_j, 'd_drift': d, 'dist': dist,
            'cooler_i': X_i[0], 'pump_i': X_i[1], 'accum_i': X_i[2],
            'cooler_j': X_j[0], 'pump_j': X_j[1], 'accum_j': X_j[2],
        })

print(f"  Total matched pairs (calib): {len(matched_pairs_calib)}")
if matched_pairs_calib:
    dists = [p['dist'] for p in matched_pairs_calib]
    print(f"  stable_idx distance: min={min(dists)}, mean={np.mean(dists):.0f}, max={max(dists)}")

# Verify all pairs meet criteria
all_ok = all(p['cooler_i'] == p['cooler_j'] and p['pump_i'] == p['pump_j']
             for p in matched_pairs_calib)
print(f"  All pairs same cooler & pump: {all_ok}")
if not all_ok:
    bad = [p for p in matched_pairs_calib
           if p['cooler_i'] != p['cooler_j'] or p['pump_i'] != p['pump_j']]
    for p in bad:
        print(f"    BAD: X_i={p['X_i']} X_j={p['X_j']}")

# ============================================================
# STEP 3: Recalibrate
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Recalibration")
print("=" * 70)

d_drift_new = np.array(sorted([p['d_drift'] for p in matched_pairs_calib]))
print(f"  Sorted d_drift ({len(d_drift_new)} values, bar):")
print(f"  {np.array2string(d_drift_new, precision=4, max_line_width=120)}")

p50_new = np.percentile(d_drift_new, 50)
p90_new = np.percentile(d_drift_new, 90)
p95_new = np.percentile(d_drift_new, 95)
tau_dp_new = p95_new
print(f"\n  P50={p50_new:.4f}, P90={p90_new:.4f}, P95={tau_dp_new:.4f}")
print(f"  tau_dp = {tau_dp_new:.4f} bar")

# Check bimodality
d_sorted = sorted(d_drift_new)
gaps = [d_sorted[i+1] - d_sorted[i] for i in range(len(d_sorted)-1)]
max_gap = max(gaps) if gaps else 0
median_gap = np.median(gaps) if gaps else 0
is_bimodal_new = max_gap > 3 * median_gap if median_gap > 0 else False
print(f"  Distribution still bimodal? {is_bimodal_new} (max_gap/median_gap={max_gap/median_gap:.1f})")
if not is_bimodal_new:
    print(f"  => Single-peak. Pump WAS the cause of bimodality in the previous round.")

spread_ratio = p95_new / (p50_new + 1e-9)
print(f"  P95/P50 ratio = {spread_ratio:.2f}")
print(f"  {'Still heavy-tailed' if spread_ratio > 2 else 'Reasonably concentrated'}")

# Compare with fault offsets
print(f"\n  Threshold vs fault offsets:")
fault_offsets = {73: 0.820, 80: 0.471, 90: 0.189}
for v in [73, 80, 90]:
    detectable = fault_offsets[v] > tau_dp_new
    print(f"    valve={v}: fault |Δdp|={fault_offsets[v]:.3f} > tau_dp={tau_dp_new:.4f} ? {detectable}")

# ============================================================
# STEP 4: Evaluation (test 18 X)
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Evaluation (test 18 X)")
print("=" * 70)

# A: fault detection
print(f"\n  A (Fault) detection:")
det_by_v = {}
for v in [73, 80, 90]:
    n_tot = 0; n_det = 0
    for X in X_test:
        b_h = healthy_map.get(X); b_f = get_fault_block(X, v)
        if b_h is None or b_f is None: continue
        if abs(block_info[b_f]['dp_median'] - block_info[b_h]['dp_median']) > tau_dp_new:
            n_det += 1
        n_tot += 1
    det_by_v[v] = n_det / n_tot if n_tot > 0 else 0
    print(f"    valve={v}: {n_det}/{n_tot} = {det_by_v[v]:.4f}")

# B: near healthy (within-block 5 vs 5)
print(f"\n  B (Near healthy) false alarm:")
n_B_tot = 0; n_B_fa = 0
for X in X_test:
    b_h = healthy_map.get(X)
    if b_h is None: continue
    indices = block_info[b_h]['indices']
    if len(indices) < 10: continue
    if abs(np.median(dp_scalar[indices[5:10]]) -
           np.median(dp_scalar[indices[:5]])) > tau_dp_new:
        n_B_fa += 1
    n_B_tot += 1
b_fpr = n_B_fa / n_B_tot if n_B_tot > 0 else 0
print(f"    {n_B_fa}/{n_B_tot} = {b_fpr:.4f}")

# C: matched controls (same cooler, same pump, different accum) on test set
print(f"\n  C (Matched healthy) false alarm:")
matched_pairs_test = []
for i, X_i in enumerate(list(X_test)):
    for j, X_j in enumerate(list(X_test)):
        if j <= i: continue
        if X_i[0] != X_j[0]: continue
        if X_i[1] != X_j[1]: continue
        if X_i[2] == X_j[2]: continue
        b_i = healthy_map.get(X_i); b_j = healthy_map.get(X_j)
        if b_i is None or b_j is None: continue
        d = abs(block_info[b_j]['dp_median'] - block_info[b_i]['dp_median'])
        matched_pairs_test.append(d)

n_C_tot = len(matched_pairs_test)
n_C_fa = sum(1 for d in matched_pairs_test if d > tau_dp_new)
c_fpr = n_C_fa / n_C_tot if n_C_tot > 0 else 0
print(f"    {n_C_fa}/{n_C_tot} = {c_fpr:.4f}")
print(f"    Near 5%? {'yes' if abs(c_fpr - 0.05) < 0.05 else 'NO'}")

# ============================================================
# STEP 5: Cross-round comparison
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Cross-round comparison")
print("=" * 70)

# Previous round results for valve
# Chronos ps1_mean: A=0.00, B=0.00, C=0.10
# Chronos ps2_std:  A=0.00, B=0.00, C=0.00
# Trivial mean:     A=0.00, B=0.00, C=0.10
# ΔP v1:            A=0.00, B=0.00, C=0.00

# Current round: A det rates per valve
a73 = det_by_v[73]; a80 = det_by_v[80]; a90 = det_by_v[90]

methods = [
    ('ΔP matched (this round)', a73, a80, a90, b_fpr, c_fpr),
    ('ΔP nearest (prev round)', 0.00, 0.00, 0.00, 0.00, 0.00),
    ('Chronos ps1_mean',        0.00, 0.00, 0.00, 0.00, 0.10),
    ('Chronos ps2_std',         0.00, 0.00, 0.00, 0.00, 0.00),
    ('Trivial mean',            0.00, 0.00, 0.00, 0.00, 0.10),
]

print(f"  {'Method':<25} {'A(73)':>8} {'A(80)':>8} {'A(90)':>8} {'B FPR':>8} {'C FPR':>8}")
print(f"  {'-'*65}")
for name, a73v, a80v, a90v, bv, cv in methods:
    print(f"  {name:<25} {a73v:>8.4f} {a80v:>8.4f} {a90v:>8.4f} {bv:>8.4f} {cv:>8.4f}")

# Which matters: channel or model?
print(f"\n  Key insight:")
print(f"  - Chronos on ps1_mean: 0% detection — channel cannot separate valve")
print(f"  - Chronos on ps2_std:  0% detection — channel passes pre-check but Chronos residuals fail")
print(f"  - ΔP with dp12 + wrong pairing: 0% detection")
print(f"  - ΔP with dp12 + matched pairing: {a73:.0%}/{a80:.0%}/{a90:.0%} detection")
print(f"  => What works: dp12 channel + correct control group (same cooler & pump).")
print(f"  => The model (Chronos) added ZERO value across all valve experiments.")

# ============================================================
# STEP 6: Stage 1 cross-validation
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Stage 1 GroupKFold comparison")
print("=" * 70)

stage1 = {73: 0.98, 80: 0.91, 90: 0.89}
print(f"  {'Valve':<10} {'ΔP Det':>10} {'GKF F1':>10}")
print(f"  {'-'*32}")
for v in [73, 80, 90]:
    print(f"  {v:<10} {det_by_v[v]:>10.4f} {stage1[v]:>10.2f}")

order_dp2 = sorted([73, 80, 90], key=lambda v: det_by_v[v], reverse=True)
order_s1 = sorted([73, 80, 90], key=lambda v: stage1[v], reverse=True)
print(f"  ΔP difficulty order:     {order_dp2}")
print(f"  Stage1 GKF F1 order:     {order_s1}")
print(f"  Same ordering: {order_dp2 == order_s1}")

# ============================================================
# 7. Plot
# ============================================================
print("\n[7] Plot...")
os.makedirs('reports/figures', exist_ok=True)

# Collect per-X dp medians for test set
plot_data = {'matched_healthy': [], 'valve90': [], 'valve80': [], 'valve73': []}

# Matched healthy pairs (test set)
for pair_d in matched_pairs_test:
    plot_data['matched_healthy'].append(pair_d)

# Fault per X
for X in X_test:
    b_h = healthy_map.get(X)
    if b_h is None: continue
    h_med = block_info[b_h]['dp_median']
    for v in [90, 80, 73]:
        b_f = get_fault_block(X, v)
        if b_f:
            plot_data[f'valve{v}'].append(abs(block_info[b_f]['dp_median'] - h_med))

fig, ax = plt.subplots(figsize=(9, 6))
categories = ['matched\nhealthy', 'valve=90', 'valve=80', 'valve=73']
colors_list = ['#4575b4', '#fdae61', '#fc8d59', '#d73027']
positions = range(4)

key_map = {'matched\nhealthy': 'matched_healthy', 'valve=90': 'valve90',
           'valve=80': 'valve80', 'valve=73': 'valve73'}
for pos, cat, col in zip(positions, categories, colors_list):
    vals = plot_data[key_map[cat]]
    if vals:
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, c=col, alpha=0.6, s=30,
                   edgecolors='white', linewidths=0.3, zorder=5)

ax.axhline(y=tau_dp_new, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'tau_dp = {tau_dp_new:.4f} bar')

ax.set_xticks(list(positions))
ax.set_xticklabels(['Matched\nHealthy', 'valve=90', 'valve=80', 'valve=73'])
ax.set_ylabel('|Δdp| (bar)', fontsize=12)
n_pairs = len(matched_pairs_test)
ax.set_title(f'ΔP Pairwise Deviation by Group (Test Set)\n'
             f'{n_pairs} matched pairs, tau_dp=P95={tau_dp_new:.4f} bar',
             fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig('reports/figures/valve_dp12_matched_pairs.png', dpi=150)
plt.close(fig)
print("    Saved reports/figures/valve_dp12_matched_pairs.png")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
