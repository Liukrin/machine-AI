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
print(f"    Calib: {len(X_calib)}, Test: {len(X_test)}")

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
# STEP 1: Print all calib matched pairs from previous round
# ============================================================
print("\n" + "=" * 70)
print("STEP 1: All 21 matched pairs (previous round)")
print("=" * 70)

pairs_prev = []
for i, X_i in enumerate(list(X_calib)):
    for j, X_j in enumerate(list(X_calib)):
        if j <= i: continue
        if X_i[0] != X_j[0]: continue
        if X_i[1] != X_j[1]: continue
        if X_i[2] == X_j[2]: continue
        b_i = healthy_map.get(X_i); b_j = healthy_map.get(X_j)
        if b_i is None or b_j is None: continue
        d = abs(block_info[b_j]['dp_median'] - block_info[b_i]['dp_median'])
        pairs_prev.append((X_i[0], X_i[1], X_i[2], X_j[2], d))

pairs_prev.sort(key=lambda x: x[4])
print(f"  {'cooler':<8} {'pump':<6} {'acc_i':<8} {'acc_j':<8} {'d_drift':>10}")
print(f"  {'-'*48}")
for p in pairs_prev:
    flag = " <-- HIGH" if p[4] > 0.5 else ""
    print(f"  {p[0]:<8} {p[1]:<6} {p[2]:<8} {p[3]:<8} {p[4]:>10.4f}{flag}")

# Identify high-value pairs
high_pairs = [p for p in pairs_prev if p[4] > 0.5]
low_pairs  = [p for p in pairs_prev if p[4] <= 0.5]
print(f"\n  6 high-value pairs (d_drift > 0.5):")
acc_combos_high = set()
for p in high_pairs:
    combo = tuple(sorted([p[2], p[3]]))
    acc_combos_high.add(combo)
    print(f"    (cooler={p[0]}, pump={p[1]}, acc_i={p[2]}, acc_j={p[3]}) "
          f"-> d_drift={p[4]:.4f}")

all_involve_90 = all(90 in (p[2], p[3]) for p in high_pairs)
print(f"\n  All high-value pairs involve acc=90? {all_involve_90}")
if not all_involve_90:
    for p in high_pairs:
        if 90 not in (p[2], p[3]):
            print(f"    COUNTEREXAMPLE: ({p[2]}, {p[3]})")

# ============================================================
# STEP 2: Redefine systemic healthy control (exclude acc=90)
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Systemic healthy control (acc in {100,115,130})")
print("=" * 70)

pairs_sys = []
for i, X_i in enumerate(list(X_calib)):
    if X_i[2] == 90: continue  # exclude acc=90
    for j, X_j in enumerate(list(X_calib)):
        if j <= i: continue
        if X_j[2] == 90: continue  # exclude acc=90
        if X_i[0] != X_j[0]: continue  # same cooler
        if X_i[1] != X_j[1]: continue  # same pump
        if X_i[2] == X_j[2]: continue  # different accum
        b_i = healthy_map.get(X_i); b_j = healthy_map.get(X_j)
        if b_i is None or b_j is None: continue
        d = abs(block_info[b_j]['dp_median'] - block_info[b_i]['dp_median'])
        pairs_sys.append(d)

pairs_sys_sorted = sorted(pairs_sys)
print(f"  Total pairs: {len(pairs_sys_sorted)}")
print(f"  Sorted d_drift: {np.array2string(np.array(pairs_sys_sorted), precision=4, max_line_width=120)}")

p50_sys = np.percentile(pairs_sys_sorted, 50)
p90_sys = np.percentile(pairs_sys_sorted, 90)
p95_sys = np.percentile(pairs_sys_sorted, 95)
tau_sys = p95_sys
print(f"  P50={p50_sys:.4f}, P90={p90_sys:.4f}, P95={tau_sys:.4f}")
print(f"  tau_dp = {tau_sys:.4f} bar")

# Check bimodality
gaps_sys = [pairs_sys_sorted[i+1] - pairs_sys_sorted[i]
            for i in range(len(pairs_sys_sorted)-1)]
max_gap_sys = max(gaps_sys) if gaps_sys else 0
med_gap_sys = np.median(gaps_sys) if gaps_sys else 1
is_bimodal = max_gap_sys > 3 * med_gap_sys
print(f"  Single-peak now? {not is_bimodal} (max_gap/med_gap={max_gap_sys/med_gap_sys:.1f})")
print(f"  P95/P50 ratio = {p95_sys/(p50_sys+1e-9):.2f}")

# ============================================================
# STEP 3: Evaluation (test set)
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Evaluation (test 18 X)")
print("=" * 70)

# A: fault detection
print(f"\n  A (Fault) detection:")
det_by_v = {}
for v in [73, 80, 90]:
    n_tot = 0; n_det = 0
    for X in X_test:
        b_h = healthy_map.get(X); b_f = get_fault_block(X, v)
        if b_h is None or b_f is None: continue
        if abs(block_info[b_f]['dp_median'] - block_info[b_h]['dp_median']) > tau_sys:
            n_det += 1
        n_tot += 1
    det_by_v[v] = n_det / n_tot if n_tot > 0 else 0
    print(f"    valve={v}: {n_det}/{n_tot} = {det_by_v[v]:.4f}")

# B: near healthy
print(f"\n  B (Near healthy) false alarm:")
n_B_tot = 0; n_B_fa = 0
for X in X_test:
    b_h = healthy_map.get(X)
    if b_h is None: continue
    indices = block_info[b_h]['indices']
    if len(indices) < 10: continue
    if abs(np.median(dp_scalar[indices[5:10]]) -
           np.median(dp_scalar[indices[:5]])) > tau_sys:
        n_B_fa += 1
    n_B_tot += 1
b_fpr = n_B_fa / n_B_tot if n_B_tot > 0 else 0
print(f"    {n_B_fa}/{n_B_tot} = {b_fpr:.4f}")

# C1: systemic healthy (acc in {100,115,130})
print(f"\n  C1 (Systemic healthy) false alarm (acc in {{100,115,130}}):")
pairs_c1 = []
for i, X_i in enumerate(list(X_test)):
    if X_i[2] == 90: continue
    for j, X_j in enumerate(list(X_test)):
        if j <= i: continue
        if X_j[2] == 90: continue
        if X_i[0] != X_j[0]: continue
        if X_i[1] != X_j[1]: continue
        if X_i[2] == X_j[2]: continue
        b_i = healthy_map.get(X_i); b_j = healthy_map.get(X_j)
        if b_i is None or b_j is None: continue
        d = abs(block_info[b_j]['dp_median'] - block_info[b_i]['dp_median'])
        pairs_c1.append(d)
n_c1 = len(pairs_c1); n_c1_fa = sum(1 for d in pairs_c1 if d > tau_sys)
c1_fpr = n_c1_fa / n_c1 if n_c1 > 0 else 0
print(f"    {n_c1_fa}/{n_c1} = {c1_fpr:.4f}")
print(f"    Near 5%? {'yes' if abs(c1_fpr - 0.05) < 0.05 else 'NO'}")

# C2: acc=90 cross-check
print(f"\n  C2 (acc=90 cross detection rate):")
pairs_c2 = []
for i, X_i in enumerate(list(X_test)):
    if X_i[2] != 90: continue  # X_i must have acc=90
    for j, X_j in enumerate(list(X_test)):
        if X_i == X_j: continue
        if X_j[2] == 90: continue  # X_j must NOT have acc=90
        if X_i[0] != X_j[0]: continue
        if X_i[1] != X_j[1]: continue
        b_i = healthy_map.get(X_i); b_j = healthy_map.get(X_j)
        if b_i is None or b_j is None: continue
        d = abs(block_info[b_j]['dp_median'] - block_info[b_i]['dp_median'])
        pairs_c2.append(d)
n_c2 = len(pairs_c2); n_c2_det = sum(1 for d in pairs_c2 if d > tau_sys)
c2_rate = n_c2_det / n_c2 if n_c2 > 0 else 0
print(f"    {n_c2_det}/{n_c2} = {c2_rate:.4f}")
print(f"    C2 >> C1? {c2_rate > c1_fpr * 2}")

# ============================================================
# STEP 4: Graded response
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Graded |Δdp| by valve level")
print("=" * 70)

print(f"  {'Valve':<12} {'|Δdp| mean ± std (bar)':>30}")
print(f"  {'-'*45}")
for v_label, v_val in [('100 (healthy)', None), ('90', 90), ('80', 80), ('73', 73)]:
    vals = []
    for X in X_test:
        if v_val is None:
            # Use within-block variability: std of healthy block dp_values
            b_h = healthy_map.get(X)
            if b_h:
                vals.append(np.std(block_info[b_h]['dp_values']))
        else:
            b_h = healthy_map.get(X); b_f = get_fault_block(X, v_val)
            if b_h and b_f:
                vals.append(abs(block_info[b_f]['dp_median'] - block_info[b_h]['dp_median']))
    vals = np.array(vals)
    print(f"  {v_label:<12} {np.mean(vals):>10.4f} ± {np.std(vals):.4f}")

# Monotonic check
means_ordered = []
for v in [90, 80, 73]:
    vals = []
    for X in X_test:
        b_h = healthy_map.get(X); b_f = get_fault_block(X, v)
        if b_h and b_f: vals.append(abs(block_info[b_f]['dp_median'] - block_info[b_h]['dp_median']))
    means_ordered.append(np.mean(vals))
monotonic = means_ordered[0] <= means_ordered[1] <= means_ordered[2]
print(f"  Monotonically increasing with opening loss? {monotonic}")

# ============================================================
# STEP 5: Five-round summary
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Five-round valve summary")
print("=" * 70)

rounds = [
    ('Chronos ps1_mean',     0.00, 0.00, 0.00, 0.00, 0.10,
     'ps1_mean dominated by cooler; valve fault ~0 bar shift'),
    ('Chronos ps1_ptp',      0.60, 0.00, 0.00, 0.67, 1.00,
     'A/B/C s_level all overlap; no separation'),
    ('Chronos ps2_std',      0.00, 0.00, 0.00, 0.00, 0.00,
     'passed pre-check but Chronos residuals fail'),
    ('DeltaP nearest',       0.00, 0.00, 0.00, 0.00, 0.00,
     'tau inflated by pump-crossing pairs'),
    ('DeltaP matched (acc all)', 0.00, 0.00, 0.00, 0.00, 0.00,
     'tau inflated by acc=90 cross pairs; bimodal'),
    ('DeltaP systemic (excl 90)', *[det_by_v[v] for v in [73,80,90]], b_fpr, c1_fpr,
     f'excl acc=90; tau={tau_sys:.4f}; C1={"OK" if abs(c1_fpr-0.05)<0.05 else "NO"}'),
]

print(f"  {'Method':<28} {'A(73)':>7} {'A(80)':>7} {'A(90)':>7} {'B FPR':>7} {'C FPR':>7}  Cause")
print(f"  {'-'*100}")
for r in rounds:
    name, a73, a80, a90, bf, cf, cause = r
    print(f"  {name:<28} {a73:>7.4f} {a80:>7.4f} {a90:>7.4f} {bf:>7.4f} {cf:>7.4f}  {cause}")

# ============================================================
# 6. Plot
# ============================================================
print("\n[6] Plot...")
os.makedirs('reports/figures', exist_ok=True)

plot_data = {'sys_healthy': pairs_c1, 'acc90_cross': pairs_c2,
             'valve90': [], 'valve80': [], 'valve73': []}
for X in X_test:
    b_h = healthy_map.get(X)
    if b_h is None: continue
    h_med = block_info[b_h]['dp_median']
    for v in [90, 80, 73]:
        b_f = get_fault_block(X, v)
        if b_f:
            plot_data[f'valve{v}'].append(abs(block_info[b_f]['dp_median'] - h_med))

fig, ax = plt.subplots(figsize=(10, 6))
categories = ['Systemic\nHealthy', 'acc=90\nCross', 'valve=90', 'valve=80', 'valve=73']
colors_list = ['#4575b4', '#abd9e9', '#fdae61', '#fc8d59', '#d73027']
keys = ['sys_healthy', 'acc90_cross', 'valve90', 'valve80', 'valve73']

for pos, (cat, key, col) in enumerate(zip(categories, keys, colors_list)):
    vals = plot_data[key]
    if vals:
        jitter = np.random.default_rng(42).uniform(-0.18, 0.18, len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, c=col, alpha=0.55, s=28,
                   edgecolors='white', linewidths=0.3, zorder=5)

ax.axhline(y=tau_sys, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'tau_dp = {tau_sys:.4f} bar')

ax.set_xticks(range(5))
ax.set_xticklabels(categories, fontsize=9)
ax.set_ylabel('|Δdp| (bar)', fontsize=12)
ax.set_title(f'ΔP Pairwise Deviation — Test Set (18 X)\n'
             f'Healthy pairs: {len(pairs_c1)} (acc in {{100,115,130}}), '
             f'P95 threshold = {tau_sys:.4f} bar', fontsize=12)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')
fig.tight_layout()
fig.savefig('reports/figures/valve_dp12_systemic.png', dpi=150)
plt.close(fig)
print("    Saved reports/figures/valve_dp12_systemic.png")

print("\n" + "=" * 70)
print("VALVE COMPLETE")
print("=" * 70)
