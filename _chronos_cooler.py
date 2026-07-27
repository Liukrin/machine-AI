import numpy as np
import pandas as pd
import torch
import time, os, warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from chronos import BaseChronosPipeline

# ============================================================
# 0. Load data + model
# ============================================================
print("[0] Loading data...")
data = np.load('data/processed/tensor_31ch_60steps.npz', allow_pickle=True)
tensor   = data['tensor']
stable_mask = data['stable_mask']
labels   = data['labels']
block_id = data['block_id']
ch_names = list(data['channel_names'])

tensor_s = tensor[stable_mask]    # (1449, 31, 60)
labels_s = labels[stable_mask]    # (1449, 4)
block_s  = block_id[stable_mask]  # (1449,)

ts2_idx = ch_names.index('ts2')
print(f"    ts2 channel index: {ts2_idx}")

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
    }

# Build X triples
valve_vals = sorted(set(labels_s[:, 1]))
pump_vals  = sorted(set(labels_s[:, 2]))
accum_vals = sorted(set(labels_s[:, 3]))
all_X = [(v, p, a) for v in valve_vals for p in pump_vals for a in accum_vals]

print(f"[0] Loading Chronos-Bolt-Small...")
pipe = BaseChronosPipeline.from_pretrained(
    'amazon/chronos-bolt-small',
    local_files_only=True,
    device_map='cpu',
    dtype=torch.float32,
)
print(f"    Pipeline class: {type(pipe).__name__}")
print(f"    Module: {type(pipe).__module__}")

# ============================================================
# 1. Build A/B/C windows
# ============================================================
print("\n[1] Building windows...")

# --- Type A: Fault windows ---
contexts_A = []   # list of (context_300, target_cycles, cooler_target, X)
for X in all_X:
    v, p, a = X
    b100 = b20 = b3 = None
    for b, info in block_info.items():
        if info['valve'] == v and info['pump'] == p and info['accum'] == a:
            if info['cooler'] == 100: b100 = b
            elif info['cooler'] == 20: b20 = b
            elif info['cooler'] == 3: b3 = b

    if b100 is None: continue
    ctx_indices = block_info[b100]['indices'][:5]
    ctx_ts2 = tensor_s[ctx_indices, ts2_idx, :].flatten().astype(np.float32)  # (300,)

    if b3 is not None:
        tgt = tensor_s[block_info[b3]['indices'], ts2_idx, :].astype(np.float32)  # (n, 60)
        contexts_A.append((ctx_ts2, tgt, 3, X, ctx_indices, block_info[b3]['indices']))
    if b20 is not None:
        tgt = tensor_s[block_info[b20]['indices'], ts2_idx, :].astype(np.float32)
        contexts_A.append((ctx_ts2, tgt, 20, X, ctx_indices, block_info[b20]['indices']))

# --- Type B: Near healthy (same block, 1-5 -> 6-10) ---
contexts_B = []
for X in all_X:
    v, p, a = X
    for b, info in block_info.items():
        if info['valve'] == v and info['pump'] == p and info['accum'] == a and info['cooler'] == 100:
            if info['n'] >= 10:
                ctx_indices = info['indices'][:5]
                tgt_indices = info['indices'][5:10]
                ctx_ts2 = tensor_s[ctx_indices, ts2_idx, :].flatten().astype(np.float32)
                tgt = tensor_s[tgt_indices, ts2_idx, :].astype(np.float32)
                contexts_B.append((ctx_ts2, tgt, X, ctx_indices, tgt_indices))
            break

# --- Type C: Far healthy (different healthy blocks, max distance) ---
# Build list of cooler=100 blocks sorted by stable_idx
c100_blocks = sorted(
    [(b, info) for b, info in block_info.items() if info['cooler'] == 100],
    key=lambda x: x[1]['start']
)
# Map X -> position index in c100_blocks
X_to_c100_idx = {}
for idx, (b, info) in enumerate(c100_blocks):
    X_to_c100_idx[(info['valve'], info['pump'], info['accum'])] = idx

contexts_C = []
n_c100 = len(c100_blocks)
for X_i in all_X:
    if X_i not in X_to_c100_idx: continue
    i_idx = X_to_c100_idx[X_i]
    # Find farthest healthy block: max |i_idx - j_idx|
    best_j = None
    best_dist = -1
    for X_j in all_X:
        if X_j not in X_to_c100_idx: continue
        j_idx = X_to_c100_idx[X_j]
        dist = abs(i_idx - j_idx)
        if dist > best_dist:
            best_dist = dist
            best_j = j_idx
    if best_j is None: continue

    # context from X_i
    b_i = c100_blocks[i_idx][0]
    ctx_indices = block_info[b_i]['indices'][:5]
    ctx_ts2 = tensor_s[ctx_indices, ts2_idx, :].flatten().astype(np.float32)

    # target from X_j (the farthest block) — first 10 cycles only
    b_j = c100_blocks[best_j][0]
    tgt_indices = block_info[b_j]['indices'][:10]
    tgt = tensor_s[tgt_indices, ts2_idx, :].astype(np.float32)  # (10, 60)
    contexts_C.append((ctx_ts2, tgt, X_i, c100_blocks[best_j][1], ctx_indices, tgt_indices))

print(f"    Type A (fault):    {len(contexts_A)} windows")
print(f"    Type B (near healthy): {len(contexts_B)} windows")
print(f"    Type C (far healthy):  {len(contexts_C)} windows")

# Verify: contexts_A has 96 entries (48 X × 2 fault types), each with 10 target cycles = 960 decisions
# contexts_B: 48 entries × 5 target cycles = 240 decisions
# contexts_C: 48 entries × 10 target cycles = 480 decisions
assert len(contexts_A) == 96, f"Expected 96 fault context batches, got {len(contexts_A)}"
assert len(contexts_B) == 48, f"Expected 48 near healthy context batches, got {len(contexts_B)}"
assert len(contexts_C) == 48, f"Expected 48 far healthy context batches, got {len(contexts_C)}"

# ============================================================
# 2. Detection rule (deterministic Python)
# ============================================================
def detect(q10, q50, q90, actual):
    """
    q10, q90, actual: (60,) arrays
    Returns: 'danger' (>=3 consecutive out), 'warning' (out but <3 consec), 'normal' (no out)
    """
    outside = (actual < q10) | (actual > q90)
    n = len(outside)
    max_consec = 0
    current = 0
    for i in range(n):
        if outside[i]:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0
    if max_consec >= 3:
        return 'danger'
    elif max_consec > 0:
        return 'warning'
    else:
        return 'normal'

# ============================================================
# 3. Run Chronos predictions
# ============================================================
print("\n[2] Running Chronos predictions...")

def chronos_predict_batch(contexts_list):
    """Batch predict for a list of (ctx_300, ...)"""
    if len(contexts_list) == 0:
        return []
    ctx_batch = np.stack([c[0] for c in contexts_list])  # (N, 300)
    with torch.no_grad():
        result = pipe.predict_quantiles(
            torch.tensor(ctx_batch),
            prediction_length=60,
            quantile_levels=[0.1, 0.5, 0.9],
        )
    # result[0]: (N, 60, 3) -> [q10, q50, q90]
    quantiles = result[0].numpy()  # (N, 60, 3)
    q10_all = quantiles[:, :, 0]
    q50_all = quantiles[:, :, 1]
    q90_all = quantiles[:, :, 2]
    return list(zip(q10_all, q50_all, q90_all))

t0 = time.time()
pred_A = chronos_predict_batch(contexts_A)
pred_B = chronos_predict_batch(contexts_B)
pred_C = chronos_predict_batch(contexts_C)
t1 = time.time()
print(f"    Prediction time: {t1-t0:.2f}s")

# ============================================================
# 4. Apply detection + compute metrics
# ============================================================

def evaluate_window_set(contexts_list, preds, label):
    """Apply detection rule and return per-cycle results + aggregate stats."""
    n_total = 0
    n_danger = 0
    n_warning = 0
    n_normal = 0
    cycle_results = []  # list of {verdict, cooler_target, X, ...}
    all_point_in_band = []
    all_point_out_band = []
    all_band_widths = []

    for idx, (ctx_data, pred) in enumerate(zip(contexts_list, preds)):
        ctx_arr = ctx_data[0]   # already q10
        targets = ctx_data[1]   # (m, 60)
        q10_arr = pred[0]   # (60,)
        q50_arr = pred[1]   # (60,)
        q90_arr = pred[2]   # (60,)

        band_width = np.mean(q90_arr - q10_arr)
        all_band_widths.append(band_width)

        m = targets.shape[0]
        for ci in range(m):
            actual = targets[ci]
            verdict = detect(q10_arr, q50_arr, q90_arr, actual)

            # Per-point band coverage
            in_band = (actual >= q10_arr) & (actual <= q90_arr)
            all_point_in_band.append(in_band.sum())
            all_point_out_band.append((~in_band).sum())

            n_total += 1
            if verdict == 'danger':
                n_danger += 1
            elif verdict == 'warning':
                n_warning += 1
            else:
                n_normal += 1

            cycle_results.append({
                'verdict': verdict,
                'ctx_data': ctx_data,
                'q10': q10_arr, 'q50': q50_arr, 'q90': q90_arr,
                'actual': actual,
                'band_width': band_width,
            })

    return {
        'label': label,
        'n_total': n_total,
        'n_danger': n_danger,
        'n_warning': n_warning,
        'n_normal': n_normal,
        'danger_rate': n_danger / n_total if n_total > 0 else 0,
        'warning_rate': n_warning / n_total if n_total > 0 else 0,
        'normal_rate': n_normal / n_total if n_total > 0 else 0,
        'point_coverage': np.sum(all_point_in_band) / (np.sum(all_point_in_band) + np.sum(all_point_out_band)),
        'mean_bandwidth': np.mean(all_band_widths),
        'cycle_results': cycle_results,
        'all_band_widths': all_band_widths,
    }

stats_A = evaluate_window_set(contexts_A, pred_A, 'A (Fault)')
stats_B = evaluate_window_set(contexts_B, pred_B, 'B (Near Healthy)')
stats_C = evaluate_window_set(contexts_C, pred_C, 'C (Far Healthy)')

def print_stats(stats):
    print(f"  {stats['label']}:")
    print(f"    Total cycles: {stats['n_total']}")
    print(f"    Danger:  {stats['n_danger']:>4} ({stats['danger_rate']:.4f})")
    print(f"    Warning: {stats['n_warning']:>4} ({stats['warning_rate']:.4f})")
    print(f"    Normal:  {stats['n_normal']:>4} ({stats['normal_rate']:.4f})")

print("\n[3] Chronos Detection Results:")
print_stats(stats_A)
print_stats(stats_B)
print_stats(stats_C)

# ============================================================
# 5. Naive last-value baseline
# ============================================================
print("\n[4] Naive last-value baseline...")

def naive_predict(context_300):
    """Last cycle of context as median, ±1 std as band."""
    last_cycle = context_300[-60:]  # (60,)
    std_val = np.std(last_cycle)
    q50 = last_cycle
    q10 = last_cycle - std_val
    q90 = last_cycle + std_val
    return q10, q50, q90

def evaluate_naive(contexts_list):
    n_total = 0; n_danger = 0; n_warning = 0; n_normal = 0
    for ctx_data in contexts_list:
        ctx_arr = ctx_data[0]
        targets = ctx_data[1]
        q10, q50, q90 = naive_predict(ctx_arr)
        m = targets.shape[0]
        for ci in range(m):
            actual = targets[ci]
            verdict = detect(q10, q50, q90, actual)
            n_total += 1
            if verdict == 'danger': n_danger += 1
            elif verdict == 'warning': n_warning += 1
            else: n_normal += 1
    return {
        'n_total': n_total, 'n_danger': n_danger, 'n_warning': n_warning, 'n_normal': n_normal,
        'danger_rate': n_danger/n_total if n_total>0 else 0,
        'warning_rate': n_warning/n_total if n_total>0 else 0,
        'normal_rate': n_normal/n_total if n_total>0 else 0,
    }

naive_A = evaluate_naive(contexts_A)
naive_B = evaluate_naive(contexts_B)
naive_C = evaluate_naive(contexts_C)

print(f"  A (Fault):    danger={naive_A['danger_rate']:.4f}, warning={naive_A['warning_rate']:.4f}")
print(f"  B (Near):     danger={naive_B['danger_rate']:.4f}, warning={naive_B['warning_rate']:.4f}")
print(f"  C (Far):      danger={naive_C['danger_rate']:.4f}, warning={naive_C['warning_rate']:.4f}")

# ============================================================
# 6. Three diagnostic quantities
# ============================================================
print("\n[5] DIAGNOSTIC QUANTITIES")

# D1: Empirical band coverage (type B)
print(f"  D1: Point coverage (Type B): {stats_B['point_coverage']:.4f} "
      f"(ground truth points within [q10,q90])")

# D2: Mean band width
print(f"  D2: Mean band width: {stats_B['mean_bandwidth']:.4f} °C")
print(f"      Ratio to 1.058°C (healthy block drift range): "
      f"{stats_B['mean_bandwidth'] / 1.0582:.4f}")

# D3: Type C target-context ts2 mean difference
ts2_diffs_C = []
for ctx_data in contexts_C:
    ctx_300 = ctx_data[0]
    ctx_last_60_mean = ctx_300[-60:].mean()
    targets = ctx_data[1]  # (m, 60)
    tgt_mean = targets.mean()
    ts2_diffs_C.append(abs(tgt_mean - ctx_last_60_mean))

ts2_diffs_C = np.array(ts2_diffs_C)
print(f"  D3: Type C |target_mean - context_tail_mean| (abs):")
print(f"      min={ts2_diffs_C.min():.4f}, mean={ts2_diffs_C.mean():.4f}, max={ts2_diffs_C.max():.4f} °C")
print(f"      Ratio to mean band width: {ts2_diffs_C.mean() / stats_B['mean_bandwidth']:.4f}")

# ============================================================
# 7. Plots
# ============================================================
print("\n[6] Generating plots...")
os.makedirs('reports/figures', exist_ok=True)

def plot_example(ctx_data, pred, stats_label, verdict, cooler_target, X, filename):
    """Plot one example: context tail 60pts, q10-q90 band, q50, actual, outliers."""
    ctx_300 = ctx_data[0]
    targets = ctx_data[1]  # (m, 60)
    q10 = pred[0]
    q50 = pred[1]
    q90 = pred[2]
    actual = targets[0]  # first target cycle

    fig, ax = plt.subplots(figsize=(12, 5.5))
    time_full = np.arange(-60, 60)

    # Context tail (last 60 points)
    ctx_tail = ctx_300[-60:]
    ax.plot(np.arange(-60, 0), ctx_tail, color='gray', linewidth=1.5, label='Context (tail 60 pts)', alpha=0.8)

    # Quantile band
    pred_time = np.arange(60)
    ax.fill_between(pred_time, q10, q90, alpha=0.25, color='#4575b4', label='[q10, q90] band')
    ax.plot(pred_time, q50, color='#4575b4', linewidth=1.8, label='q50 median')

    # Actual
    ax.plot(pred_time, actual, color='#d73027', linewidth=1.5, label='Actual target', zorder=5)

    # Mark outside-band points
    outside = (actual < q10) | (actual > q90)
    if outside.any():
        ax.scatter(pred_time[outside], actual[outside], color='red', s=25, zorder=10,
                   edgecolors='darkred', linewidths=0.5, label=f'Outside band ({outside.sum()} pts)')

    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

    v, p, a = X
    ax.set_xlabel('Time (seconds relative to prediction start)', fontsize=11)
    ax.set_ylabel('TS2 Temperature (°C)', fontsize=11)
    ax.set_title(f'{stats_label}  |  X=(valve={v}, pump_leak={p}, accum={a}bar)  |  '
                 f'cooler_target={cooler_target}  |  verdict={verdict}', fontsize=12)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"    Saved {filename}")

# Pick examples
# A: first danger verdict in fault windows
ex_A = None
for i, cr in enumerate(stats_A['cycle_results']):
    if cr['verdict'] == 'danger':
        w = contexts_A[i]
        ex_A = (w, pred_A[i], 'A (Fault)', cr['verdict'], w[2], w[3], 'reports/figures/chronos_A_fault_example.png')
        break
if ex_A:
    plot_example(*ex_A)
else:
    # Fallback: use first A window
    w = contexts_A[0]
    cr = stats_A['cycle_results'][0]
    plot_example(w, pred_A[0], 'A (Fault)', cr['verdict'], w[2], w[3],
                 'reports/figures/chronos_A_fault_example.png')

# B: first example
w = contexts_B[0]
cr = stats_B['cycle_results'][0]
plot_example(w, pred_B[0], 'B (Near Healthy)', cr['verdict'], 100, w[2],
             'reports/figures/chronos_B_near_healthy_example.png')

# C: first example
w = contexts_C[0]
cr = stats_C['cycle_results'][0]
# Determine cooler_target: it's the cooler value of the target block
c_target_info = w[3]
cooler_target_C = c_target_info['cooler']  # always 100
plot_example(w, pred_C[0], 'C (Far Healthy)', cr['verdict'], cooler_target_C, w[2],
             'reports/figures/chronos_C_far_healthy_example.png')

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
