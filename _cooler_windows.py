import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi'] = 120

# ============================================================
# 0. Load
# ============================================================
data = np.load('data/processed/tensor_31ch_60steps.npz', allow_pickle=True)
tensor   = data['tensor']         # (2205, 31, 60)
stable_mask = data['stable_mask']
labels   = data['labels']         # (2205, 4)
block_id = data['block_id']       # (2205,)
ch_names = list(data['channel_names'])

tensor_s = tensor[stable_mask]    # (1449, 31, 60)
labels_s = labels[stable_mask]    # (1449, 4)
block_s  = block_id[stable_mask]  # (1449,)

N_STABLE = 1449

# ============================================================
# 1. ts2 channel index
# ============================================================
ts2_idx = ch_names.index('ts2')
print(f"[1] ts2 channel index: {ts2_idx}")
print(f"    channel_names[{ts2_idx}] = '{ch_names[ts2_idx]}'")

# ============================================================
# 2. Build stable_idx → block mapping
# ============================================================
# For each block_id (in stable space), get:
#   - stable_idx range (start, end) inclusive
#   - the 4-label combo
#   - number of cycles

unique_blocks = np.unique(block_s)
block_info = {}  # block_id -> {start, end, n, cooler, valve, pump, accum}

for b in unique_blocks:
    mask = block_s == b
    indices = np.where(mask)[0]  # stable_idx positions
    block_info[b] = {
        'start': indices[0],
        'end': indices[-1],
        'n': len(indices),
        'cooler': int(labels_s[indices[0], 0]),
        'valve': int(labels_s[indices[0], 1]),
        'pump': int(labels_s[indices[0], 2]),
        'accum': int(labels_s[indices[0], 3]),
        'indices': indices,  # all stable_idx positions
    }

n_blocks = len(block_info)
print(f"\nTotal blocks (stable): {n_blocks}")

# ============================================================
# 3. Window construction
# ============================================================

# X = (valve, pump_leakage, accumulator) triple
valve_vals = sorted(set(labels_s[:, 1]))   # [73, 80, 90, 100]
pump_vals  = sorted(set(labels_s[:, 2]))   # [0, 1, 2]
accum_vals = sorted(set(labels_s[:, 3]))   # [90, 100, 115, 130]

all_X = [(v, p, a) for v in valve_vals for p in pump_vals for a in accum_vals]
print(f"Total X triples: {len(all_X)} (expected 4x3x4=48)")

# Fault windows: context=(X,cooler=100), target=(X,cooler=3) or (X,cooler=20)
fault_windows = []  # each: {X, cooler_target, context_cycles, target_cycles, context_block, target_block}
pure_healthy_windows = []  # each: {X, context_cycles, target_cycle, block}

missing_X = []

for X in all_X:
    v, p, a = X
    # Find the three blocks
    b100 = None; b20 = None; b3 = None
    for b, info in block_info.items():
        if info['valve'] == v and info['pump'] == p and info['accum'] == a:
            if info['cooler'] == 100:
                b100 = b
            elif info['cooler'] == 20:
                b20 = b
            elif info['cooler'] == 3:
                b3 = b

    if b100 is None:
        missing_X.append((X, 'missing cooler=100'))
    if b20 is None:
        missing_X.append((X, 'missing cooler=20'))
    if b3 is None:
        missing_X.append((X, 'missing cooler=3'))

    if b100 is None:
        continue  # can't build windows without context

    # Context: first 5 cycles of (X, cooler=100) block
    ctx_indices = block_info[b100]['indices'][:5]
    ctx_ts2 = tensor_s[ctx_indices, ts2_idx, :]  # (5, 60)
    ctx_flat = ctx_ts2.flatten()  # (300,)

    # Fault targets
    if b3 is not None:
        tgt_indices_3 = block_info[b3]['indices']
        tgt_ts2_3 = tensor_s[tgt_indices_3, ts2_idx, :]  # (n, 60)
        fault_windows.append({
            'X': X,
            'cooler_target': 3,
            'context_stable_idx': ctx_indices,
            'target_stable_idx': tgt_indices_3,
            'context_ts2': ctx_flat,
            'target_ts2': tgt_ts2_3,
            'context_block': b100,
            'target_block': b3,
        })

    if b20 is not None:
        tgt_indices_20 = block_info[b20]['indices']
        tgt_ts2_20 = tensor_s[tgt_indices_20, ts2_idx, :]  # (n, 60)
        fault_windows.append({
            'X': X,
            'cooler_target': 20,
            'context_stable_idx': ctx_indices,
            'target_stable_idx': tgt_indices_20,
            'context_ts2': ctx_flat,
            'target_ts2': tgt_ts2_20,
            'context_block': b100,
            'target_block': b20,
        })

    # Pure healthy: first 5 → 6th cycle within same (X, cooler=100) block
    if block_info[b100]['n'] >= 6:
        ctx_indices_ph = block_info[b100]['indices'][:5]
        tgt_idx_ph = block_info[b100]['indices'][5]
        ctx_ts2_ph = tensor_s[ctx_indices_ph, ts2_idx, :].flatten()
        tgt_ts2_ph = tensor_s[tgt_idx_ph, ts2_idx, :]  # (60,)
        pure_healthy_windows.append({
            'X': X,
            'context_stable_idx': ctx_indices_ph,
            'target_stable_idx': np.array([tgt_idx_ph]),
            'context_ts2': ctx_ts2_ph,
            'target_ts2': tgt_ts2_ph.reshape(1, 60),
            'block': b100,
        })
    else:
        print(f"    WARNING: X={X} cooler=100 block has only {block_info[b100]['n']} cycles, "
              f"cannot build pure healthy window (need >=6)")

# ============================================================
# 4. Print required stats
# ============================================================
print(f"\n[2] Fault windows: {len(fault_windows)}")
print(f"    Pure healthy windows: {len(pure_healthy_windows)}")

# Deduplicate unique X in fault windows
cooler3_X = set(tuple(w['X']) for w in fault_windows if w['cooler_target'] == 3)
cooler20_X = set(tuple(w['X']) for w in fault_windows if w['cooler_target'] == 20)
healthy_X = set(tuple(w['X']) for w in pure_healthy_windows)

print(f"    X with cooler=3 target: {len(cooler3_X)}/48")
print(f"    X with cooler=20 target: {len(cooler20_X)}/48")
print(f"    X with pure healthy: {len(healthy_X)}/48")

print(f"\n[3] All 48 X triples complete?")
if missing_X:
    print(f"    NO. {len(missing_X)} missing entries:")
    for miss in missing_X:
        print(f"      X={miss[0]}, reason={miss[1]}")
else:
    # Check if fault windows cover all 48
    missing_cooler3 = set(all_X) - cooler3_X
    missing_cooler20 = set(all_X) - cooler20_X
    missing_healthy = set(all_X) - healthy_X
    all_ok = True
    if missing_cooler3:
        print(f"    cooler=3 missing for X: {sorted(missing_cooler3)}")
        all_ok = False
    if missing_cooler20:
        print(f"    cooler=20 missing for X: {sorted(missing_cooler20)}")
        all_ok = False
    if missing_healthy:
        print(f"    pure healthy missing for X: {sorted(missing_healthy)}")
        all_ok = False
    if all_ok:
        print(f"    YES. All 48 X triples have cooler 100, 20, and 3 blocks. Pure healthy also complete.")

# [4] stable_idx gap between context block start and target block start
print(f"\n[4] stable_idx gap (context block start → target block start):")
gaps = []
for w in fault_windows:
    ctx_start = w['context_stable_idx'][0]
    tgt_start = w['target_stable_idx'][0]
    gap = tgt_start - ctx_start
    gaps.append(gap)

gaps = np.array(gaps)
print(f"    min={gaps.min()}, mean={gaps.mean():.1f}, max={gaps.max()}")
# Check if stable
print(f"    All gaps positive? {(gaps > 0).all()} (target always after context)")
# The gap is large because cooler blocks are far apart in time
# cooler segments: [0,479]=3, [480,959]=20, [960,1448]=100
# So gap from cooler=100 block to cooler=3/20 block is always negative if target before context
# Actually: context=(X,cooler=100) is in [960,1448], target=(X,cooler=3) is in [0,479]
# So gap = target_start - ctx_start is negative (target earlier!)
neg_count = (gaps < 0).sum()
pos_count = (gaps > 0).sum()
print(f"    Negative gaps: {neg_count}/{len(gaps)} (target before context in time)")
print(f"    Positive gaps: {pos_count}/{len(gaps)} (target after context in time)")
print(f"    Gap is NOT stable — cooler=100 is last segment, cooler=3/20 are earlier segments.")
print(f"    |gap| stats: min={abs(gaps).min():.0f}, mean={abs(gaps).mean():.0f}, max={abs(gaps).max():.0f}")

# [5] 19-cycle anomaly block
print(f"\n[5] 19-cycle block check:")
for b, info in block_info.items():
    if info['n'] == 19:
        X_19 = (info['valve'], info['pump'], info['accum'])
        print(f"    Block {b}: X={X_19}, cooler={info['cooler']}, n={info['n']}")
        print(f"    stable_idx: [{info['start']}, {info['end']}]")
        # Check pure healthy: should use first 5 → 6th
        ph_for_this = [w for w in pure_healthy_windows if w['X'] == X_19]
        if ph_for_this:
            w = ph_for_this[0]
            print(f"    Pure healthy: context cycles {list(w['context_stable_idx'])}, "
                  f"target cycle {w['target_stable_idx'][0]}")
            print(f"    First 5 used = {list(w['context_stable_idx'])}, "
                  f"6th = {w['target_stable_idx'][0]}")
        # Check fault windows for this X
        fw_for_this = [w for w in fault_windows if w['X'] == X_19]
        for w in fw_for_this:
            print(f"    Fault cooler={w['cooler_target']}: "
                  f"context first 5 = {list(w['context_stable_idx'])}, "
                  f"target cycles = {list(w['target_stable_idx'])}")
        break
else:
    print("    No 19-cycle block found!")

# Check: first 5 cycles from the 19-cycle block are used as context
for b, info in block_info.items():
    if info['n'] == 19 and info['cooler'] == 100:
        X_19 = (info['valve'], info['pump'], info['accum'])
        print(f"    cooler=100 block with 19 cycles at X={X_19}")
        print(f"    Taking first 5 stable_idx: {list(info['indices'][:5])}")
        print(f"    Remaining 14 cycles (indices 5..18) are NOT used as context.")
        break

# ============================================================
# 6. Drift check
# ============================================================
print(f"\n[6] DRIFT CHECK")
print(f"    a. ts2 cycle means for 48 (X, cooler=100) blocks, in stable_idx order:")

# Collect all cooler=100 blocks with their ts2 mean
cooler100_blocks = []
for b, info in block_info.items():
    if info['cooler'] == 100:
        indices = info['indices']
        ts2_block = tensor_s[indices, ts2_idx, :]  # (n_cycles, 60)
        cycle_mean = ts2_block.mean(axis=1).mean()  # mean over cycles and time
        cooler100_blocks.append({
            'block': b,
            'X': (info['valve'], info['pump'], info['accum']),
            'stable_start': info['start'],
            'n': info['n'],
            'ts2_mean': cycle_mean,
        })

cooler100_blocks.sort(key=lambda x: x['stable_start'])
for i, cb in enumerate(cooler100_blocks):
    print(f"      stable_idx={cb['stable_start']:>5}  X=(v={cb['X'][0]},"
          f"p={cb['X'][1]},a={cb['X'][2]:>3})  n={cb['n']:>2}  "
          f"ts2_mean={cb['ts2_mean']:.4f}")

ts2_means_arr = np.array([cb['ts2_mean'] for cb in cooler100_blocks])
print(f"\n    ts2 mean across 48 healthy blocks:")
print(f"      min={ts2_means_arr.min():.4f}, max={ts2_means_arr.max():.4f}, "
      f"range={ts2_means_arr.max() - ts2_means_arr.min():.4f}")

# Cooler=100 vs cooler=3 temperature difference
cooler100_all = tensor_s[labels_s[:, 0] == 100, ts2_idx, :]
cooler3_all   = tensor_s[labels_s[:, 0] == 3, ts2_idx, :]
temp_100 = cooler100_all.mean()
temp_3   = cooler3_all.mean()
delta_temp = temp_3 - temp_100  # cooler=3 is hotter

print(f"\n    b. Comparison:")
print(f"      cooler=100 ts2 mean (all cycles): {temp_100:.4f}")
print(f"      cooler=3   ts2 mean (all cycles): {temp_3:.4f}")
print(f"      delta (3 - 100): {delta_temp:.4f} °C")
print(f"      Healthy-block drift range: {ts2_means_arr.max() - ts2_means_arr.min():.4f} °C")
print(f"      Ratio (drift / delta): "
      f"{(ts2_means_arr.max() - ts2_means_arr.min()) / delta_temp:.4f}")
if (ts2_means_arr.max() - ts2_means_arr.min()) > delta_temp:
    print(f"      => Drift range EXCEEDS cooler 3-vs-100 delta. Large time gap effect.")
else:
    print(f"      => Drift range is SMALLER than cooler 3-vs-100 delta.")
print(f"    c. Numbers reported as-is. No adjustment made.")

# ============================================================
# 7. Plot: same X, cooler=100 vs cooler=3, and cooler=100 vs cooler=20
# ============================================================
print(f"\n[7] Generating plots...")
os.makedirs('reports/figures', exist_ok=True)

# Pick a representative X - use one where all three cooler blocks exist
# Choose the X from the first complete triple
plot_X = None
for X in all_X:
    if X in cooler3_X and X in cooler20_X and X in healthy_X:
        plot_X = X
        break

if plot_X is None:
    print("ERROR: No complete X found for plotting!")
else:
    v, p, a = plot_X
    print(f"    Plot X: valve={v}, pump_leakage={p}, accumulator={a}")

    # Get ts2 data
    b100_plot = None; b20_plot = None; b3_plot = None
    for b, info in block_info.items():
        if (info['valve'] == v and info['pump'] == p and info['accum'] == a):
            if info['cooler'] == 100: b100_plot = b
            elif info['cooler'] == 20: b20_plot = b
            elif info['cooler'] == 3: b3_plot = b

    # Extract up to 10 cycles each
    n_show = 10
    idx100 = block_info[b100_plot]['indices'][:n_show]
    idx3   = block_info[b3_plot]['indices'][:n_show]
    idx20  = block_info[b20_plot]['indices'][:n_show]

    ts2_100 = tensor_s[idx100, ts2_idx, :]  # (n_show, 60)
    ts2_3   = tensor_s[idx3, ts2_idx, :]
    ts2_20  = tensor_s[idx20, ts2_idx, :]

    time_axis = np.arange(60)

    # --- Plot 1: cooler=100 vs cooler=3 ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(len(ts2_100)):
        ax.plot(time_axis, ts2_100[i], color='#4575b4', alpha=0.6, linewidth=1.0)
    for i in range(len(ts2_3)):
        ax.plot(time_axis, ts2_3[i], color='#d73027', alpha=0.6, linewidth=1.0)
    # Legend dummy
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#4575b4', lw=2, label='cooler=100 (healthy)'),
        Line2D([0], [0], color='#d73027', lw=2, label='cooler=3 (fault)'),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('TS2 Temperature (°C)', fontsize=12)
    ax.set_title(f'TS2: cooler=100 vs cooler=3  |  '
                 f'valve={v}, pump_leak={p}, accum={a}bar', fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('reports/figures/cooler_100_vs_3_ts2.png', dpi=150)
    plt.close(fig)
    print(f"    Saved reports/figures/cooler_100_vs_3_ts2.png")

    # --- Plot 2: cooler=100 vs cooler=20 ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(len(ts2_100)):
        ax.plot(time_axis, ts2_100[i], color='#4575b4', alpha=0.6, linewidth=1.0)
    for i in range(len(ts2_20)):
        ax.plot(time_axis, ts2_20[i], color='#fc8d59', alpha=0.6, linewidth=1.0)
    legend_elements2 = [
        Line2D([0], [0], color='#4575b4', lw=2, label='cooler=100 (healthy)'),
        Line2D([0], [0], color='#fc8d59', lw=2, label='cooler=20 (reduced)'),
    ]
    ax.legend(handles=legend_elements2, fontsize=10)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('TS2 Temperature (°C)', fontsize=12)
    ax.set_title(f'TS2: cooler=100 vs cooler=20  |  '
                 f'valve={v}, pump_leak={p}, accum={a}bar', fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig('reports/figures/cooler_100_vs_20_ts2.png', dpi=150)
    plt.close(fig)
    print(f"    Saved reports/figures/cooler_100_vs_20_ts2.png")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
