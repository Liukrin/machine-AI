import numpy as np
import pandas as pd
import os, gc, sys

DATA = 'dataset'
N_CYCLES = 2205

# ============================================================
# 0. Load profile
# ============================================================
profile = pd.read_csv(os.path.join(DATA, 'profile.txt'), sep='\t', header=None)
profile.columns = ['cooler','valve','pump_leakage','accumulator','stable_flag']
print(f"profile shape: {profile.shape}")

stable_mask = (profile['stable_flag'].values == 0).astype(bool)
labels = profile[['cooler','valve','pump_leakage','accumulator']].values.astype(np.int32)
print(f"stable_mask sum: {stable_mask.sum()}")
print(f"labels shape: {labels.shape}")

# ============================================================
# 1. Build tensor (2205, 31, 60) float32
# ============================================================
channel_names = []
channel_blocks = []  # list of (2205, 60) float32 arrays

# --- A. 100Hz sensors: PS1-6, EPS1 -> mean/std/ptp ---
for sensor in ['PS1','PS2','PS3','PS4','PS5','PS6','EPS1']:
    print(f"Reading {sensor}...", end=' ', flush=True)
    raw = pd.read_csv(os.path.join(DATA, f'{sensor}.txt'), sep='\t', header=None).values.astype(np.float64)
    print(f"raw shape={raw.shape} dtype={raw.dtype}", end=' ', flush=True)
    # reshape to (2205, 60, 100)
    reshaped = raw.reshape(N_CYCLES, 60, 100)
    del raw; gc.collect()

    ch_mean = reshaped.mean(axis=2).astype(np.float32)
    ch_std  = reshaped.std(axis=2).astype(np.float32)
    ch_ptp  = (reshaped.max(axis=2) - reshaped.min(axis=2)).astype(np.float32)
    del reshaped; gc.collect()

    channel_blocks.extend([ch_mean, ch_std, ch_ptp])
    sname = sensor.lower()
    channel_names.extend([f'{sname}_mean', f'{sname}_std', f'{sname}_ptp'])
    print(f"-> 3 channels [{sname}_mean/_std/_ptp] shape={ch_mean.shape}")

# --- B. 10Hz sensors: FS1, FS2 -> mean/std ---
for sensor in ['FS1','FS2']:
    print(f"Reading {sensor}...", end=' ', flush=True)
    raw = pd.read_csv(os.path.join(DATA, f'{sensor}.txt'), sep='\t', header=None).values.astype(np.float64)
    print(f"raw shape={raw.shape}", end=' ', flush=True)
    # reshape to (2205, 60, 10)
    reshaped = raw.reshape(N_CYCLES, 60, 10)
    del raw; gc.collect()

    ch_mean = reshaped.mean(axis=2).astype(np.float32)
    ch_std  = reshaped.std(axis=2).astype(np.float32)
    del reshaped; gc.collect()

    channel_blocks.extend([ch_mean, ch_std])
    sname = sensor.lower()
    channel_names.extend([f'{sname}_mean', f'{sname}_std'])
    print(f"-> 2 channels [{sname}_mean/_std] shape={ch_mean.shape}")

# --- C. 1Hz sensors: TS1-4, VS1, SE -> keep as-is ---
for sensor in ['TS1','TS2','TS3','TS4','VS1','SE']:
    print(f"Reading {sensor}...", end=' ', flush=True)
    raw = pd.read_csv(os.path.join(DATA, f'{sensor}.txt'), sep='\t', header=None).values.astype(np.float32)
    print(f"shape={raw.shape}")
    channel_blocks.append(raw)
    channel_names.append(sensor.lower())

# --- CE, CP: skip ---
print(f"\nTotal channels collected: {len(channel_blocks)}")
print(f"Channel names count: {len(channel_names)}")

# Stack into final tensor
tensor = np.stack(channel_blocks, axis=1).astype(np.float32)  # (2205, 31, 60)
del channel_blocks; gc.collect()
print(f"Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")

# ============================================================
# 2. Build combo_id and block_id
# ============================================================
# combo_id: map each unique 4-tuple to int
# Encode 4-tuple as single int: c0*1000000 + c1*10000 + c2*100 + c3
# (values are small so no overflow risk)
combo_key = (labels[:, 0].astype(np.int64) * 1000000 +
             labels[:, 1].astype(np.int64) * 10000 +
             labels[:, 2].astype(np.int64) * 100 +
             labels[:, 3].astype(np.int64))
unique_keys, combo_id = np.unique(combo_key, return_inverse=True)
combo_id = combo_id.astype(np.int32)
print(f"combo_id shape: {combo_id.shape}, dtype: {combo_id.dtype}")
print(f"Unique combos (all 2205): {len(unique_keys)}")
print(f"Unique combos (stable only): {np.unique(combo_id[stable_mask]).shape[0]}")

# block_id: only on stable rows, adjacent same-label runs get same block
block_id = np.full(N_CYCLES, -1, dtype=np.int32)
stable_indices = np.where(stable_mask)[0]  # indices in full 0..2204 space
stable_combo = combo_id[stable_mask]        # combo_id only for stable rows (1D)
print(f"stable_combo shape: {stable_combo.shape}")

if len(stable_indices) > 0:
    current_block = 0
    block_id[stable_indices[0]] = current_block
    for i in range(1, len(stable_indices)):
        if stable_combo[i] != stable_combo[i-1]:
            current_block += 1
        block_id[stable_indices[i]] = current_block
    n_blocks = current_block + 1
else:
    n_blocks = 0
print(f"Number of blocks (stable): {n_blocks}")

# ============================================================
# 3. Save
# ============================================================
out_path = 'data/processed/tensor_31ch_60steps.npz'
os.makedirs('data/processed', exist_ok=True)
np.savez_compressed(
    out_path,
    tensor=tensor,
    channel_names=np.array(channel_names),
    stable_mask=stable_mask,
    labels=labels,
    combo_id=combo_id,
    block_id=block_id,
)
print(f"\nSaved to {out_path}")

# ============================================================
# 4. Reload and verify
# ============================================================
print("\n--- Reload verification ---")
data = np.load(out_path, allow_pickle=True)
tensor_r = data['tensor']
channel_names_r = data['channel_names']
stable_mask_r = data['stable_mask']
labels_r = data['labels']
combo_id_r = data['combo_id']
block_id_r = data['block_id']
print(f"tensor shape: {tensor_r.shape}, dtype: {tensor_r.dtype}")
print(f"stable_mask sum: {stable_mask_r.sum()}")
print(f"All arrays loaded OK.")

# ============================================================
# 5. SELF-CHECKS
# ============================================================
print("\n" + "="*70)
print("SELF-CHECKS")
print("="*70)

# Check 1: shape & dtype
print("\n[1] tensor.shape & dtype")
print(f"    tensor.shape = {tensor_r.shape}")
print(f"    Expected (2205, 31, 60): {tensor_r.shape == (2205, 31, 60)}")
print(f"    dtype = {tensor_r.dtype}")
print(f"    Expected float32: {tensor_r.dtype == np.float32}")

# Check 2: NaN/Inf
print("\n[2] NaN & Inf scan")
nan_count = np.isnan(tensor_r).sum()
inf_count = np.isinf(tensor_r).sum()
print(f"    NaN total: {nan_count}")
print(f"    Inf total: {inf_count}")
if nan_count > 0:
    nan_locs = np.argwhere(np.isnan(tensor_r))
    for loc in nan_locs[:20]:
        cycle, ch, step = loc
        nm = channel_names_r[ch]
        print(f"    NaN at cycle={cycle}, channel={nm} (idx={ch}), step={step}")
if inf_count > 0:
    inf_locs = np.argwhere(np.isinf(tensor_r))
    for loc in inf_locs[:20]:
        cycle, ch, step = loc
        nm = channel_names_r[ch]
        print(f"    Inf at cycle={cycle}, channel={nm} (idx={ch}), step={step}")
print(f"    Both zero: {nan_count == 0 and inf_count == 0}")

# Check 3: stable_mask sum
print("\n[3] stable_mask.sum()")
print(f"    stable_mask.sum() = {stable_mask_r.sum()}")
print(f"    Expected 1449: {stable_mask_r.sum() == 1449}")

# Check 4: combo_id unique count
print("\n[4] combo_id unique count")
n_unique_all = len(np.unique(combo_id_r))
n_unique_stable = len(np.unique(combo_id_r[stable_mask_r]))
print(f"    All 2205 rows: {n_unique_all} unique combos")
print(f"    Stable rows only: {n_unique_stable} unique combos")

# Check 5: block_id stats (stable only)
print("\n[5] block_id stats (stable rows)")
stable_block_ids = block_id_r[stable_mask_r]
block_vc = pd.Series(stable_block_ids).value_counts().sort_index()
print(f"    Total blocks: {len(block_vc)}")
print(f"    Block length: min={block_vc.min()}, max={block_vc.max()}, "
      f"mean={block_vc.mean():.1f}, median={block_vc.median():.1f}")
print(f"    All block lengths: {list(block_vc.values)}")
len_dist = pd.Series(block_vc.values).value_counts().sort_index()
print(f"    Length distribution (length:count): {dict(len_dist)}")

# Check 6: Structure verification in stable_idx space
print("\n[6] Structure verification (stable_idx space)")
stable_labels = labels_r[stable_mask_r]  # (1449, 4)
stable_idx_cooler = stable_labels[:, 0]
stable_idx_accum  = stable_labels[:, 3]
stable_idx_pump   = stable_labels[:, 2]

# 6a: three cooler segment lengths
segments_cooler = []
current_val = stable_idx_cooler[0]
current_start = 0
for i in range(1, len(stable_idx_cooler)):
    if stable_idx_cooler[i] != current_val:
        segments_cooler.append((current_val, current_start, i-1, i - current_start))
        current_val = stable_idx_cooler[i]
        current_start = i
segments_cooler.append((current_val, current_start, len(stable_idx_cooler)-1,
                         len(stable_idx_cooler) - current_start))

print(f"    6a. Cooler segments in stable space:")
cooler_lengths = []
for val, start, end, length in segments_cooler:
    print(f"        cooler={val}: stable_idx [{start}, {end}], length={length}")
    cooler_lengths.append(length)
print(f"        Expected 480/480/489: {cooler_lengths == [480, 480, 489]}")

# 6b: cooler==100 segment bounds
for val, start, end, length in segments_cooler:
    if val == 100:
        print(f"    6b. cooler==100 stable_idx range: [{start}, {end}]")
        print(f"        Expected [960, 1448]: {[start, end] == [960, 1448]}")

# 6c: accumulator segments within one cooler segment
# Use the second cooler segment (cooler=20, 480 rows) for clean 120 split
cooler_val, c_start, c_end, c_len = segments_cooler[1]
subset_accum = stable_idx_accum[c_start:c_end+1]
segments_accum = []
current_val = subset_accum[0]
current_start = 0
for i in range(1, len(subset_accum)):
    if subset_accum[i] != current_val:
        segments_accum.append((current_val, c_start + current_start,
                               c_start + i - 1, i - current_start))
        current_val = subset_accum[i]
        current_start = i
segments_accum.append((current_val, c_start + current_start,
                       c_start + len(subset_accum) - 1,
                       len(subset_accum) - current_start))

print(f"    6c. Accumulator segments within cooler={cooler_val} "
      f"(stable_idx {c_start}-{c_end}):")
accum_lengths = []
for val, start, end, length in segments_accum:
    print(f"        accum={val}: stable_idx [{start}, {end}], length={length}")
    accum_lengths.append(length)
print(f"        Expected all 120: {all(l == 120 for l in accum_lengths)}")

# 6d: pump segments within one accumulator segment
acc_val, a_start, a_end, a_len = segments_accum[0]
subset_pump = stable_idx_pump[a_start:a_end+1]
segments_pump = []
current_val = subset_pump[0]
current_start = 0
for i in range(1, len(subset_pump)):
    if subset_pump[i] != current_val:
        segments_pump.append((current_val, a_start + current_start,
                              a_start + i - 1, i - current_start))
        current_val = subset_pump[i]
        current_start = i
segments_pump.append((current_val, a_start + current_start,
                       a_start + len(subset_pump) - 1,
                       len(subset_pump) - current_start))

print(f"    6d. Pump segments within accum={acc_val} "
      f"(stable_idx {a_start}-{a_end}):")
pump_lengths = []
for val, start, end, length in segments_pump:
    print(f"        pump={val}: stable_idx [{start}, {end}], length={length}")
    pump_lengths.append(length)
print(f"        Expected all 40: {all(l == 40 for l in pump_lengths)}")

# Check 7: Physical sanity -- ts2 by cooler group
print("\n[7] Physical sanity: ts2 channel mean by cooler group (stable only)")
ts2_idx = list(channel_names_r).index('ts2')
ts2_data = tensor_r[stable_mask_r, ts2_idx, :]  # (1449, 60)
ts2_cycle_mean = ts2_data.mean(axis=1)  # (1449,)

for c_val in [3, 20, 100]:
    mask_c = stable_labels[:, 0] == c_val
    m = ts2_cycle_mean[mask_c].mean()
    s = ts2_cycle_mean[mask_c].std()
    print(f"    cooler={c_val}: mean={m:.3f} C, std={s:.3f} C (n={mask_c.sum()})")

means_list = []
for c_val in [3, 20, 100]:
    mask_c = stable_labels[:, 0] == c_val
    means_list.append(ts2_cycle_mean[mask_c].mean())
sep = means_list[0] - means_list[2]
print(f"    Separation (cooler 3 vs 100): {sep:.3f} C")
print(f"    Well-separated (>1 C): {abs(sep) > 1.0}")

# Check 8: channel names
print("\n[8] Channel names (31 total):")
for i, name in enumerate(channel_names_r):
    print(f"    [{i:2d}] {name}")

print("\n" + "="*70)
print("ALL CHECKS COMPLETE")
print("="*70)
