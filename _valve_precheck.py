import numpy as np
import pandas as pd
import os, sys, warnings, time
warnings.filterwarnings('ignore')

# ============================================================
# 0. Load
# ============================================================
print("[0] Loading data...")
data = np.load('data/processed/tensor_31ch_60steps.npz', allow_pickle=True)
tensor   = data['tensor']         # (2205, 31, 60) float32
stable_mask = data['stable_mask']
labels   = data['labels']
block_id = data['block_id']
ch_names = list(data['channel_names'])

tensor_s = tensor[stable_mask]    # (1449, 31, 60)
labels_s = labels[stable_mask]    # (1449, 4)
block_s  = block_id[stable_mask]  # (1449,)

# Build dp12 = ps1_mean - ps2_mean
idx_ps1 = ch_names.index('ps1_mean')
idx_ps2 = ch_names.index('ps2_mean')
dp12 = tensor_s[:, idx_ps1, :] - tensor_s[:, idx_ps2, :]  # (1449, 60)
# Append to tensor_s
tensor_s_all = np.concatenate([tensor_s, dp12[:, np.newaxis, :]], axis=1).astype(np.float32)
ch_names_all = ch_names + ['dp12']
print(f"    Channels: {len(ch_names_all)} (31 original + dp12)")
print(f"    Tensor shape: {tensor_s_all.shape}")

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

# X triples and ordering
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

X_calib = X_ordered[:18]
X_test  = X_ordered[18:]
print(f"    Calibration X: {len(X_calib)}, Test X: {len(X_test)}, Overlap: {len(set(X_calib) & set(X_test))}")

# ============================================================
# 1. Build healthy blocks index for drift computation
# ============================================================
# Map X -> (block_id, stable_start) for valve=100
healthy_map = {}
for X in all_X:
    c, p, a = X
    for b, info in block_info.items():
        if info['cooler'] == c and info['pump'] == p and info['accum'] == a and info['valve'] == 100:
            healthy_map[X] = b
            break

# ============================================================
# 2. Separability pre-check (calibration only)
# ============================================================
print("\n[1] Separability pre-check (calibration X only)...")

def cycle_means(block_b, ch_idx):
    """Mean over 60 time steps for each cycle in the block."""
    indices = block_info[block_b]['indices']
    data = tensor_s_all[indices, ch_idx, :]  # (n_cycles, 60)
    return data.mean(axis=1)  # (n_cycles,)

results_table = []

for ch_name in ch_names_all:
    ch_idx = ch_names_all.index(ch_name)

    sep73_list = []; sep80_list = []; sep90_list = []; drift_list = []
    drift_dist_list = []
    drift_label_diff_list = []

    for X_i in X_calib:
        b_healthy = healthy_map.get(X_i)
        if b_healthy is None: continue
        healthy_means = cycle_means(b_healthy, ch_idx)
        h_med = np.median(healthy_means)
        h_std = np.std(healthy_means)

        # Fault separations
        fault_blocks = {}
        for b, info in block_info.items():
            if (info['cooler'] == X_i[0] and info['pump'] == X_i[1] and
                info['accum'] == X_i[2] and info['valve'] in [73, 80, 90]):
                fault_blocks[info['valve']] = b

        seps = {}
        for v in [73, 80, 90]:
            if v in fault_blocks:
                f_means = cycle_means(fault_blocks[v], ch_idx)
                sep_v = np.abs(np.median(f_means) - h_med) / (h_std + 1e-9)
                seps[v] = sep_v
            else:
                seps[v] = np.nan

        # Drift: nearest OTHER healthy block (valve=100)
        # Find X_j != X_i with valve=100, closest in stable_idx
        h_start = block_info[b_healthy]['start']
        best_dist = 999999
        best_X_j = None
        for X_j in all_X:
            if X_j == X_i: continue
            b_j = healthy_map.get(X_j)
            if b_j is None: continue
            dist = abs(block_info[b_j]['start'] - h_start)
            if dist < best_dist:
                best_dist = dist
                best_X_j = X_j

        if best_X_j is not None:
            near_means = cycle_means(healthy_map[best_X_j], ch_idx)
            near_med = np.median(near_means)
            drift_val = np.abs(near_med - h_med) / (h_std + 1e-9)
            drift_list.append(drift_val)
            drift_dist_list.append(best_dist)

            # Which labels differ?
            diffs = []
            if X_i[0] != best_X_j[0]: diffs.append(f'cooler({X_i[0]}->{best_X_j[0]})')
            if X_i[1] != best_X_j[1]: diffs.append(f'pump({X_i[1]}->{best_X_j[1]})')
            if X_i[2] != best_X_j[2]: diffs.append(f'accum({X_i[2]}->{best_X_j[2]})')
            drift_label_diff_list.append(';'.join(diffs) if diffs else 'same')

        for v in [73, 80, 90]:
            if v == 73: sep73_list.append(seps.get(73, np.nan))
            elif v == 80: sep80_list.append(seps.get(80, np.nan))
            elif v == 90: sep90_list.append(seps.get(90, np.nan))

    # Median across 18 X
    med73 = np.nanmedian(sep73_list) if sep73_list else np.nan
    med80 = np.nanmedian(sep80_list) if sep80_list else np.nan
    med90 = np.nanmedian(sep90_list) if sep90_list else np.nan
    med_drift = np.nanmedian(drift_list) if drift_list else np.nan
    avg_dist = np.mean(drift_dist_list) if drift_dist_list else np.nan
    # What labels typically differ in nearest healthy pairs?
    label_diff_summary = {}
    for d in drift_label_diff_list:
        for part in d.split(';'):
            if part and part != 'same':
                label_diff_summary[part] = label_diff_summary.get(part, 0) + 1

    # Pass check
    passed = (med73 >= 3 and med80 >= 3 and med90 >= 3 and
              med73 >= 2 * med_drift and med80 >= 2 * med_drift and med90 >= 2 * med_drift)

    results_table.append({
        'channel': ch_name,
        'sep73': med73, 'sep80': med80, 'sep90': med90,
        'drift': med_drift, 'avg_dist': avg_dist,
        'label_diffs': label_diff_summary,
        'passed': passed,
    })

# Sort by sep73 descending
results_table.sort(key=lambda r: r['sep73'], reverse=True)

# Print full 32-row table
print(f"\n{'Channel':<22} {'sep(73)':>9} {'sep(80)':>9} {'sep(90)':>9} {'drift':>9} {'pass':>6}")
print("-" * 70)
passed_channels = []
for r in results_table:
    flag = "YES" if r['passed'] else ""
    print(f"{r['channel']:<22} {r['sep73']:>9.2f} {r['sep80']:>9.2f} "
          f"{r['sep90']:>9.2f} {r['drift']:>9.2f} {flag:>6}")
    if r['passed']:
        passed_channels.append(r['channel'])

print(f"\n  Drift info (typical nearest healthy pair):")
# Show one example
for r in results_table[:1]:
    print(f"    avg nearest distance: {r['avg_dist']:.0f} stable_idx")
    print(f"    typical label differences: {r['label_diffs']}")

print(f"\n  Pass condition: sep(73,80,90) >= 3 AND all >= 2*drift")
if passed_channels:
    print(f"  PASSED channels: {passed_channels}")
else:
    print(f"  NO CHANNEL PASSED.")
    print(f"\n  valve pre-check FAILED. Stage 2 valve ends here.")
    print(f"  Best channel by sep(73): {results_table[0]['channel']} "
          f"(sep73={results_table[0]['sep73']:.2f}, drift={results_table[0]['drift']:.2f})")
    sys.exit(0)

# ============================================================
# 3. If passed: run Chronos on best channel
# ============================================================
print("\n" + "="*60)
print("STEP 2: Chronos on best passing channel")
print("="*60)

# Use the actual PASSED channel with highest sep73, not just top-sorted
passed_sorted = [r for r in results_table if r['passed']]
best_ch = passed_sorted[0]['channel']
best_idx = ch_names_all.index(best_ch)
print(f"  Best channel: {best_ch} (idx={best_idx})")

import torch
from chronos import BaseChronosPipeline

print("[3] Loading Chronos-Bolt-Small...")
pipe = BaseChronosPipeline.from_pretrained(
    'amazon/chronos-bolt-small', local_files_only=True,
    device_map='cpu', dtype=torch.float32,
)

# Build windows
print("\n[4] Building windows...")

# Use NEAREST healthy block for C-class (not farthest)
ctx_A = []; ctx_B = []; ctx_C = []

for X_i in all_X:
    c, p, a = X_i
    b_healthy = healthy_map.get(X_i)
    if b_healthy is None: continue

    ctx_indices = block_info[b_healthy]['indices'][:5]
    ctx_arr = tensor_s_all[ctx_indices, best_idx, :].flatten().astype(np.float32)

    # A: fault targets
    for v in [73, 80, 90]:
        for b, info in block_info.items():
            if (info['cooler'] == c and info['pump'] == p and
                info['accum'] == a and info['valve'] == v):
                tgt = tensor_s_all[info['indices'], best_idx, :].astype(np.float32)
                ctx_A.append({'ctx': ctx_arr, 'tgt': tgt, 'valve_target': v, 'X': X_i,
                              'ctx_start': block_info[b_healthy]['start'],
                              'tgt_start': info['start']})
                break

    # B: near healthy (cycles 6-10)
    if block_info[b_healthy]['n'] >= 10:
        tgt_indices = block_info[b_healthy]['indices'][5:10]
        tgt = tensor_s_all[tgt_indices, best_idx, :].astype(np.float32)
        ctx_B.append({'ctx': ctx_arr, 'tgt': tgt, 'X': X_i,
                      'ctx_start': block_info[b_healthy]['start'],
                      'tgt_start': block_info[b_healthy]['start']})

    # C: NEAREST other healthy block (not farthest)
    h_start = block_info[b_healthy]['start']
    best_dist = 999999
    best_X_j = None
    for X_j in all_X:
        if X_j == X_i: continue
        b_j = healthy_map.get(X_j)
        if b_j is None: continue
        dist = abs(block_info[b_j]['start'] - h_start)
        if dist < best_dist:
            best_dist = dist
            best_X_j = X_j
    if best_X_j is not None:
        b_j = healthy_map[best_X_j]
        tgt_indices = block_info[b_j]['indices'][:10]
        tgt = tensor_s_all[tgt_indices, best_idx, :].astype(np.float32)
        ctx_C.append({'ctx': ctx_arr, 'tgt': tgt, 'X_i': X_i, 'X_j': best_X_j,
                      'ctx_start': block_info[b_healthy]['start'],
                      'tgt_start': block_info[b_j]['start'],
                      'gap': best_dist})

print(f"  A: {len(ctx_A)}, B: {len(ctx_B)}, C: {len(ctx_C)}")

# C-class gap stats
c_gaps = [c['gap'] for c in ctx_C]
print(f"  C gap: min={min(c_gaps)}, mean={np.mean(c_gaps):.1f}, max={max(c_gaps)}")
print(f"  Same magnitude as A gap (10-30)? {max(c_gaps) <= 50 and min(c_gaps) >= 5}")

# C cross-cooler count
n_cross = 0
for c in ctx_C:
    if c['X_i'][0] != c['X_j'][0]:
        n_cross += 1
print(f"  C pairs crossing cooler levels: {n_cross}/{len(ctx_C)} (expected 0)")

# Chronos predictions
print("\n[5] Chronos predictions...")
def chronos_predict_all(ctx_list):
    if len(ctx_list) == 0: return []
    ctx_batch = np.stack([c['ctx'] for c in ctx_list])
    with torch.no_grad():
        result = pipe.predict_quantiles(
            torch.tensor(ctx_batch), prediction_length=60,
            quantile_levels=[0.1, 0.5, 0.9],
        )
    q = result[0].numpy()
    return list(zip(q[:, :, 0], q[:, :, 1], q[:, :, 2]))

t0 = time.time()
for wtype, ctx_list in [('A', ctx_A), ('B', ctx_B), ('C', ctx_C)]:
    preds = chronos_predict_all(ctx_list)
    for i, p in enumerate(preds):
        ctx_list[i]['pred'] = p
print(f"  Time: {time.time()-t0:.2f}s")

# Threshold calibration
print("\n[6] Threshold calibration (calib C-class only)...")
s_lev_calib = []; s_shp_calib = []
for c in ctx_C:
    if c['X_i'] not in X_calib: continue
    q10, q50, q90 = c['pred']
    tgt = c['tgt']
    band_half = np.mean((q90 - q10) / 2.0)
    for ci in range(tgt.shape[0]):
        resid = tgt[ci] - q50
        sl = np.abs(np.mean(resid))
        ss = np.mean(np.abs(resid - np.mean(resid)))
        if band_half > 1e-10: ss = ss / band_half
        s_lev_calib.append(sl); s_shp_calib.append(ss)

s_lev_calib = np.array(s_lev_calib); s_shp_calib = np.array(s_shp_calib)
tau_level = np.percentile(s_lev_calib, 95)
tau_shape = np.percentile(s_shp_calib, 95)
print(f"  tau_level={tau_level:.4f}, tau_shape={tau_shape:.4f}")

# Evaluate on test
print("\n[7] Test evaluation...")

def classify_test(ctx_entry, tau_level, tau_shape):
    q10, q50, q90 = ctx_entry['pred']
    tgt = ctx_entry['tgt']
    band_half = np.mean((q90 - q10) / 2.0)
    results = []
    for ci in range(tgt.shape[0]):
        resid = tgt[ci] - q50
        sl = np.abs(np.mean(resid))
        ss = np.mean(np.abs(resid - np.mean(resid)))
        if band_half > 1e-10: ss = ss / band_half
        danger = (sl > tau_level) or (ss > tau_shape)
        results.append({'danger': danger, 's_level': sl, 's_shape': ss})
    return results

def eval_test(ctx_list, filter_fn=None):
    all_r = []
    for c in ctx_list:
        X_key = c.get('X_i', c.get('X', None))
        if X_key is None: X_key = c['X']
        if X_key not in X_test: continue
        if filter_fn and not filter_fn(c): continue
        all_r.extend(classify_test(c, tau_level, tau_shape))
    return all_r

print(f"\n  Channel: {best_ch}")
for vf in [73, 80, 90]:
    res = eval_test(ctx_A, lambda c, v=vf: c['valve_target'] == v)
    n = len(res); n_d = sum(r['danger'] for r in res)
    print(f"  A (valve={vf}): danger={n_d}/{n} = {n_d/n:.4f}" if n > 0 else f"  A (valve={vf}): NO DATA")

res_A = eval_test(ctx_A)
n_A = len(res_A); n_Ad = sum(r['danger'] for r in res_A)
print(f"  A (combined): danger={n_Ad}/{n_A} = {n_Ad/n_A:.4f}" if n_A > 0 else "A: empty")

res_B = eval_test(ctx_B)
n_B = len(res_B); n_Bd = sum(r['danger'] for r in res_B)
print(f"  B (Near):     danger={n_Bd}/{n_B} = {n_Bd/n_B:.4f}" if n_B > 0 else "B: empty")

res_C = eval_test(ctx_C)
n_C = len(res_C); n_Cd = sum(r['danger'] for r in res_C)
print(f"  C (Far/Near): danger={n_Cd}/{n_C} = {n_Cd/n_C:.4f}" if n_C > 0 else "C: empty")

# Three-way comparison
print(f"\n  {'Method':<25} {'A Det':>10} {'B FPR':>10} {'C FPR':>10}")
print(f"  {'-'*55}")
chronos_A = n_Ad/n_A if n_A>0 else 0
chronos_B = n_Bd/n_B if n_B>0 else 0
chronos_C = n_Cd/n_C if n_C>0 else 0

# Naive
def calibrate_naive(ctx_C_list):
    s_lev=[]; s_shp=[]
    for c in ctx_C_list:
        if c['X_i'] not in X_calib: continue
        lc=c['ctx'][-60:]; std_lc=np.std(lc); tgt=c['tgt']
        for ci in range(tgt.shape[0]):
            r=tgt[ci]-lc; sl=np.abs(np.mean(r)); ss=np.mean(np.abs(r-np.mean(r)))
            if std_lc>1e-10: ss/=std_lc
            s_lev.append(sl); s_shp.append(ss)
    return np.percentile(s_lev,95), np.percentile(s_shp,95)

tl_n, ts_n = calibrate_naive(ctx_C)
def eval_n(ctx_list, filter_fn=None):
    n_t=0; n_d=0
    for c in ctx_list:
        Xk=c.get('X_i',c.get('X',None)) or c['X']
        if Xk not in X_test: continue
        if filter_fn and not filter_fn(c): continue
        lc=c['ctx'][-60:]; std_lc=np.std(lc); tgt=c['tgt']
        for ci in range(tgt.shape[0]):
            r=tgt[ci]-lc; sl=np.abs(np.mean(r)); ss=np.mean(np.abs(r-np.mean(r)))
            if std_lc>1e-10: ss/=std_lc
            if (sl>tl_n)or(ss>ts_n): n_d+=1
            n_t+=1
    return n_d/n_t if n_t>0 else 0

def calibrate_trivial(ctx_C_list):
    s_lev=[]; s_shp=[]
    for c in ctx_C_list:
        if c['X_i'] not in X_calib: continue
        lc=c['ctx'][-60:]; q10,q50,q90=c['pred']; tgt=c['tgt']
        bh=np.mean((q90-q10)/2.0)
        for ci in range(tgt.shape[0]):
            sl=np.abs(np.mean(tgt[ci]-lc))
            r=tgt[ci]-q50; ss=np.mean(np.abs(r-np.mean(r)))
            if bh>1e-10: ss/=bh
            s_lev.append(sl); s_shp.append(ss)
    return np.percentile(s_lev,95), np.percentile(s_shp,95)

tl_t, ts_t = calibrate_trivial(ctx_C)
def eval_t(ctx_list, filter_fn=None):
    n_t=0; n_d=0
    for c in ctx_list:
        Xk=c.get('X_i',c.get('X',None)) or c['X']
        if Xk not in X_test: continue
        if filter_fn and not filter_fn(c): continue
        lc=c['ctx'][-60:]; q10,q50,q90=c['pred']; tgt=c['tgt']
        bh=np.mean((q90-q10)/2.0)
        for ci in range(tgt.shape[0]):
            sl=np.abs(np.mean(tgt[ci]-lc))
            r=tgt[ci]-q50; ss=np.mean(np.abs(r-np.mean(r)))
            if bh>1e-10: ss/=bh
            if (sl>tl_t)or(ss>ts_t): n_d+=1
            n_t+=1
    return n_d/n_t if n_t>0 else 0

na_A = eval_n(ctx_A); na_B = eval_n(ctx_B); na_C = eval_n(ctx_C)
tr_A = eval_t(ctx_A); tr_B = eval_t(ctx_B); tr_C = eval_t(ctx_C)

print(f"  {'Chronos decomp':<25} {chronos_A:>10.4f} {chronos_B:>10.4f} {chronos_C:>10.4f}")
print(f"  {'Naive last-value':<25} {na_A:>10.4f} {na_B:>10.4f} {na_C:>10.4f}")
print(f"  {'Trivial mean':<25} {tr_A:>10.4f} {tr_B:>10.4f} {tr_C:>10.4f}")

print("\n" + "="*60)
print("DONE")
print("="*60)
