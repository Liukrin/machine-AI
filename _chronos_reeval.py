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

ts2_idx = ch_names.index('ts2')

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

valve_vals = [73, 80, 90, 100]
pump_vals  = [0, 1, 2]
accum_vals = [90, 100, 115, 130]
all_X = [(v, p, a) for v in valve_vals for p in pump_vals for a in accum_vals]

# Sort X by stable_idx of their cooler=100 block
X_with_stable_start = []
for X in all_X:
    v, p, a = X
    for b, info in block_info.items():
        if info['valve'] == v and info['pump'] == p and info['accum'] == a and info['cooler'] == 100:
            X_with_stable_start.append((X, info['start']))
            break
X_with_stable_start.sort(key=lambda x: x[1])
X_ordered = [x[0] for x in X_with_stable_start]
print(f"    {len(X_ordered)} X triples ordered by stable_idx")

print("[0] Loading Chronos-Bolt-Small...")
pipe = BaseChronosPipeline.from_pretrained(
    'amazon/chronos-bolt-small', local_files_only=True,
    device_map='cpu', dtype=torch.float32,
)
print(f"    Pipeline: {type(pipe).__name__}")

# ============================================================
# 1. Build windows (identical to previous round)
# ============================================================
print("\n[1] Building windows...")

contexts_A = []
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
    ctx_ts2 = tensor_s[ctx_indices, ts2_idx, :].flatten().astype(np.float32)
    if b3 is not None:
        tgt = tensor_s[block_info[b3]['indices'], ts2_idx, :].astype(np.float32)
        contexts_A.append({'ctx': ctx_ts2, 'tgt': tgt, 'cooler_target': 3, 'X': X,
                           'ctx_start': block_info[b100]['start'], 'tgt_start': block_info[b3]['start']})
    if b20 is not None:
        tgt = tensor_s[block_info[b20]['indices'], ts2_idx, :].astype(np.float32)
        contexts_A.append({'ctx': ctx_ts2, 'tgt': tgt, 'cooler_target': 20, 'X': X,
                           'ctx_start': block_info[b100]['start'], 'tgt_start': block_info[b20]['start']})

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
                contexts_B.append({'ctx': ctx_ts2, 'tgt': tgt, 'X': X,
                                   'ctx_start': info['start'], 'tgt_start': info['start']})
            break

c100_blocks = sorted(
    [(b, info) for b, info in block_info.items() if info['cooler'] == 100],
    key=lambda x: x[1]['start']
)
X_to_c100_idx = {}
for idx, (b, info) in enumerate(c100_blocks):
    X_to_c100_idx[(info['valve'], info['pump'], info['accum'])] = idx

contexts_C = []
for X_i in all_X:
    if X_i not in X_to_c100_idx: continue
    i_idx = X_to_c100_idx[X_i]
    best_j = None; best_dist = -1
    for X_j in all_X:
        if X_j not in X_to_c100_idx: continue
        j_idx = X_to_c100_idx[X_j]
        dist = abs(i_idx - j_idx)
        if dist > best_dist:
            best_dist = dist; best_j = j_idx
    if best_j is None: continue
    b_i = c100_blocks[i_idx][0]
    ctx_indices = block_info[b_i]['indices'][:5]
    ctx_ts2 = tensor_s[ctx_indices, ts2_idx, :].flatten().astype(np.float32)
    b_j = c100_blocks[best_j][0]
    tgt_indices = block_info[b_j]['indices'][:10]
    tgt = tensor_s[tgt_indices, ts2_idx, :].astype(np.float32)
    contexts_C.append({'ctx': ctx_ts2, 'tgt': tgt, 'X_i': X_i,
                       'X_j': (c100_blocks[best_j][1]['valve'], c100_blocks[best_j][1]['pump'],
                               c100_blocks[best_j][1]['accum']),
                       'ctx_start': block_info[b_i]['start'],
                       'tgt_start': c100_blocks[best_j][1]['start']})

print(f"    A: {len(contexts_A)} batches, B: {len(contexts_B)} batches, C: {len(contexts_C)} batches")

# ============================================================
# 2. Chronos predictions (identical to previous round)
# ============================================================
print("\n[2] Chronos predictions...")

def chronos_predict_batch(ctx_list):
    if len(ctx_list) == 0: return []
    ctx_batch = np.stack([c['ctx'] for c in ctx_list])
    with torch.no_grad():
        result = pipe.predict_quantiles(
            torch.tensor(ctx_batch), prediction_length=60,
            quantile_levels=[0.1, 0.5, 0.9],
        )
    quantiles = result[0].numpy()
    return list(zip(quantiles[:, :, 0], quantiles[:, :, 1], quantiles[:, :, 2]))

t0 = time.time()
pred_A = chronos_predict_batch(contexts_A)
pred_B = chronos_predict_batch(contexts_B)
pred_C = chronos_predict_batch(contexts_C)
print(f"    Time: {time.time()-t0:.2f}s")

# Attach predictions to window dicts
for i, p in enumerate(pred_A): contexts_A[i]['pred'] = p
for i, p in enumerate(pred_B): contexts_B[i]['pred'] = p
for i, p in enumerate(pred_C): contexts_C[i]['pred'] = p

# ============================================================
# STEP 1: Point prediction accuracy (B and C only)
# ============================================================
print("\n" + "="*60)
print("STEP 1: Point prediction accuracy")
print("="*60)

def point_mae_chronos(ctx_list):
    """MAE of Chronos q50 vs ground truth, per-point."""
    all_abs_err = []
    for c in ctx_list:
        q50 = c['pred'][1]  # (60,)
        tgt = c['tgt']       # (m, 60)
        for ci in range(tgt.shape[0]):
            all_abs_err.append(np.abs(tgt[ci] - q50))
    return np.mean(np.concatenate(all_abs_err))

def point_mae_naive(ctx_list):
    """MAE of naive last-value vs ground truth, per-point."""
    all_abs_err = []
    for c in ctx_list:
        last_cycle = c['ctx'][-60:]
        tgt = c['tgt']
        for ci in range(tgt.shape[0]):
            all_abs_err.append(np.abs(tgt[ci] - last_cycle))
    return np.mean(np.concatenate(all_abs_err))

mae_B_chronos = point_mae_chronos(contexts_B)
mae_B_naive   = point_mae_naive(contexts_B)
mae_C_chronos = point_mae_chronos(contexts_C)
mae_C_naive   = point_mae_naive(contexts_C)

print(f"  Type B (Near Healthy):")
print(f"    Chronos q50 MAE: {mae_B_chronos:.4f} °C")
print(f"    Naive last-value MAE: {mae_B_naive:.4f} °C")
print(f"    Ratio (Chronos/Naive): {mae_B_chronos/mae_B_naive:.4f}")
print(f"  Type C (Far Healthy):")
print(f"    Chronos q50 MAE: {mae_C_chronos:.4f} °C")
print(f"    Naive last-value MAE: {mae_C_naive:.4f} °C")
print(f"    Ratio (Chronos/Naive): {mae_C_chronos/mae_C_naive:.4f}")

if mae_B_chronos < mae_B_naive:
    impr_B = (mae_B_naive - mae_B_chronos) / mae_B_naive * 100
    print(f"  Chronos IS better than naive on B: {impr_B:.1f}% lower MAE")
else:
    print(f"  Chronos is NOT better than naive on B")

if mae_C_chronos < mae_C_naive:
    impr_C = (mae_C_naive - mae_C_chronos) / mae_C_naive * 100
    print(f"  Chronos IS better than naive on C: {impr_C:.1f}% lower MAE")
else:
    print(f"  Chronos is NOT better than naive on C")

# ============================================================
# STEP 2: Deviation decomposition
# ============================================================
print("\n" + "="*60)
print("STEP 2: Deviation decomposition (s_level, s_shape)")
print("="*60)

def compute_deviation(ctx_list):
    """For each window ctx entry, compute s_level and s_shape per target cycle."""
    results = []
    for c in ctx_list:
        q10 = c['pred'][0]
        q50 = c['pred'][1]
        q90 = c['pred'][2]
        tgt = c['tgt']
        band_half = (q90 - q10) / 2.0  # (60,)
        mean_band_half = np.mean(band_half)
        for ci in range(tgt.shape[0]):
            resid = tgt[ci] - q50  # (60,)
            s_level = np.abs(np.mean(resid))
            # s_shape: mean absolute deviation from mean residual, normalized by mean half-band
            s_shape = np.mean(np.abs(resid - np.mean(resid)))
            if mean_band_half > 1e-10:
                s_shape = s_shape / mean_band_half
            else:
                s_shape = 999.0
            results.append({
                's_level': s_level, 's_shape': s_shape,
                'type': c.get('cooler_target', 100),
            })
    return results

dev_A = compute_deviation(contexts_A)
dev_B = compute_deviation(contexts_B)
dev_C = compute_deviation(contexts_C)

print(f"  A (Fault): {len(dev_A)} cycles")
print(f"    s_level: min={min(d['s_level'] for d in dev_A):.4f}, "
      f"median={np.median([d['s_level'] for d in dev_A]):.4f}, "
      f"max={max(d['s_level'] for d in dev_A):.4f}")
print(f"    s_shape: min={min(d['s_shape'] for d in dev_A):.4f}, "
      f"median={np.median([d['s_shape'] for d in dev_A]):.4f}, "
      f"max={max(d['s_shape'] for d in dev_A):.4f}")
print(f"  B (Near Healthy): {len(dev_B)} cycles")
print(f"    s_level: min={min(d['s_level'] for d in dev_B):.4f}, "
      f"median={np.median([d['s_level'] for d in dev_B]):.4f}, "
      f"max={max(d['s_level'] for d in dev_B):.4f}")
print(f"    s_shape: min={min(d['s_shape'] for d in dev_B):.4f}, "
      f"median={np.median([d['s_shape'] for d in dev_B]):.4f}, "
      f"max={max(d['s_shape'] for d in dev_B):.4f}")
print(f"  C (Far Healthy): {len(dev_C)} cycles")
print(f"    s_level: min={min(d['s_level'] for d in dev_C):.4f}, "
      f"median={np.median([d['s_level'] for d in dev_C]):.4f}, "
      f"max={max(d['s_level'] for d in dev_C):.4f}")
print(f"    s_shape: min={min(d['s_shape'] for d in dev_C):.4f}, "
      f"median={np.median([d['s_shape'] for d in dev_C]):.4f}, "
      f"max={max(d['s_shape'] for d in dev_C):.4f}")

# ============================================================
# STEP 3: Threshold calibration (C-class windows, first 24 X)
# ============================================================
print("\n" + "="*60)
print("STEP 3: Threshold calibration")
print("="*60)

# Split X: first 24 = calibration, last 24 = test
X_calib = set(X_ordered[:24])
X_test  = set(X_ordered[24:])
print(f"  Calibration X count: {len(X_calib)}")
print(f"  Test X count: {len(X_test)}")
# Verify no overlap
assert len(X_calib & X_test) == 0, "OVERLAP between calib and test!"

# Collect C-class s_level, s_shape for calibration X only
# C windows: for X_i in calib, use the window where X_i is the context owner
calib_s_level = []
calib_s_shape = []
for c, dev_list in [(contexts_C, dev_C)]:
    for idx, d in enumerate(dev_list):
        X_owner = contexts_C[idx // len(dev_list)]['X_i'] if idx < len(contexts_C) * len(dev_list) else None
        # Actually dev_C is a flat list. Need to map back.
        pass

# Better approach: iterate through contexts_C
calib_s_level = []
calib_s_shape = []
for ci, c in enumerate(contexts_C):
    X_i = c['X_i']
    if X_i in X_calib:
        q10 = c['pred'][0]; q50 = c['pred'][1]; q90 = c['pred'][2]
        tgt = c['tgt']
        band_half = np.mean((q90 - q10) / 2.0)
        for cycle_i in range(tgt.shape[0]):
            resid = tgt[cycle_i] - q50
            s_level = np.abs(np.mean(resid))
            s_shape = np.mean(np.abs(resid - np.mean(resid)))
            if band_half > 1e-10:
                s_shape = s_shape / band_half
            calib_s_level.append(s_level)
            calib_s_shape.append(s_shape)

calib_s_level = np.array(calib_s_level)
calib_s_shape = np.array(calib_s_shape)
tau_level = np.percentile(calib_s_level, 95)
tau_shape = np.percentile(calib_s_shape, 95)
print(f"  tau_level (95th pct of C-class calib s_level): {tau_level:.4f} °C")
print(f"  tau_shape (95th pct of C-class calib s_shape): {tau_shape:.4f}")

# ============================================================
# STEP 4: Evaluate on test X only
# ============================================================
print("\n" + "="*60)
print("STEP 4: Test-set evaluation (24 test X)")
print("="*60)

def classify_window(c, pred_tuple, tau_level, tau_shape):
    """Returns list of verdicts (True=danger, False=normal) per target cycle."""
    q10, q50, q90 = pred_tuple
    tgt = c['tgt']
    band_half = np.mean((q90 - q10) / 2.0)
    verdicts = []
    details = []
    for ci in range(tgt.shape[0]):
        resid = tgt[ci] - q50
        s_level = np.abs(np.mean(resid))
        s_shape = np.mean(np.abs(resid - np.mean(resid)))
        if band_half > 1e-10:
            s_shape = s_shape / band_half
        danger = (s_level > tau_level) or (s_shape > tau_shape)
        verdicts.append(danger)
        details.append({'s_level': s_level, 's_shape': s_shape, 'danger': danger})
    return verdicts, details

def eval_on_test(contexts_list, tau_level, tau_shape, filter_fn=None):
    """Evaluate on test X only. filter_fn(c) returns True if should include."""
    results = []
    for i, c in enumerate(contexts_list):
        # Determine if this window belongs to test set
        X_key = c.get('X_i', c.get('X', None))
        if X_key is None:
            # For A and B, X is the key
            X_key = c['X']
        if X_key not in X_test:
            continue
        if filter_fn and not filter_fn(c):
            continue
        verdicts, details = classify_window(c, c['pred'], tau_level, tau_shape)
        results.append({'c': c, 'verdicts': verdicts, 'details': details})
    return results

# A-class, split by cooler=3 and cooler=20
res_A_test = eval_on_test(contexts_A, tau_level, tau_shape)
res_A3 = [r for r in res_A_test if r['c']['cooler_target'] == 3]
res_A20 = [r for r in res_A_test if r['c']['cooler_target'] == 20]

n_A3 = sum(len(r['verdicts']) for r in res_A3)
n_A3_danger = sum(sum(r['verdicts']) for r in res_A3)
n_A20 = sum(len(r['verdicts']) for r in res_A20)
n_A20_danger = sum(sum(r['verdicts']) for r in res_A20)

print(f"  A (cooler=3):  danger={n_A3_danger}/{n_A3} = {n_A3_danger/n_A3:.4f}")
print(f"  A (cooler=20): danger={n_A20_danger}/{n_A20} = {n_A20_danger/n_A20:.4f}")
print(f"  A (combined):  danger={(n_A3_danger+n_A20_danger)}/{n_A3+n_A20} = "
      f"{(n_A3_danger+n_A20_danger)/(n_A3+n_A20):.4f}")

res_B_test = eval_on_test(contexts_B, tau_level, tau_shape)
n_B = sum(len(r['verdicts']) for r in res_B_test)
n_B_danger = sum(sum(r['verdicts']) for r in res_B_test)
print(f"  B (Near Healthy): danger={n_B_danger}/{n_B} = {n_B_danger/n_B:.4f}")

res_C_test = eval_on_test(contexts_C, tau_level, tau_shape)
n_C = sum(len(r['verdicts']) for r in res_C_test)
n_C_danger = sum(sum(r['verdicts']) for r in res_C_test)
c_fpr = n_C_danger / n_C if n_C > 0 else 0
print(f"  C (Far Healthy):  danger={n_C_danger}/{n_C} = {c_fpr:.4f}")
print(f"  Expected near 5%: {abs(c_fpr - 0.05) < 0.03} (within ±3pp)")

# s_level / s_shape stats per class on test set
def collect_stats_on_test(contexts_list, filter_fn=None):
    s_lev = []; s_shp = []
    for i, c in enumerate(contexts_list):
        X_key = c.get('X_i', c.get('X', None))
        if X_key is None: X_key = c['X']
        if X_key not in X_test: continue
        if filter_fn and not filter_fn(c): continue
        q10, q50, q90 = c['pred']
        tgt = c['tgt']
        band_half = np.mean((q90 - q10) / 2.0)
        for ci in range(tgt.shape[0]):
            resid = tgt[ci] - q50
            sl = np.abs(np.mean(resid))
            ss = np.mean(np.abs(resid - np.mean(resid)))
            if band_half > 1e-10: ss = ss / band_half
            s_lev.append(sl); s_shp.append(ss)
    return np.array(s_lev), np.array(s_shp)

for label, ctxs, flt in [
    ('A (Fault)', contexts_A, None),
    ('B (Near)', contexts_B, None),
    ('C (Far)', contexts_C, None),
]:
    sl, ss = collect_stats_on_test(ctxs, flt)
    print(f"  {label}: s_level min={sl.min():.4f} median={np.median(sl):.4f} max={sl.max():.4f}")
    print(f"  {label}: s_shape min={ss.min():.4f} median={np.median(ss):.4f} max={ss.max():.4f}")

# ============================================================
# STEP 5: Trivial mean detector
# ============================================================
print("\n" + "="*60)
print("STEP 5: Trivial mean detector vs Chronos vs Naive")
print("="*60)

def trivial_classify(c, tau_level_trivial, tau_shape_trivial):
    """s_level only for trivial (no q50 from Chronos). Use same s_shape from Chronos band."""
    # Trivial: no Chronos prediction, compute s_level from last cycle of context
    q50_chr = c['pred'][1]
    q10_chr = c['pred'][0]
    q90_chr = c['pred'][2]
    tgt = c['tgt']
    last_cycle = c['ctx'][-60:]
    band_half = np.mean((q90_chr - q10_chr) / 2.0)  # still need Chronos band for s_shape

    verdicts = []
    for ci in range(tgt.shape[0]):
        resid_trivial = tgt[ci] - last_cycle  # trivial: just last cycle
        s_level_t = np.abs(np.mean(resid_trivial))
        # s_shape still needs Chronos q50 for residual
        resid_chr = tgt[ci] - q50_chr
        s_shape_t = np.mean(np.abs(resid_chr - np.mean(resid_chr)))
        if band_half > 1e-10:
            s_shape_t = s_shape_t / band_half
        danger = (s_level_t > tau_level_trivial) or (s_shape_t > tau_shape_trivial)
        verdicts.append(danger)
    return verdicts

# Calibrate trivial thresholds on C-class calib
calib_s_level_trivial = []
calib_s_shape_trivial = []
for ci, c in enumerate(contexts_C):
    X_i = c['X_i']
    if X_i in X_calib:
        last_cycle = c['ctx'][-60:]
        q10 = c['pred'][0]; q50 = c['pred'][1]; q90 = c['pred'][2]
        tgt = c['tgt']
        band_half = np.mean((q90 - q10) / 2.0)
        for cycle_i in range(tgt.shape[0]):
            resid_t = tgt[cycle_i] - last_cycle
            s_level_t = np.abs(np.mean(resid_t))
            resid_chr = tgt[cycle_i] - q50
            s_shape_t = np.mean(np.abs(resid_chr - np.mean(resid_chr)))
            if band_half > 1e-10:
                s_shape_t = s_shape_t / band_half
            calib_s_level_trivial.append(s_level_t)
            calib_s_shape_trivial.append(s_shape_t)

calib_s_level_trivial = np.array(calib_s_level_trivial)
calib_s_shape_trivial = np.array(calib_s_shape_trivial)
tau_level_trivial = np.percentile(calib_s_level_trivial, 95)
tau_shape_trivial = np.percentile(calib_s_shape_trivial, 95)
print(f"  Trivial tau_level: {tau_level_trivial:.4f}")
print(f"  Trivial tau_shape: {tau_shape_trivial:.4f}")

# Evaluate trivial on test set
def eval_trivial_on_test(contexts_list, tau_l, tau_s, filter_fn=None):
    total = 0; danger = 0
    for c in contexts_list:
        X_key = c.get('X_i', c.get('X', None))
        if X_key is None: X_key = c['X']
        if X_key not in X_test: continue
        if filter_fn and not filter_fn(c): continue
        verdicts = trivial_classify(c, tau_l, tau_s)
        total += len(verdicts)
        danger += sum(verdicts)
    return danger / total if total > 0 else 0

trivial_A = eval_trivial_on_test(contexts_A, tau_level_trivial, tau_shape_trivial)
trivial_B = eval_trivial_on_test(contexts_B, tau_level_trivial, tau_shape_trivial)
trivial_C = eval_trivial_on_test(contexts_C, tau_level_trivial, tau_shape_trivial)

# Naive evaluation on test set (using the same s_level/s_shape framework)
def eval_naive_on_test(contexts_list, tau_l, tau_s, filter_fn=None):
    """Naive: use last-value as q50, ±1 std of last cycle as band."""
    total = 0; danger = 0
    for c in contexts_list:
        X_key = c.get('X_i', c.get('X', None))
        if X_key is None: X_key = c['X']
        if X_key not in X_test: continue
        if filter_fn and not filter_fn(c): continue
        last_cycle = c['ctx'][-60:]
        std_last = np.std(last_cycle)
        tgt = c['tgt']
        band_half = std_last  # band = ±1 std
        for ci in range(tgt.shape[0]):
            resid = tgt[ci] - last_cycle
            s_l = np.abs(np.mean(resid))
            s_s = np.mean(np.abs(resid - np.mean(resid)))
            if band_half > 1e-10:
                s_s = s_s / band_half
            danger_verdict = (s_l > tau_l) or (s_s > tau_s)
            total += 1
            if danger_verdict: danger += 1
    return danger / total if total > 0 else 0

# Calibrate naive thresholds
calib_s_level_naive = []
calib_s_shape_naive = []
for ci, c in enumerate(contexts_C):
    X_i = c['X_i']
    if X_i in X_calib:
        last_cycle = c['ctx'][-60:]
        std_last = np.std(last_cycle)
        tgt = c['tgt']
        band_half = std_last
        for cycle_i in range(tgt.shape[0]):
            resid = tgt[cycle_i] - last_cycle
            s_l = np.abs(np.mean(resid))
            s_s = np.mean(np.abs(resid - np.mean(resid)))
            if band_half > 1e-10:
                s_s = s_s / band_half
            calib_s_level_naive.append(s_l)
            calib_s_shape_naive.append(s_s)

calib_s_level_naive = np.array(calib_s_level_naive)
calib_s_shape_naive = np.array(calib_s_shape_naive)
tau_level_naive = np.percentile(calib_s_level_naive, 95)
tau_shape_naive = np.percentile(calib_s_shape_naive, 95)

naive_A = eval_naive_on_test(contexts_A, tau_level_naive, tau_shape_naive)
naive_B = eval_naive_on_test(contexts_B, tau_level_naive, tau_shape_naive)
naive_C = eval_naive_on_test(contexts_C, tau_level_naive, tau_shape_naive)

# Chronos results on test
chronos_A = (n_A3_danger + n_A20_danger) / (n_A3 + n_A20)
chronos_B = n_B_danger / n_B
chronos_C = c_fpr

print(f"\n  {'Method':<25} {'A Detection':>12} {'B FPR':>12} {'C FPR':>12}")
print(f"  {'-'*60}")
print(f"  {'Chronos decomposition':<25} {chronos_A:>12.4f} {chronos_B:>12.4f} {chronos_C:>12.4f}")
print(f"  {'Naive last-value':<25} {naive_A:>12.4f} {naive_B:>12.4f} {naive_C:>12.4f}")
print(f"  {'Trivial mean':<25} {trivial_A:>12.4f} {trivial_B:>12.4f} {trivial_C:>12.4f}")

# Explicit answer
print(f"\n  Chronos A detection: {chronos_A:.4f}")
print(f"  Trivial A detection: {trivial_A:.4f}")
if chronos_A <= trivial_A + 0.01:
    print("  => On cooler, Chronos does NOT bring gain over trivial mean detector.")
else:
    print(f"  => Chronos brings {chronos_A - trivial_A:.4f} gain in detection over trivial.")

# ============================================================
# 6. Scatter plot
# ============================================================
print("\n[6] Scatter plot...")

# Collect all test-set deviations for scatter
all_test_dev = []
for label, ctxs in [('A', contexts_A), ('B', contexts_B), ('C', contexts_C)]:
    for i, c in enumerate(ctxs):
        X_key = c.get('X_i', c.get('X', None))
        if X_key is None: X_key = c['X']
        if X_key not in X_test: continue
        q10, q50, q90 = c['pred']
        tgt = c['tgt']
        band_half = np.mean((q90 - q10) / 2.0)
        for ci in range(tgt.shape[0]):
            resid = tgt[ci] - q50
            sl = np.abs(np.mean(resid))
            ss = np.mean(np.abs(resid - np.mean(resid)))
            if band_half > 1e-10: ss = ss / band_half
            all_test_dev.append({'s_level': sl, 's_shape': ss, 'type': label})

fig, ax = plt.subplots(figsize=(10, 8))
colors = {'A': '#d73027', 'B': '#4575b4', 'C': '#fc8d59'}
for label in ['A', 'B', 'C']:
    pts = [d for d in all_test_dev if d['type'] == label]
    sl = [p['s_level'] for p in pts]
    ss = [p['s_shape'] for p in pts]
    ax.scatter(sl, ss, c=colors[label], alpha=0.4, s=8, label=f'{label} ({len(pts)} pts)', rasterized=True)

# Threshold lines
sl_range = [min(d['s_level'] for d in all_test_dev) * 0.5,
            max(d['s_level'] for d in all_test_dev) * 2.0]
ax.axvline(x=tau_level, color='black', linestyle='--', linewidth=1.2, alpha=0.7,
           label=f'tau_level={tau_level:.3f}')
ax.axhline(y=tau_shape, color='gray', linestyle='--', linewidth=1.2, alpha=0.7,
           label=f'tau_shape={tau_shape:.3f}')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('s_level = |mean(actual − q50)|  (°C)', fontsize=12)
ax.set_ylabel('s_shape = mean(|resid − mean(resid)|) / mean((q90−q10)/2)', fontsize=12)
ax.set_title(f'Deviation Decomposition — Test Set (24 X triples)\n'
             f'A={len([d for d in all_test_dev if d["type"]=="A"])} '
             f'B={len([d for d in all_test_dev if d["type"]=="B"])} '
             f'C={len([d for d in all_test_dev if d["type"]=="C"])} pts',
             fontsize=13)
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3, which='both')
fig.tight_layout()
fig.savefig('reports/figures/chronos_deviation_scatter.png', dpi=150)
plt.close(fig)
print("    Saved reports/figures/chronos_deviation_scatter.png")

print("\n" + "="*60)
print("DONE")
print("="*60)
