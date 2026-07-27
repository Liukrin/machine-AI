import numpy as np
import pandas as pd
import torch, time, os, warnings
warnings.filterwarnings('ignore')
from chronos import BaseChronosPipeline

print("[Task 1] Building demo data package...")

# ============================================================
# 0. Load data + model
# ============================================================
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

# Order X by stable_idx
X_with_start = []
for X in all_X:
    v, p, a = X
    for b, info in block_info.items():
        if (info['valve'] == v and info['pump'] == p and
            info['accum'] == a and info['cooler'] == 100):
            X_with_start.append((X, info['start']))
            break
X_with_start.sort(key=lambda x: x[1])
X_ordered = [x[0] for x in X_with_start]
X_calib = set(X_ordered[:24])
X_test  = set(X_ordered[24:])
print(f"Calib X: {len(X_calib)}, Test X: {len(X_test)}")

# Load Chronos
print("Loading Chronos-Bolt-Small...")
pipe = BaseChronosPipeline.from_pretrained(
    'amazon/chronos-bolt-small', local_files_only=True,
    device_map='cpu', dtype=torch.float32,
)

# ============================================================
# 1. Build A/B/C windows
# ============================================================
print("Building windows...")

ctx_list_all = []  # unified list for prediction

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
    ctx_300 = tensor_s[ctx_indices, ts2_idx, :].flatten().astype(np.float32)
    ctx_tail = ctx_300[-60:].copy()
    in_calib = (X in X_calib)

    # A: fault
    for cooler_tgt, bf in [(3, b3), (20, b20)]:
        if bf is None: continue
        tgt = tensor_s[block_info[bf]['indices'], ts2_idx, :].astype(np.float32)
        ctx_list_all.append({
            'window_type': 'A', 'X': X, 'cooler_target': cooler_tgt,
            'ctx_300': ctx_300, 'ctx_tail': ctx_tail, 'tgt': tgt,
            'is_calib': in_calib,
            'ctx_start': block_info[b100]['start'],
            'tgt_start': block_info[bf]['start'],
        })

    # B: near healthy
    if block_info[b100]['n'] >= 10:
        tgt_indices = block_info[b100]['indices'][5:10]
        tgt = tensor_s[tgt_indices, ts2_idx, :].astype(np.float32)
        ctx_list_all.append({
            'window_type': 'B', 'X': X, 'cooler_target': 100,
            'ctx_300': ctx_300, 'ctx_tail': ctx_tail, 'tgt': tgt,
            'is_calib': in_calib,
            'ctx_start': block_info[b100]['start'],
            'tgt_start': block_info[b100]['start'],
        })

# C: far healthy
c100_blocks = sorted(
    [(b, info) for b, info in block_info.items() if info['cooler'] == 100],
    key=lambda x: x[1]['start'])
X_to_idx = {}
for idx, (b, info) in enumerate(c100_blocks):
    X_to_idx[(info['valve'], info['pump'], info['accum'])] = idx

for X_i in all_X:
    if X_i not in X_to_idx: continue
    i_idx = X_to_idx[X_i]
    best_j = None; best_dist = -1
    for X_j in all_X:
        if X_j not in X_to_idx: continue
        j_idx = X_to_idx[X_j]
        dist = abs(i_idx - j_idx)
        if dist > best_dist:
            best_dist = dist; best_j = j_idx
    if best_j is None: continue
    b_i = c100_blocks[i_idx][0]
    ctx_indices = block_info[b_i]['indices'][:5]
    ctx_300 = tensor_s[ctx_indices, ts2_idx, :].flatten().astype(np.float32)
    ctx_tail = ctx_300[-60:].copy()
    b_j = c100_blocks[best_j][0]
    tgt_indices = block_info[b_j]['indices'][:10]
    tgt = tensor_s[tgt_indices, ts2_idx, :].astype(np.float32)
    in_calib = (X_i in X_calib)
    ctx_list_all.append({
        'window_type': 'C', 'X': X_i, 'cooler_target': 100,
        'ctx_300': ctx_300, 'ctx_tail': ctx_tail, 'tgt': tgt,
        'is_calib': in_calib,
        'ctx_start': block_info[b_i]['start'],
        'tgt_start': block_info[b_j]['start'],
    })

# Count
nA = sum(1 for c in ctx_list_all if c['window_type'] == 'A')
nB = sum(1 for c in ctx_list_all if c['window_type'] == 'B')
nC = sum(1 for c in ctx_list_all if c['window_type'] == 'C')
print(f"Windows: A={nA} batches, B={nB} batches, C={nC} batches")

# ============================================================
# 2. Chronos predictions (batched)
# ============================================================
print("Running Chronos predictions...")
t0 = time.time()
ctx_batch = np.stack([c['ctx_300'] for c in ctx_list_all])
with torch.no_grad():
    result = pipe.predict_quantiles(
        torch.tensor(ctx_batch), prediction_length=60,
        quantile_levels=[0.1, 0.5, 0.9],
    )
q_all = result[0].numpy()  # (N, 60, 3)
print(f"Prediction time: {time.time()-t0:.2f}s for {len(ctx_list_all)} contexts")

# ============================================================
# 3. Compute deviation + calibrate thresholds (calib C only)
# ============================================================
print("Computing deviations + calibrating...")

# Collect calib C s_level/s_shape
calib_s_level = []; calib_s_shape = []
for i, c in enumerate(ctx_list_all):
    q10 = q_all[i, :, 0]; q50 = q_all[i, :, 1]; q90 = q_all[i, :, 2]
    tgt = c['tgt']; band_half = np.mean((q90 - q10) / 2.0)
    if c['window_type'] == 'C' and c['is_calib']:
        for ci in range(tgt.shape[0]):
            resid = tgt[ci] - q50
            sl = np.abs(np.mean(resid))
            ss = np.mean(np.abs(resid - np.mean(resid)))
            if band_half > 1e-10: ss /= band_half
            calib_s_level.append(sl); calib_s_shape.append(ss)

tau_level = np.percentile(calib_s_level, 95)
tau_shape = np.percentile(calib_s_shape, 95)
print(f"tau_level={tau_level:.4f}, tau_shape={tau_shape:.4f}")

# ============================================================
# 4. Classify all windows + collect per-cycle data
# ============================================================
print("Classifying all cycles...")

def detect_consecutive(outside):
    max_c = 0; cur = 0
    for o in outside:
        if o: cur += 1; max_c = max(max_c, cur)
        else: cur = 0
    return max_c

# Storage arrays
all_records = []
test_metrics = {'A_total': 0, 'A_danger': 0,
                'B_total': 0, 'B_danger': 0,
                'C_total': 0, 'C_danger': 0}

window_id = 0
for i, c in enumerate(ctx_list_all):
    q10 = q_all[i, :, 0]; q50 = q_all[i, :, 1]; q90 = q_all[i, :, 2]
    tgt = c['tgt']; band_half = np.mean((q90 - q10) / 2.0)
    n_cycles = tgt.shape[0]

    for ci in range(n_cycles):
        actual = tgt[ci]
        resid = actual - q50
        sl = np.abs(np.mean(resid))
        ss = np.mean(np.abs(resid - np.mean(resid)))
        if band_half > 1e-10: ss /= band_half

        # Decision: s_level/s_shape rule with calibrated thresholds
        danger = (sl > tau_level) or (ss > tau_shape)
        if danger: verdict = 'danger'
        else: verdict = 'normal'

        all_records.append({
            'window_id': window_id,
            'type': c['window_type'],
            'X_valve': int(c['X'][0]), 'X_pump': int(c['X'][1]), 'X_accum': int(c['X'][2]),
            'cooler_target': int(c['cooler_target']),
            'is_calib': bool(c['is_calib']),
            'ctx_tail': c['ctx_tail'].astype(np.float32),
            'q10': q10.astype(np.float32), 'q50': q50.astype(np.float32), 'q90': q90.astype(np.float32),
            'actual': actual.astype(np.float32),
            's_level': float(sl), 's_shape': float(ss),
            'verdict': verdict,
        })
        window_id += 1

        # Test metrics
        if not c['is_calib']:
            wtype = c['window_type']
            test_metrics[f'{wtype}_total'] += 1
            if verdict == 'danger': test_metrics[f'{wtype}_danger'] += 1

print(f"Total cycles recorded: {len(all_records)}")

# Report test metrics
a_det = test_metrics['A_danger'] / test_metrics['A_total'] if test_metrics['A_total'] > 0 else 0
b_fpr = test_metrics['B_danger'] / test_metrics['B_total'] if test_metrics['B_total'] > 0 else 0
c_fpr = test_metrics['C_danger'] / test_metrics['C_total'] if test_metrics['C_total'] > 0 else 0
print(f"Test metrics: A det={a_det:.4f} ({test_metrics['A_danger']}/{test_metrics['A_total']}), "
      f"B FPR={b_fpr:.4f} ({test_metrics['B_danger']}/{test_metrics['B_total']}), "
      f"C FPR={c_fpr:.4f} ({test_metrics['C_danger']}/{test_metrics['C_total']})")

# Compare with expected
print(f"Expected from previous run: A=1.0000, B=0.0000, C=0.0667")
for name, actual, expected in [('A det', a_det, 1.0), ('B FPR', b_fpr, 0.0), ('C FPR', c_fpr, 0.0667)]:
    diff = abs(actual - expected)
    if diff > 0.001:
        print(f"  DIFF: {name} actual={actual:.4f} expected={expected:.4f} diff={diff:.4f}")
    else:
        print(f"  OK: {name}={actual:.4f}")

# ============================================================
# 5. Pack into structured arrays
# ============================================================
print("Packing data...")
N = len(all_records)

# Fixed-length arrays
ctx_tail_arr  = np.array([r['ctx_tail'] for r in all_records], dtype=np.float32)
q10_arr       = np.array([r['q10'] for r in all_records], dtype=np.float32)
q50_arr       = np.array([r['q50'] for r in all_records], dtype=np.float32)
q90_arr       = np.array([r['q90'] for r in all_records], dtype=np.float32)
actual_arr    = np.array([r['actual'] for r in all_records], dtype=np.float32)
s_level_arr   = np.array([r['s_level'] for r in all_records], dtype=np.float32)
s_shape_arr   = np.array([r['s_shape'] for r in all_records], dtype=np.float32)

# Scalar arrays
window_id_arr = np.array([r['window_id'] for r in all_records], dtype=np.int32)
type_arr      = np.array([r['type'] for r in all_records], dtype='<U1')
X_valve_arr   = np.array([r['X_valve'] for r in all_records], dtype=np.int32)
X_pump_arr    = np.array([r['X_pump'] for r in all_records], dtype=np.int32)
X_accum_arr   = np.array([r['X_accum'] for r in all_records], dtype=np.int32)
cooler_arr    = np.array([r['cooler_target'] for r in all_records], dtype=np.int32)
is_calib_arr  = np.array([r['is_calib'] for r in all_records], dtype=bool)
verdict_arr   = np.array([r['verdict'] for r in all_records], dtype='<U7')

# ============================================================
# 6. Save
# ============================================================
os.makedirs('app_data', exist_ok=True)
out_path = 'app_data/demo_windows.npz'

np.savez_compressed(
    out_path,
    window_id=window_id_arr, type=type_arr,
    X_valve=X_valve_arr, X_pump=X_pump_arr, X_accum=X_accum_arr,
    cooler_target=cooler_arr, is_calib=is_calib_arr,
    ctx_tail=ctx_tail_arr, q10=q10_arr, q50=q50_arr, q90=q90_arr,
    actual=actual_arr, s_level=s_level_arr, s_shape=s_shape_arr,
    verdict=verdict_arr,
    tau_level=np.float32(tau_level), tau_shape=np.float32(tau_shape),
    a_detection=np.float32(a_det), b_fpr=np.float32(b_fpr), c_fpr=np.float32(c_fpr),
    nA=np.int32(test_metrics['A_total']), nB=np.int32(test_metrics['B_total']),
    nC=np.int32(test_metrics['C_total']),
)

size_mb = os.path.getsize(out_path) / 1024**2
print(f"\nSaved {out_path}: {size_mb:.2f} MB")

if size_mb > 25:
    print("Over 25MB, re-saving with float16...")
    # Re-save with float16 for large arrays
    np.savez_compressed(
        out_path,
        window_id=window_id_arr, type=type_arr,
        X_valve=X_valve_arr, X_pump=X_pump_arr, X_accum=X_accum_arr,
        cooler_target=cooler_arr, is_calib=is_calib_arr,
        ctx_tail=ctx_tail_arr.astype(np.float16),
        q10=q10_arr.astype(np.float16), q50=q50_arr.astype(np.float16),
        q90=q90_arr.astype(np.float16), actual=actual_arr.astype(np.float16),
        s_level=s_level_arr, s_shape=s_shape_arr,
        verdict=verdict_arr,
        tau_level=np.float32(tau_level), tau_shape=np.float32(tau_shape),
        a_detection=np.float32(a_det), b_fpr=np.float32(b_fpr), c_fpr=np.float32(c_fpr),
        nA=np.int32(test_metrics['A_total']), nB=np.int32(test_metrics['B_total']),
        nC=np.int32(test_metrics['C_total']),
        is_float16=np.bool_(True),
    )
    size_mb2 = os.path.getsize(out_path) / 1024**2
    print(f"Re-saved with float16: {size_mb2:.2f} MB")
else:
    print("Under 25MB, float32 is fine.")

# Self-check
print("\n--- Self-check ---")
verify = np.load(out_path, allow_pickle=True)
print(f"Keys: {list(verify.keys())}")
print(f"ctx_tail shape: {verify['ctx_tail'].shape}, dtype: {verify['ctx_tail'].dtype}")
print(f"tau_level: {verify['tau_level']}, tau_shape: {verify['tau_shape']}")
print(f"a_detection: {verify['a_detection']}, b_fpr: {verify['b_fpr']}, c_fpr: {verify['c_fpr']}")

# Verify verdict distribution
for wtype in ['A', 'B', 'C']:
    mask = verify['type'] == wtype
    vc = pd.Series(verify['verdict'][mask]).value_counts().to_dict()
    print(f"  {wtype} verdicts: {vc}")

print("\n[Task 1] DONE")
