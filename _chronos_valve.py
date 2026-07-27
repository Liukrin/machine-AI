import numpy as np
import torch, time, os, warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from chronos import BaseChronosPipeline

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

# Channel indices by name
ch_idx = {}
for ch_name in ['ps1_mean', 'ps1_ptp']:
    idx = ch_names.index(ch_name)
    ch_idx[ch_name] = idx
    print(f"    Channel '{ch_name}' -> index {idx}")

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

cooler_vals = [3, 20, 100]
pump_vals   = [0, 1, 2]
accum_vals  = [90, 100, 115, 130]
all_X = [(c, p, a) for c in cooler_vals for p in pump_vals for a in accum_vals]
print(f"    Total X triples: {len(all_X)} (3x3x4=36)")

# Order X by stable_idx of their valve=100 block
X_with_start = []
for X in all_X:
    c, p, a = X
    for b, info in block_info.items():
        if info['cooler'] == c and info['pump'] == p and info['accum'] == a and info['valve'] == 100:
            X_with_start.append((X, info['start']))
            break
X_with_start.sort(key=lambda x: x[1])
X_ordered = [x[0] for x in X_with_start]
print(f"    {len(X_ordered)} X triples ordered by stable_idx")

# Load Chronos
print("[0] Loading Chronos-Bolt-Small...")
pipe = BaseChronosPipeline.from_pretrained(
    'amazon/chronos-bolt-small', local_files_only=True,
    device_map='cpu', dtype=torch.float32,
)

# ============================================================
# 1. Pre-checks
# ============================================================
print("\n" + "="*60)
print("PRE-CHECK 1: X completeness")
print("="*60)

missing = []
for X in all_X:
    c, p, a = X
    found = {100: False, 73: False, 80: False, 90: False}
    for b, info in block_info.items():
        if info['cooler'] == c and info['pump'] == p and info['accum'] == a:
            found[info['valve']] = True
    for v in [100, 73, 80, 90]:
        if not found[v]:
            missing.append((X, v))
if missing:
    print(f"  MISSING: {len(missing)} entries")
    for m in missing:
        print(f"    X={m[0]}, missing valve={m[1]}")
else:
    print(f"  All 36 X have valve=100,73,80,90 blocks. OK.")

# Gap check
print("\n" + "="*60)
print("PRE-CHECK 2: stable_idx gap (context block -> target block)")
print("="*60)

gaps_all = []
for X in all_X:
    c, p, a = X
    b100 = None; b_faults = {}
    for b, info in block_info.items():
        if info['cooler'] == c and info['pump'] == p and info['accum'] == a:
            if info['valve'] == 100: b100 = b
            else: b_faults[info['valve']] = b
    if b100 is None: continue
    ctx_start = block_info[b100]['start']
    for vf, bf in b_faults.items():
        tgt_start = block_info[bf]['start']
        gaps_all.append(tgt_start - ctx_start)
gaps_all = np.array(gaps_all)
print(f"  min={gaps_all.min()}, mean={gaps_all.mean():.1f}, max={gaps_all.max()}")
print(f"  |gap| stats: min={abs(gaps_all).min():.0f}, mean={abs(gaps_all).mean():.0f}, max={abs(gaps_all).max():.0f}")
print(f"  Much smaller than cooler round (480-960): {abs(gaps_all).max() < 200}")

# Healthy block drift
print("\n" + "="*60)
print("PRE-CHECK 3: Healthy block drift (36 valve=100 blocks)")
print("="*60)

for ch_name in ['ps1_mean', 'ps1_ptp']:
    ci = ch_idx[ch_name]
    means = []
    for X in X_ordered:
        c, p, a = X
        for b, info in block_info.items():
            if info['cooler'] == c and info['pump'] == p and info['accum'] == a and info['valve'] == 100:
                ts = tensor_s[info['indices'], ci, :]
                means.append(ts.mean())
                break
    means = np.array(means)
    print(f"  {ch_name}: min={means.min():.4f}, max={means.max():.4f}, range={means.max()-means.min():.4f}")

# ============================================================
# 2. Build windows (A, B, C) for BOTH channels
# ============================================================
print("\n[2] Building windows...")

def build_windows_for_channel(ch_name):
    ci = ch_idx[ch_name]
    ctx_A = []; ctx_B = []; ctx_C = []

    for X in all_X:
        c, p, a = X
        b100 = None; b_faults = {}
        for b, info in block_info.items():
            if info['cooler'] == c and info['pump'] == p and info['accum'] == a:
                if info['valve'] == 100: b100 = b
                else: b_faults[info['valve']] = b
        if b100 is None: continue

        ctx_indices = block_info[b100]['indices'][:5]
        ctx_ts2 = tensor_s[ctx_indices, ci, :].flatten().astype(np.float32)

        # A: fault targets
        for vf in [73, 80, 90]:
            if vf in b_faults:
                bf = b_faults[vf]
                tgt = tensor_s[block_info[bf]['indices'], ci, :].astype(np.float32)
                ctx_A.append({'ctx': ctx_ts2, 'tgt': tgt, 'valve_target': vf, 'X': X,
                              'ctx_start': block_info[b100]['start'],
                              'tgt_start': block_info[bf]['start']})

        # B: near healthy (cycles 6-10)
        if block_info[b100]['n'] >= 10:
            tgt_indices = block_info[b100]['indices'][5:10]
            tgt = tensor_s[tgt_indices, ci, :].astype(np.float32)
            ctx_B.append({'ctx': ctx_ts2, 'tgt': tgt, 'X': X,
                          'ctx_start': block_info[b100]['start'],
                          'tgt_start': block_info[b100]['start']})

    # C: far healthy
    v100_blocks = sorted(
        [(b, info) for b, info in block_info.items() if info['valve'] == 100],
        key=lambda x: x[1]['start']
    )
    X_to_idx = {}
    for idx, (b, info) in enumerate(v100_blocks):
        X_to_idx[(info['cooler'], info['pump'], info['accum'])] = idx

    # Precompute cooler-level of each v100 block for C-crossing check
    v100_cooler = [info['cooler'] for b, info in v100_blocks]

    n_cross_cooler = 0
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
        b_i = v100_blocks[i_idx][0]
        ctx_indices = block_info[b_i]['indices'][:5]
        ctx_ts2 = tensor_s[ctx_indices, ci, :].flatten().astype(np.float32)
        b_j = v100_blocks[best_j][0]
        tgt_indices = block_info[b_j]['indices'][:10]
        tgt = tensor_s[tgt_indices, ci, :].astype(np.float32)

        # Check if crossing cooler level
        if v100_cooler[i_idx] != v100_cooler[best_j]:
            n_cross_cooler += 1

        ctx_C.append({'ctx': ctx_ts2, 'tgt': tgt, 'X_i': X_i,
                      'X_j': (v100_blocks[best_j][1]['cooler'], v100_blocks[best_j][1]['pump'],
                              v100_blocks[best_j][1]['accum']),
                      'ctx_start': block_info[b_i]['start'],
                      'tgt_start': block_info[b_j]['start']})

    return ctx_A, ctx_B, ctx_C, n_cross_cooler

results = {}
for ch_name in ['ps1_mean', 'ps1_ptp']:
    ctx_A, ctx_B, ctx_C, n_cross = build_windows_for_channel(ch_name)
    results[ch_name] = {'A': ctx_A, 'B': ctx_B, 'C': ctx_C, 'n_cross': n_cross}
    print(f"  {ch_name}: A={len(ctx_A)}, B={len(ctx_B)}, C={len(ctx_C)}, "
          f"C-cross-cooler={n_cross}/36")

# ============================================================
# 3. Chronos predictions
# ============================================================
print("\n[3] Chronos predictions...")

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

for ch_name in ['ps1_mean', 'ps1_ptp']:
    t0 = time.time()
    for wtype in ['A', 'B', 'C']:
        preds = chronos_predict_all(results[ch_name][wtype])
        for i, p in enumerate(preds):
            results[ch_name][wtype][i]['pred'] = p
    print(f"  {ch_name}: {time.time()-t0:.2f}s")

# ============================================================
# 4. Deviation decomposition + threshold calibration
# ============================================================
print("\n[4] Threshold calibration (first 18 X calib, last 18 X test)...")

X_calib = set(X_ordered[:18])
X_test  = set(X_ordered[18:])
print(f"  Calib X: {len(X_calib)}, Test X: {len(X_test)}, Overlap: {len(X_calib & X_test)}")

def compute_deviations(ctx_list):
    """Flat list of (s_level, s_shape) per target cycle."""
    out = []
    for c in ctx_list:
        q10, q50, q90 = c['pred']
        tgt = c['tgt']
        band_half = np.mean((q90 - q10) / 2.0)
        for ci in range(tgt.shape[0]):
            resid = tgt[ci] - q50
            sl = np.abs(np.mean(resid))
            ss = np.mean(np.abs(resid - np.mean(resid)))
            if band_half > 1e-10: ss = ss / band_half
            out.append({'s_level': sl, 's_shape': ss})
    return out

def calibrate(ctx_C_list):
    """Calibrate tau_level, tau_shape from C-class windows of calib X."""
    s_lev = []; s_shp = []
    for c in ctx_C_list:
        X_i = c['X_i']
        if X_i not in X_calib: continue
        q10, q50, q90 = c['pred']
        tgt = c['tgt']
        band_half = np.mean((q90 - q10) / 2.0)
        for ci in range(tgt.shape[0]):
            resid = tgt[ci] - q50
            sl = np.abs(np.mean(resid))
            ss = np.mean(np.abs(resid - np.mean(resid)))
            if band_half > 1e-10: ss = ss / band_half
            s_lev.append(sl); s_shp.append(ss)
    s_lev = np.array(s_lev); s_shp = np.array(s_shp)
    return np.percentile(s_lev, 95), np.percentile(s_shp, 95)

thresholds = {}
for ch_name in ['ps1_mean', 'ps1_ptp']:
    tl, ts = calibrate(results[ch_name]['C'])
    thresholds[ch_name] = (tl, ts)
    print(f"  {ch_name}: tau_level={tl:.4f}, tau_shape={ts:.4f}")

# ============================================================
# 5. Evaluate on test X
# ============================================================
print("\n[5] Evaluation on test set...")

def classify_test(ctx_entry, tau_level, tau_shape):
    q10, q50, q90 = ctx_entry['pred']
    tgt = ctx_entry['tgt']
    band_half = np.mean((q90 - q10) / 2.0)
    results_per_cycle = []
    for ci in range(tgt.shape[0]):
        resid = tgt[ci] - q50
        sl = np.abs(np.mean(resid))
        ss = np.mean(np.abs(resid - np.mean(resid)))
        if band_half > 1e-10: ss = ss / band_half
        over_level = sl > tau_level
        over_shape = ss > tau_shape
        danger = over_level or over_shape
        results_per_cycle.append({
            'danger': danger, 's_level': sl, 's_shape': ss,
            'only_level': over_level and not over_shape,
            'only_shape': over_shape and not over_level,
            'both': over_level and over_shape,
        })
    return results_per_cycle

def eval_on_test(ctx_list, tau_level, tau_shape, filter_fn=None):
    all_res = []
    for c in ctx_list:
        X_key = c.get('X_i', c.get('X', None))
        if X_key is None: X_key = c['X']
        if X_key not in X_test: continue
        if filter_fn and not filter_fn(c): continue
        cyc = classify_test(c, tau_level, tau_shape)
        all_res.extend(cyc)
    return all_res

for ch_name in ['ps1_mean', 'ps1_ptp']:
    tl, ts = thresholds[ch_name]
    print(f"\n{'='*60}")
    print(f"  Channel: {ch_name}")
    print(f"{'='*60}")

    # A: by valve value
    for vf in [73, 80, 90]:
        res_A = eval_on_test(results[ch_name]['A'], tl, ts,
                            filter_fn=lambda c, v=vf: c['valve_target'] == v)
        n = len(res_A)
        n_d = sum(r['danger'] for r in res_A)
        print(f"  A (valve={vf}): danger={n_d}/{n} = {n_d/n:.4f}" if n > 0 else f"  A (valve={vf}): NO DATA")

    res_A_all = eval_on_test(results[ch_name]['A'], tl, ts)
    n_A = len(res_A_all); n_A_d = sum(r['danger'] for r in res_A_all)
    print(f"  A (combined): danger={n_A_d}/{n_A} = {n_A_d/n_A:.4f}" if n_A > 0 else "  A: NO DATA")

    res_B = eval_on_test(results[ch_name]['B'], tl, ts)
    n_B = len(res_B); n_B_d = sum(r['danger'] for r in res_B)
    print(f"  B (Near):     danger={n_B_d}/{n_B} = {n_B_d/n_B:.4f}" if n_B > 0 else "  B: NO DATA")

    res_C = eval_on_test(results[ch_name]['C'], tl, ts)
    n_C = len(res_C); n_C_d = sum(r['danger'] for r in res_C)
    print(f"  C (Far):      danger={n_C_d}/{n_C} = {n_C_d/n_C:.4f}" if n_C > 0 else "  C: NO DATA")

    # s_level / s_shape stats
    for label, res_list in [('A', res_A_all), ('B', res_B), ('C', res_C)]:
        sl = np.array([r['s_level'] for r in res_list])
        ss = np.array([r['s_shape'] for r in res_list])
        if len(sl) > 0:
            print(f"  {label}: s_level min={sl.min():.4f} median={np.median(sl):.4f} max={sl.max():.4f}")
            print(f"  {label}: s_shape min={ss.min():.4f} median={np.median(ss):.4f} max={ss.max():.4f}")

    # s_shape contribution breakdown for A
    only_level = sum(r['only_level'] for r in res_A_all)
    only_shape = sum(r['only_shape'] for r in res_A_all)
    both = sum(r['both'] for r in res_A_all)
    print(f"  A breakdown: only_level={only_level}/{n_A_d}={only_level/n_A_d:.4f} "
          f"only_shape={only_shape}/{n_A_d}={only_shape/n_A_d:.4f} "
          f"both={both}/{n_A_d}={both/n_A_d:.4f}" if n_A_d > 0 else "  A breakdown: N/A")
    print(f"  => s_shape contribution beyond s_level: "
          f"{'YES' if only_shape > 0 else 'NONE'} "
          f"({only_shape}/{n_A_d} danger cycles from s_shape alone)" if n_A_d > 0 else "")

# ============================================================
# 6. Three-way comparison
# ============================================================
print("\n" + "="*60)
print("STEP 6: Three-way comparison (Chronos vs Naive vs Trivial)")
print("="*60)

for ch_name in ['ps1_mean', 'ps1_ptp']:
    print(f"\n--- {ch_name} ---")
    tl, ts = thresholds[ch_name]

    # Naive: calibrate on C-class calib
    def calibrate_naive(ctx_C_list):
        s_lev = []; s_shp = []
        for c in ctx_C_list:
            X_i = c['X_i']
            if X_i not in X_calib: continue
            last_cycle = c['ctx'][-60:]
            std_last = np.std(last_cycle)
            tgt = c['tgt']
            band_half = std_last
            for ci in range(tgt.shape[0]):
                resid = tgt[ci] - last_cycle
                sl = np.abs(np.mean(resid))
                ss = np.mean(np.abs(resid - np.mean(resid)))
                if band_half > 1e-10: ss = ss / band_half
                s_lev.append(sl); s_shp.append(ss)
        s_lev = np.array(s_lev); s_shp = np.array(s_shp)
        return np.percentile(s_lev, 95), np.percentile(s_shp, 95)

    tl_n, ts_n = calibrate_naive(results[ch_name]['C'])

    def eval_naive(ctx_list, tl_n, ts_n, filter_fn=None):
        n_tot = 0; n_d = 0
        for c in ctx_list:
            X_key = c.get('X_i', c.get('X', None))
            if X_key is None: X_key = c['X']
            if X_key not in X_test: continue
            if filter_fn and not filter_fn(c): continue
            last_cycle = c['ctx'][-60:]
            std_last = np.std(last_cycle)
            tgt = c['tgt']
            band_half = std_last
            for ci in range(tgt.shape[0]):
                resid = tgt[ci] - last_cycle
                sl = np.abs(np.mean(resid))
                ss = np.mean(np.abs(resid - np.mean(resid)))
                if band_half > 1e-10: ss = ss / band_half
                if (sl > tl_n) or (ss > ts_n): n_d += 1
                n_tot += 1
        return n_d / n_tot if n_tot > 0 else 0

    # Trivial: calibrate on C-class calib
    def calibrate_trivial(ctx_C_list):
        s_lev = []; s_shp = []
        for c in ctx_C_list:
            X_i = c['X_i']
            if X_i not in X_calib: continue
            last_cycle = c['ctx'][-60:]
            q10, q50, q90 = c['pred']
            tgt = c['tgt']
            band_half = np.mean((q90 - q10) / 2.0)
            for ci in range(tgt.shape[0]):
                resid_t = tgt[ci] - last_cycle
                sl = np.abs(np.mean(resid_t))
                resid_chr = tgt[ci] - q50
                ss = np.mean(np.abs(resid_chr - np.mean(resid_chr)))
                if band_half > 1e-10: ss = ss / band_half
                s_lev.append(sl); s_shp.append(ss)
        s_lev = np.array(s_lev); s_shp = np.array(s_shp)
        return np.percentile(s_lev, 95), np.percentile(s_shp, 95)

    tl_t, ts_t = calibrate_trivial(results[ch_name]['C'])

    def eval_trivial(ctx_list, tl_t, ts_t, filter_fn=None):
        n_tot = 0; n_d = 0
        for c in ctx_list:
            X_key = c.get('X_i', c.get('X', None))
            if X_key is None: X_key = c['X']
            if X_key not in X_test: continue
            if filter_fn and not filter_fn(c): continue
            last_cycle = c['ctx'][-60:]
            q10, q50, q90 = c['pred']
            tgt = c['tgt']
            band_half = np.mean((q90 - q10) / 2.0)
            for ci in range(tgt.shape[0]):
                resid_t = tgt[ci] - last_cycle
                sl = np.abs(np.mean(resid_t))
                resid_chr = tgt[ci] - q50
                ss = np.mean(np.abs(resid_chr - np.mean(resid_chr)))
                if band_half > 1e-10: ss = ss / band_half
                if (sl > tl_t) or (ss > ts_t): n_d += 1
                n_tot += 1
        return n_d / n_tot if n_tot > 0 else 0

    # Chronos results on test
    res_A_chr = eval_on_test(results[ch_name]['A'], tl, ts)
    chronos_A = sum(r['danger'] for r in res_A_chr) / len(res_A_chr)
    res_B_chr = eval_on_test(results[ch_name]['B'], tl, ts)
    chronos_B = sum(r['danger'] for r in res_B_chr) / len(res_B_chr)
    res_C_chr = eval_on_test(results[ch_name]['C'], tl, ts)
    chronos_C = sum(r['danger'] for r in res_C_chr) / len(res_C_chr)

    na_A = eval_naive(results[ch_name]['A'], tl_n, ts_n)
    na_B = eval_naive(results[ch_name]['B'], tl_n, ts_n)
    na_C = eval_naive(results[ch_name]['C'], tl_n, ts_n)
    tr_A = eval_trivial(results[ch_name]['A'], tl_t, ts_t)
    tr_B = eval_trivial(results[ch_name]['B'], tl_t, ts_t)
    tr_C = eval_trivial(results[ch_name]['C'], tl_t, ts_t)

    print(f"  {'Method':<25} {'A Det':>10} {'B FPR':>10} {'C FPR':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Chronos decomp':<25} {chronos_A:>10.4f} {chronos_B:>10.4f} {chronos_C:>10.4f}")
    print(f"  {'Naive last-value':<25} {na_A:>10.4f} {na_B:>10.4f} {na_C:>10.4f}")
    print(f"  {'Trivial mean':<25} {tr_A:>10.4f} {tr_B:>10.4f} {tr_C:>10.4f}")

    # Correlation: s_level vs s_trivial
    s_level_vals = []; s_trivial_vals = []
    q50_means = []; ctx_last_means = []
    for c in results[ch_name]['C']:
        X_i = c['X_i']
        if X_i not in X_test: continue
        last_cycle = c['ctx'][-60:]
        q50 = c['pred'][1]
        tgt = c['tgt']
        for ci in range(tgt.shape[0]):
            resid = tgt[ci] - q50
            sl = np.abs(np.mean(resid))
            resid_t = tgt[ci] - last_cycle
            st = np.abs(np.mean(resid_t))
            s_level_vals.append(sl); s_trivial_vals.append(st)
            q50_means.append(np.mean(q50))
            ctx_last_means.append(np.mean(last_cycle))

    s_level_vals = np.array(s_level_vals); s_trivial_vals = np.array(s_trivial_vals)
    corr = np.corrcoef(s_level_vals, s_trivial_vals)[0, 1]
    print(f"  Pearson r(s_level, s_trivial): {corr:.4f}")

    q50_means = np.array(q50_means); ctx_last_means = np.array(ctx_last_means)
    mad = np.mean(np.abs(q50_means - ctx_last_means))
    print(f"  MAD(mean(q50), mean(ctx_last_cycle)): {mad:.6f}")
    print(f"  => Chronos q50 degenerates to flat extrapolation: "
          f"{'YES' if mad < 0.01 else 'NO'} (MAD={mad:.6f})")

# ============================================================
# 7. Plots
# ============================================================
print("\n[7] Generating plots...")
os.makedirs('reports/figures', exist_ok=True)

for ch_name in ['ps1_mean', 'ps1_ptp']:
    tl, ts = thresholds[ch_name]

    # Scatter
    all_pts = []
    for label, ctx_list in [('A', results[ch_name]['A']), ('B', results[ch_name]['B']),
                             ('C', results[ch_name]['C'])]:
        for c in ctx_list:
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
                all_pts.append({'s_level': sl, 's_shape': ss, 'type': label})

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = {'A': '#d73027', 'B': '#4575b4', 'C': '#fc8d59'}
    for label in ['A', 'B', 'C']:
        pts = [p for p in all_pts if p['type'] == label]
        sl = [p['s_level'] for p in pts]
        ss = [p['s_shape'] for p in pts]
        ax.scatter(sl, ss, c=colors[label], alpha=0.4, s=8,
                   label=f'{label} ({len(pts)})', rasterized=True)

    ax.axvline(x=tl, color='black', linestyle='--', linewidth=1.2, alpha=0.7,
               label=f'tau_level={tl:.3f}')
    ax.axhline(y=ts, color='gray', linestyle='--', linewidth=1.2, alpha=0.7,
               label=f'tau_shape={ts:.3f}')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('s_level = |mean(actual − q50)|', fontsize=12)
    ax.set_ylabel('s_shape = mean(|resid − mean(resid)|) / mean((q90−q10)/2)', fontsize=12)
    ax.set_title(f'{ch_name} — Deviation Decomposition (Test Set, 18 X)', fontsize=13)
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()
    fname = f'reports/figures/valve_scatter_{ch_name}.png'
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"    Saved {fname}")

# Waveform overlay: valve=100 vs valve=73, ps1_mean
print("    Generating waveform overlay...")
# Find first test X
plot_X = None
for X in X_ordered:
    if X in X_test:
        plot_X = X; break
if plot_X:
    c, p, a = plot_X
    ci = ch_idx['ps1_mean']
    b100 = b73 = None
    for b, info in block_info.items():
        if info['cooler'] == c and info['pump'] == p and info['accum'] == a:
            if info['valve'] == 100: b100 = b
            elif info['valve'] == 73: b73 = b
    if b100 and b73:
        idx100 = block_info[b100]['indices'][:10]
        idx73 = block_info[b73]['indices'][:10]
        ts100 = tensor_s[idx100, ci, :]
        ts73 = tensor_s[idx73, ci, :]
        fig, ax = plt.subplots(figsize=(10, 5))
        time_axis = np.arange(60)
        for i in range(len(ts100)):
            ax.plot(time_axis, ts100[i], color='#4575b4', alpha=0.6, linewidth=1.0)
        for i in range(len(ts73)):
            ax.plot(time_axis, ts73[i], color='#d73027', alpha=0.6, linewidth=1.0)
        legend_elements = [
            Line2D([0], [0], color='#4575b4', lw=2, label='valve=100 (healthy)'),
            Line2D([0], [0], color='#d73027', lw=2, label='valve=73 (fault)'),
        ]
        ax.legend(handles=legend_elements, fontsize=10)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('PS1 Mean Pressure (bar)', fontsize=12)
        ax.set_title(f'PS1 Mean: valve=100 vs valve=73  |  '
                     f'cooler={c}, pump_leak={p}, accum={a}bar', fontsize=13)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig('reports/figures/valve_waveform_ps1_mean.png', dpi=150)
        plt.close(fig)
        print(f"    Saved reports/figures/valve_waveform_ps1_mean.png")

print("\n" + "="*60)
print("DONE")
print("="*60)
