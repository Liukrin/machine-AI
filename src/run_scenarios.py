"""Run four scenarios through the agent graph."""
import os, sys, json, time
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from agent_graph import build_graph, AgentState, TAU_LEVEL, TAU_SHAPE
import numpy as np

DATA = np.load(
    os.path.join(os.path.dirname(__file__), "..", "app_data", "demo_windows.npz"),
    allow_pickle=True,
)
OUT_WO = os.path.join(os.path.dirname(__file__), "..", "output", "workorders")
OUT_RUNS = os.path.join(os.path.dirname(__file__), "..", "output", "runs")
os.makedirs(OUT_WO, exist_ok=True)
os.makedirs(OUT_RUNS, exist_ok=True)

graph = build_graph()


def run_scenario(label, initial_state):
    print(f"\n{'='*60}")
    print(f"SCENARIO {label}")
    if "window_id" in initial_state:
        wid = initial_state["window_id"]
        print(f"  window_id={wid}")
        # Show raw data
        sl = float(DATA["s_level"][wid])
        ss = float(DATA["s_shape"][wid])
        print(f"  s_level={sl:.4f}, s_shape={ss:.4f}")
        print(f"  cooler_target={int(DATA['cooler_target'][wid])}, "
              f"verdict={str(DATA['verdict'][wid])}")
    print(f"{'='*60}")

    t0 = time.time()
    result = graph.invoke(initial_state)
    elapsed = time.time() - t0

    status = result["status"]
    alert = result["alert_level"]
    print(f"  Path:        watcher", end="")
    if alert != "normal":
        print(f" -> diagnostician", end="")
        if result.get("retrieval_hit"):
            print(f" -> reporter -> verify -> END")
        else:
            print(f" -> END (rejected)")
    else:
        print(f" -> END (normal)")
    print(f"  Status:      {status}")
    print(f"  Alert level: {alert}")
    print(f"  Retrieval:   hit={result['retrieval_hit']}, "
          f"score={result.get('retrieval_score')}, "
          f"items={result.get('retrieved', [])}")
    print(f"  LLM called:  {bool(result.get('diagnosis_raw') and not result['diagnosis_raw'].startswith('[LLM'))}")
    print(f"  Elapsed:     {elapsed:.2f}s")

    # Save State JSON (always)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    state_serializable = {}
    for k, v in result.items():
        if isinstance(v, (str, int, float, bool, list, dict, type(None))):
            state_serializable[k] = v
        else:
            state_serializable[k] = str(v)
    run_path = os.path.join(OUT_RUNS, f"{label}_{ts}.json")
    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(state_serializable, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"  State saved: {run_path}")

    # Save .md only if status is completed (not normal, not rejected without output)
    if status == "completed" and result.get("work_order_md"):
        wo_path = os.path.join(OUT_WO, f"{label}_{wid}.md")
        with open(wo_path, "w", encoding="utf-8") as f:
            f.write(result["work_order_md"])
        print(f"  Work order:  {wo_path}")
    elif status == "rejected":
        wo_path = os.path.join(OUT_WO, f"{label}_rejected.md")
        with open(wo_path, "w", encoding="utf-8") as f:
            f.write(result.get("work_order_md", "## 诊断结论\n\n知识库无对应条目，需人工介入。"))
        print(f"  Work order:  {wo_path}")
    else:
        print(f"  Work order:  (skipped — status={status}, no .md for normal)")

    # Validate JSON
    if result.get("work_order_json"):
        try:
            json.loads(result["work_order_json"])
            print(f"  JSON valid:  OK")
        except Exception as e:
            print(f"  JSON valid:  FAIL — {e}")
    else:
        print(f"  JSON valid:  (no JSON for normal path)")

    return result


# ============================================================
print("=" * 60)
print("TASK 5: Four scenarios")
print("=" * 60)

# ---- Scenario A: fault (cooler=3) ----
mask_a = (DATA["type"] == "A") & (DATA["verdict"] == "danger") & (~DATA["is_calib"])
idx_a = int(np.where(mask_a)[0][0])
res_a = run_scenario("A_fault_cooler3", {
    "window_id": idx_a, "component": "冷却器",
    "alert_level": "", "s_level": 0.0, "s_shape": 0.0,
    "X_valve": 0, "X_pump": 0, "X_accum": 0, "cooler_target": 0,
    "verdict_original": "", "retrieval_hit": False,
    "retrieved": [], "retrieved_texts": [], "retrieval_score": None,
    "diagnosis_raw": "", "work_order_md": "", "work_order_json": "",
    "verification": {}, "status": "running",
})

# ---- Scenario B: normal ----
mask_b = (DATA["type"] == "B") & (DATA["verdict"] == "normal") & (~DATA["is_calib"])
idx_b = int(np.where(mask_b)[0][0])
res_b = run_scenario("B_normal", {
    "window_id": idx_b, "component": "冷却器",
    "alert_level": "", "s_level": 0.0, "s_shape": 0.0,
    "X_valve": 0, "X_pump": 0, "X_accum": 0, "cooler_target": 0,
    "verdict_original": "", "retrieval_hit": False,
    "retrieved": [], "retrieved_texts": [], "retrieval_score": None,
    "diagnosis_raw": "", "work_order_md": "", "work_order_json": "",
    "verification": {}, "status": "running",
})

# ---- Scenario C: fault (cooler=20) ----
mask_c = (DATA["type"] == "A") & (DATA["verdict"] == "danger") & (~DATA["is_calib"])
# Find one with cooler_target=20
for i in np.where(mask_c)[0]:
    if int(DATA["cooler_target"][i]) == 20:
        idx_c = int(i)
        break
else:
    idx_c = int(np.where(mask_c)[0][10])
res_c = run_scenario("C_fault_cooler20", {
    "window_id": idx_c, "component": "冷却器",
    "alert_level": "", "s_level": 0.0, "s_shape": 0.0,
    "X_valve": 0, "X_pump": 0, "X_accum": 0, "cooler_target": 0,
    "verdict_original": "", "retrieval_hit": False,
    "retrieved": [], "retrieved_texts": [], "retrieval_score": None,
    "diagnosis_raw": "", "work_order_md": "", "work_order_json": "",
    "verification": {}, "status": "running",
})

# ---- Scenario D: rejection (component not in KB) ----
# Go directly to Diagnostician, bypassing Watcher
from agent_graph import diagnostician
state_d = {
    "window_id": -1,  # synthetic, not from demo data
    "component": "齿轮箱",
    "alert_level": "danger",  # force past Watcher
    "s_level": 15.0,
    "s_shape": 0.5,
    "X_valve": 73, "X_pump": 0, "X_accum": 90, "cooler_target": 3,
    "verdict_original": "synthetic",
    "retrieval_hit": False, "retrieved": [], "retrieved_texts": [],
    "retrieval_score": None, "diagnosis_raw": "",
    "work_order_md": "", "work_order_json": "",
    "verification": {}, "status": "running",
}
print(f"\n{'='*60}")
print(f"SCENARIO D_reject_gearbox")
print(f"  component='齿轮箱' (not in knowledge base)")
print(f"{'='*60}")
t0 = time.time()
result_d = diagnostician(state_d)
elapsed = time.time() - t0
print(f"  Path:        watcher(skipped) -> diagnostician -> END (rejected)")
print(f"  Status:      {result_d['status']}")
print(f"  Retrieval:   hit={result_d['retrieval_hit']}, "
      f"score={result_d.get('retrieval_score')}, items={result_d.get('retrieved', [])}")
print(f"  LLM called:  False")
print(f"  Elapsed:     {elapsed:.2f}s")
# Save state
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
state_d_ser = {k: (str(v) if not isinstance(v, (str,int,float,bool,list,dict,type(None))) else v)
               for k, v in result_d.items()}
run_path_d = os.path.join(OUT_RUNS, f"D_reject_gearbox_{ts}.json")
with open(run_path_d, "w", encoding="utf-8") as f:
    json.dump(state_d_ser, f, ensure_ascii=False, indent=2, allow_nan=False)
print(f"  State saved: {run_path_d}")
if result_d.get("work_order_md"):
    wo_path_d = os.path.join(OUT_WO, f"D_reject_gearbox.md")
    with open(wo_path_d, "w", encoding="utf-8") as f:
        f.write(result_d["work_order_md"])
    print(f"  Work order:  {wo_path_d}")
if result_d.get("work_order_json"):
    try:
        json.loads(result_d["work_order_json"])
        print(f"  JSON valid:  OK")
    except Exception as e:
        print(f"  JSON valid:  FAIL — {e}")
res_d = result_d

# Summary
print(f"\n{'='*60}")
print("SCENARIO SUMMARY")
print(f"{'='*60}")
for label, res in [("A", res_a), ("B", res_b), ("C", res_c), ("D", res_d)]:
    llm_called = bool(res.get('diagnosis_raw') and not str(res['diagnosis_raw']).startswith('[LLM'))
    print(f"  {label}: status={res['status']}, alert={res.get('alert_level','?')}, "
          f"retrieval={res['retrieval_hit']}, llm={llm_called}, "
          f"top_kb={res.get('retrieved', [])[:2]}")
