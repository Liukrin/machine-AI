"""Unit tests for verify_citation — no LLM, deterministic assertions."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from agent_graph import verify_citation

# Reference text: cooler-02 body (severe cooler failure)
COOLER_02_TEXT = (
    "冷却效率 CE 降至正常水平的 3%，接近完全失效。"
    "TS2 通道较健康态（cooler=100）升高约 18.6°C"
    "（本项目阶段一实测），"
    "四个温度传感器均指示系"
    "统处于高温状态。"
    "立即停机降温，拆检冷却"
    "器芯体，更换破裂管束或"
    "整体更换冷却器。"
    "检查冷却水供水系统全链"
    "路，修复中断点并确认压"
    "力恢复。"
)

COOLER_02_TEXT = """冷却效率 CE 降至正常水平的 3%，接近完全失效。TS2 通道较健康态（cooler=100）升高约 18.6�C（本项目阶段一实测），四个温度传感器均指示系统处于高温状态。立即停机降温，拆检冷却器芯体，更换破裂管束或整体更换冷却器。检查冷却水供水系统全链路，修复中断点并确认压力恢复。更换或修复风扇驱动装置，确认散热风量符合设计值。强制切换或更换旁通阀，确保热油全部流经冷却器。"""

EXTRA = "s_level:20.06 s_shape:2.5267 tau_level:0.9884 tau_shape:1.3531 cooler_target:3 alert_level:danger"

results = []

# === Case 1: Viscosity hallucination → should be suspicious ===
case1 = "压力波形畸变可能由高温导致油液黏度下降引起"
v1 = verify_citation(case1, [COOLER_02_TEXT])
flagged1 = v1.get("suspicious_count", 0) > 0
results.append(("Case 1: viscosity hallucination", flagged1, True, v1))

# === Case 2: Pure fabrication → should be suspicious ===
case2 = "建议每 500 小时更换液压油滤芯"
v2 = verify_citation(case2, [COOLER_02_TEXT])
flagged2 = v2.get("suspicious_count", 0) > 0
results.append(("Case 2: pure fabrication", flagged2, True, v2))

# === Case 3: Verbatim quote → should be clean ===
case3 = "立即停机降温，拆检冷却器芯体，更换破裂管束或整体更换冷却器"
v3 = verify_citation(case3, [COOLER_02_TEXT])
flagged3 = v3.get("suspicious_count", 0) > 0
results.append(("Case 3: verbatim quote", flagged3, False, v3))

# === Case 4: Measured value from Watcher → should be clean ===
case4 = "温度基线偏移 20.1°C，符合冷却器严重失效特征"
v4 = verify_citation(case4, [COOLER_02_TEXT], extra_grounding=EXTRA)
flagged4 = v4.get("suspicious_count", 0) > 0
results.append(("Case 4: measured value with extra_grounding", flagged4, False, v4))

# === Case 5: Heading line → should be skipped by sentence splitter ===
case5 = "建议处置措施："
v5 = verify_citation(case5, [COOLER_02_TEXT])
flagged5 = v5.get("suspicious_count", 0) > 0
results.append(("Case 5: heading line", flagged5, False, v5))

# Print results
print("=" * 60)
print("VERIFICATION UNIT TESTS")
print("=" * 60)
passed = 0
failed = 0
for name, actual, expected, v in results:
    ok = (actual == expected)
    if ok:
        passed += 1
    else:
        failed += 1
    status = "PASS" if ok else "FAIL"
    print(f"\n[{status}] {name}")
    print(f"  expected={'suspicious' if expected else 'clean'} actual={'suspicious' if actual else 'clean'}")
    if not ok:
        print(f"  sentences checked: {v.get('total_sentences')}")
        print(f"  suspicious: {v.get('suspicious')}")

print(f"\n{'='*60}")
print(f"RESULTS: {passed} passed, {failed} failed")
assert failed == 0, f"{failed} test(s) FAILED"
