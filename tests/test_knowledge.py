"""Regression tests for knowledge base numerical accuracy and attribution.

Strict baseline: only experimentally verified values are allowed as "project-measured".
"""

import os, re, yaml, sys

KNOWLEDGE_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge")

# === Strict baseline (single source of truth) ===
# These are the ONLY numbers that may be claimed as "项目实测"
MEASURED_COOLER = {
    18.6: "cooler=3 TS2 offset (cooler-02)",
    8.0:  "cooler=20 TS2 offset (cooler-01)",
    41.6: "cooler=100 TS2 baseline (system-01)",
}

MEASURED_VALVE = {
    0.197: "valve=90 deltaP (valve-01)",
    0.479: "valve=80 deltaP (valve-02)",
    0.826: "valve=73 deltaP (valve-03)",
}

# UCI official label values (not project-measured, but valid UCI-sourced)
UCI_LABELS = {
    3, 20, 100,           # cooler states
    73, 80, 90, 100,      # valve states (100 reused)
    0, 1, 2,               # pump leakage
    90, 100, 115, 130,    # accumulator
}

ALL_MEASURED = {**MEASURED_COOLER, **MEASURED_VALVE}


def find_numbers(body: str):
    """Extract (number, unit, context) from body text."""
    out = []
    for m in re.finditer(r"(约\s*)?(\d+\.?\d*)\s*(°C|bar|%)", body):
        num = float(m.group(2))
        ctx = body[max(0, m.start()-30):m.end()+30].replace("\n", " ").strip()
        out.append((num, m.group(2), ctx))
    return out


# === Load entries ===
entries = {}
for fname in sorted(os.listdir(KNOWLEDGE_DIR)):
    if not fname.endswith(".md"):
        continue
    with open(os.path.join(KNOWLEDGE_DIR, fname), encoding="utf-8") as f:
        content = f.read()
    parts = content.split("---", 2)
    fm = yaml.safe_load(parts[1]) if len(parts) >= 3 else {}
    body = parts[2] if len(parts) >= 3 else ""
    entries[fm["id"]] = {"fm": fm, "body": body, "file": fname}


# === Test 1: Core numerical assertions ===
print("=" * 60)
print("TEST 1: Core numerical assertions")
print("=" * 60)

checks = [
    ("cooler-01", "8.0", "cooler=20 TS2 offset ~8.0 C"),
    ("cooler-02", "18.6", "cooler=3 TS2 offset ~18.6 C"),
    ("valve-01", "0.197", "valve=90 deltaP ~0.197 bar"),
    ("valve-02", "0.479", "valve=80 deltaP ~0.479 bar"),
    ("valve-03", "0.826", "valve=73 deltaP ~0.826 bar"),
    ("system-01", "41.6", "cooler=100 TS2 baseline ~41.6 C"),
]

passed = 0
failed = 0
for eid, expected_val, desc in checks:
    body = entries[eid]["body"]
    if expected_val in body:
        print(f"  PASS: {eid} contains '{expected_val}' ({desc})")
        passed += 1
    else:
        print(f"  FAIL: {eid} does NOT contain '{expected_val}' ({desc})")
        failed += 1


# === Test 2: Strict attribution ===
print("\n" + "=" * 60)
print("TEST 2: Strict attribution (project-measured claims)")
print("=" * 60)

for eid, entry in entries.items():
    fm = entry["fm"]
    src = fm.get("source", "")
    body = entry["body"]
    is_measured = "实测" in src

    if not is_measured:
        continue

    nums = find_numbers(body)
    for num, raw, ctx in nums:
        # Check if this number is in the allowed measured list
        is_allowed = any(abs(num - a) < 0.015 for a in ALL_MEASURED)
        # UCI labels don't count as "project measured" — they're from the dataset
        is_uci_label = any(abs(num - u) < 0.01 for u in UCI_LABELS if u >= 10)

        if is_allowed:
            print(f"  PASS: {eid} claims '{num}' — in strict baseline ({ALL_MEASURED.get(int(num) if num == int(num) else num, '?')})")
            passed += 1
        elif is_uci_label:
            # UCI labels cited as project-measured is WRONG
            print(f"  FAIL: {eid} claims '{num}' as project-measured — this is a UCI label, NOT a project measurement")
            failed += 1
        else:
            print(f"  FAIL: {eid} claims '{num}' as project-measured — NOT in strict baseline")
            print(f"    context: \"{ctx}\"")
            failed += 1


# === Test 3: Mild-severity entries must not claim severe numbers ===
print("\n" + "=" * 60)
print("TEST 3: Mild entries free of severe/fault numbers")
print("=" * 60)

MILD_IDS = [eid for eid, entry in entries.items()
            if entry["fm"].get("severity") == "轻度"]
for eid in MILD_IDS:
    body = entries[eid]["body"]
    has_186 = "18.6" in body
    has_80 = "8.0" in body
    if has_186 or has_80:
        print(f"  FAIL: {eid} (severity=轻度) contains severe-condition number(s): "
              f"{'18.6' if has_186 else ''} {'8.0' if has_80 else ''}")
        failed += 1
    else:
        print(f"  PASS: {eid} is clean")
        passed += 1

# === Test 4: UCI-sourced entries (no project claims) ===
print("\n" + "=" * 60)
print("TEST 4: UCI-sourced entries (no project claims)")
print("=" * 60)

for eid in ["pump-01", "pump-02", "accum-01", "accum-02", "accum-03"]:
    fm = entries[eid]["fm"]
    src = fm.get("source", "")
    if "实测" in src:
        print(f"  FAIL: {eid} has project-measured claim but should be UCI-only")
        failed += 1
    else:
        print(f"  PASS: {eid} correctly uses UCI source")
        passed += 1


# === Summary ===
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 60)

# === Analysis for system-01 and cooler-03 ===
print("\n" + "=" * 60)
print("ANALYSIS FOR MANUAL REVIEW")
print("=" * 60)

print("\nsystem-01 numbers:")
for num, raw, ctx in find_numbers(entries["system-01"]["body"]):
    is_allowed = any(abs(num - a) < 0.015 for a in ALL_MEASURED)
    print(f"  {num}: allowed={is_allowed}")

print("\ncooler-03 numbers:")
for num, raw, ctx in find_numbers(entries["cooler-03"]["body"]):
    is_allowed = any(abs(num - a) < 0.015 for a in ALL_MEASURED)
    print(f"  {num}: allowed={is_allowed}")

sys.exit(0 if failed == 0 else 1)
