"""Deterministic severity mapping — pure Python, no LLM.

Thresholds are midpoint values from Phase 2 measurements:
  cooler=3  ~ 18.6 C s_level
  cooler=20 ~  8.0 C s_level (approx)
  healthy   ~  0.0 C s_level

Split points: (0+8)/2=4, (8+18.6)/2=13.3 -> rounded to 3 and 14 for clean boundaries.
These are reproducible, auditable rules. No model inference involved.
"""


def map_severity(component: str, s_level: float) -> str | None:
    """Return severity level for a given component and s_level value.

    Returns None for components without calibrated thresholds.
    Accepts both English ("cooler") and Chinese ("冷却器") component names.
    """
    comp_lower = component.lower()
    if comp_lower in ("cooler", "冷却器"):
        if s_level < 3:
            return "轻度"
        elif s_level < 14:
            return "中度"
        else:
            return "严重"
    # No calibrated thresholds for other components
    return None


if __name__ == "__main__":
    for sl in [1.0, 8.0, 20.0]:
        sev = map_severity("cooler", sl)
        print(f"  cooler s_level={sl:5.1f} C -> severity={sev}")
