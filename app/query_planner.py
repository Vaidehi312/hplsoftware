from typing import Dict, Any
import re

def build_query_plan(query: str) -> Dict[str, Any]:
    q = query.lower()

    # -------------------------
    # Entity detection (reuse your logic)
    # -------------------------
    tile = re.findall(r"(tile[_\-]?\d+|\w+\.(jpg|jpeg|png|tif))", q)
    slide = re.findall(r"(tcga[-\w]+dx\d+)", q)
    hpc = re.findall(r"hpc[\s\-]*([0-9]+)", q)
    sample = re.findall(r"sample[\s\-]*([a-z0-9]+)", q)

    # -------------------------
    # Flags
    # -------------------------
    malignant = "malignant" in q and "not malignant" not in q
    non_malignant = "not malignant" in q or "non malignant" in q
    survival = any(k in q for k in ["survival", "cox", "hazard", "prognosis"])

    # -------------------------
    # Intent (simple + explicit)
    # -------------------------
    if slide:
        intent = "slide_query"
    elif tile:
        intent = "tile_query"
    elif hpc:
        intent = "hpc_query"
    elif sample:
        intent = "sample_query"
    else:
        intent = "general_query"

    # -------------------------
    # UI actions
    # -------------------------
    ui_actions = {
        "open_slide_viewer": bool(slide),
        "show_tile_preview": bool(slide or tile)
    }

    # -------------------------
    # Operations (what computation is needed)
    # -------------------------
    operations = []
    if malignant or non_malignant:
        operations.append("malignancy_lookup")
    if survival:
        operations.append("survival_analysis")

    # -------------------------
    # Final JSON plan
    # -------------------------
    plan = {
        "intent": intent,
        "entities": {
            "tile": [t[0] for t in tile],
            "slide": slide,
            "hpc": hpc,
            "sample": sample
        },
        "flags": {
            "malignant": malignant or None,
            "non_malignant": non_malignant or None,
            "survival": survival or None
        },
        "operations": operations,
        "ui_actions": ui_actions,
        "confidence": 0.9   # static for now
    }

    return plan
