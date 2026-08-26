import json
from pathlib import Path

def save_query_plan_to_file(plan: dict, slide_id: str | None = None):
    out_dir = Path("query_plans")
    out_dir.mkdir(exist_ok=True)

    # filename
    name = f"{slide_id}_plan.json" if slide_id else "plan.json"
    path = out_dir / name

    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, ensure_ascii=False)

    return str(path)