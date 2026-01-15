import json
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_file = os.path.join(base, "merged_all_data.json")
output_dir = os.path.join(base, "output_by_platform")

os.makedirs(output_dir, exist_ok=True)

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

platform_groups = {}

for item in data:
    platform = item.get("platform", "UNKNOWN")
    platform_groups.setdefault(platform, []).append(item)

for platform, items in platform_groups.items():
    file_path = os.path.join(
        output_dir,
        f"{platform.lower().replace(' ', '_')}.json"
    )
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"{platform}: {len(items)} records → {file_path}")
