import json
import os
from clean_nan import clean_nan_object

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_file = os.path.join(base, "data/raw/tiki_all_2026-01-07_12-26.json")
output_file = os.path.join(base, "data/preliminary/tiki_all_2026-01-07_12-26.json")

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

data = clean_nan_object(data)

new_data = []
for item in data if isinstance(data, list) else []:
    if not isinstance(item, dict):
        continue

    new_item = {
        k: v for k, v in item.items()
        if k != "badges" and k!= "thumbnail_url"
    }

    if "crawl_date" in new_item:
        new_item["platform"] = "Tiki"

    new_data.append(new_item)


with open(output_file, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print("✅ Đã chuẩn hóa dữ liệu Tiki")
