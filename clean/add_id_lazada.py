import json
import re
import os
from clean_nan import clean_nan_object

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_file = os.path.join(base, "data/raw/lazada_all_2026-01-08_12-04.json")
output_file = os.path.join(base, "data/preliminary/lazada_all_2026-01-08_12-04.json")


def extract_id_from_url(url):
    if not url:
        return None

    match = re.search(r"pdp-i(\d+)", url)
    if match:
        return int(match.group(1))
    return None


with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

data = clean_nan_object(data)

new_data = []
for item in data if isinstance(data, list) else []:
    if not isinstance(item, dict):
        continue
    new_item = {}
    extracted_id = extract_id_from_url(item.get("url"))

    for key, value in item.items():
        new_item[key] = value
        if key == "category_name":
            new_item["id"] = extracted_id

    new_data.append(new_item)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print("✅ Đã chuẩn hóa dữ liệu Lazada")
