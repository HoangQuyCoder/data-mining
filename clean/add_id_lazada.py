import json
import re
import os
from clean_nan import clean_nan_object

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_file = os.path.join(base, "data/raw/lazada_all_2026-01-08_12-04.json")
output_file = os.path.join(base, "data/preliminary/lazada_all_2026-01-08_12-04.json")

FIELD_RENAME_MAP = {
    "discount": "discount_rate",
    "rating": "rating_average",
    "sold_value": "quantity_sold_value",
    "sold_text": "quantity_sold_text",
}


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

    for k, v in item.items():
        if k == "image":
            continue

        new_key = FIELD_RENAME_MAP.get(k, k)
        new_item[new_key] = v

    # extract id từ url
    extracted_id = extract_id_from_url(item.get("url"))

    if extracted_id is not None:
        new_item["id"] = extracted_id

    new_data.append(new_item)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)

print("✅ Đã chuẩn hóa dữ liệu Lazada")
