import json
import re
import os
from datetime import date

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_file = os.path.join(base, "data/raw/shopee_all_2026-01-13_21-15.json")
output_file = os.path.join(base, "data/clean/shopee_all_2026-01-13_21-15.json")

def parse_discount_rate(discount):
    if not discount:
        return None
    match = re.search(r"(\d+)", discount)
    return int(match.group(1)) if match else None

with open(input_file, "r", encoding="utf-8") as f:
    shopee_data = json.load(f)

normalized_data = []

for item in shopee_data:
    normalized_item = {
        "crawl_date": date.today().isoformat(),
        "platform": "Shopee",
        "category_name": "",
        "id": int(item["itemid"]),
        "name": item.get("name"),
        "price": item.get("price"),
        "original_price": None,
        "discount_rate": parse_discount_rate(item.get("discount")),
        "rating_average": item.get("rating_star"),
        "review_count": None,
        "quantity_sold_value": item.get("historical_sold"),
        "quantity_sold_text": (
            f"Đã bán {item['historical_sold']}"
            if item.get("historical_sold") is not None
            else None
        ),
        "url": item.get("url")
    }

    normalized_data.append(normalized_item)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(normalized_data, f, ensure_ascii=False, indent=2)

print("✅ Đã chuẩn hóa dữ liệu Shopee")
