import json
import re
import os
from datetime import date
import pandas as pd

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_file = os.path.join(base, "data/raw/shopee_all_2026-01-13_21-15.json")
output_file = os.path.join(base, "data/preliminary/shopee_all_2026-01-13_21-15.json")

category_map = {
    100001: "Máy tính & Laptop",
    100009: "Máy ảnh & Máy quay phim",
    100010: "Điện thoại & Phụ kiện",
    100011: "Thiết bị điện tử",
    100012: "Thiết bị điện gia dụng",
    100013: "Thể thao & Du lịch",
    100015: "Đồ chơi",
    100016: "Xe máy, Xe đạp & Phụ kiện",
    100017: "Mẹ & Bé",
    100532: "Thời trang Nam",
    100533: "Giày dép Nam",
    100534: "Đồng hồ",
    100535: "Túi ví Nam",
    100629: "Thời trang Nữ",
    100630: "Giày dép Nữ",
    100631: "Túi ví Nữ",
    100632: "Phụ kiện & Trang sức Nữ",
    100633: "Sức khỏe",
    100634: "Sắc đẹp",
    100635: "Nhà cửa & Đời sống",
    100636: "Bách hóa Online",
    100637: "Văn phòng phẩm",
    100638: "Nhà sách Online",
    100639: "Thú cưng",
    100640: "Voucher & Dịch vụ",
    100641: "Ô tô",
    100643: "Thời trang Trẻ em",
    100644: "Giày dép Trẻ em"
}

df = pd.read_json(input_file)

df['category_name'] = df['category_id'].map(category_map).fillna("Không xác định")

with open(input_file, "r", encoding="utf-8") as f:
    shopee_data = json.load(f)

normalized_data = []

for item in shopee_data:
    normalized_item = {
        "crawl_date": date.today().isoformat(),
        "platform": "Shopee",
        "category_name": item.get("category_name"),
        "id": int(item["itemid"]),
        "name": item.get("name"),
        "price": item.get("price"),
        "original_price": item.get("price_original"),
        "discount_rate": item.get("discount_rate"),
        "rating_average": item.get("rating_star"),
        "review_count": item.get("review_count"),
        "quantity_sold_value": item.get("historical_sold"),
        "quantity_sold_text": (
            f"Đã bán {item['historical_sold']}"
            if item.get("historical_sold") is not None
            else None
        ),
        "brand": item.get("brand"),
        "location": item.get("location"),
        "url": item.get("url")
    }

    normalized_data.append(normalized_item)


with open(output_file, "w", encoding="utf-8") as f:
    json.dump(normalized_data, f, ensure_ascii=False, indent=2)

print("✅ Đã chuẩn hóa dữ liệu Shopee")
