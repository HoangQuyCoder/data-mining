import json
import os
from datetime import date

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

input_file = os.path.join(base, "data/raw/shopee_all_2026-01-13_21-15.json")
output_file = os.path.join(base, "data/preliminary/shopee_all_2026-01-13_21-15.json")

SHOPEE_CATEGORY_MAP = {
    100001: "Bách hóa Online",           # Kem đánh răng, vệ sinh cá nhân
    100009: "Phụ kiện & Trang sức Nữ",   # Khăn quàng, phụ kiện thời trang
    100010: "Thiết bị điện gia dụng",    # Máy rửa bát, nồi hấp, máy hút ẩm
    100011: "Thời trang Trẻ em",         # Đồ bộ, áo thun cho bé
    100012: "Giày dép Trẻ em",           # Giày dép trẻ em, sandal
    100013: "Đồng hồ",                   # Đồng hồ thông minh, đồng hồ thể thao
    100015: "Sắc đẹp",                   # Mỹ phẩm, tuýp chiết, kem dưỡng
    100016: "Máy tính & Laptop",         # Túi chống sốc laptop, phụ kiện laptop
    100017: "Mẹ & Bé",                   # Quần lót, áo gia đình, váy mẹ bé
    100532: "Giày dép Nữ",               # Dép, giày quai hậu nữ
    100533: "Túi ví Nữ",                 # Túi đựng laptop, túi chống sốc (dành cho nữ)
    100534: "Đồng hồ",                   # Đồng hồ nam, đồng hồ thời trang
    100535: "Thiết bị điện tử",          # Jack, đế nối, phụ kiện điện tử
    100629: "Bách hóa Online",           # Đồ ăn vặt, snack, bánh tráng
    100630: "Sắc đẹp",                   # Dầu gội, xà phòng, mỹ phẩm
    100631: "Thú cưng",                  # Súp cho mèo, đồ chơi cho mèo
    100632: "Mẹ & Bé",                   # Tăm bông, khăn ướt cho bé
    100633: "Đồng hồ",                   # Đồng hồ học sinh, đồng hồ thể thao
    100634: "Đồ chơi",                   # Máy chơi game, tay cầm game
    100635: "Máy ảnh & Máy quay phim",   # Bộ vệ sinh máy ảnh
    100636: "Bách hóa Online",           # Nước giặt, bột giặt, nước rửa chén
    100637: "Thể thao & Du lịch",        # Lưới tàng hình, ống nhòm, đồ dùng outdoor
    100638: "Bách hóa Online",           # Băng keo đóng gói, đồ văn phòng
    100639: "Đồ chơi",                   # Board game, UNO, trò chơi
    100640: "Ô tô",                      # Máy bơm lốp, dầu nhớt, ví đựng giấy tờ xe
    100641: "Xe máy, Xe đạp & Phụ kiện", # Còi xe
    100643: "Nhà sách Online",           # Sách các loại
    100644: "Văn phòng phẩm",            # Giá đỡ laptop, pad công thức, mực máy bấm giá
}

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

normalized_data = []

for item in data:
    category_id = item.get("category_id")
    
    normalized_item = {
        "crawl_date": date.today().isoformat(),
        "platform": "Shopee",
        "category_name": SHOPEE_CATEGORY_MAP.get(category_id, "Không xác định"),
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
