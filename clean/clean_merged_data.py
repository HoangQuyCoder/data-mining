import pandas as pd
import numpy as np
import json
import re
import os
from datetime import datetime


def extract_price(value):
    """Trích xuất giá trị số từ chuỗi giá (vd: '499.000 ₫' -> 499000)"""

    if value is None or pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    value = re.sub(r"[^\d]", "", str(value))
    return float(value) if value else None


def extract_discount(value):
    """Trích xuất tỷ lệ giảm giá (vd: '17% Off' -> 17)"""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = re.search(r"\d+", value)
    return float(match.group()) if match else None


def extract_sold_value(sold_text):
    """Trích xuất số lượng bán (vd: '1.2K Sold' -> 1200)"""
    try:
        if pd.isna(sold_text):
            return None
    except (TypeError, ValueError):
        pass

    if sold_text is None:
        return None

    sold_text = str(sold_text).upper().strip()
    match = re.search(r'([\d.]+)\s*([KMB]?)', sold_text)

    if match:
        value = float(match.group(1))
        unit = match.group(2)

        if unit == 'K':
            return int(value * 1000)
        elif unit == 'M':
            return int(value * 1000000)
        elif unit == 'B':
            return int(value * 1000000000)
        else:
            return int(value)

    return None


def safe_to_numeric(value):
    """
    Ép kiểu an toàn:
    - number -> number
    - string số -> number
    - dict / list / khác -> NaN
    """
    if isinstance(value, (int, float)):
        return value

    if isinstance(value, str):
        try:
            return float(value)
        except:
            return None

    return None


def normalize_product(item: dict) -> dict:
    platform = item.get("platform", "").lower()

    normalized = {
        "crawl_date": item.get("crawl_date"),
        "platform": item.get("platform"),
        "category": item.get("category_name"),
        "id": item.get("id"),
        "product_name": item.get("name"),
        "current_price": None,
        "original_price": None,
        "discount_rate": None,
        "rating_average": None,
        "num_reviews": None,
        "quantity_sold": None,
        "quantity_sold_text": None,
        "brand": None,
        "seller_name": None,
        'seller_location': None,
        "product_url": item.get("url"),
    }

    # ===== TIKI =====
    if platform == "tiki":
        normalized.update({
            "current_price": item.get("price"),
            "original_price": item.get("original_price"),
            "discount_rate": item.get("discount_rate"),
            "rating_average": item.get("rating_average"),
            "num_reviews": item.get("review_count"),
            "quantity_sold": item.get("quantity_sold_value"),
            "quantity_sold_text": item.get("quantity_sold_text"),
            "brand": item.get("brand"),
            "seller_name": item.get("seller_name"),
            'seller_location': item.get("location"),
        })

    # ===== LAZADA =====
    elif platform == "lazada":
        normalized.update({
            "current_price": item.get("price"),
            "original_price": item.get("original_price"),
            "discount_rate": item.get("discount"),
            "rating_average": item.get("rating"),
            "num_reviews": item.get("review_count"),
            "quantity_sold": item.get("sold_value"),
            "quantity_sold_text": item.get("sold_text"),
            "brand": item.get("brand"),
            "seller_name": item.get("seller_name"),
            'seller_location': item.get("location"),
        })

    # ===== SHOPEE =====
    elif platform == "shopee":
        normalized.update({
            "current_price": item.get("price"),
            "original_price": item.get("original_price"),
            "discount_rate": item.get("discount_rate"),
            "rating_average": item.get("rating_average"),
            "num_reviews": item.get("review_count"),
            "quantity_sold": item.get("quantity_sold_value"),
            "quantity_sold_text": item.get("quantity_sold_text"),
            "brand": item.get("brand"),
            "seller_name": item.get("seller_name"),
            'seller_location': item.get("location"),
        })

    return normalized


def normalize_dataset(data: list[dict]) -> pd.DataFrame:
    normalized_data = [normalize_product(item) for item in data]
    df = pd.DataFrame(normalized_data)
    return df


def clean_merged_data(input_file, output_file=None):
    """
    Làm sạch dữ liệu từ file merged_raw_data.json

    Parameters:
    - input_file: đường dẫn file .json đầu vào
    - output_file: đường dẫn file output (mặc định: data/clean/merged_cleaned_data.json)
    """

    # Đọc dữ liệu
    print(f"📂 Đang đọc file: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"✓ Đã load {df.shape[0]} records, {df.shape[1]} cột")

    # 1. Chuẩn hóa tên cột
    print("🔧 Bước 1: Chuẩn hóa tên cột...")
    print(f"  - Cột trước rename: {list(df.columns)}")

    df = normalize_dataset(data)

    print(f"✓ Cột sau rename: {list(df.columns)}\n")

    # 2. Xử lý giá tiền
    print("💰 Bước 2: Xử lý giá tiền...")
    if 'current_price' in df.columns:
        df['current_price'] = df['current_price'].apply(extract_price)
    if 'original_price' in df.columns:
        df['original_price'] = df['original_price'].apply(extract_price)

    df[['current_price', 'original_price']] = df[
        ['current_price', 'original_price']
    ].apply(pd.to_numeric, errors='coerce')

    print(f"✓ Giá tiền đã được chuẩn hóa\n")

    # 3. Xử lý discount
    print("📉 Bước 3: Xử lý discount rate...")
    if 'discount_rate' in df.columns:
        df['discount_rate'] = df['discount_rate'].apply(extract_discount)
    print(f"✓ Discount rate đã được chuẩn hóa\n")

    # 4. Xử lý rating và review_count
    print("⭐ Bước 4: Xử lý rating và số review...")
    if 'rating_average' in df.columns:
        df['rating_average'] = df['rating_average'].apply(safe_to_numeric)

    if 'num_reviews' in df.columns:
        df['num_reviews'] = df['num_reviews'].apply(safe_to_numeric)

    print(f"✓ Rating và review_count đã được chuẩn hóa\n")

    # 5. Xử lý quantity sold
    print("📦 Bước 5: Xử lý số lượng đã bán...")
    if 'quantity_sold_text' in df.columns:
        df['quantity_sold'] = df['quantity_sold_text'].apply(
            lambda x: extract_sold_value(x) if isinstance(x, str) else None
        )
    print(f"✓ Quantity sold đã được chuẩn hóa\n")

    # 6. Xử lý dữ liệu thiếu
    print("🧹 Bước 6: Xử lý dữ liệu thiếu...")
    print(f"  - Dữ liệu thiếu trước xử lý:")
    print(df.isnull().sum())

    # # Điền giá trị mặc định
    if 'rating_average' in df.columns:
        df["rating_average_missing"] = df["rating_average"].isna().astype(int)
        df["rating_average"].fillna(
            df["rating_average"].median(), inplace=True)
    if 'discount_rate' in df.columns:
        df["discount_rate_missing"] = df["discount_rate"].isna().astype(int)
        df["discount_rate"].fillna(df["discount_rate"].median(), inplace=True)
    if 'num_reviews' in df.columns:
        df['num_reviews'] = df['num_reviews'].fillna(0)
    if 'quantity_sold' in df.columns:
        df['quantity_sold'] = df['quantity_sold'].fillna(0)

    df['original_price'] = pd.to_numeric(df['original_price'], errors="coerce")

    text_columns = [
        "quantity_sold_text",
        "brand",
        "seller_name",
        "seller_location"
    ]
    df[text_columns] = df[text_columns].fillna("UNKNOWN")
    print(f"✓ Dữ liệu thiếu đẫ được xử lí\n")

    print("🗑️  Bước 7: Loại bỏ dữ liệu không hợp lệ...")
    print("Số record trước khi loại bỏ", len(df))

    df = df.sort_values(
        by=[
            "quantity_sold",
            "num_reviews",
            "rating_average_missing"
        ],
        ascending=[False, False, True]
    )

    DEDUP_KEYS = ["platform", "id"]

    df_dedup = df.drop_duplicates(
        subset=DEDUP_KEYS,
        keep="first"
    )
    df_dedup = df_dedup.reset_index(drop=True)

    # Loại bỏ record không có tên sản phẩm
    if 'product_name' in df.columns:
        df = df[df['product_name'].notna()]

    # Loại bỏ record có giá <= 0 hoặc null
    if 'current_price' in df.columns:
        df = df[df['current_price'] > 0]
        df = df[df['current_price'].notna()]

    print("Số record sau khi loại bỏ", len(df), "\n")

    # 8. Sắp xếp và chọn cột cần thiết
    print("📋 Bước 9: Chọn cột cần thiết...")

    # Danh sách cột cuối cùng
    final_columns = [
        'id', 'crawl_date', 'platform', 'category', 'product_name',
        'current_price', 'original_price', 'discount_rate',
        'rating_average', 'quality_category', 'num_reviews', 'popularity_category',
        'quantity_sold', 'quantity_sold_text',
        'brand', 'seller_name', 'seller_location',
        'product_url', 'rating_average_missing', 'discount_rate_missing'
    ]

    # Chỉ lấy các cột tồn tại
    final_columns = [col for col in final_columns if col in df.columns]
    df = df[final_columns]

    print(f"✓ Cột cuối cùng: {len(df.columns)} cột\n")

    # 9. Lưu dữ liệu
    if output_file is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_file = os.path.join(base, 'data/clean/merged_cleaned_data.json')

    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"💾 Bước 10: Lưu dữ liệu...")
    print(f"  - Output file: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(df.to_dict('records'), f, ensure_ascii=False, indent=2)

    print(f"✓ Dữ liệu đã được lưu\n")

    # 10. Thống kê tóm tắt
    print("=" * 60)
    print("📊 THỐNG KÊ TÓM TẮT")
    print("=" * 60)
    print(f"Tổng records: {len(df)}")
    print(f"\nThông tin giá:")
    if 'current_price' in df.columns:
        print(
            f"  - Giá hiện tại: {df['current_price'].min():.0f} - {df['current_price'].max():.0f}")
        print(f"  - Trung bình: {df['current_price'].mean():.0f}")
    print(f"\nThông tin đánh giá:")
    if 'rating_average' in df.columns:
        print(f"  - Rating trung bình: {df['rating_average'].mean():.2f}")
    if 'num_reviews' in df.columns:
        print(f"  - Review trung bình: {df['num_reviews'].mean():.0f}")
    if 'platform' in df.columns:
        print(f"\nPlatform:")
        print(df['platform'].value_counts())
    if 'category' in df.columns:
        print(f"\nTop 5 Categories:")
        print(df['category'].value_counts().head())
    if 'brand' in df.columns:
        print(f"\nTop 5 Brands:")
        print(df['brand'].value_counts().head())
    if 'quality_category' in df.columns:
        print(f"\nQuality Distribution:")
        print(df['quality_category'].value_counts())
    if 'popularity_category' in df.columns:
        print(f"\nPopularity Distribution:")
        print(df['popularity_category'].value_counts())
    print("=" * 60)
    print("\n💡 Để xem biểu đồ trực quan hóa, chạy:")
    print("   python main/visualize_cleaning_results.py")
    print("=" * 60)

    return df


if __name__ == "__main__":
    # Đường dẫn input/output
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(
        base, 'data/preliminary/merged_preliminary_data.json')
    output_file = os.path.join(base, 'data/clean/merged_cleaned_data.json')

    print("\n🚀 BẮT ĐẦU LÀMS SẠCH DỮ LIỆU")
    print("=" * 60 + "\n")

    # Chạy hàm làm sạch
    df_cleaned = clean_merged_data(input_file, output_file)

    print("\n✅ HOÀN THÀNH!\n")
