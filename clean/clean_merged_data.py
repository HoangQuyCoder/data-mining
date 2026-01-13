import pandas as pd
import numpy as np
import json
import re
import os
from datetime import datetime


def extract_price(price_str):
    """Trích xuất giá trị số từ chuỗi giá (vd: '499.000 ₫' -> 499000)"""
    try:
        if pd.isna(price_str):
            return None
    except (TypeError, ValueError):
        pass

    if price_str is None:
        return float(price_str)

    # Loại bỏ ký tự đơn vị tiền tệ và khoảng trắng
    price_str = str(price_str).replace('₫', '').strip()
    # Loại bỏ dấu chấm phân cách hàng nghìn
    price_str = price_str.replace('.', '').replace(',', '.')

    try:
        return float(price_str)
    except:
        return None


def extract_discount(discount_str):
    """Trích xuất tỷ lệ giảm giá (vd: '17% Off' -> 17)"""
    try:
        if pd.isna(discount_str):
            return None
    except (TypeError, ValueError):
        pass

    if discount_str is None:
        return None

    discount_str = str(discount_str)
    match = re.search(r'(\d+)%', discount_str)

    return int(match.group(1)) if match else None


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

    # Loại bỏ cột trùng lặp ngay từ đầu
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    print(f"✓ Sau loại bỏ duplicate columns: {df.shape[1]} cột\n")

    # 1. Chuẩn hóa tên cột
    print("🔧 Bước 1: Chuẩn hóa tên cột...")
    print(f"  - Cột trước rename: {list(df.columns)}")

    column_mapping = {
        'category_name': 'category',
        'name': 'product_name',
        'price': 'current_price',
        'original_price': 'original_price',
        'discount': 'discount_rate',
        'rating': 'rating_average',
        'review_count': 'num_reviews',
        'sold_text': 'quantity_sold_text',
        'location': 'seller_location',
        'url': 'product_url',
        'image': 'image_url'
    }

    # Chỉ rename cột tồn tại
    existing_mapping = {k: v for k,
                        v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_mapping)

    # Loại bỏ cột trùng lặp nếu có
    df = df.loc[:, ~df.columns.duplicated(keep='first')]

    print(f"✓ Cột sau rename: {list(df.columns)}\n")

    # 2. Xử lý giá tiền
    print("💰 Bước 2: Xử lý giá tiền...")
    if 'current_price' in df.columns:
        df['current_price'] = df['current_price'].apply(extract_price)
    if 'original_price' in df.columns:
        df['original_price'] = pd.to_numeric(
            df['original_price'], errors='coerce')
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
            lambda x: extract_sold_value(x) if isinstance(x, str) else 0
        )
    print(f"✓ Quantity sold đã được chuẩn hóa\n")

    # 6. Xử lý dữ liệu thiếu
    print("🧹 Bước 6: Xử lý dữ liệu thiếu...")
    print(f"  - Dữ liệu thiếu trước xử lý:")
    print(df.isnull().sum())

    # Điền giá trị mặc định
    if 'rating_average' in df.columns:
        df['rating_average'] = df['rating_average'].fillna(0)
    if 'num_reviews' in df.columns:
        df['num_reviews'] = df['num_reviews'].fillna(0)
    if 'quantity_sold' in df.columns:
        df['quantity_sold'] = df['quantity_sold'].fillna(0)
    if 'discount_rate' in df.columns:
        df['discount_rate'] = df['discount_rate'].fillna(0)
    print("🗑️  Bước 7: Loại bỏ dữ liệu không hợp lệ...")
    initial_count = len(df)

    # Loại bỏ record không có id
    if 'id' in df.columns:
        df = df[df['id'].notna()]

    # Loại bỏ record không có tên sản phẩm
    if 'product_name' in df.columns:
        df = df[df['product_name'].notna()]

    # Loại bỏ record có giá <= 0 hoặc null
    if 'current_price' in df.columns:
        df = df[df['current_price'] > 0]
        df = df[df['current_price'].notna()]
    # 8. Thêm các cột tiêu chí
    print("➕ Bước 8: Thêm các cột tiêu chí...")

    # Tính giá khuyến mại thực tế
    if 'original_price' in df.columns and 'discount_rate' in df.columns:
        df['sale_price'] = df['original_price'] - \
            df['original_price'] * df['discount_rate'] / 100

    # Phân loại sản phẩm dựa trên rating
    def categorize_rating(rating):
        if rating >= 4.5:
            return 'Excellent'
        elif rating >= 4.0:
            return 'Very Good'
        elif rating >= 3.5:
            return 'Good'
        elif rating >= 3.0:
            return 'Average'
        else:
            return 'Poor'

    if 'rating_average' in df.columns:
        df['quality_category'] = df['rating_average'].apply(categorize_rating)

    # Phân loại độ phổ biến dựa trên số review
    def categorize_popularity(reviews):
        if reviews >= 1000:
            return 'Very Popular'
        elif reviews >= 500:
            return 'Popular'
        elif reviews >= 100:
            return 'Moderate'
        elif reviews >= 10:
            return 'Low'
        else:
            return 'Very Low'

    if 'num_reviews' in df.columns:
        df['popularity_category'] = df['num_reviews'].apply(
            categorize_popularity)

    print(f"✓ Đã thêm các cột tiêu chí\n")

    # 9. Sắp xếp và chọn cột cần thiết
    print("📋 Bước 9: Chọn cột cần thiết...")

    # Danh sách cột cuối cùng
    final_columns = [
        'id', 'crawl_date', 'platform', 'category', 'product_name',
        'current_price', 'original_price', 'sale_price', 'discount_rate',
        'rating_average', 'quality_category', 'num_reviews', 'popularity_category',
        'quantity_sold', 'quantity_sold_text',
        'brand', 'seller_name', 'seller_location',
        'product_url', 'image_url'
    ]

    # Chỉ lấy các cột tồn tại
    final_columns = [col for col in final_columns if col in df.columns]
    df = df[final_columns]

    print(f"✓ Cột cuối cùng: {len(df.columns)} cột\n")

    # 10. Lưu dữ liệu
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

    # 11. Thống kê tóm tắt
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
    input_file = os.path.join(base, 'data/raw/merged_raw_data.json')
    output_file = os.path.join(base, 'data/clean/merged_cleaned_data.json')

    print("\n🚀 BẮT ĐẦU LÀMS SẠCH DỮ LIỆU")
    print("=" * 60 + "\n")

    # Chạy hàm làm sạch
    df_cleaned = clean_merged_data(input_file, output_file)

    print("\n✅ HOÀN THÀNH!\n")
