import pandas as pd
import numpy as np
import json
import re
from datetime import datetime


def clean_lazada_data(input_file, output_file=None):
    """
    Clean dữ liệu crawl từ Lazada

    Parameters:
    - input_file: đường dẫn file .json hoặc .csv
    - output_file: nếu muốn lưu kết quả (mặc định: thêm _cleaned vào tên file gốc)
    """

    # 1. Đọc dữ liệu
    if input_file.endswith('.json'):
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)  # tự động load toàn bộ file
        df = pd.DataFrame(data)  # giả sử mỗi dòng là 1 json object
    elif input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    else:
        raise ValueError("File phải là .json hoặc .csv")

    print(f"Trước khi clean: {df.shape[0]} sản phẩm, {df.shape[1]} cột")

    # Chuẩn hóa tên cột (đồng bộ các tên khác nhau)
    column_mapping = {
        'category_name': 'category_name',
        'name': 'product_name',
        'price': 'current_price',
        'original_price': 'original_price',
        'discount': 'discount_rate',
        'discount_rate': 'discount_rate',
        'rating': 'rating_average',
        'rating_average': 'rating_average',
        'ratingScore': 'rating_average',
        'review_count': 'review_count',
        'review': 'review_count',
        'sold_value': 'quantity_sold_value',
        'sold_text': 'quantity_sold_text',
        'quantity_sold_value': 'quantity_sold_value',
        'quantity_sold_text': 'quantity_sold_text',
        'brand': 'brand',
        'location': 'seller_location',
        'seller_name': 'seller_name',
        'url': 'product_url',
        'image': 'image_url'
    }

    df = df.rename(columns=column_mapping)

    # Chỉ giữ lại các cột cần thiết (bạn có thể chỉnh lại danh sách này)
    desired_columns = [
        'crawl_date', 'platform', 'category_name', 'product_name',
        'current_price', 'original_price', 'discount_rate',
        'rating_average', 'review_count',
        'quantity_sold_value', 'quantity_sold_text',
        'brand', 'seller_location', 'seller_name',
        'product_url'
    ]

    # Giữ lại cột nào có trong df
    existing_cols = [col for col in desired_columns if col in df.columns]
    df = df[existing_cols]

    # 2. Xử lý kiểu dữ liệu và missing values

    # Price fields → chuyển thành số, loại bỏ dấu phẩy, ký tự lạ
    price_cols = ['current_price', 'original_price']
    for col in price_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(
                ',', '').str.replace('đ', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Discount rate → chuyển thành số (loại % nếu có)
    if 'discount_rate' in df.columns:
        df['discount_rate'] = df['discount_rate'].astype(
            str).str.replace('%', '').str.strip()
        df['discount_rate'] = pd.to_numeric(
            df['discount_rate'], errors='coerce')
        # Nếu discount_rate > 100 hoặc âm → đặt về 0
        df['discount_rate'] = df['discount_rate'].clip(lower=0, upper=100)

    # Rating & Review
    if 'rating_average' in df.columns:
        df['rating_average'] = pd.to_numeric(
            df['rating_average'], errors='coerce')
        df['rating_average'] = df['rating_average'].clip(lower=0, upper=5)

    if 'review_count' in df.columns:
        df['review_count'] = pd.to_numeric(
            df['review_count'], errors='coerce').fillna(0).astype(int)

    # Sold quantity
    if 'quantity_sold_value' in df.columns:
        df['quantity_sold_value'] = pd.to_numeric(
            df['quantity_sold_value'], errors='coerce').fillna(0).astype(int)

    # 3. Điền giá trị mặc định cho các trường thiếu

    fill_defaults = {
        'current_price': 0,
        'original_price': lambda x: x['current_price'] if pd.notna(x['current_price']) else 0,
        'discount_rate': 0,
        'rating_average': 0.0,
        'review_count': 0,
        'quantity_sold_value': 0,
        'quantity_sold_text': '',
        'brand': 'No Brand',
        'seller_location': 'Không xác định',
        'seller_name': 'Không xác định',
        'product_url': ''
    }

    for col, default in fill_defaults.items():
        if col in df.columns:
            if callable(default):
                # Trường hợp original_price lấy từ current_price
                df[col] = df.apply(lambda row: default(
                    row) if pd.isna(row[col]) else row[col], axis=1)
            else:
                df[col] = df[col].fillna(default)

    # 4. Sửa URL lỗi (ví dụ: chứa //www.lazada.vn//)
    if 'product_url' in df.columns:
        df['product_url'] = df['product_url'].str.replace(
            r'//+', '/', regex=True)
        df['product_url'] = df['product_url'].str.replace(
            r'^/', 'https://www.lazada.vn/', regex=True)

    # 5. Thêm một số cột hữu ích (tùy chọn)
    df['has_discount'] = (df['discount_rate'] > 0).astype(int)
    df['is_sold'] = (df['quantity_sold_value'] > 0).astype(int)
    df['clean_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 6. Sắp xếp lại cột cho đẹp
    final_columns_order = [
        'crawl_date', 'platform', 'category_name', 'product_name',
        'current_price', 'original_price', 'discount_rate', 'has_discount',
        'rating_average', 'review_count',
        'quantity_sold_value', 'quantity_sold_text', 'is_sold',
        'brand', 'seller_location', 'seller_name',
        'product_url', 'clean_date'
    ]

    final_columns = [col for col in final_columns_order if col in df.columns]
    df = df[final_columns]

    initial_rows = len(df)
    print(f"Trước khi xóa trùng: {initial_rows:,} dòng")

    # Cách 1 - Ưu tiên cao nhất: dùng product_url (nếu có)
    if 'product_url' in df.columns:
        # Chuẩn hóa URL trước khi so sánh
        df['product_url'] = df['product_url'].str.strip()
        df['product_url'] = df['product_url'].str.replace(
            r'^https?://(www\.)?', 'https://', regex=True)
        df['product_url'] = df['product_url'].str.rstrip('/')

        # Xóa trùng, giữ bản ghi đầu tiên
        df = df.drop_duplicates(subset=['product_url'], keep='first')

        print(f"Sau khi xóa trùng theo product_url: {len(df):,} dòng "
              f"(xóa {initial_rows - len(df):,} bản ghi trùng)")

    # Cách 2 - Dự phòng: nếu không có product_url hoặc còn nhiều trùng
    else:
        print("Không tìm thấy cột product_url → dùng cách dự phòng")

        # Kết hợp các cột quan trọng
        duplicate_subset = ['product_name',
                            'current_price', 'seller_name', 'category_name']
        available_subset = [
            col for col in duplicate_subset if col in df.columns]

        if available_subset:
            df = df.drop_duplicates(subset=available_subset, keep='first')
            print(
                f"Sau khi xóa trùng theo {available_subset}: {len(df):,} dòng")
        else:
            print("Không đủ thông tin để xác định trùng lặp → giữ nguyên dữ liệu")

    # Nếu có cột id sản phẩm (item id) → nên ưu tiên dùng nó
    if 'id' in df.columns:
        df['id'] = df['id'].astype(str).str.strip()
        df = df.drop_duplicates(subset=['id'], keep='first')
        print(f"Sau khi xóa trùng theo id: {len(df):,} dòng")

    # 7. Lưu kết quả
    if output_file is None:
        output_file = input_file.replace(
            '.csv', '_cleaned.csv').replace('.json', '_cleaned.csv')

    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Đã clean và lưu thành công: {output_file}")
    print(f"Sau khi clean: {df.shape[0]} sản phẩm")

    # Thống kê missing values sau clean (nên = 0)
    print("\nMissing values sau clean:")
    print(df.isnull().sum())

    return df


# Ví dụ sử dụng
if __name__ == "__main__":
    # Thay bằng đường dẫn file của bạn
    file_path = "merged_cleaned.json"  # hoặc .csv

    cleaned_df = clean_lazada_data(file_path)

    # Xem 5 dòng đầu sau clean
    print("\nDữ liệu sau khi clean (5 dòng đầu):")
    print(cleaned_df.head())
