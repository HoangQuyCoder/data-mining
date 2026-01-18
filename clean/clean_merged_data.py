import pandas as pd
import numpy as np
import json
import re
import os
from datetime import datetime


class ValueExtractor:
    """Lớp trích xuất và chuyển đổi các giá trị từ dữ liệu thô"""

    @staticmethod
    def extract_price(value):
        """Trích xuất giá trị số từ chuỗi giá (vd: '499.000 ₫' -> 499000)"""
        if value is None or pd.isna(value):
            return None

        if isinstance(value, (int, float)):
            return float(value)

        value = re.sub(r"[^\d]", "", str(value))
        return float(value) if value else None

    @staticmethod
    def extract_discount(value):
        """Trích xuất tỷ lệ giảm giá (vd: '17% Off' -> 17)"""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        match = re.search(r"\d+", value)
        return float(match.group()) if match else None

    @staticmethod
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

    @staticmethod
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


class ProductNormalizer:
    """Lớp chuẩn hóa dữ liệu sản phẩm từ các platform khác nhau"""

    PLATFORM_MAPPING = {
        "tiki": {
            "current_price": "price",
            "original_price": "original_price",
            "discount_rate": "discount_rate",
            "rating_average": "rating_average",
            "num_reviews": "review_count",
            "quantity_sold": "quantity_sold_value",
            "quantity_sold_text": "quantity_sold_text",
            "brand": "brand",
            "seller_location": "location",
        },
        "lazada": {
            "current_price": "price",
            "original_price": "original_price",
            "discount_rate": "discount",
            "rating_average": "rating",
            "num_reviews": "review_count",
            "quantity_sold": "sold_value",
            "quantity_sold_text": "sold_text",
            "brand": "brand",
            "seller_location": "location",
        },
        "shopee": {
            "current_price": "price",
            "original_price": "original_price",
            "discount_rate": "discount_rate",
            "rating_average": "rating_average",
            "num_reviews": "review_count",
            "quantity_sold": "quantity_sold_value",
            "quantity_sold_text": "quantity_sold_text",
            "brand": "brand",
            "seller_location": "location",
        },
    }

    @classmethod
    def normalize_product(cls, item: dict) -> dict:
        """Chuẩn hóa một sản phẩm từ bất kỳ platform nào"""
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
            "seller_location": None,
            "product_url": item.get("url"),
        }

        # Áp dụng mapping cho platform
        if platform in cls.PLATFORM_MAPPING:
            mapping = cls.PLATFORM_MAPPING[platform]
            for standard_key, source_key in mapping.items():
                if standard_key in normalized:
                    normalized[standard_key] = item.get(source_key)

        return normalized

    @classmethod
    def normalize_dataset(cls, data: list[dict]) -> pd.DataFrame:
        """Chuẩn hóa toàn bộ dataset"""
        normalized_data = [cls.normalize_product(item) for item in data]
        df = pd.DataFrame(normalized_data)
        return df


class DataCleaner:
    """Lớp chính để làm sạch và xử lý dữ liệu merged"""

    # Danh sách cột cuối cùng cần giữ lại
    FINAL_COLUMNS = [
        'id', 'crawl_date', 'platform', 'category', 'product_name',
        'current_price', 'discount_rate',
        'rating_average', 'quality_category', 'num_reviews', 'popularity_category',
        'quantity_sold',
        'brand', 'seller_location',
        'product_url', 'rating_average_missing', 'discount_rate_missing'
    ]

    # Cột text cần điền giá trị mặc định
    TEXT_COLUMNS = [
        "quantity_sold_text",
        "brand",
        "seller_location"
    ]

    # Key để loại bỏ duplicate
    DEDUP_KEYS = ["platform", "id"]

    def __init__(self, input_file, output_file=None):
        """Khởi tạo DataCleaner"""
        self.input_file = input_file
        self.output_file = output_file or self._get_default_output_file()
        # self.df = None
        self.raw_data = None

    def _get_default_output_file(self):
        """Lấy đường dẫn output mặc định"""
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, 'data/clean/merged_cleaned_data.json')

    def load_data(self):
        """Bước 1: Đọc dữ liệu từ file JSON"""
        print(f"📂 Đang đọc file: {self.input_file}")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)

        print(f"✓ Đã load {len(self.raw_data)} records\n")

    def normalize_data(self):
        """Bước 2: Chuẩn hóa tên cột từ các platform khác nhau"""
        print("🔧 Bước 1: Chuẩn hóa dữ liệu...")
        if self.raw_data is None:
            raise ValueError("Raw data is not loaded. Call load_data() first.")
        self.df = ProductNormalizer.normalize_dataset(self.raw_data)
        print(f"✓ Đã chuẩn hóa {len(self.df)} records\n")

    def clean_prices(self):
        """Bước 3: Xử lý giá tiền"""
        print("💰 Bước 2: Xử lý giá tiền...")
        if 'current_price' in self.df.columns:
            self.df['current_price'] = self.df['current_price'].apply(
                ValueExtractor.extract_price
            )
        if 'original_price' in self.df.columns:
            self.df['original_price'] = self.df['original_price'].apply(
                ValueExtractor.extract_price
            )

        self.df[['current_price', 'original_price']] = self.df[
            ['current_price', 'original_price']
        ].apply(pd.to_numeric, errors='coerce')

        print(f"✓ Giá tiền đã được chuẩn hóa\n")

    def clean_discount(self):
        """Bước 4: Xử lý discount rate"""
        print("📉 Bước 3: Xử lý discount rate...")
        if 'discount_rate' in self.df.columns:
            self.df['discount_rate'] = self.df['discount_rate'].apply(
                ValueExtractor.extract_discount
            )
        print(f"✓ Discount rate đã được chuẩn hóa\n")

    def clean_ratings(self):
        """Bước 5: Xử lý rating và review count"""
        print("⭐ Bước 4: Xử lý rating và số review...")
        if 'rating_average' in self.df.columns:
            self.df['rating_average'] = self.df['rating_average'].apply(
                ValueExtractor.safe_to_numeric
            )

        if 'num_reviews' in self.df.columns:
            self.df['num_reviews'] = self.df['num_reviews'].apply(
                ValueExtractor.safe_to_numeric
            )

        print(f"✓ Rating và review_count đã được chuẩn hóa\n")

    def clean_quantity_sold(self):
        """Bước 6: Xử lý số lượng đã bán"""
        print("📦 Bước 5: Xử lý số lượng đã bán...")
        if 'quantity_sold_text' in self.df.columns:
            self.df['quantity_sold'] = self.df['quantity_sold_text'].apply(
                lambda x: ValueExtractor.extract_sold_value(
                    x) if isinstance(x, str) else None
            )
        print(f"✓ Quantity sold đã được chuẩn hóa\n")

    def clean_brand(self):
        """Bước 6.5: Xử lý brand - normalize các biến thể 'No Brand'"""
        print("🏷️  Bước 5.5: Xử lý brand...")
        if 'brand' in self.df.columns:
            def normalize_brand(value):
                if value is None or pd.isna(value):
                    return "UNKNOWN"

                value_str = str(value).strip()
                # Normalize các biến thể của "No Brand"
                if value_str.lower() in ["no brand", "no.brand", "nobrand", "none", "n/a", ""]:
                    return "UNKNOWN"

                return value_str if value_str else "UNKNOWN"

            self.df['brand'] = self.df['brand'].apply(normalize_brand)
        print(f"✓ Brand đã được chuẩn hóa\n")

    def handle_missing_data(self):
        """Bước 7: Xử lý dữ liệu thiếu"""
        print("🧹 Bước 6: Xử lý dữ liệu thiếu...")
        print(f"  - Dữ liệu thiếu trước xử lý:")
        print(self.df.isnull().sum())

        # Xử lý rating_average
        if 'rating_average' in self.df.columns:
            self.df["rating_average_missing"] = self.df["rating_average"].isna().astype(
                int)
            self.df["rating_average"].fillna(
                self.df["rating_average"].median(), inplace=True
            )

        # Xử lý discount_rate
        if 'discount_rate' in self.df.columns:
            self.df["discount_rate_missing"] = self.df["discount_rate"].isna().astype(
                int)
            self.df["discount_rate"].fillna(
                self.df["discount_rate"].median(), inplace=True
            )

        # Xử lý num_reviews và quantity_sold
        if 'num_reviews' in self.df.columns:
            self.df['num_reviews'] = self.df['num_reviews'].fillna(0)
        if 'quantity_sold' in self.df.columns:
            self.df['quantity_sold'] = self.df['quantity_sold'].fillna(0)

        self.df['original_price'] = pd.to_numeric(
            self.df['original_price'], errors="coerce")

        # Điền giá trị mặc định cho cột text
        self.df[self.TEXT_COLUMNS] = self.df[self.TEXT_COLUMNS].fillna(
            "UNKNOWN")
        print(f"✓ Dữ liệu thiếu đã được xử lý\n")

    def remove_duplicates_and_invalid(self):
        """Bước 8: Loại bỏ dữ liệu trùng lặp và không hợp lệ"""
        print("🗑️  Bước 7: Loại bỏ dữ liệu không hợp lệ...")
        print(f"  - Số record trước khi loại bỏ: {len(self.df)}")

        # Sắp xếp theo chất lượng dữ liệu
        self.df = self.df.sort_values(
            by=[
                "quantity_sold",
                "num_reviews",
                "rating_average_missing" if "rating_average_missing" in self.df.columns else "id"
            ],
            ascending=[False, False, True]
        )

        # Loại bỏ trùng lặp
        self.df = self.df.drop_duplicates(
            subset=self.DEDUP_KEYS,
            keep="first"
        )
        self.df = self.df.reset_index(drop=True)

        # Loại bỏ record không có tên sản phẩm
        if 'product_name' in self.df.columns:
            self.df = self.df[self.df['product_name'].notna()]

        # Loại bỏ record có giá <= 0 hoặc null
        if 'current_price' in self.df.columns:
            self.df = self.df[self.df['current_price'] > 0]
            self.df = self.df[self.df['current_price'].notna()]

        print(f"  - Số record sau khi loại bỏ: {len(self.df)}\n")

    def select_final_columns(self):
        """Bước 9: Chọn các cột cần thiết"""
        print("📋 Bước 8: Chọn cột cần thiết...")

        # Chỉ lấy các cột tồn tại
        available_columns = [
            col for col in self.FINAL_COLUMNS if col in self.df.columns]
        self.df = self.df[available_columns]

        print(f"✓ Cột cuối cùng: {len(self.df.columns)} cột\n")

    def save_data(self):
        """Bước 10: Lưu dữ liệu"""
        print(f"💾 Bước 9: Lưu dữ liệu...")
        print(f"  - Output file: {self.output_file}")

        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.df.to_dict('records'), f,
                      ensure_ascii=False, indent=2)

        print(f"✓ Dữ liệu đã được lưu\n")

    def print_statistics(self):
        """Bước 11: In thống kê tóm tắt"""
        print("=" * 60)
        print("📊 THỐNG KÊ TÓM TẮT")
        print("=" * 60)
        print(f"Tổng records: {len(self.df)}")

        print(f"\nThông tin giá:")
        if 'current_price' in self.df.columns:
            print(
                f"  - Giá hiện tại: {self.df['current_price'].min():.0f} - {self.df['current_price'].max():.0f}")
            print(f"  - Trung bình: {self.df['current_price'].mean():.0f}")

        print(f"\nThông tin đánh giá:")
        if 'rating_average' in self.df.columns:
            print(
                f"  - Rating trung bình: {self.df['rating_average'].mean():.2f}")
        if 'num_reviews' in self.df.columns:
            print(
                f"  - Review trung bình: {self.df['num_reviews'].mean():.0f}")

        if 'platform' in self.df.columns:
            print(f"\nPlatform:")
            print(self.df['platform'].value_counts())

        if 'category' in self.df.columns:
            print(f"\nTop 5 Categories:")
            print(self.df['category'].value_counts().head())

        if 'brand' in self.df.columns:
            print(f"\nTop 5 Brands:")
            print(self.df['brand'].value_counts().head())

        if 'quality_category' in self.df.columns:
            print(f"\nQuality Distribution:")
            print(self.df['quality_category'].value_counts())

        if 'popularity_category' in self.df.columns:
            print(f"\nPopularity Distribution:")
            print(self.df['popularity_category'].value_counts())

        print("=" * 60)
        print("\n💡 Để xem biểu đồ trực quan hóa, chạy:")
        print("   python main/visualize_cleaning_results.py")
        print("=" * 60)

    def clean(self):
        """Thực hiện toàn bộ quá trình làm sạch dữ liệu"""
        print("\n🚀 BẮT ĐẦU LÀM SẠCH DỮ LIỆU")
        print("=" * 60 + "\n")

        self.load_data()
        self.normalize_data()
        self.clean_prices()
        self.clean_discount()
        self.clean_ratings()
        self.clean_quantity_sold()
        self.clean_brand()
        self.handle_missing_data()
        self.remove_duplicates_and_invalid()
        self.select_final_columns()
        self.save_data()
        self.print_statistics()

        print("\n✅ HOÀN THÀNH!\n")
        return self.df


# Hàm wrapper để tương thích với code cũ
def clean_merged_data(input_file, output_file=None):
    """
    Làm sạch dữ liệu từ file merged_raw_data.json

    Parameters:
    - input_file: đường dẫn file .json đầu vào
    - output_file: đường dẫn file output (mặc định: data/clean/merged_cleaned_data.json)
    """
    cleaner = DataCleaner(input_file, output_file)
    return cleaner.clean()


if __name__ == "__main__":
    # Đường dẫn input/output
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(
        base, 'data/preliminary/merged_preliminary_data.json')
    output_file = os.path.join(base, 'data/clean/merged_cleaned_data.json')

    # Sử dụng class DataCleaner
    cleaner = DataCleaner(input_file, output_file)
    df_cleaned = cleaner.clean()
