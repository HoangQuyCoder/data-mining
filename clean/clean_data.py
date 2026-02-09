
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
import re
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Thiết lập font tiếng Việt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False


class OutlierHandler:
    """Class xử lý ngoại lệ trong dữ liệu"""

    def __init__(self, df):
        self.df = df.copy()
        self.df_original = df.copy()
        self.outliers_log = []

    def log_changes(self, step, before, after, details=""):
        """Ghi lại các thay đổi"""
        log_entry = {
            'step': step,
            'records_before': before,
            'records_after': after,
            'records_removed': before - after,
            'removal_rate': f"{((before - after) / before * 100):.2f}%",
            'details': details
        }
        self.outliers_log.append(log_entry)
        print(f"\n{'='*80}")
        print(f"[{step}]")
        print(f"   Records trước: {before:,}")
        print(f"   Records sau: {after:,}")
        print(
            f"   Đã loại bỏ: {before - after:,} ({log_entry['removal_rate']})")
        if details:
            print(f"   Chi tiết: {details}")

    def handle_price_outliers(self, lower_percentile=0.1, upper_percentile=99.9):
        """
        Xử lý ngoại lệ về giá
        - Loại bỏ giá quá thấp (dưới 1,000 VNĐ - có thể là lỗi)
        - Loại bỏ giá quá cao (trên percentile 99.9)
        """
        print("\n" + "="*80)
        print("XỬ LÝ NGOẠI LỆ VỀ GIÁ")
        print("="*80)

        before = len(self.df)

        # Tính ngưỡng
        min_valid_price = 1000  # Giá tối thiểu hợp lệ
        price_upper = self.df['current_price'].quantile(upper_percentile / 100)

        print(f"\nThống kê giá ban đầu:")
        print(f"   - Min: {self.df['current_price'].min():,.0f} VNĐ")
        print(f"   - Max: {self.df['current_price'].max():,.0f} VNĐ")
        print(f"   - Mean: {self.df['current_price'].mean():,.0f} VNĐ")
        print(f"   - Median: {self.df['current_price'].median():,.0f} VNĐ")
        print(f"   - Percentile {upper_percentile}%: {price_upper:,.0f} VNĐ")

        # Lọc giá
        price_outliers = self.df[
            (self.df['current_price'] < min_valid_price) |
            (self.df['current_price'] > price_upper)
        ]

        print(f"\nPhát hiện {len(price_outliers):,} ngoại lệ về giá:")
        print(
            f"   - Giá < {min_valid_price:,} VNĐ: {len(self.df[self.df['current_price'] < min_valid_price]):,} records")
        print(
            f"   - Giá > {price_upper:,.0f} VNĐ: {len(self.df[self.df['current_price'] > price_upper]):,} records")

        self.df = self.df[
            (self.df['current_price'] >= min_valid_price) &
            (self.df['current_price'] <= price_upper)
        ]

        after = len(self.df)
        self.log_changes(
            "Xử lý giá outliers",
            before,
            after,
            f"Loại bỏ giá < {min_valid_price:,} VNĐ và > {price_upper:,.0f} VNĐ"
        )

        print(f"\nThống kê giá sau xử lý:")
        print(f"   - Min: {self.df['current_price'].min():,.0f} VNĐ")
        print(f"   - Max: {self.df['current_price'].max():,.0f} VNĐ")
        print(f"   - Mean: {self.df['current_price'].mean():,.0f} VNĐ")
        print(f"   - Median: {self.df['current_price'].median():,.0f} VNĐ")

        return self

    def handle_quantity_sold_outliers(self, upper_percentile=99.5):
        """
        Xử lý ngoại lệ về số lượng bán
        - Loại bỏ các giá trị bất thường quá cao
        """
        print("\n" + "="*80)
        print("XỬ LÝ NGOẠI LỆ VỀ SỐ LƯỢNG BÁN")
        print("="*80)

        before = len(self.df)

        # Chỉ xử lý các records có dữ liệu quantity_sold_value
        has_qty = self.df['quantity_sold'].notna()

        if has_qty.sum() == 0:
            print("Không có dữ liệu quantity_sold để xử lý")
            return self

        qty_upper = self.df.loc[has_qty, 'quantity_sold'].quantile(
            upper_percentile / 100)

        print(f"\nThống kê số lượng bán ban đầu (có dữ liệu):")
        print(f"   - Số records có dữ liệu: {has_qty.sum():,}")
        print(f"   - Min: {self.df.loc[has_qty, 'quantity_sold'].min():,.0f}")
        print(f"   - Max: {self.df.loc[has_qty, 'quantity_sold'].max():,.0f}")
        print(
            f"   - Mean: {self.df.loc[has_qty, 'quantity_sold'].mean():,.0f}")
        print(
            f"   - Median: {self.df.loc[has_qty, 'quantity_sold'].median():,.0f}")
        print(f"   - Percentile {upper_percentile}%: {qty_upper:,.0f}")

        # Lọc outliers
        qty_outliers = (has_qty) & (self.df['quantity_sold'] > qty_upper)

        print(
            f"\nPhát hiện {qty_outliers.sum():,} ngoại lệ về số lượng bán (> {qty_upper:,.0f})")

        self.df = self.df[~qty_outliers]

        after = len(self.df)
        self.log_changes(
            "Xử lý số lượng bán outliers",
            before,
            after,
            f"Loại bỏ quantity_sold > {qty_upper:,.0f}"
        )

        has_qty_after = self.df['quantity_sold'].notna()
        if has_qty_after.sum() > 0:
            print(f"\nThống kê số lượng bán sau xử lý:")
            print(f"   - Số records có dữ liệu: {has_qty_after.sum():,}")
            print(
                f"   - Min: {self.df.loc[has_qty_after, 'quantity_sold'].min():,.0f}")
            print(
                f"   - Max: {self.df.loc[has_qty_after, 'quantity_sold'].max():,.0f}")
            print(
                f"   - Mean: {self.df.loc[has_qty_after, 'quantity_sold'].mean():,.0f}")
            print(
                f"   - Median: {self.df.loc[has_qty_after, 'quantity_sold'].median():,.0f}")

        return self

    def handle_discount_outliers(self, max_discount=95):
        """
        Xử lý ngoại lệ về discount rate
        - Loại bỏ discount > 95% (thường là lỗi hoặc chiêu trò marketing)
        """
        print("\n" + "="*80)
        print("XỬ LÝ NGOẠI LỆ VỀ DISCOUNT")
        print("="*80)

        before = len(self.df)

        has_discount = self.df['discount_rate'].notna()

        if has_discount.sum() == 0:
            print("Không có dữ liệu discount_rate để xử lý")
            return self

        print(f"\nThống kê discount ban đầu:")
        print(f"   - Số records có dữ liệu: {has_discount.sum():,}")
        print(
            f"   - Min: {self.df.loc[has_discount, 'discount_rate'].min():.1f}%")
        print(
            f"   - Max: {self.df.loc[has_discount, 'discount_rate'].max():.1f}%")
        print(
            f"   - Mean: {self.df.loc[has_discount, 'discount_rate'].mean():.1f}%")
        print(
            f"   - Median: {self.df.loc[has_discount, 'discount_rate'].median():.1f}%")

        # Lọc discount bất thường
        discount_outliers = (has_discount) & (
            self.df['discount_rate'] > max_discount)

        print(
            f"\nPhát hiện {discount_outliers.sum():,} ngoại lệ về discount (> {max_discount}%)")

        self.df = self.df[~discount_outliers]

        after = len(self.df)
        self.log_changes(
            "Xử lý discount outliers",
            before,
            after,
            f"Loại bỏ discount > {max_discount}%"
        )

        has_discount_after = self.df['discount_rate'].notna()
        if has_discount_after.sum() > 0:
            print(f"\nThống kê discount sau xử lý:")
            print(f"   - Số records có dữ liệu: {has_discount_after.sum():,}")
            print(
                f"   - Min: {self.df.loc[has_discount_after, 'discount_rate'].min():.1f}%")
            print(
                f"   - Max: {self.df.loc[has_discount_after, 'discount_rate'].max():.1f}%")
            print(
                f"   - Mean: {self.df.loc[has_discount_after, 'discount_rate'].mean():.1f}%")
            print(
                f"   - Median: {self.df.loc[has_discount_after, 'discount_rate'].median():.1f}%")

        return self

    def handle_num_reviews_outliers(
        self,
        upper_percentile=99.7,
        max_review_per_sold_ratio=1.0,
        strategy="cap"  # "cap" | "drop"
    ):
        """
        Xử lý outlier cho num_reviews theo nghiệp vụ

        - Chỉ xử lý sản phẩm đã bán
        - Cap hoặc drop review quá cao
        - Đảm bảo num_reviews <= quantity_sold * ratio
        """

        print("\n" + "="*80)
        print("XỬ LÝ OUTLIER NUM_REVIEWS")
        print("="*80)

        before = len(self.df)

        # Chỉ xử lý sản phẩm đã bán
        mask_sold = (
            self.df['num_reviews'].notna() &
            self.df['quantity_sold'].notna() &
            (self.df['quantity_sold'] > 0)
        )

        if mask_sold.sum() == 0:
            print("Không có dữ liệu num_reviews hợp lệ để xử lý")
            return self

        # ================================
        # Thống kê ban đầu
        # ================================
        print(f"\nThống kê num_reviews ban đầu:")
        print(f"   - Records xử lý: {mask_sold.sum():,}")
        print(f"   - Min: {self.df.loc[mask_sold, 'num_reviews'].min():,}")
        print(f"   - Max: {self.df.loc[mask_sold, 'num_reviews'].max():,}")
        print(
            f"   - Mean: {self.df.loc[mask_sold, 'num_reviews'].mean():,.1f}")
        print(
            f"   - Median: {self.df.loc[mask_sold, 'num_reviews'].median():,.1f}")

        # ================================
        # 1. Percentile cap
        # ================================
        review_upper = self.df.loc[mask_sold, 'num_reviews'].quantile(
            upper_percentile / 100)

        print(f"   - Percentile {upper_percentile}%: {review_upper:,.0f}")

        # ================================
        # 2. Logic nghiệp vụ: review <= sold * ratio
        # ================================
        max_allowed_review = self.df['quantity_sold'] * \
            max_review_per_sold_ratio

        outlier_mask = (
            mask_sold &
            (
                (self.df['num_reviews'] > review_upper) |
                (self.df['num_reviews'] > max_allowed_review)
            )
        )

        outlier_count = outlier_mask.sum()
        print(f"\nPhát hiện {outlier_count:,} outlier num_reviews")

        if outlier_count == 0:
            print("   ✅ Không có outlier cần xử lý")
            return self

        # ================================
        # 3. Xử lý
        # ================================
        if strategy == "drop":
            self.df = self.df[~outlier_mask]
            print(f"   → Drop {outlier_count:,} records")
        else:
            # Cap về ngưỡng hợp lệ nhất
            cap_value = np.minimum(review_upper, max_allowed_review)
            self.df.loc[outlier_mask,
                        'num_reviews'] = cap_value[outlier_mask].astype('int64')
            print(f"   → Cap num_reviews về min(percentile, sold×ratio)")

        after = len(self.df)

        self.log_changes(
            "Xử lý num_reviews outliers",
            before,
            after,
            f"strategy={strategy}, percentile={upper_percentile}, max_review_per_sold={max_review_per_sold_ratio}"
        )

        # ================================
        # Thống kê sau xử lý
        # ================================
        mask_after = (
            self.df['num_reviews'].notna() &
            (self.df['quantity_sold'] > 0)
        )

        print(f"\nThống kê num_reviews sau xử lý:")
        print(f"   - Max: {self.df.loc[mask_after, 'num_reviews'].max():,}")
        print(
            f"   - Mean: {self.df.loc[mask_after, 'num_reviews'].mean():,.1f}")
        print(
            f"   - Median: {self.df.loc[mask_after, 'num_reviews'].median():,.1f}")

        return self

    def handle_rating_outliers(self):
        """
        Xử lý ngoại lệ về rating
        - Loại bỏ rating = 0 (thường là sản phẩm không có đánh giá thực sự)
        - Giữ lại records có rating từ 0.1 đến 5.0
        """
        print("\n" + "="*80)
        print("XỬ LÝ NGOẠI LỆ VỀ RATING")
        print("="*80)

        before = len(self.df)

        has_rating = self.df['rating_average'].notna()

        if has_rating.sum() == 0:
            print("Không có dữ liệu rating_average để xử lý")
            return self

        print(f"\nThống kê rating ban đầu:")
        print(f"   - Số records có rating: {has_rating.sum():,}")
        print(f"   - Rating = 0: {(self.df['rating_average'] == 0).sum():,}")
        print(
            f"   - Rating trung bình: {self.df.loc[has_rating, 'rating_average'].mean():.2f}")

        # Loại bỏ rating = 0
        rating_zero = (has_rating) & (self.df['rating_average'] == 0)
        print(f"\nLoại bỏ {rating_zero.sum():,} records có rating = 0")

        self.df = self.df[~rating_zero]

        after = len(self.df)
        self.log_changes(
            "Xử lý rating outliers",
            before,
            after,
            "Loại bỏ rating = 0"
        )

        has_rating_after = self.df['rating_average'].notna()
        if has_rating_after.sum() > 0:
            print(f"\nThống kê rating sau xử lý:")
            print(f"   - Số records có rating: {has_rating_after.sum():,}")
            print(
                f"   - Rating trung bình: {self.df.loc[has_rating_after, 'rating_average'].mean():.2f}")

        return self

    def generate_comparison_report(self, output_dir='outlier_analysis'):
        """Tạo báo cáo so sánh trước và sau xử lý"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*80)
        print("TẠO BÁO CÁO SO SÁNH")
        print("="*80)

        # 1. So sánh phân bố giá
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Biểu đồ 1: Phân bố giá (log scale)
        axes[0, 0].hist(self.df_original['current_price'],
                        bins=100, alpha=0.5, label='Trước xử lý', color='red')
        axes[0, 0].hist(self.df['current_price'], bins=100,
                        alpha=0.5, label='Sau xử lý', color='green')
        axes[0, 0].set_xlabel('Giá (VNĐ)')
        axes[0, 0].set_ylabel('Số lượng')
        axes[0, 0].set_title('So sánh phân bố giá')
        axes[0, 0].set_yscale('log')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Biểu đồ 2: Boxplot giá
        box_data = [
            self.df_original['current_price'].values,
            self.df['current_price'].values
        ]
        axes[0, 1].boxplot(box_data, labels=['Trước xử lý', 'Sau xử lý'])
        axes[0, 1].set_ylabel('Giá (VNĐ)')
        axes[0, 1].set_title('Boxplot so sánh giá')
        axes[0, 1].grid(True, alpha=0.3)

        # Biểu đồ 3: Phân bố discount
        has_discount_orig = self.df_original['discount_rate'].notna()
        has_discount_new = self.df['discount_rate'].notna()

        if has_discount_orig.sum() > 0 and has_discount_new.sum() > 0:
            axes[1, 0].hist(self.df_original.loc[has_discount_orig, 'discount_rate'],
                            bins=50, alpha=0.5, label='Trước xử lý', color='red')
            axes[1, 0].hist(self.df.loc[has_discount_new, 'discount_rate'],
                            bins=50, alpha=0.5, label='Sau xử lý', color='green')
            axes[1, 0].set_xlabel('Discount (%)')
            axes[1, 0].set_ylabel('Số lượng')
            axes[1, 0].set_title('So sánh phân bố Discount')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Biểu đồ 4: So sánh category distribution
        top_cats_orig = self.df_original['category'].value_counts().head(10)
        top_cats_new = self.df['category'].value_counts().head(10)

        x = np.arange(len(top_cats_orig))
        width = 0.35

        axes[1, 1].bar(x - width/2, top_cats_orig.values, width,
                       label='Trước xử lý', alpha=0.8, color='red')
        axes[1, 1].bar(x + width/2, [top_cats_new.get(cat, 0) for cat in top_cats_orig.index],
                       width, label='Sau xử lý', alpha=0.8, color='green')
        axes[1, 1].set_xlabel('Category')
        axes[1, 1].set_ylabel('Số lượng')
        axes[1, 1].set_title('So sánh Top 10 Categories')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([cat[:20] + '...' if len(cat) > 20 else cat
                                    for cat in top_cats_orig.index], rotation=45, ha='right')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/comparison_report.png',
                    dpi=300, bbox_inches='tight')
        print(f"   ✓ Đã lưu: {output_dir}/comparison_report.png")
        plt.close()

        # 2. Tạo bảng tóm tắt
        summary_df = pd.DataFrame(self.outliers_log)
        summary_df.to_csv(
            f'{output_dir}/outlier_handling_summary.csv', index=False, encoding='utf-8-sig')
        print(f"   ✓ Đã lưu: {output_dir}/outlier_handling_summary.csv")

        # 3. In báo cáo cuối cùng
        print("\n" + "="*80)
        print("TÓM TẮT CUỐI CÙNG")
        print("="*80)
        print(f"\nDữ liệu ban đầu: {len(self.df_original):,} records")
        print(f"Dữ liệu sau xử lý: {len(self.df):,} records")
        print(
            f"Tổng đã loại bỏ: {len(self.df_original) - len(self.df):,} records")
        print(
            f"Tỷ lệ giữ lại: {len(self.df) / len(self.df_original) * 100:.2f}%")

        print("\nChi tiết các bước xử lý:")
        for log in self.outliers_log:
            print(f"\n[{log['step']}]")
            print(
                f"   - Loại bỏ: {log['records_removed']:,} records ({log['removal_rate']})")
            if log['details']:
                print(f"   - {log['details']}")

        return self

    def get_cleaned_data(self):
        """Trả về dữ liệu đã làm sạch"""
        return self.df


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
            "discount_rate": "discount_rate",
            "rating_average": "rating_average",
            "num_reviews": "review_count",
            "quantity_sold": "quantity_sold_value",
            "quantity_sold_text": "quantity_sold_text",
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
        'rating_average', 'num_reviews',
        'quantity_sold',
        'brand', 'seller_location',
        'product_url',
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
                    return "No Brand"

                value_str = str(value).strip()
                # Normalize các biến thể của "No Brand"
                if value_str.lower() in ["No brand", "no brand", "no.brand", "nobrand", "none", "n/a", ""]:
                    return "No Brand"

                return value_str if value_str else "No Brand"

            self.df['brand'] = self.df['brand'].apply(normalize_brand)
        print(f"✓ Brand đã được chuẩn hóa\n")

    def handle_missing_data(self):
        """Bước 6: Xử lý missing values"""
        print("🧹 Bước 6: Xử lý missing values...")

        before_total = len(self.df)
        print(f"   Trước xử lý: {before_total:,} records")

        CRITICAL_COLUMNS = [
            'quantity_sold',
            'num_reviews',
            'rating_average',
            'discount_rate',
        ]

        critical_cols_present = [
            col for col in CRITICAL_COLUMNS if col in self.df.columns]

        if not critical_cols_present:
            print("   ⚠️ Không tìm thấy cột critical nào → không drop theo missing")
        else:
            print(
                f"   Các cột bắt buộc kiểm tra: {', '.join(critical_cols_present)}")

            # Đếm missing ở các cột critical
            missing_critical = self.df[critical_cols_present].isna().sum(
                axis=1)
            rows_to_drop = missing_critical > 0
            num_drop_critical = rows_to_drop.sum()

            if num_drop_critical > 0:
                print(
                    f"   → Drop {num_drop_critical:,} records thiếu ít nhất 1 cột critical")
                self.df = self.df[~rows_to_drop].copy()

        # đảm bảo là số nguyên, không âm
        if 'quantity_sold' in self.df.columns:
            self.df['quantity_sold'] = self.df['quantity_sold'].clip(
                lower=0).astype('int64')

        # ép logic + tạo feature
        if 'num_reviews' in self.df.columns:
            mask_not_sold = self.df['quantity_sold'] == 0
            conflict = self.df.loc[mask_not_sold & (
                self.df['num_reviews'] > 0)].shape[0]
            if conflict > 0:
                print(
                    f"   ⚠️ Fix {conflict:,} records: sold=0 nhưng review>0 → set review=0")
                self.df.loc[mask_not_sold, 'num_reviews'] = 0

            self.df['num_reviews'] = self.df['num_reviews'].clip(
                lower=0).astype('int64')
            self.df['has_reviews'] = (
                self.df['num_reviews'] > 0).astype('int8')

            zero_rev = (self.df['num_reviews'] == 0).sum()
            print(
                f"   - num_reviews = 0: {zero_rev:,} ({zero_rev/len(self.df)*100:.1f}%)")

        # discount_rate
        if 'discount_rate' in self.df.columns:
            self.df['discount_rate'] = self.df['discount_rate'].clip(0, 100)
            self.df['has_discount'] = (
                self.df['discount_rate'] > 0).astype('int8')

        # ép logic + giới hạn
        if 'rating_average' in self.df.columns:
            mask_no_review = (self.df['num_reviews'] == 0)
            invalid = self.df.loc[mask_no_review &
                                  self.df['rating_average'].notna()].shape[0]
            if invalid > 0:
                print(
                    f"   ⚠️ Fix {invalid:,} records: no review nhưng có rating → set NaN")
                self.df.loc[mask_no_review, 'rating_average'] = np.nan

            # Giới hạn giá trị
            self.df['rating_average'] = self.df['rating_average'].clip(1, 5)
            
            # Fill rating_average NaN → 0 (không có review)
            num_filled = self.df['rating_average'].isna().sum()
            if num_filled > 0:
                self.df['rating_average'] = self.df['rating_average'].fillna(0)
                print(f"   - rating_average: fill {num_filled:,} NaN → 0 (no review)")

        TEXT_FILL = {
            'brand': 'No Brand',
            'seller_location': 'Unknown Location',
            'quantity_sold_text': 'Chưa có thông tin bán'
        }

        for col, val in TEXT_FILL.items():
            if col in self.df.columns:
                miss = self.df[col].isna().sum()
                if miss > 0:
                    self.df[col] = self.df[col].fillna(val).str.strip()
                    print(f"   - {col}: fill {miss:,} missing → '{val}'")


        after_total = len(self.df)
        dropped = before_total - after_total

        print("   ✓ Hoàn thành preprocessing")
        print("\n" + "="*60)
        print("📊 KẾT QUẢ SAU XỬ LÝ MISSING (DROP)")
        print("="*60)
        print(f"   Trước xử lý     : {before_total:>12,} records")
        print(f"   Sau khi drop    : {after_total:>12,} records")
        print(
            f"   Đã loại bỏ      : {dropped:>12,} records ({dropped/before_total*100:.1f}% nếu >0)")
        print("✓ Hoàn thành xử lý missing values – chỉ giữ record có dữ liệu đầy đủ\n")

    def remove_duplicates_and_invalid(self):
        """Bước 8: Loại bỏ dữ liệu trùng lặp"""
        print("🗑️  Bước 7: Loại bỏ dữ liệu không hợp lệ...")
        print(f"  - Số record trước khi loại bỏ: {len(self.df)}")

        # Sắp xếp theo chất lượng dữ liệu
        self.df = self.df.sort_values(
            by=[
                "quantity_sold",
                "num_reviews",
            ],
            ascending=[False, False]
        )

        # Loại bỏ trùng lặp
        self.df = self.df.drop_duplicates(
            subset=self.DEDUP_KEYS,
            keep="first"
        )
        self.df = self.df.reset_index(drop=True)

        print(f"  - Số record sau khi loại bỏ: {len(self.df)}\n")

    def handle_outliers(self):
        """Bước 9: Xử lý outlier sau khi đã làm sạch cơ bản"""
        print("\n" + "="*60)
        print("🗑️ Bước 8: Xử lý outliers...")
        print("="*60 + "\n")

        handler = OutlierHandler(self.df)

        handler\
            .handle_price_outliers(upper_percentile=99.5)\
            .handle_quantity_sold_outliers(upper_percentile=99.0)\
            .handle_discount_outliers(max_discount=80)\
            .handle_rating_outliers()\
            .handle_num_reviews_outliers(
                upper_percentile=99.7,
                max_review_per_sold_ratio=1.0,
                strategy="drop"
            )

        # Lưu báo cáo (tuỳ chọn)
        handler.generate_comparison_report(output_dir='data/outlier_analysis')

        self.df = handler.get_cleaned_data()
        print(f"→ Sau xử lý outlier: {len(self.df):,} records\n")
        return self

    def select_final_columns(self):
        """Bước 10: Chọn các cột cần thiết"""
        print("📋 Bước 9: Chọn cột cần thiết...")

        # Chỉ lấy các cột tồn tại
        available_columns = [
            col for col in self.FINAL_COLUMNS if col in self.df.columns]
        self.df = self.df[available_columns]

        print(f"✓ Cột cuối cùng: {len(self.df.columns)} cột\n")

    def save_data(self):
        """Bước 11: Lưu dữ liệu"""
        print(f"💾 Bước 10: Lưu dữ liệu...")
        print(f"  - Output file: {self.output_file}")

        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(self.df.to_dict('records'), f,
                      ensure_ascii=False, indent=2)

        print(f"✓ Dữ liệu đã được lưu\n")

    def print_statistics(self):
        """Bước 12: In thống kê tóm tắt"""
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

        print("=" * 60)
        print("\n💡 Để xem biểu đồ trực quan hóa, chạy:")
        print("   python visualizations/visualize_cleaned_data.py")
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
        self.handle_outliers()
        self.select_final_columns()
        self.save_data()
        self.print_statistics()

        print("\n✅ HOÀN THÀNH!\n")
        return self.df


if __name__ == "__main__":
    # Đường dẫn input/output
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(
        base, 'data/raw/merged_raw_data.json')
    output_file = os.path.join(base, 'data/cleaned/merged_cleaned_data.json')

    # Sử dụng class DataCleaner
    cleaner = DataCleaner(input_file, output_file)
    df_cleaned = cleaner.clean()
