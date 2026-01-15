import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional


class FeatureEngineer:
    # Feature Engineering cho phân vùng sản phẩm (Product Segmentation)

    def __init__(self, df: pd.DataFrame):
        """
        Khởi tạo Feature Engineer

        Parameters:
        - df: DataFrame đã được làm sạch từ clean_merged_data
        """
        self.df = df.copy()
        self.stats = {}

    def engineer_features(self) -> pd.DataFrame:
        """Chạy tất cả feature engineering steps"""

        print("🔧 BẮT ĐẦU FEATURE ENGINEERING")
        print("=" * 70)

        # 1. Quality Segmentation
        print("\n✓ Bước 1: Xây dựng Quality Category...")
        self._create_quality_category()

        # 2. Popularity Segmentation
        print("✓ Bước 2: Xây dựng Popularity Category...")
        self._create_popularity_category()

        # 3. Price Segmentation
        print("✓ Bước 3: Xây dựng Price Segment...")
        self._create_price_segment()

        # 4. Seller Tier
        print("✓ Bước 4: Xây dựng Seller Tier...")
        self._create_seller_tier()

        # 5. Brand Strength
        print("✓ Bước 5: Xây dựng Brand Strength...")
        self._create_brand_strength()

        # 6. Engagement Score
        print("✓ Bước 6: Xây dựng Engagement Score...")
        self._create_engagement_score()

        # 7. Value Score
        print("✓ Bước 7: Xây dựng Value Score...")
        self._create_value_score()

        # 8. Product Lifecycle Status
        print("✓ Bước 8: Xây dựng Product Lifecycle Status...")
        self._create_lifecycle_status()

        # 9. Discount Intensity
        print("✓ Bước 9: Xây dựng Discount Intensity...")
        self._create_discount_intensity()

        print("\n" + "=" * 70)
        print(f"✅ HOÀN THÀNH FEATURE ENGINEERING")
        print(f"   Tổng features mới: {self._count_new_features()}")

        return self.df

    def _create_quality_category(self):
        """
        Phân vùng chất lượng dựa trên rating_average và num_reviews

        Categories:
        - Premium: rating >= 4.5 && reviews >= median
        - Excellent: rating >= 4.3 && reviews >= Q1
        - Good: rating >= 3.8 && reviews >= Q1
        - Average: rating >= 3.0 && reviews >= 0
        - Unknown: rating is missing
        - Poor: rating < 3.0
        """

        # Tính toán thống kê
        median_reviews = self.df['num_reviews'].median()
        q1_reviews = self.df['num_reviews'].quantile(0.25)

        self.stats['quality'] = {
            'median_reviews': median_reviews,
            'q1_reviews': q1_reviews
        }

        def categorize_quality(row):
            rating = row['rating_average']
            reviews = row['num_reviews']

            # Xử lý missing rating
            if pd.isna(rating) or row['rating_average_missing'] == 1:
                return 'Unknown'

            # Phân loại dựa trên rating
            if rating >= 4.5:
                if reviews >= median_reviews:
                    return 'Premium'
                else:
                    return 'Excellent'
            elif rating >= 4.3:
                return 'Excellent'
            elif rating >= 3.8:
                if reviews >= q1_reviews:
                    return 'Good'
                else:
                    return 'Good'
            elif rating >= 3.0:
                return 'Average'
            else:
                return 'Poor'

        self.df['quality_category'] = self.df.apply(categorize_quality, axis=1)

        print(f"   Distribution:")
        for cat, count in self.df['quality_category'].value_counts().items():
            print(f"     - {cat}: {count:,} ({count/len(self.df)*100:.1f}%)")

    def _create_popularity_category(self):
        """
        Phân vùng độ phổ biến dựa trên quantity_sold, rating, num_reviews

        Categories:
        - Viral: quantity_sold >= P90 (top 10%)
        - Hot: quantity_sold >= P75 (top 25%)
        - Trending: quantity_sold >= P50 (top 50%)
        - Normal: quantity_sold >= P25 (bottom 50%)
        - New: quantity_sold < P25 && (low reviews or new)
        """

        # Tính percentiles
        p90 = self.df['quantity_sold'].quantile(0.90)
        p75 = self.df['quantity_sold'].quantile(0.75)
        p50 = self.df['quantity_sold'].quantile(0.50)
        p25 = self.df['quantity_sold'].quantile(0.25)

        self.stats['popularity'] = {
            'p90': p90, 'p75': p75, 'p50': p50, 'p25': p25
        }

        def categorize_popularity(row):
            sold = row['quantity_sold']
            rating = row['rating_average']
            reviews = row['num_reviews']

            if sold >= p90:
                return 'Viral'
            elif sold >= p75:
                return 'Hot'
            elif sold >= p50:
                return 'Trending'
            elif sold >= p25:
                return 'Normal'
            else:
                # New products có ít review và sold
                if reviews < 10:
                    return 'New'
                else:
                    return 'Normal'

        self.df['popularity_category'] = self.df.apply(
            categorize_popularity, axis=1
        )

        print(f"   Distribution:")
        for cat, count in self.df['popularity_category'].value_counts().items():
            print(f"     - {cat}: {count:,} ({count/len(self.df)*100:.1f}%)")

    def _create_price_segment(self):
        """
        Phân vùng giá dựa trên mức giá tuyệt đối

        Sử dụng quartiles của current_price để chia thành 4 segments
        """

        # Tính quartiles
        q1 = self.df['current_price'].quantile(0.25)
        q2 = self.df['current_price'].quantile(0.50)
        q3 = self.df['current_price'].quantile(0.75)

        self.stats['price_segment'] = {
            'q1': q1, 'q2': q2, 'q3': q3
        }

        def categorize_price(price):
            if price <= q1:
                return 'Budget'
            elif price <= q2:
                return 'Economy'
            elif price <= q3:
                return 'Premium'
            else:
                return 'Luxury'

        self.df['price_segment'] = self.df['current_price'].apply(
            categorize_price
        )

        print(f"   Distribution:")
        for seg, count in self.df['price_segment'].value_counts().items():
            print(f"     - {seg}: {count:,} ({count/len(self.df)*100:.1f}%)")

    def _create_seller_tier(self):
        """
        Phân vùng người bán dựa trên chất lượng và khối lượng bán hàng

        Tiers:
        - Elite: rating >= 4.5 && quantity_sold >= P75
        - Pro: rating >= 4.0 && quantity_sold >= P50
        - Standard: rating >= 3.5 && quantity_sold >= 0
        - New: quantity_sold < P25 or rating < 3.5
        """

        p75_sold = self.df['quantity_sold'].quantile(0.75)
        p50_sold = self.df['quantity_sold'].quantile(0.50)
        p25_sold = self.df['quantity_sold'].quantile(0.25)

        def categorize_seller(row):
            rating = row['rating_average']
            sold = row['quantity_sold']

            if pd.isna(rating):
                return 'New'

            if rating >= 4.5 and sold >= p75_sold:
                return 'Elite'
            elif rating >= 4.0 and sold >= p50_sold:
                return 'Pro'
            elif rating >= 3.5:
                return 'Standard'
            else:
                return 'New'

        self.df['seller_tier'] = self.df.apply(categorize_seller, axis=1)

        print(f"   Distribution:")
        for tier, count in self.df['seller_tier'].value_counts().items():
            print(f"     - {tier}: {count:,} ({count/len(self.df)*100:.1f}%)")

    def _create_brand_strength(self):
        """
        Xác định sức mạnh của thương hiệu dựa trên:
        - Price premium so với trung bình category
        - Rating consistency
        - Market share (số lượng sản phẩm)

        Categories:
        - Strong: price premium + high rating
        - Moderate: neutral price + decent rating
        - Weak: price discount or low rating
        """

        # Pre-calculate category average prices (tránh tính lại nhiều lần)
        category_avg_prices = self.df.groupby(
            'category')['current_price'].mean().to_dict()

        def categorize_brand(row):
            brand = row['brand']

            # Unknown brand
            if brand is None or brand == 'UNKNOWN':
                return 'Unknown'

            # Tính price premium so với category average
            category = row['category']
            category_avg_price = category_avg_prices.get(category, 0)

            price = row['current_price']
            rating = row['rating_average']

            if pd.isna(category_avg_price) or category_avg_price == 0:
                price_premium = 0
            else:
                price_premium = (price - category_avg_price) / \
                    category_avg_price

            # Phân loại
            if (price_premium > 0.1) and (rating >= 4.2):
                return 'Strong'
            elif (price_premium <= 0.1) and (rating >= 3.8):
                return 'Moderate'
            else:
                return 'Weak'

        self.df['brand_strength'] = self.df.apply(categorize_brand, axis=1)

        print(f"   Distribution:")
        for strength, count in self.df['brand_strength'].value_counts().items():
            print(
                f"     - {strength}: {count:,} ({count/len(self.df)*100:.1f}%)")

    def _create_engagement_score(self):
        """
        Tính điểm engagement (0-100) dựa trên:
        - Rating (40%)
        - Review count (40%)
        - Quantity sold (20%)
        """

        # Normalize mỗi metric
        max_rating = 5.0
        max_reviews = self.df['num_reviews'].quantile(0.95)
        max_sold = self.df['quantity_sold'].quantile(0.95)

        self.stats['engagement'] = {
            'max_reviews': max_reviews,
            'max_sold': max_sold
        }

        rating_score = (
            (self.df['rating_average'] / max_rating) * 100
        ).fillna(0)

        review_score = (
            (self.df['num_reviews'] / max_reviews) * 100
        ).clip(0, 100)

        sold_score = (
            (self.df['quantity_sold'] / max_sold) * 100
        ).clip(0, 100)

        self.df['engagement_score'] = (
            (rating_score * 0.4) +
            (review_score * 0.4) +
            (sold_score * 0.2)
        ).round(2)

        print(
            f"   Range: {self.df['engagement_score'].min():.2f} - {self.df['engagement_score'].max():.2f}")
        print(f"   Mean: {self.df['engagement_score'].mean():.2f}")

    def _create_value_score(self):
        """
        Tính giá trị sản phẩm (0-100) dựa trên:
        - Discount rate (30%)
        - Price-to-rating ratio (40%)
        - Popularity (30%)
        """

        # Normalize metrics
        max_discount = self.df['discount_rate'].quantile(0.95)

        discount_score = (
            (self.df['discount_rate'] / max_discount) * 100
        ).clip(0, 100)

        # Price-to-rating ratio: lower price with higher rating = better value
        max_price = self.df['current_price'].quantile(0.95)
        price_norm = (self.df['current_price'] / max_price).clip(0, 1)
        rating_norm = (self.df['rating_average'] / 5.0).fillna(0)

        # Inverse of price norm (lower price = higher score)
        price_rating_score = ((1 - price_norm) + rating_norm) / 2 * 100

        # Popularity score
        popularity_score = self.df['popularity_category'].map({
            'Viral': 100,
            'Hot': 80,
            'Trending': 60,
            'Normal': 40,
            'New': 20
        }).fillna(50)

        self.df['value_score'] = (
            (discount_score * 0.3) +
            (price_rating_score * 0.4) +
            (popularity_score * 0.3)
        ).round(2)

        print(
            f"   Range: {self.df['value_score'].min():.2f} - {self.df['value_score'].max():.2f}")
        print(f"   Mean: {self.df['value_score'].mean():.2f}")

    def _create_lifecycle_status(self):
        """
        Xác định giai đoạn vòng đời sản phẩm

        Giai đoạn:
        - Introduction: New product (low sales, few reviews)
        - Growth: Increasing popularity (good rating, moderate sales)
        - Maturity: Peak performance (high sales, high rating)
        - Decline: Old product or declining (low engagement)
        """

        p75_sold = self.df['quantity_sold'].quantile(0.75)
        p25_sold = self.df['quantity_sold'].quantile(0.25)
        median_reviews = self.df['num_reviews'].median()

        def categorize_lifecycle(row):
            sold = row['quantity_sold']
            reviews = row['num_reviews']
            rating = row['rating_average']

            if sold < p25_sold and reviews < 10:
                return 'Introduction'
            elif sold < p75_sold and reviews > median_reviews * 0.5:
                if rating >= 3.8:
                    return 'Growth'
                else:
                    return 'Decline'
            elif sold >= p75_sold and rating >= 4.0:
                return 'Maturity'
            else:
                return 'Decline'

        self.df['lifecycle_status'] = self.df.apply(
            categorize_lifecycle, axis=1
        )

        print(f"   Distribution:")
        for status, count in self.df['lifecycle_status'].value_counts().items():
            print(
                f"     - {status}: {count:,} ({count/len(self.df)*100:.1f}%)")

    def _create_discount_intensity(self):
        """
        Xác định mức độ giảm giá của sản phẩm

        Categories:
        - No Discount: discount_rate < 5%
        - Mild: discount_rate 5-15%
        - Moderate: discount_rate 15-30%
        - Aggressive: discount_rate 30-50%
        - Heavy: discount_rate >= 50%
        """

        def categorize_discount(discount_rate):
            if pd.isna(discount_rate) or discount_rate == 0:
                return 'No Discount'
            elif discount_rate < 5:
                return 'No Discount'
            elif discount_rate < 15:
                return 'Mild'
            elif discount_rate < 30:
                return 'Moderate'
            elif discount_rate < 50:
                return 'Aggressive'
            else:
                return 'Heavy'

        self.df['discount_intensity'] = self.df['discount_rate'].apply(
            categorize_discount
        )

        # Tính toán discount intensity score (0-100)
        max_discount = self.df['discount_rate'].quantile(0.95)
        self.df['discount_intensity_score'] = (
            (self.df['discount_rate'] / max_discount) * 100
        ).clip(0, 100).round(2)

        self.stats['discount_intensity'] = {
            'max_discount': max_discount
        }

        print(f"   Distribution:")
        for intensity, count in self.df['discount_intensity'].value_counts().items():
            print(
                f"     - {intensity}: {count:,} ({count/len(self.df)*100:.1f}%)")
        print(
            f"   Score Range: {self.df['discount_intensity_score'].min():.2f} - {self.df['discount_intensity_score'].max():.2f}")
        print(
            f"   Score Mean: {self.df['discount_intensity_score'].mean():.2f}")

    def _count_new_features(self) -> int:
        """Đếm số features mới được tạo"""
        new_features = [
            'quality_category', 'popularity_category', 'price_segment',
            'seller_tier', 'brand_strength', 'engagement_score',
            'value_score', 'lifecycle_status', 'discount_intensity',
            'discount_intensity_score'
        ]
        return len([f for f in new_features if f in self.df.columns])

    def get_feature_statistics(self) -> Dict:
        """Lấy thống kê của các features"""
        return self.stats

    def get_dataframe(self) -> pd.DataFrame:
        """Trả về DataFrame với features đã được engineering"""
        return self.df


def create_feature_engineering(input_file: str, output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Hàm main cho feature engineering

    Parameters:
    - input_file: đường dẫn file cleaned data (JSON)
    - output_file: đường dẫn file output (default: data/transformation/engineered_features.json)

    Returns:
    - DataFrame với features đã được engineering
    """

    print("\n" + "=" * 70)
    print("🎯 FEATURE ENGINEERING - PHÂN VÙNG SẢN PHẨM")
    print("=" * 70)

    # 1. Đọc dữ liệu
    print("\n📂 Bước 0: Đọc dữ liệu...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"✓ Đã load {len(df):,} records")

    # 2. Feature engineering
    engineer = FeatureEngineer(df)
    df_engineered = engineer.engineer_features()

    # 3. Lưu dữ liệu
    if output_file is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base, 'data/transformation')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'engineered_features.json')

    print(f"\n💾 Lưu dữ liệu vào: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(df_engineered.to_dict('records'),
                  f, ensure_ascii=False, indent=2)

    print(f"✅ Đã lưu {len(df_engineered):,} records\n")

    # 4. Thống kê tóm tắt
    print("=" * 70)
    print("📊 THỐNG KÊ TÓM TẮT")
    print("=" * 70)

    print("\n✨ Các Features mới được tạo:")
    features = [
        'quality_category', 'popularity_category', 'price_segment',
        'seller_tier', 'brand_strength', 'engagement_score',
        'value_score', 'lifecycle_status', 'discount_intensity',
        'discount_intensity_score'
    ]
    for feature in features:
        print(f"  ✓ {feature}")

    print("\n📈 Thông tin từng feature:")

    print("\n1️⃣  Quality Category:")
    print(df_engineered['quality_category'].value_counts().to_string())

    print("\n2️⃣  Popularity Category:")
    print(df_engineered['popularity_category'].value_counts().to_string())

    print("\n3️⃣  Price Segment:")
    print(df_engineered['price_segment'].value_counts().to_string())

    print("\n4️⃣  Seller Tier:")
    print(df_engineered['seller_tier'].value_counts().to_string())

    print("\n5️⃣  Engagement Score Statistics:")
    print(f"  Min: {df_engineered['engagement_score'].min():.2f}")
    print(f"  Max: {df_engineered['engagement_score'].max():.2f}")
    print(f"  Mean: {df_engineered['engagement_score'].mean():.2f}")
    print(f"  Median: {df_engineered['engagement_score'].median():.2f}")

    print("\n6️⃣  Value Score Statistics:")
    print(f"  Min: {df_engineered['value_score'].min():.2f}")
    print(f"  Max: {df_engineered['value_score'].max():.2f}")
    print(f"  Mean: {df_engineered['value_score'].mean():.2f}")
    print(f"  Median: {df_engineered['value_score'].median():.2f}")

    print("\n7️⃣  Lifecycle Status:")
    print(df_engineered['lifecycle_status'].value_counts().to_string())

    print("\n8️⃣  Discount Intensity:")
    print(df_engineered['discount_intensity'].value_counts().to_string())

    print("\n9️⃣  Discount Intensity Score Statistics:")
    print(f"  Min: {df_engineered['discount_intensity_score'].min():.2f}")
    print(f"  Max: {df_engineered['discount_intensity_score'].max():.2f}")
    print(f"  Mean: {df_engineered['discount_intensity_score'].mean():.2f}")
    print(
        f"  Median: {df_engineered['discount_intensity_score'].median():.2f}")

    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH FEATURE ENGINEERING\n")

    return df_engineered


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base, 'data/clean/merged_cleaned_data.json')
    output_file = os.path.join(base, 'data/transformation/engineered_features.json')

    df_result = create_feature_engineering(input_file, output_file)
