import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class FeatureEngineer:
    """
    Feature Engineering tối ưu cho phân vùng sản phẩm:
    - 🔥 HOT TREND: Sản phẩm đang viral, tăng trưởng nhanh
    - 💰 ƯU ĐÃI: Sản phẩm có giảm giá tốt, giá trị cao
    - 🏆 BÁN CHẠY: Sản phẩm có doanh số cao, ổn định
    """

    def __init__(self, df: pd.DataFrame):
        """
        Khởi tạo Feature Engineer

        Parameters:
        - df: DataFrame đã được làm sạch từ clean_merged_data
        """
        self.df = df.copy()
        self.stats = {}

        # Validation
        required_cols = ['current_price', 'discount_rate', 'rating_average',
                         'num_reviews', 'quantity_sold', 'crawl_date']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"❌ Thiếu các cột bắt buộc: {missing}")

    def engineer_features(self, visualize: bool = True) -> pd.DataFrame:
        """Thực hiện feature engineering"""
        print("🔧 BẮT ĐẦU FEATURE ENGINEERING")
        print("=" * 80)
        print("🎯 Mục tiêu: Tạo features cho phân vùng HOT TREND | ƯU ĐÃI | BÁN CHẠY")
        print("=" * 80)

        # Core features (quan trọng nhất)
        print("\n📊 BƯỚC 1: TẠO CORE FEATURES")
        self._prepare_time_features()
        self._create_sales_velocity()
        self._create_review_velocity()
        self._create_price_features()

        # Popularity & Engagement (cho HOT TREND & BÁN CHẠY)
        print("\n🔥 BƯỚC 2: TẠO POPULARITY & ENGAGEMENT FEATURES")
        self._create_popularity_score()
        self._create_engagement_score()
        self._create_trend_momentum()

        # Value & Discount (cho ƯU ĐÃI)
        print("\n💰 BƯỚC 3: TẠO VALUE & DISCOUNT FEATURES")
        self._create_discount_intensity()
        self._create_value_score()
        self._create_deal_quality_score()

        # Categorical features (hỗ trợ)
        print("\n📂 BƯỚC 4: TẠO CATEGORICAL FEATURES")
        self._create_popularity_category()
        self._create_price_segment()
        self._create_quality_tier()

        # Context features (theo danh mục)
        print("\n🏷️ BƯỚC 5: TẠO CONTEXT FEATURES")
        self._create_category_context()

        # Final selection
        self._select_final_features()

        print(f"\n✅ HOÀN THÀNH. Tổng cộng: {len(self.df.columns)} features")
        print("=" * 80)

        if visualize:
            self._visualize_features()

        return self.df

    # ========================================================================
    # BƯỚC 1: CORE FEATURES (Nền tảng)
    # ========================================================================

    def _prepare_time_features(self):
        """Chuẩn bị các đặc trưng về thời gian - xử lý bias crawl_date hạn chế"""
        print("  ⏰ Tính toán time features...")

        self.df['crawl_date'] = pd.to_datetime(self.df['crawl_date'])
        max_date = self.df['crawl_date'].max()

        self.df['days_active'] = (max_date - self.df['crawl_date']).dt.days + 1

        # Kiểm tra số ngày crawl duy nhất
        unique_crawl_days = self.df['crawl_date'].nunique()
        print(f"     Số ngày crawl duy nhất: {unique_crawl_days}")

        if unique_crawl_days <= 5:  # ngưỡng tùy chỉnh
            print(
                "     ⚠️ Dữ liệu crawl quá ít ngày → product_age bị bias → cố định 'Brand New'")
            self.df['product_age'] = 'Brand New'
        else:
            # Chỉ dùng cut khi dữ liệu đủ đa dạng
            self.df['product_age'] = pd.cut(
                self.df['days_active'],
                bins=[0, 7, 30, 90, 365, float('inf')],
                labels=['Brand New', 'New', 'Recent', 'Established', 'Mature'],
                include_lowest=True
            )

        print(
            f"     ✓ Days active range: {self.df['days_active'].min()} - {self.df['days_active'].max()}")
        print(
            f"     ✓ product_age distribution:\n{self.df['product_age'].value_counts(normalize=True).round(3)}")

    def _create_sales_velocity(self):
        """
        🏆 Tốc độ bán hàng - QUAN TRỌNG cho BÁN CHẠY
        Đơn vị: sản phẩm/ngày
        """
        print("  📈 Tính Sales Velocity (sản phẩm/ngày)...")

        self.df['sales_velocity'] = (
            self.df['quantity_sold'] / self.df['days_active']
        ).round(2)

        # Normalized sales velocity (0-100)
        max_velocity = self.df['sales_velocity'].quantile(0.98)
        self.df['sales_velocity_normalized'] = (
            (self.df['sales_velocity'] / max_velocity) * 100
        ).clip(0, 100).round(2)

        self.stats['sales_velocity'] = {
            'mean': self.df['sales_velocity'].mean(),
            'median': self.df['sales_velocity'].median(),
            'p95': self.df['sales_velocity'].quantile(0.95)
        }

        print(
            f"     ✓ Mean: {self.stats['sales_velocity']['mean']:.2f} sản phẩm/ngày")
        print(
            f"     ✓ P95: {self.stats['sales_velocity']['p95']:.2f} sản phẩm/ngày")

    def _create_review_velocity(self):
        """
        🔥 Tốc độ nhận review - QUAN TRỌNG cho HOT TREND
        Đơn vị: reviews/ngày
        """
        print("  💬 Tính Review Velocity (reviews/ngày)...")

        self.df['review_velocity'] = (
            self.df['num_reviews'] / self.df['days_active']
        ).round(2)

        # Normalized review velocity (0-100)
        max_review_velocity = self.df['review_velocity'].quantile(0.98)
        self.df['review_velocity_normalized'] = (
            (self.df['review_velocity'] / max_review_velocity) * 100
        ).clip(0, 100).round(2)

        self.stats['review_velocity'] = {
            'mean': self.df['review_velocity'].mean(),
            'median': self.df['review_velocity'].median(),
            'p95': self.df['review_velocity'].quantile(0.95)
        }

        print(
            f"     ✓ Mean: {self.stats['review_velocity']['mean']:.2f} reviews/ngày")
        print(
            f"     ✓ P95: {self.stats['review_velocity']['p95']:.2f} reviews/ngày")

    def _create_price_features(self):
        """
        💰 Các đặc trưng về giá - QUAN TRỌNG cho ƯU ĐÃI
        """
        print("  💵 Tính Price Features...")

        # Original price (giá gốc)
        self.df['original_price'] = np.where(
            self.df['discount_rate'] > 0,
            self.df['current_price'] / (1 - self.df['discount_rate'] / 100),
            self.df['current_price']
        ).round(2)

        # Absolute saving (tiết kiệm tuyệt đối)
        self.df['absolute_saving'] = (
            self.df['original_price'] - self.df['current_price']
        ).round(2)

        # Price per rating point (giá trị tiền/điểm rating)
        # Giá trị thấp = tốt hơn (rẻ mà chất lượng cao)
        self.df['price_per_rating'] = np.where(
            self.df['rating_average'] > 0,
            self.df['current_price'] / self.df['rating_average'],
            float('inf')
        ).round(2)

        print(
            f"     ✓ Mean absolute saving: {self.df['absolute_saving'].mean():.2f} VNĐ")

    # ========================================================================
    # BƯỚC 2: POPULARITY & ENGAGEMENT (HOT TREND & BÁN CHẠY)
    # ========================================================================

    def _create_popularity_score(self):
        """
        🏆 Điểm phổ biến tổng hợp (0-100)
        - Quan trọng cho: BÁN CHẠY

        Công thức:
        - Quantity sold (50%)
        - Reviews (30%)
        - Rating (20%)
        """
        print("  🏆 Tính Popularity Score (0-100)...")

        # Normalize từng thành phần
        max_sold = self.df['quantity_sold'].quantile(0.98)
        sold_score = (self.df['quantity_sold'] / max_sold * 100).clip(0, 100)

        max_reviews = self.df['num_reviews'].quantile(0.98)
        review_score = (self.df['num_reviews'] /
                        max_reviews * 100).clip(0, 100)

        rating_score = (self.df['rating_average'] / 5.0 * 100).fillna(0)

        # Tổng hợp
        self.df['popularity_score'] = (
            sold_score * 0.50 +
            review_score * 0.30 +
            rating_score * 0.20
        ).round(2)

        print(f"     ✓ Mean: {self.df['popularity_score'].mean():.2f}")
        print(
            f"     ✓ Range: {self.df['popularity_score'].min():.0f} - {self.df['popularity_score'].max():.0f}")

    def _create_engagement_score(self):
        """
        🔥 Điểm tương tác (0-100)
        - Quan trọng cho: HOT TREND

        Công thức:
        - Review velocity (40%) - tốc độ nhận review
        - Sales velocity (40%) - tốc độ bán
        - Rating (20%) - chất lượng
        """
        print("  🔥 Tính Engagement Score (0-100)...")

        # Đã có normalized velocity từ trước
        review_vel_score = self.df['review_velocity_normalized']
        sales_vel_score = self.df['sales_velocity_normalized']
        rating_score = (self.df['rating_average'] / 5.0 * 100).fillna(0)

        # Tổng hợp
        self.df['engagement_score'] = (
            review_vel_score * 0.40 +
            sales_vel_score * 0.40 +
            rating_score * 0.20
        ).round(2)

        print(f"     ✓ Mean: {self.df['engagement_score'].mean():.2f}")
        print(
            f"     ✓ Range: {self.df['engagement_score'].min():.0f} - {self.df['engagement_score'].max():.0f}")

    def _create_trend_momentum(self):
        """
        🚀 Momentum xu hướng
        - Quan trọng cho: HOT TREND

        Kết hợp:
        - High engagement + New/Recent product = High momentum
        """
        print("  🚀 Tính Trend Momentum...")

        # Age factor: sản phẩm mới có momentum cao hơn
        age_factor = self.df['product_age'].map({
            'Brand New': 1.5,
            'New': 1.3,
            'Recent': 1.0,
            'Established': 0.7,
            'Mature': 0.5
        }).astype(float)

        # Trend momentum = engagement * age_factor
        self.df['trend_momentum'] = (
            self.df['engagement_score'] * age_factor
        ).clip(0, 150).round(2)

        print(f"     ✓ Mean: {self.df['trend_momentum'].mean():.2f}")

    # ========================================================================
    # BƯỚC 3: VALUE & DISCOUNT (ƯU ĐÃI)
    # ========================================================================

    def _create_discount_intensity(self):
        """
        💰 Mức độ giảm giá
        - Quan trọng cho: ƯU ĐÃI
        """
        print("  💸 Phân loại Discount Intensity...")

        def categorize_discount(rate):
            if pd.isna(rate) or rate < 5:
                return 'No Discount'
            elif rate < 15:
                return 'Mild'
            elif rate < 30:
                return 'Moderate'
            elif rate < 50:
                return 'Aggressive'
            else:
                return 'Heavy'

        self.df['discount_intensity'] = self.df['discount_rate'].apply(
            categorize_discount)

        # Discount score (0-100)
        max_discount = self.df['discount_rate'].quantile(0.98)
        self.df['discount_score'] = (
            (self.df['discount_rate'] / max_discount) * 100
        ).clip(0, 100).round(2)

        distribution = self.df['discount_intensity'].value_counts()
        for cat, count in distribution.items():
            print(f"     - {cat}: {count:,} ({count/len(self.df)*100:.1f}%)")

    def _create_value_score(self):
        """
        💎 Điểm giá trị tổng hợp (0-100)
        - Quan trọng cho: ƯU ĐÃI
        """
        print("  💎 Tính Value Score (0-100)...")

        # Discount score (đã có) — core signal cho Best Deal
        discount_component = self.df['discount_score']

        # Rating component: RELATIVE vs category median (bias fix)
        if 'category' in self.df.columns:
            cat_rating_median = self.df.groupby(
                'category')['rating_average'].transform('median')
            cat_rating_std = self.df.groupby(
                'category')['rating_average'].transform('std').clip(lower=0.01)
            rating_z = (self.df['rating_average'] -
                        cat_rating_median) / cat_rating_std
            # Map z-score → 0-100: z=-2→10, z=0→50, z=+2→90
            rating_component = ((rating_z * 20) + 50).clip(0, 100)
        else:
            rating_component = (
                self.df['rating_average'] / 5.0 * 100).fillna(0)
        rating_component = rating_component.fillna(50)  # neutral default

        # Price competitiveness: percentile rank WITHIN category (thấp hơn → điểm cao)
        if 'category' in self.df.columns:
            price_percentile_in_cat = self.df.groupby(
                'category')['current_price'].rank(pct=True)
            price_competitiveness = (
                (1 - price_percentile_in_cat) * 100).clip(0, 100)
        else:
            price_competitiveness = (
                1 - self.df['current_price'].rank(pct=True)) * 100

        # Tổng hợp
        self.df['value_score'] = (
            discount_component * 0.45 +
            rating_component * 0.20 +
            price_competitiveness * 0.35
        ).round(2)

        print(f"     ✓ Mean: {self.df['value_score'].mean():.2f}")
        print(
            f"     ✓ Range: {self.df['value_score'].min():.0f} - {self.df['value_score'].max():.0f}")

    def _create_deal_quality_score(self):
        """
        🎁 Chất lượng deal (0-100)
        - Quan trọng cho: ƯU ĐÃI
        """
        print("  🎁 Tính Deal Quality Score (0-100)...")

        # Normalize absolute saving
        max_saving = self.df['absolute_saving'].quantile(0.98)
        saving_score = (self.df['absolute_saving'] /
                        max_saving * 100).clip(0, 100)

        # Rating: RELATIVE vs category median (bias fix)
        if 'category' in self.df.columns:
            cat_rating_median = self.df.groupby(
                'category')['rating_average'].transform('median')
            cat_rating_std = self.df.groupby(
                'category')['rating_average'].transform('std').clip(lower=0.01)
            rating_z = (self.df['rating_average'] -
                        cat_rating_median) / cat_rating_std
            rating_score = ((rating_z * 20) + 50).clip(0, 100)
        else:
            rating_score = (self.df['rating_average'] / 5.0 * 100).fillna(0)
        rating_score = rating_score.fillna(50)

        # Review credibility
        max_reviews = self.df['num_reviews'].quantile(0.98)
        credibility_score = (
            self.df['num_reviews'] / max_reviews * 100).clip(0, 100)

        # Discount gate bonus: chỉ có điểm nếu thực sự có discount đáng kể
        discount_gate = np.where(self.df['discount_rate'] >= 30, 100,   # Aggressive+
                                 np.where(self.df['discount_rate'] >= 15, 50,    # Moderate
                                 0.0))                                            # No/Mild → 0

        # Tổng hợp
        self.df['deal_quality_score'] = (
            saving_score * 0.45 +
            rating_score * 0.20 +
            credibility_score * 0.20 +
            discount_gate * 0.15
        ).round(2)

        # HARD GATE: discount < 15% → deal_quality = 0 (không thể là Best Deal)
        self.df.loc[self.df['discount_rate'] < 15, 'deal_quality_score'] = 0.0

        print(f"     ✓ Mean: {self.df['deal_quality_score'].mean():.2f}")
        print(
            f"     ✓ Products with deal_quality > 0: {(self.df['deal_quality_score'] > 0).sum():,} ({(self.df['deal_quality_score'] > 0).mean()*100:.1f}%)")

    # ========================================================================
    # BƯỚC 4: CATEGORICAL FEATURES (Hỗ trợ)
    # ========================================================================

    def _create_popularity_category(self):
        """Phân loại độ phổ biến dựa trên percentiles"""
        print("  📊 Phân loại Popularity Category...")

        # Dựa trên popularity_score
        p90 = self.df['popularity_score'].quantile(0.90)
        p75 = self.df['popularity_score'].quantile(0.75)
        p50 = self.df['popularity_score'].quantile(0.50)
        p25 = self.df['popularity_score'].quantile(0.25)

        def categorize(score):
            if score >= p90:
                return 'Viral'
            elif score >= p75:
                return 'Hot'
            elif score >= p50:
                return 'Trending'
            elif score >= p25:
                return 'Normal'
            else:
                return 'Low'

        self.df['popularity_category'] = self.df['popularity_score'].apply(
            categorize)

        distribution = self.df['popularity_category'].value_counts()
        for cat, count in distribution.items():
            print(f"     - {cat}: {count:,} ({count/len(self.df)*100:.1f}%)")

    def _create_price_segment(self):
        """Phân khúc giá dựa trên quartiles"""
        print("  💵 Phân khúc Price Segment...")

        q1 = self.df['current_price'].quantile(0.25)
        q2 = self.df['current_price'].quantile(0.50)
        q3 = self.df['current_price'].quantile(0.75)

        def categorize(price):
            if price <= q1:
                return 'Budget'
            elif price <= q2:
                return 'Economy'
            elif price <= q3:
                return 'Mid-Range'
            else:
                return 'Premium'

        self.df['price_segment'] = self.df['current_price'].apply(categorize)

        distribution = self.df['price_segment'].value_counts()
        for seg, count in distribution.items():
            print(f"     - {seg}: {count:,} ({count/len(self.df)*100:.1f}%)")

    def _create_quality_tier(self):
        """Phân tầng chất lượng dựa trên rating và reviews"""
        print("  ⭐ Phân tầng Quality Tier...")

        median_reviews = self.df['num_reviews'].median()

        def categorize(row):
            rating = row['rating_average']
            reviews = row['num_reviews']

            if rating >= 4.5 and reviews >= median_reviews:
                return 'Premium'
            elif rating >= 4.0:
                return 'High'
            elif rating >= 3.5:
                return 'Good'
            elif rating >= 3.0:
                return 'Average'
            else:
                return 'Low'

        self.df['quality_tier'] = self.df.apply(categorize, axis=1)

        distribution = self.df['quality_tier'].value_counts()
        for tier, count in distribution.items():
            print(f"     - {tier}: {count:,} ({count/len(self.df)*100:.1f}%)")

    # ========================================================================
    # BƯỚC 5: CONTEXT FEATURES (Theo danh mục)
    # ========================================================================

    def _create_category_context(self):
        """Tạo features theo ngữ cảnh danh mục"""
        print("  🏷️ Tạo Category Context Features...")

        if 'category' not in self.df.columns:
            print("     ⚠️ Không có cột 'category', bỏ qua")
            self.df['category_popularity_rank'] = 50.0
            self.df['category_price_percentile'] = 50.0
            return

        # 1. Category popularity rank (0-100)
        # Ranking danh mục dựa trên tổng quantity_sold
        category_total_sold = self.df.groupby(
            'category')['quantity_sold'].sum()
        category_rank = category_total_sold.rank(pct=True) * 100
        self.df['category_popularity_rank'] = self.df['category'].map(
            category_rank).round(2)

        # 2. Price position trong category (0-100)
        # Percentile của giá sản phẩm trong category của nó
        self.df['category_price_percentile'] = (
            self.df.groupby('category')['current_price']
            .rank(pct=True) * 100
        ).round(2)

        print(
            f"     ✓ Category popularity rank mean: {self.df['category_popularity_rank'].mean():.2f}")
        print(
            f"     ✓ Category price percentile mean: {self.df['category_price_percentile'].mean():.2f}")

    # ========================================================================
    # FINAL SELECTION
    # ========================================================================

    def _select_final_features(self):
        """Chọn các features cuối cùng cho output"""
        print("\n📋 Chọn final features...")

        # Metadata & identifiers
        metadata = ['id', 'crawl_date', 'platform']

        # Context (nếu có)
        context = []
        if 'category' in self.df.columns:
            context.append('category')
        if 'brand' in self.df.columns:
            context.append('brand')
        if 'product_name' in self.df.columns:
            context.append('product_name')

        # Raw numerical features
        raw_numerical = [
            'current_price', 'original_price', 'absolute_saving',
            'discount_rate', 'rating_average', 'num_reviews', 'quantity_sold',
            'days_active'
        ]

        # Engineered numerical features (scores)
        engineered_numerical = [
            'sales_velocity', 'sales_velocity_normalized',
            'review_velocity', 'review_velocity_normalized',
            'popularity_score', 'engagement_score', 'trend_momentum',
            'discount_score', 'value_score', 'deal_quality_score',
            'category_popularity_rank', 'category_price_percentile'
        ]

        # Categorical features
        categorical = [
            'popularity_category', 'price_segment', 'quality_tier',
            'discount_intensity', 'product_age'
        ]

        # Combine all
        final_columns = metadata + context + \
            raw_numerical + engineered_numerical + categorical

        # Filter chỉ giữ columns tồn tại
        final_columns = [
            col for col in final_columns if col in self.df.columns]

        self.df = self.df[final_columns]

        print(f"     ✓ Metadata: {len(metadata)}")
        print(f"     ✓ Context: {len(context)}")
        print(f"     ✓ Raw numerical: {len(raw_numerical)}")
        print(f"     ✓ Engineered numerical: {len(engineered_numerical)}")
        print(f"     ✓ Categorical: {len(categorical)}")
        print(f"     ✓ Total: {len(final_columns)} features")

    # ========================================================================
    # VISUALIZATION
    # ========================================================================

    def _visualize_features(self, output_dir: str | None = None):
        """Tạo biểu đồ trực quan hóa cho các features chính"""
        if output_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base, 'data/visualizations/feature')

        os.makedirs(output_dir, exist_ok=True)
        print(f"\n📊 Tạo visualizations → {output_dir}")

        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)

        total = len(self.df)

        # Helper functions
        def plot_categorical(feature, title, filename, order=None):
            if feature not in self.df.columns:
                return

            fig, ax = plt.subplots()
            data = self.df[feature].value_counts()
            if order:
                data = data.reindex(order, fill_value=0)

            data.plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(feature.replace('_', ' ').title())
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)

            for rect in ax.containers[0]:
                height = rect.get_height()
                pct = height / total * 100
                ax.annotate(
                    f'{int(height):,}\n({pct:.1f}%)',
                    (rect.get_x() + rect.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=9
                )

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename),
                        dpi=150, bbox_inches='tight')
            plt.close()

        def plot_numerical(feature, title, filename, color='royalblue'):
            if feature not in self.df.columns:
                return

            fig, ax = plt.subplots()
            self.df[feature].hist(
                bins=50, ax=ax, color=color, edgecolor='black', alpha=0.7)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(feature.replace('_', ' ').title())
            ax.set_ylabel('Count')

            # Add stats
            mean_val = self.df[feature].mean()
            median_val = self.df[feature].median()
            ax.axvline(mean_val, color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {mean_val:.1f}')
            ax.axvline(median_val, color='green', linestyle='--',
                       linewidth=2, label=f'Median: {median_val:.1f}')
            ax.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename),
                        dpi=150, bbox_inches='tight')
            plt.close()

        # Generate plots
        print("  🎨 Generating charts...")

        # Categorical
        plot_categorical('popularity_category', '1. Popularity Category Distribution',
                         '01_popularity_category.png',
                         order=['Viral', 'Hot', 'Trending', 'Normal', 'Low'])

        plot_categorical('discount_intensity', '2. Discount Intensity Distribution',
                         '02_discount_intensity.png',
                         order=['Heavy', 'Aggressive', 'Moderate', 'Mild', 'No Discount'])

        plot_categorical('quality_tier', '3. Quality Tier Distribution',
                         '03_quality_tier.png',
                         order=['Premium', 'High', 'Good', 'Average', 'Low'])

        plot_categorical('price_segment', '4. Price Segment Distribution',
                         '04_price_segment.png',
                         order=['Budget', 'Economy', 'Mid-Range', 'Premium'])

        plot_categorical('product_age', '5. Product Age Distribution',
                         '05_product_age.png',
                         order=['Brand New', 'New', 'Recent', 'Established', 'Mature'])

        # Numerical - Core scores
        plot_numerical('popularity_score', '6. Popularity Score Distribution (BÁN CHẠY)',
                       '06_popularity_score.png', color='green')

        plot_numerical('engagement_score', '7. Engagement Score Distribution (HOT TREND)',
                       '07_engagement_score.png', color='red')

        plot_numerical('value_score', '8. Value Score Distribution (ƯU ĐÃI)',
                       '08_value_score.png', color='orange')

        plot_numerical('deal_quality_score', '9. Deal Quality Score Distribution (ƯU ĐÃI)',
                       '09_deal_quality_score.png', color='purple')

        plot_numerical('trend_momentum', '10. Trend Momentum Distribution (HOT TREND)',
                       '10_trend_momentum.png', color='crimson')

        # Numerical - Velocities
        plot_numerical('sales_velocity', '11. Sales Velocity Distribution',
                       '11_sales_velocity.png', color='teal')

        plot_numerical('review_velocity', '12. Review Velocity Distribution',
                       '12_review_velocity.png', color='indigo')

        print(f"  ✅ Saved 12 visualization charts")


# ========================================================================
# MAIN FUNCTION
# ========================================================================

def create_feature_engineering(
    input_file: str,
    output_file: Optional[str] = None,
    visualize: bool = True
) -> pd.DataFrame:
    """
    Main function cho feature engineering

    Parameters:
    - input_file: đường dẫn file cleaned data (JSON)
    - output_file: đường dẫn file output (mặc định: data/transformation/engineered_features.json)
    - visualize: có tạo biểu đồ hay không

    Returns:
    - DataFrame với features đã được engineering
    """

    print("\n" + "=" * 80)
    print("🎯 FEATURE ENGINEERING - PHÂN VÙNG SẢN PHẨM")
    print("   HOT TREND 🔥 | ƯU ĐÃI 💰 | BÁN CHẠY 🏆")
    print("=" * 80)

    # 1. Load data
    print("\n📂 Loading data...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"✓ Loaded {len(df):,} records")
    print(f"✓ Columns: {list(df.columns)}")

    # 2. Feature engineering
    engineer = FeatureEngineer(df)
    df_engineered = engineer.engineer_features(visualize=visualize)

    # 3. Save output
    if output_file is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base, 'data/transformation')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'engineered_features.json')

    print(f"\n💾 Saving to: {output_file}")

    # Convert datetime to string
    datetime_cols = df_engineered.select_dtypes(
        include=['datetime64[ns]']).columns
    for col in datetime_cols:
        df_engineered[col] = df_engineered[col].astype(str)

    # Convert category to string (for JSON serialization)
    category_cols = df_engineered.select_dtypes(include=['category']).columns
    for col in category_cols:
        df_engineered[col] = df_engineered[col].astype(str)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(df_engineered.to_dict('records'),
                  f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(df_engineered):,} records")

    # 4. Summary statistics
    print("\n" + "=" * 80)
    print("📊 SUMMARY STATISTICS")
    print("=" * 80)

    print("\n🔢 KEY METRICS:")
    key_metrics = [
        'popularity_score', 'engagement_score', 'value_score',
        'deal_quality_score', 'trend_momentum'
    ]

    for metric in key_metrics:
        if metric in df_engineered.columns:
            print(f"\n  {metric.upper()}:")
            print(f"    Min:    {df_engineered[metric].min():.2f}")
            print(f"    Max:    {df_engineered[metric].max():.2f}")
            print(f"    Mean:   {df_engineered[metric].mean():.2f}")
            print(f"    Median: {df_engineered[metric].median():.2f}")
            print(f"    Std:    {df_engineered[metric].std():.2f}")

    print("\n📊 CATEGORICAL DISTRIBUTIONS:")
    categorical_features = [
        'popularity_category', 'discount_intensity', 'quality_tier', 'price_segment'
    ]

    for feature in categorical_features:
        if feature in df_engineered.columns:
            print(f"\n  {feature.upper()}:")
            counts = df_engineered[feature].value_counts()
            for val, count in counts.items():
                pct = count / len(df_engineered) * 100
                print(f"    {val}: {count:,} ({pct:.1f}%)")

    print("\n" + "=" * 80)
    print("✅ FEATURE ENGINEERING COMPLETED")
    print("=" * 80 + "\n")

    return df_engineered


if __name__ == "__main__":
    # Example usage
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base, 'data/clean/merged_cleaned_data.json')
    output_file = os.path.join(
        base, 'data/transformation/engineered_features.json')

    df_result = create_feature_engineering(
        input_file, output_file, visualize=True)

    print(f"📋 Final shape: {df_result.shape}")
    print(f"📋 Final columns: {list(df_result.columns)}")
