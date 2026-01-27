import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Cấu hình matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'

# Đường dẫn
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / 'data' / 'clean' / 'merged_cleaned_data.json'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'visualizations' / 'cleaned_data'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 80)
print("TRỰC QUAN HÓA DỮ LIỆU SAU KHI LÀM SẠCH".center(80))
print("=" * 80)

# 1. Đọc dữ liệu
print("\n[1/12] Đang đọc dữ liệu đã làm sạch...")
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(f"   ✓ Đã đọc {len(df):,} records từ {len(df.columns)} columns")
print(f"   ✓ Columns: {list(df.columns)}")

# 2. Thống kê tổng quan
print("\n[2/12] Thống kê tổng quan...")
print(f"   - Tổng số records: {len(df):,}")
print(f"   - Platforms: {df['platform'].unique().tolist()}")
print(
    f"   - Khoảng thời gian: {df['crawl_date'].min()} đến {df['crawl_date'].max()}")
print(f"   - Số categories: {df['category'].nunique()}")
print(f"   - Số brands: {df['brand'].nunique()}")

# 3. Phân bố theo Platform
print("\n[3/12] Trực quan hóa phân bố Platform...")
platform_counts = df['platform'].value_counts()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Pie chart
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
explode = [0.05] * len(platform_counts)
wedges, texts, autotexts = axes[0].pie(
    platform_counts.values,
    labels=platform_counts.index,
    autopct='%1.1f%%',
    colors=colors,
    explode=explode,
    shadow=True,
    startangle=90
)
axes[0].set_title('Phân bố theo Platform', fontsize=14,
                  fontweight='bold', pad=20)
for text in texts:
    text.set_fontsize(12)
    text.set_fontweight('bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(11)
    autotext.set_fontweight('bold')

# Bar chart với số liệu
bars = axes[1].bar(platform_counts.index, platform_counts.values,
                   color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
axes[1].set_ylabel('Số lượng sản phẩm', fontsize=12, fontweight='bold')
axes[1].set_title('Số lượng sản phẩm theo Platform',
                  fontsize=14, fontweight='bold', pad=20)
axes[1].grid(axis='y', alpha=0.3)
for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height):,}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

# Stacked bar - phân bố theo category
platform_category = df.groupby(
    ['platform', 'category']).size().unstack(fill_value=0)
top_categories = df['category'].value_counts().head(10).index
platform_category_top = platform_category[top_categories]
platform_category_top.plot(kind='bar', stacked=True, ax=axes[2],
                           colormap='tab20', edgecolor='black', linewidth=0.5)
axes[2].set_ylabel('Số lượng sản phẩm', fontsize=11, fontweight='bold')
axes[2].set_title('Phân bố Top 10 Category theo Platform',
                  fontsize=12, fontweight='bold', pad=20)
axes[2].legend(title='Category', bbox_to_anchor=(
    1.05, 1), loc='upper left', fontsize=8)
axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=0)
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_platform_analysis.png',
            dpi=300, bbox_inches='tight')
print(f"   ✓ Đã lưu: 01_platform_analysis.png")
plt.close()

# 4. Phân tích giá sản phẩm
print("\n[4/12] Phân tích phân bố giá...")
price_stats = df['current_price'].describe()
print(f"   - Giá MIN: {price_stats['min']:,.0f} VNĐ")
print(f"   - Giá MAX: {price_stats['max']:,.0f} VNĐ")
print(f"   - Giá TB: {price_stats['mean']:,.0f} VNĐ")
print(f"   - Giá Median: {price_stats['50%']:,.0f} VNĐ")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Histogram - Toàn bộ
ax1 = axes[0, 0]
df['current_price'].hist(
    bins=100, ax=ax1, color='#4ECDC4', edgecolor='black', alpha=0.8)
ax1.set_xlabel('Giá (VNĐ)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Số lượng', fontsize=11, fontweight='bold')
ax1.set_title('Phân bố giá (Toàn bộ)', fontsize=12, fontweight='bold')
ax1.axvline(price_stats['mean'], color='red', linestyle='--',
            linewidth=2, label=f'Mean: {price_stats["mean"]:,.0f}')
ax1.axvline(price_stats['50%'], color='orange', linestyle='--',
            linewidth=2, label=f'Median: {price_stats["50%"]:,.0f}')
ax1.legend()
ax1.grid(alpha=0.3)

# Histogram - Giá < 10 triệu (để nhìn rõ hơn)
ax2 = axes[0, 1]
df[df['current_price'] < 10000000]['current_price'].hist(
    bins=100, ax=ax2, color='#FF6B6B', edgecolor='black', alpha=0.8
)
ax2.set_xlabel('Giá (VNĐ)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Số lượng', fontsize=11, fontweight='bold')
ax2.set_title('Phân bố giá (< 10 triệu VNĐ)', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)

# Boxplot theo platform
ax3 = axes[1, 0]
platforms = df['platform'].unique()
price_by_platform = [df[df['platform'] == p]['current_price']
                     for p in platforms]
bp = ax3.boxplot(price_by_platform, labels=platforms,
                 patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax3.set_ylabel('Giá (VNĐ)', fontsize=11, fontweight='bold')
ax3.set_title('Phân bố giá theo Platform (Boxplot)',
              fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Giá trung bình theo platform
ax4 = axes[1, 1]
price_mean = df.groupby('platform')['current_price'].mean().sort_values()
bars = ax4.barh(price_mean.index, price_mean.values,
                color=colors, alpha=0.8, edgecolor='black')
ax4.set_xlabel('Giá trung bình (VNĐ)', fontsize=11, fontweight='bold')
ax4.set_title('Giá trung bình theo Platform', fontsize=12, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, price_mean.values)):
    ax4.text(val + price_mean.max()*0.02, i, f'{val:,.0f}',
             va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_price_analysis.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Đã lưu: 02_price_analysis.png")
plt.close()

# 5. Phân tích Rating
print("\n[5/12] Phân tích Rating...")
rating_stats = df['rating_average'].describe()
print(f"   - Rating TB: {rating_stats['mean']:.2f}")
print(f"   - Rating Median: {rating_stats['50%']:.2f}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Distribution
ax1 = axes[0, 0]
df['rating_average'].hist(
    bins=50, ax=ax1, color='#FFD93D', edgecolor='black', alpha=0.8)
ax1.set_xlabel('Rating', fontsize=11, fontweight='bold')
ax1.set_ylabel('Số lượng', fontsize=11, fontweight='bold')
ax1.set_title('Phân bố Rating', fontsize=12, fontweight='bold')
ax1.axvline(rating_stats['mean'], color='red', linestyle='--',
            linewidth=2, label=f'Mean: {rating_stats["mean"]:.2f}')
ax1.legend()
ax1.grid(alpha=0.3)

# Rating theo platform
ax2 = axes[0, 1]
rating_by_platform = df.groupby(
    'platform')['rating_average'].mean().sort_values()
bars = ax2.barh(rating_by_platform.index,
                rating_by_platform.values, color=colors, alpha=0.8)
ax2.set_xlabel('Rating trung bình', fontsize=11, fontweight='bold')
ax2.set_title('Rating trung bình theo Platform',
              fontsize=12, fontweight='bold')
ax2.set_xlim(0, 5)
ax2.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, rating_by_platform.values)):
    ax2.text(val + 0.05, i, f'{val:.2f}',
             va='center', fontsize=10, fontweight='bold')

# Phân bố quality_category nếu có
if 'quality_category' in df.columns:
    ax3 = axes[1, 0]
    quality_counts = df['quality_category'].value_counts()
    colors_quality = ['#6BCF7F', '#FFD93D', '#FF6B6B'][:len(quality_counts)]
    bars = ax3.bar(quality_counts.index, quality_counts.values, color=colors_quality,
                   edgecolor='black', alpha=0.8)
    ax3.set_ylabel('Số lượng', fontsize=11, fontweight='bold')
    ax3.set_title('Phân bố Quality Category', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Boxplot rating theo platform
ax4 = axes[1, 1]
rating_by_platform_list = [df[df['platform'] == p]
                           ['rating_average'] for p in platforms]
bp = ax4.boxplot(rating_by_platform_list, labels=platforms, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax4.set_ylabel('Rating', fontsize=11, fontweight='bold')
ax4.set_title('Phân bố Rating theo Platform', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_rating_analysis.png',
            dpi=300, bbox_inches='tight')
print(f"   ✓ Đã lưu: 03_rating_analysis.png")
plt.close()

# 6. Phân tích Review Count
print("\n[6/12] Phân tích số lượng Review...")
review_stats = df['num_reviews'].describe()
print(f"   - Review TB: {review_stats['mean']:,.0f}")
print(f"   - Review MAX: {review_stats['max']:,.0f}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Histogram (log scale)
ax1 = axes[0, 0]
review_data = df[df['num_reviews'] > 0]['num_reviews']
ax1.hist(np.log10(review_data + 1), bins=50,
         color='#95E1D3', edgecolor='black', alpha=0.8)
ax1.set_xlabel('Log10(Review Count + 1)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Số lượng', fontsize=11, fontweight='bold')
ax1.set_title('Phân bố Review Count (Log scale)',
              fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)

# Review theo platform
ax2 = axes[0, 1]
review_by_platform = df.groupby('platform')['num_reviews'].mean().sort_values()
bars = ax2.barh(review_by_platform.index,
                review_by_platform.values, color=colors, alpha=0.8)
ax2.set_xlabel('Review trung bình', fontsize=11, fontweight='bold')
ax2.set_title('Review trung bình theo Platform',
              fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, review_by_platform.values)):
    ax2.text(val + review_by_platform.max()*0.02, i, f'{val:,.0f}',
             va='center', fontsize=10, fontweight='bold')

# Phân bố popularity_category nếu có
if 'popularity_category' in df.columns:
    ax3 = axes[1, 0]
    pop_counts = df['popularity_category'].value_counts()
    colors_pop = ['#FF6B6B', '#FFD93D', '#6BCF7F'][:len(pop_counts)]
    bars = ax3.bar(pop_counts.index, pop_counts.values, color=colors_pop,
                   edgecolor='black', alpha=0.8)
    ax3.set_ylabel('Số lượng', fontsize=11, fontweight='bold')
    ax3.set_title('Phân bố Popularity Category',
                  fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Scatter: Rating vs Review Count
ax4 = axes[1, 1]
scatter_data = df[df['num_reviews'] < 5000]  # Giới hạn để nhìn rõ
ax4.scatter(scatter_data['num_reviews'], scatter_data['rating_average'],
            alpha=0.3, s=10, color='#F38181')
ax4.set_xlabel('Review Count', fontsize=11, fontweight='bold')
ax4.set_ylabel('Rating', fontsize=11, fontweight='bold')
ax4.set_title('Mối quan hệ Rating vs Review Count',
              fontsize=12, fontweight='bold')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_review_analysis.png',
            dpi=300, bbox_inches='tight')
print(f"   ✓ Đã lưu: 04_review_analysis.png")
plt.close()

# 7. Phân tích Quantity Sold
print("\n[7/12] Phân tích số lượng đã bán...")
sold_stats = df['quantity_sold'].describe()
print(f"   - Sold TB: {sold_stats['mean']:,.0f}")
print(f"   - Sold MAX: {sold_stats['max']:,.0f}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Histogram (log scale)
ax1 = axes[0, 0]
sold_data = df[df['quantity_sold'] > 0]['quantity_sold']
ax1.hist(np.log10(sold_data + 1), bins=50,
         color='#A8E6CF', edgecolor='black', alpha=0.8)
ax1.set_xlabel('Log10(Quantity Sold + 1)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Số lượng', fontsize=11, fontweight='bold')
ax1.set_title('Phân bố Quantity Sold (Log scale)',
              fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)

# Sold theo platform
ax2 = axes[0, 1]
sold_by_platform = df.groupby('platform')['quantity_sold'].mean().sort_values()
bars = ax2.barh(sold_by_platform.index,
                sold_by_platform.values, color=colors, alpha=0.8)
ax2.set_xlabel('Quantity Sold trung bình', fontsize=11, fontweight='bold')
ax2.set_title('Quantity Sold trung bình theo Platform',
              fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, sold_by_platform.values)):
    ax2.text(val + sold_by_platform.max()*0.02, i, f'{val:,.0f}',
             va='center', fontsize=10, fontweight='bold')

# Scatter: Price vs Quantity Sold
ax3 = axes[1, 0]
scatter_data = df[(df['quantity_sold'] < 10000) &
                  (df['current_price'] < 20000000)]
ax3.scatter(scatter_data['current_price'], scatter_data['quantity_sold'],
            alpha=0.3, s=10, color='#FF6B6B')
ax3.set_xlabel('Giá (VNĐ)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Quantity Sold', fontsize=11, fontweight='bold')
ax3.set_title('Mối quan hệ Giá vs Quantity Sold',
              fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)

# Top 10 sản phẩm bán chạy nhất
ax4 = axes[1, 1]
top_sold = df.nlargest(10, 'quantity_sold')[['product_name', 'quantity_sold']]
product_names = [
    name[:30] + '...' if len(name) > 30 else name for name in top_sold['product_name']]
bars = ax4.barh(range(len(top_sold)), top_sold['quantity_sold'].values,
                color=plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(top_sold))))
ax4.set_yticks(range(len(top_sold)))
ax4.set_yticklabels(product_names, fontsize=9)
ax4.set_xlabel('Quantity Sold', fontsize=11, fontweight='bold')
ax4.set_title('Top 10 sản phẩm bán chạy nhất', fontsize=12, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, top_sold['quantity_sold'].values)):
    ax4.text(val + top_sold['quantity_sold'].max()*0.02, i, f'{val:,.0f}',
             va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_quantity_sold_analysis.png',
            dpi=300, bbox_inches='tight')
print(f"   ✓ Đã lưu: 05_quantity_sold_analysis.png")
plt.close()

# 8. Phân tích Discount
print("\n[8/12] Phân tích Discount...")
discount_stats = df['discount_rate'].describe()
print(f"   - Discount TB: {discount_stats['mean']:.1f}%")
print(f"   - Discount MAX: {discount_stats['max']:.1f}%")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Histogram
ax1 = axes[0]
df['discount_rate'].hist(bins=50, ax=ax1, color='#A29BFE',
                         edgecolor='black', alpha=0.8)
ax1.set_xlabel('Discount Rate (%)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Số lượng', fontsize=11, fontweight='bold')
ax1.set_title('Phân bố Discount Rate', fontsize=12, fontweight='bold')
ax1.axvline(discount_stats['mean'], color='red', linestyle='--', linewidth=2,
            label=f'Mean: {discount_stats["mean"]:.1f}%')
ax1.legend()
ax1.grid(alpha=0.3)

# Discount theo platform
ax2 = axes[1]
discount_by_platform = df.groupby(
    'platform')['discount_rate'].mean().sort_values()
bars = ax2.barh(discount_by_platform.index,
                discount_by_platform.values, color=colors, alpha=0.8)
ax2.set_xlabel('Discount Rate trung bình (%)', fontsize=11, fontweight='bold')
ax2.set_title('Discount Rate TB theo Platform', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, discount_by_platform.values)):
    ax2.text(val + 1, i, f'{val:.1f}%', va='center',
             fontsize=10, fontweight='bold')

# Scatter: Discount vs Price
ax3 = axes[2]
scatter_data = df[df['current_price'] < 20000000]
ax3.scatter(scatter_data['discount_rate'], scatter_data['current_price'],
            alpha=0.3, s=10, color='#FFA07A')
ax3.set_xlabel('Discount Rate (%)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Giá (VNĐ)', fontsize=11, fontweight='bold')
ax3.set_title('Mối quan hệ Discount vs Giá', fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06_discount_analysis.png',
            dpi=300, bbox_inches='tight')
print(f"   ✓ Đã lưu: 06_discount_analysis.png")
plt.close()

# 9. Phân tích Category
print("\n[9/12] Phân tích Category...")
category_counts = df['category'].value_counts().head(15)
print(f"   - Tổng số categories: {df['category'].nunique()}")
print(f"   - Top 5 categories:")
for cat, count in category_counts.head().items():
    print(f"      {cat}: {count:,}")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Top 15 categories
ax1 = axes[0, 0]
colors_cat = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(category_counts)))
bars = ax1.barh(range(len(category_counts)),
                category_counts.values, color=colors_cat)
ax1.set_yticks(range(len(category_counts)))
ax1.set_yticklabels(category_counts.index, fontsize=10)
ax1.set_xlabel('Số lượng', fontsize=11, fontweight='bold')
ax1.set_title('Top 15 Categories', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, category_counts.values)):
    ax1.text(val + category_counts.max()*0.01, i, f'{int(val):,}',
             va='center', fontsize=9, fontweight='bold')

# Giá trung bình theo category (top 10)
ax2 = axes[0, 1]
price_by_cat = df.groupby('category')['current_price'].mean().nlargest(10)
bars = ax2.barh(range(len(price_by_cat)), price_by_cat.values,
                color=plt.cm.get_cmap('Oranges')(np.linspace(0.4, 0.9, len(price_by_cat))))
ax2.set_yticks(range(len(price_by_cat)))
ax2.set_yticklabels(price_by_cat.index, fontsize=10)
ax2.set_xlabel('Giá trung bình (VNĐ)', fontsize=11, fontweight='bold')
ax2.set_title('Top 10 Categories có giá TB cao nhất',
              fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, price_by_cat.values)):
    ax2.text(val + price_by_cat.max()*0.01, i, f'{val:,.0f}',
             va='center', fontsize=9, fontweight='bold')

# Rating trung bình theo category (top 10)
ax3 = axes[1, 0]
rating_by_cat = df.groupby('category')['rating_average'].mean().nlargest(10)
bars = ax3.barh(range(len(rating_by_cat)), rating_by_cat.values,
                color=plt.cm.get_cmap('Greens')(np.linspace(0.4, 0.9, len(rating_by_cat))))
ax3.set_yticks(range(len(rating_by_cat)))
ax3.set_yticklabels(rating_by_cat.index, fontsize=10)
ax3.set_xlabel('Rating trung bình', fontsize=11, fontweight='bold')
ax3.set_title('Top 10 Categories có Rating TB cao nhất',
              fontsize=12, fontweight='bold')
ax3.set_xlim(0, 5)
ax3.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, rating_by_cat.values)):
    ax3.text(val + 0.05, i, f'{val:.2f}',
             va='center', fontsize=9, fontweight='bold')

# Số lượng bán theo category (top 10)
ax4 = axes[1, 1]
sold_by_cat = df.groupby('category')['quantity_sold'].sum().nlargest(10)
bars = ax4.barh(range(len(sold_by_cat)), sold_by_cat.values,
                color=plt.cm.get_cmap('Blues')(np.linspace(0.4, 0.9, len(sold_by_cat))))
ax4.set_yticks(range(len(sold_by_cat)))
ax4.set_yticklabels(sold_by_cat.index, fontsize=10)
ax4.set_xlabel('Tổng số lượng bán', fontsize=11, fontweight='bold')
ax4.set_title('Top 10 Categories có số lượng bán cao nhất',
              fontsize=12, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, sold_by_cat.values)):
    ax4.text(val + sold_by_cat.max()*0.01, i, f'{int(val):,}',
             va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '07_category_analysis.png',
            dpi=300, bbox_inches='tight')
print(f"   ✓ Đã lưu: 07_category_analysis.png")
plt.close()

# 10. Phân tích Brand
print("\n[10/12] Phân tích Brand...")
brand_counts = df[df['brand'] != 'UNKNOWN']['brand'].value_counts().head(20)
unknown_count = (df['brand'] == 'UNKNOWN').sum()
print(
    f"   - Tổng số brands (không kể UNKNOWN): {df[df['brand'] != 'UNKNOWN']['brand'].nunique()}")
print(f"   - Số lượng UNKNOWN: {unknown_count:,}")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Top 20 brands
ax1 = axes[0, 0]
colors_brand = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(brand_counts)))
bars = ax1.barh(range(len(brand_counts)),
                brand_counts.values, color=colors_brand)
ax1.set_yticks(range(len(brand_counts)))
ax1.set_yticklabels(brand_counts.index, fontsize=10)
ax1.set_xlabel('Số lượng', fontsize=11, fontweight='bold')
ax1.set_title('Top 20 Brands', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, brand_counts.values)):
    ax1.text(val + brand_counts.max()*0.01, i, f'{int(val):,}',
             va='center', fontsize=9, fontweight='bold')

# Tỷ lệ UNKNOWN vs có brand
ax2 = axes[0, 1]
brand_summary = pd.Series({
    'Có Brand': len(df[df['brand'] != 'UNKNOWN']),
    'UNKNOWN': unknown_count
})
colors_summary = ['#6BCF7F', '#FF6B6B']
explode = (0.05, 0.05)
wedges, texts, autotexts = ax2.pie(
    brand_summary.values,
    labels=brand_summary.index,
    autopct='%1.1f%%',
    colors=colors_summary,
    explode=explode,
    shadow=True,
    startangle=90
)
ax2.set_title('Tỷ lệ sản phẩm có/không có Brand',
              fontsize=12, fontweight='bold')
for text in texts:
    text.set_fontsize(11)
    text.set_fontweight('bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(11)
    autotext.set_fontweight('bold')

# Giá TB theo brand (top 10)
ax3 = axes[1, 0]
price_by_brand = df[df['brand'] != 'UNKNOWN'].groupby(
    'brand')['current_price'].mean().nlargest(10)
bars = ax3.barh(range(len(price_by_brand)), price_by_brand.values,
                color=plt.cm.get_cmap('Purples')(np.linspace(0.4, 0.9, len(price_by_brand))))
ax3.set_yticks(range(len(price_by_brand)))
ax3.set_yticklabels(price_by_brand.index, fontsize=10)
ax3.set_xlabel('Giá trung bình (VNĐ)', fontsize=11, fontweight='bold')
ax3.set_title('Top 10 Brands có giá TB cao nhất',
              fontsize=12, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, price_by_brand.values)):
    ax3.text(val + price_by_brand.max()*0.01, i, f'{val:,.0f}',
             va='center', fontsize=9, fontweight='bold')

# Số lượng bán theo brand (top 10)
ax4 = axes[1, 1]
sold_by_brand = df[df['brand'] != 'UNKNOWN'].groupby(
    'brand')['quantity_sold'].sum().nlargest(10)
bars = ax4.barh(range(len(sold_by_brand)), sold_by_brand.values,
                color=plt.cm.get_cmap('Reds')(np.linspace(0.4, 0.9, len(sold_by_brand))))
ax4.set_yticks(range(len(sold_by_brand)))
ax4.set_yticklabels(sold_by_brand.index, fontsize=10)
ax4.set_xlabel('Tổng số lượng bán', fontsize=11, fontweight='bold')
ax4.set_title('Top 10 Brands bán chạy nhất', fontsize=12, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, sold_by_brand.values)):
    ax4.text(val + sold_by_brand.max()*0.01, i, f'{int(val):,}',
             va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '08_brand_analysis.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Đã lưu: 08_brand_analysis.png")
plt.close()

# 11. Phân tích Location
print("\n[11/12] Phân tích Location...")
location_counts = df[df['seller_location'] !=
                     'UNKNOWN']['seller_location'].value_counts().head(20)
unknown_loc = (df['seller_location'] == 'UNKNOWN').sum()
print(
    f"   - Tổng số locations: {df[df['seller_location'] != 'UNKNOWN']['seller_location'].nunique()}")
print(f"   - Số lượng UNKNOWN: {unknown_loc:,}")

fig, ax = plt.subplots(figsize=(14, 10))
colors_loc = plt.cm.get_cmap('viridis')(
    np.linspace(0, 1, len(location_counts)))
bars = ax.barh(range(len(location_counts)),
               location_counts.to_numpy(dtype=int), color=colors_loc)
ax.set_yticks(range(len(location_counts)))
ax.set_yticklabels(location_counts.index, fontsize=10)
ax.set_xlabel('Số lượng', fontsize=11, fontweight='bold')
ax.set_title('Top 20 Địa điểm bán hàng', fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, location_counts.values)):
    ax.text(val + location_counts.max()*0.01, i, f'{int(val):,}',
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '09_location_analysis.png',
            dpi=300, bbox_inches='tight')
print(f"   ✓ Đã lưu: 09_location_analysis.png")
plt.close()

# 12. Correlation Matrix
print("\n[12/12] Tạo Correlation Matrix...")
numeric_cols = ['current_price', 'discount_rate',
                'rating_average', 'num_reviews', 'quantity_sold']
corr_data = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
            vmin=-1, vmax=1, annot_kws={'size': 11, 'weight': 'bold'})
ax.set_title('Correlation Matrix - Các biến số',
             fontsize=14, fontweight='bold', pad=20)
plt.yticks(rotation=0)
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '10_correlation_matrix.png',
            dpi=300, bbox_inches='tight')
print(f"   ✓ Đã lưu: 10_correlation_matrix.png")
plt.close()

# Tổng kết
print("\n" + "=" * 80)
print("HOÀN THÀNH".center(80))
print("=" * 80)
print(f"\n✅ Đã tạo 10 biểu đồ trực quan hóa cho dữ liệu đã làm sạch")
print(f"✅ Tất cả files đã được lưu trong: {OUTPUT_DIR}")
print("\nCÁC FILE ĐÃ TẠO:")
print("  1. 01_platform_analysis.png - Phân tích platform")
print("  2. 02_price_analysis.png - Phân tích giá")
print("  3. 03_rating_analysis.png - Phân tích rating")
print("  4. 04_review_analysis.png - Phân tích review count")
print("  5. 05_quantity_sold_analysis.png - Phân tích quantity sold")
print("  6. 06_discount_analysis.png - Phân tích discount")
print("  7. 07_category_analysis.png - Phân tích category")
print("  8. 08_brand_analysis.png - Phân tích brand")
print("  9. 09_location_analysis.png - Phân tích location")
print(" 10. 10_correlation_matrix.png - Ma trận tương quan")
print("\n" + "=" * 80)
print("\n💡 Dữ liệu đã sẵn sàng cho bước TRANSFORMATION!")
print("=" * 80)
