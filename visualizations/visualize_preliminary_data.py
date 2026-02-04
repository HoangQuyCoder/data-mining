import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Cấu hình matplotlib để hiển thị tiếng Việt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# Đường dẫn file
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / 'data' / 'preliminary' / 'merged_preliminary_data.json'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'visualizations' / 'raw'
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("TRỰC QUAN HÓA DỮ LIỆU PRELIMINARY - PHỤC VỤ LÀM SẠCH DỮ LIỆU".center(80))
print("=" * 80)

# 1. Đọc dữ liệu
print("\n[1/12] Đang đọc dữ liệu...")
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(f"   ✓ Đã đọc {len(df):,} records từ {len(df.columns)} columns")

# 2. Thống kê tổng quan
print("\n[2/12] Thống kê tổng quan về dữ liệu...")
print(f"   - Tổng số records: {len(df):,}")
print(f"   - Tổng số columns: {len(df.columns)}")
print(f"   - Các platforms: {df['platform'].unique().tolist()}")
print(
    f"   - Khoảng thời gian: {df['crawl_date'].min()} đến {df['crawl_date'].max()}")

# 3. Phân tích giá trị null/missing
print("\n[3/12] Phân tích giá trị NULL/Missing...")
null_analysis = pd.DataFrame({
    'Column': df.columns,
    'Null_Count': df.isnull().sum().values,
    'Null_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
})
null_analysis = null_analysis.sort_values('Null_Count', ascending=False)
print(null_analysis.to_string(index=False))

# Vẽ biểu đồ NULL values
fig, ax = plt.subplots(figsize=(14, 8))
colors = ['#ff6b6b' if x > 50 else '#ffd93d' if x > 20 else '#6bcf7f'
          for x in null_analysis['Null_Percentage']]
bars = ax.barh(null_analysis['Column'],
               null_analysis['Null_Percentage'], color=colors)
ax.set_xlabel('Phần trăm giá trị NULL (%)', fontsize=12, fontweight='bold')
ax.set_title('Tỷ lệ giá trị NULL/Missing theo từng trường dữ liệu',
             fontsize=14, fontweight='bold', pad=20)
ax.axvline(x=50, color='red', linestyle='--', alpha=0.3, label='50% threshold')
ax.axvline(x=20, color='orange', linestyle='--',
           alpha=0.3, label='20% threshold')
ax.legend()
ax.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, null_analysis['Null_Percentage'])):
    if val > 0:
        ax.text(val + 1, i, f'{val:.1f}%', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_null_values_analysis.png',
            dpi=300, bbox_inches='tight')
print(f"   ✓ Đã lưu: 01_null_values_analysis.png")
plt.close()

# 4. Phân tích theo Platform
print("\n[4/12] Phân tích phân bố theo Platform...")
platform_counts = df['platform'].value_counts()
print(platform_counts)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Pie chart
colors_pie = ['#ff6b6b', '#4ecdc4', '#45b7d1']
explode = [0.05] * len(platform_counts)
wedges, texts, autotexts = ax1.pie(platform_counts.values,
                                   labels=platform_counts.index,
                                   autopct='%1.1f%%',
                                   colors=colors_pie,
                                   explode=explode,
                                   shadow=True,
                                   startangle=90)
ax1.set_title('Phân bố dữ liệu theo Platform',
              fontsize=14, fontweight='bold', pad=20)
for text in texts:
    text.set_fontsize(12)
    text.set_fontweight('bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(11)
    autotext.set_fontweight('bold')

# Bar chart
bars = ax2.bar(platform_counts.index, platform_counts.values, color=colors_pie,
               edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Số lượng sản phẩm', fontsize=12, fontweight='bold')
ax2.set_title('Số lượng sản phẩm theo Platform',
              fontsize=14, fontweight='bold', pad=20)
ax2.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height):,}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_platform_distribution.png',
            dpi=300, bbox_inches='tight')
print(f"   ✓ Đã lưu: 02_platform_distribution.png")
plt.close()

# 5. Phân tích Category
print("\n[5/12] Phân tích Category...")
category_counts = df['category_name'].value_counts().head(15)
print(f"   - Tổng số categories: {df['category_name'].nunique()}")
print(f"   - Top 15 categories phổ biến nhất:")
print(category_counts)

fig, ax = plt.subplots(figsize=(14, 8))
bars = ax.barh(range(len(category_counts)), category_counts.to_numpy(dtype=int),
               color=plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(category_counts))))
ax.set_yticks(range(len(category_counts)))
ax.set_yticklabels(category_counts.index, fontsize=10)
ax.set_xlabel('Số lượng sản phẩm', fontsize=12, fontweight='bold')
ax.set_title('Top 15 Danh mục sản phẩm phổ biến nhất',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, category_counts.values)):
    ax.text(val + max(category_counts.values)*0.01, i, f'{int(val):,}',
            va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_category_distribution.png',
            dpi=300, bbox_inches='tight')
print(f"   ✓ Đã lưu: 03_category_distribution.png")
plt.close()

# 6. Phân tích giá (price)
print("\n[6/12] Phân tích phân bố giá sản phẩm...")

# Làm sạch và chuyển đổi giá


def clean_price(price_str):
    if pd.isna(price_str):
        return None
    try:
        # Loại bỏ ký tự không phải số
        price_clean = str(price_str).replace(
            '₫', '').replace('.', '').replace(',', '').strip()
        return float(price_clean)
    except:
        return None


df['price_cleaned'] = df['price'].apply(clean_price)
df['original_price_cleaned'] = df['original_price']

# Thống kê giá
price_stats = df['price_cleaned'].describe()
print(f"   Thống kê giá (VNĐ):")
print(
    f"   - Số lượng có giá: {df['price_cleaned'].notna().sum():,} / {len(df):,}")
print(f"   - Giá thấp nhất: {price_stats['min']:,.0f} VNĐ")
print(f"   - Giá trung bình: {price_stats['mean']:,.0f} VNĐ")
print(f"   - Giá trung vị: {price_stats['50%']:,.0f} VNĐ")
print(f"   - Giá cao nhất: {price_stats['max']:,.0f} VNĐ")

# Vẽ biểu đồ phân bố giá
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Histogram
ax1 = axes[0, 0]
df['price_cleaned'].dropna().hist(
    bins=100, ax=ax1, color='#4ecdc4', edgecolor='black')
ax1.set_xlabel('Giá (VNĐ)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Số lượng sản phẩm', fontsize=11, fontweight='bold')
ax1.set_title('Phân bố giá sản phẩm (Toàn bộ)', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)

# Histogram (giá < 20 triệu để nhìn rõ hơn)
ax2 = axes[0, 1]
df[df['price_cleaned'] < 20000000]['price_cleaned'].hist(bins=100, ax=ax2,
                                                         color='#ff6b6b', edgecolor='black')
ax2.set_xlabel('Giá (VNĐ)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Số lượng sản phẩm', fontsize=11, fontweight='bold')
ax2.set_title('Phân bố giá sản phẩm (< 20 triệu VNĐ)',
              fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3)

# Boxplot theo platform
ax3 = axes[1, 0]
price_by_platform = [df[df['platform'] == p]['price_cleaned'].dropna()
                     for p in df['platform'].unique()]
bp = ax3.boxplot(price_by_platform,
                 labels=df['platform'].unique(), patch_artist=True, showfliers=False)
for patch, color in zip(bp['boxes'], ['#ff6b6b', '#4ecdc4', '#45b7d1']):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax3.set_ylabel('Giá (VNĐ)', fontsize=11, fontweight='bold')
ax3.set_title('Phân bố giá theo Platform (Boxplot)',
              fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# Violin plot
ax4 = axes[1, 1]
platforms = df['platform'].unique()
positions = range(len(platforms))
for i, platform in enumerate(platforms):
    data = df[df['platform'] == platform]['price_cleaned'].dropna()
    # Giới hạn ở giá < 50 triệu để nhìn rõ hơn
    data = data[data < 50000000]
    if len(data) > 0:
        parts = ax4.violinplot([data], positions=[i],
                               widths=0.7, showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor(['#ff6b6b', '#4ecdc4', '#45b7d1'][i])
            pc.set_alpha(0.7)

ax4.set_xticks(positions)
ax4.set_xticklabels(platforms)
ax4.set_ylabel('Giá (VNĐ)', fontsize=11, fontweight='bold')
ax4.set_title('Phân bố giá theo Platform (< 50 triệu VNĐ)',
              fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_price_distribution.png',
            dpi=300, bbox_inches='tight')
print(f"   ✓ Đã lưu: 04_price_distribution.png")
plt.close()

# 7. Phân tích Rating và Review
print("\n[7/12] Phân tích Rating và Review Count...")

rating_stats = df['rating_average'].describe()
review_stats = df['review_count'].describe()

print(f"   Rating:")
print(f"   - Có rating: {df['rating_average'].notna().sum():,} / {len(df):,}")
print(f"   - Rating trung bình: {rating_stats['mean']:.2f}")
print(f"   - Rating thấp nhất: {rating_stats['min']:.2f}")
print(f"   - Rating cao nhất: {rating_stats['max']:.2f}")

print(f"\n   Review Count:")
print(f"   - Có review: {df['review_count'].notna().sum():,} / {len(df):,}")
print(f"   - Review trung bình: {review_stats['mean']:.0f}")
print(f"   - Review thấp nhất: {review_stats['min']:.0f}")
print(f"   - Review cao nhất: {review_stats['max']:.0f}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Rating distribution
ax1 = axes[0, 0]
df['rating_average'].dropna().hist(
    bins=50, ax=ax1, color='#ffd93d', edgecolor='black')
ax1.set_xlabel('Rating', fontsize=11, fontweight='bold')
ax1.set_ylabel('Số lượng sản phẩm', fontsize=11, fontweight='bold')
ax1.set_title('Phân bố Rating', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)

# Rating theo platform
ax2 = axes[0, 1]
rating_by_platform = df.groupby(
    'platform')['rating_average'].mean().sort_values()
bars = ax2.barh(rating_by_platform.index, rating_by_platform.values,
                color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
ax2.set_xlabel('Rating trung bình', fontsize=11, fontweight='bold')
ax2.set_title('Rating trung bình theo Platform',
              fontsize=12, fontweight='bold')
ax2.set_xlim(0, 5)
ax2.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, rating_by_platform.values)):
    ax2.text(val + 0.05, i, f'{val:.2f}',
             va='center', fontsize=10, fontweight='bold')

# Review count distribution (log scale)
ax3 = axes[1, 0]
review_data = df['review_count'].dropna()
review_data = review_data[review_data > 0]
ax3.hist(np.log10(review_data + 1), bins=50,
         color='#95e1d3', edgecolor='black')
ax3.set_xlabel('Log10(Review Count + 1)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Số lượng sản phẩm', fontsize=11, fontweight='bold')
ax3.set_title('Phân bố Review Count (Log scale)',
              fontsize=12, fontweight='bold')
ax3.grid(alpha=0.3)

# Scatter: Rating vs Review Count
ax4 = axes[1, 1]
scatter_data = df[['rating_average', 'review_count']].dropna()
# Giới hạn để nhìn rõ
scatter_data = scatter_data[scatter_data['review_count'] < 10000]
ax4.scatter(scatter_data['review_count'], scatter_data['rating_average'],
            alpha=0.3, s=10, color='#f38181')
ax4.set_xlabel('Review Count', fontsize=11, fontweight='bold')
ax4.set_ylabel('Rating', fontsize=11, fontweight='bold')
ax4.set_title('Mối quan hệ giữa Rating và Review Count',
              fontsize=12, fontweight='bold')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '05_rating_review_analysis.png',
            dpi=300, bbox_inches='tight')
print(f"   ✓ Đã lưu: 05_rating_review_analysis.png")
plt.close()

# 8. Phân tích Discount
print("\n[8/12] Phân tích Discount...")

# Parse discount percentage


def parse_discount(discount_str):
    if pd.isna(discount_str):
        return None
    try:
        return float(str(discount_str).replace('%', '').replace('Off', '').strip())
    except:
        return None


df['discount_pct'] = df['discount_rate'].apply(parse_discount)

discount_stats = df['discount_pct'].describe()
print(f"   - Có discount: {df['discount_pct'].notna().sum():,} / {len(df):,}")
print(f"   - Discount trung bình: {discount_stats['mean']:.1f}%")
print(f"   - Discount thấp nhất: {discount_stats['min']:.1f}%")
print(f"   - Discount cao nhất: {discount_stats['max']:.1f}%")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Discount distribution
ax1 = axes[0]
df['discount_pct'].dropna().hist(
    bins=50, ax=ax1, color='#a29bfe', edgecolor='black')
ax1.set_xlabel('Discount (%)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Số lượng sản phẩm', fontsize=11, fontweight='bold')
ax1.set_title('Phân bố mức Discount', fontsize=12, fontweight='bold')
ax1.grid(alpha=0.3)

# Discount theo platform
ax2 = axes[1]
discount_by_platform = df.groupby(
    'platform')['discount_pct'].mean().sort_values()
bars = ax2.barh(discount_by_platform.index, discount_by_platform.values,
                color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
ax2.set_xlabel('Discount trung bình (%)', fontsize=11, fontweight='bold')
ax2.set_title('Discount trung bình theo Platform',
              fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, discount_by_platform.values)):
    ax2.text(val + 1, i, f'{val:.1f}%', va='center',
             fontsize=10, fontweight='bold')

# Phần trăm sản phẩm có discount theo platform
ax3 = axes[2]
discount_pct_by_platform = df.groupby('platform').apply(
    lambda x: (x['discount_pct'].notna().sum() / len(x) * 100)
).sort_values()
bars = ax3.barh(discount_pct_by_platform.index, discount_pct_by_platform.values,
                color=['#ff6b6b', '#4ecdc4', '#45b7d1'])
ax3.set_xlabel('% sản phẩm có discount', fontsize=11, fontweight='bold')
ax3.set_title('Tỷ lệ sản phẩm có Discount theo Platform',
              fontsize=12, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)
for i, (bar, val) in enumerate(zip(bars, discount_pct_by_platform.values)):
    ax3.text(val + 1, i, f'{val:.1f}%', va='center',
             fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '06_discount_analysis.png',
            dpi=300, bbox_inches='tight')
print(f"   ✓ Đã lưu: 06_discount_analysis.png")
plt.close()

# 9. Phản tích Brand
print("\n[9/12] Phân tích Brand...")

brand_counts = df['brand'].value_counts().head(20)
print(f"   - Tổng số brands: {df['brand'].nunique()}")
print(f"   - Top 20 brands:")
print(brand_counts)

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Top brands
ax1 = axes[0]
colors_brand = plt.cm.get_cmap('tab20')(np.linspace(0, 1, len(brand_counts)))
bars = ax1.barh(range(len(brand_counts)),
                brand_counts.values, color=colors_brand)
ax1.set_yticks(range(len(brand_counts)))
ax1.set_yticklabels(brand_counts.index, fontsize=10)
ax1.set_xlabel('Số lượng sản phẩm', fontsize=11, fontweight='bold')
ax1.set_title('Top 20 Brands có nhiều sản phẩm nhất',
              fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, brand_counts.values)):
    ax1.text(val + max(brand_counts.values)*0.01, i, f'{int(val):,}',
             va='center', fontsize=9, fontweight='bold')

# Tỷ lệ No Brand
ax2 = axes[1]
no_brand_count = df[df['brand'] == 'No Brand'].shape[0]
has_brand_count = df[df['brand'] != 'No Brand'].shape[0]
sizes = [no_brand_count, has_brand_count]
labels = ['No Brand', 'Có Brand']
colors = ['#ff6b6b', '#6bcf7f']
explode = (0.1, 0)

wedges, texts, autotexts = ax2.pie(sizes, labels=labels, autopct='%1.1f%%',
                                   colors=colors, explode=explode, shadow=True,
                                   startangle=90)
ax2.set_title('Tỷ lệ sản phẩm có/không có Brand',
              fontsize=12, fontweight='bold')
for text in texts:
    text.set_fontsize(11)
    text.set_fontweight('bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(11)
    autotext.set_fontweight('bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '07_brand_analysis.png', dpi=300, bbox_inches='tight')
print(f"   ✓ Đã lưu: 07_brand_analysis.png")
plt.close()

# 10. Phân tích Location
print("\n[10/12] Phân tích Location...")

location_counts = df['location'].value_counts().head(20)
print(f"   - Tổng số locations: {df['location'].nunique()}")
print(f"   - Top 20 locations:")
print(location_counts)

fig, ax = plt.subplots(figsize=(14, 10))
colors_loc = plt.cm.get_cmap('viridis')(
    np.linspace(0, 1, len(location_counts)))
bars = ax.barh(range(len(location_counts)),
               location_counts.to_numpy(dtype=int), color=colors_loc)
ax.set_yticks(range(len(location_counts)))
ax.set_yticklabels(location_counts.index, fontsize=10)
ax.set_xlabel('Số lượng sản phẩm', fontsize=11, fontweight='bold')
ax.set_title('Top 20 Địa điểm bán hàng phổ biến nhất',
             fontsize=12, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars, location_counts.values)):
    ax.text(val + max(location_counts.values)*0.01, i, f'{int(val):,}',
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '08_location_analysis.png',
            dpi=300, bbox_inches='tight')
print(f"   ✓ Đã lưu: 08_location_analysis.png")
plt.close()

# 11. Phân tích Quantity Sold
print("\n[11/12] Phân tích Quantity Sold...")

sold_df = df.dropna(subset=['quantity_sold_value'])
sold_df = sold_df[sold_df['quantity_sold_value'] > 0]

top_sold = sold_df.sort_values(
    'quantity_sold_value', ascending=False
).head(20)

print(f"   - Số sản phẩm có dữ liệu bán: {len(sold_df)}")
print("   - Top 20 sản phẩm bán chạy nhất:")
print(top_sold[['name', 'quantity_sold_value']])

fig, ax = plt.subplots(figsize=(14, 10))
bars = ax.barh(
    range(len(top_sold)),
    top_sold['quantity_sold_value'].astype(int)
)

ax.set_yticks(range(len(top_sold)))
ax.set_yticklabels(top_sold['name'], fontsize=9)
ax.set_xlabel('Số lượng bán', fontsize=11, fontweight='bold')
ax.set_title('Top 20 Sản phẩm bán chạy nhất',
             fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

for i, val in enumerate(top_sold['quantity_sold_value']):
    ax.text(val * 1.01, i, f'{int(val):,}',
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '09_top_quantity_sold.png',
            dpi=300, bbox_inches='tight')
print("   ✓ Đã lưu: 09_top_quantity_sold.png")
plt.close()

# 12. Phân tích Seller
print("\n[12/12] Phân tích Seller...")

seller_counts = (
    df['seller_name']
    .dropna()
    .value_counts()
    .head(20)
)

print(f"   - Tổng số seller: {df['seller_name'].nunique()}")
print("   - Top 20 seller nhiều sản phẩm nhất:")
print(seller_counts)

fig, ax = plt.subplots(figsize=(14, 10))
bars = ax.barh(
    range(len(seller_counts)),
    seller_counts.to_numpy(dtype=int)
)

ax.set_yticks(range(len(seller_counts)))
ax.set_yticklabels(seller_counts.index, fontsize=9)
ax.set_xlabel('Số lượng sản phẩm', fontsize=11, fontweight='bold')
ax.set_title('Top 20 Seller có nhiều sản phẩm nhất',
             fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

for i, val in enumerate(seller_counts.values):
    ax.text(val * 1.01, i, f'{int(val):,}',
            va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '10_seller_analysis.png',
            dpi=300, bbox_inches='tight')
print("   ✓ Đã lưu: 10_seller_analysis.png")
plt.close()


# 11. Tạo báo cáo Data Quality
print("\n" + "=" * 80)
print("TẠO BÁO CÁO DATA QUALITY".center(80))
print("=" * 80)

quality_report = []

# Kiểm tra các vấn đề về chất lượng dữ liệu
quality_report.append({
    'Issue': 'Missing Values',
    'Severity': 'HIGH' if null_analysis['Null_Percentage'].max() > 50 else 'MEDIUM',
    'Description': f"{null_analysis[null_analysis['Null_Percentage'] > 0].shape[0]} columns có giá trị NULL",
    'Recommendation': 'Cần xem xét fill hoặc xóa các records thiếu dữ liệu quan trọng'
})

# Kiểm tra giá
invalid_prices = df[(df['price_cleaned'].notna()) &
                    (df['price_cleaned'] <= 0)].shape[0]
if invalid_prices > 0:
    quality_report.append({
        'Issue': 'Invalid Prices',
        'Severity': 'HIGH',
        'Description': f'{invalid_prices:,} sản phẩm có giá <= 0',
        'Recommendation': 'Cần kiểm tra và sửa hoặc loại bỏ các giá không hợp lệ'
    })

# Kiểm tra rating
invalid_ratings = df[(df['rating_average'].notna()) & (
    (df['rating_average'] < 0) | (df['rating_average'] > 5))].shape[0]
if invalid_ratings > 0:
    quality_report.append({
        'Issue': 'Invalid Ratings',
        'Severity': 'MEDIUM',
        'Description': f'{invalid_ratings:,} sản phẩm có rating ngoài khoảng 0-5',
        'Recommendation': 'Cần kiểm tra và sửa hoặc loại bỏ các rating không hợp lệ'
    })

# Kiểm tra duplicate IDs
duplicate_ids = df[df.duplicated(subset=['id', 'platform'], keep=False)]
if len(duplicate_ids) > 0:
    quality_report.append({
        'Issue': 'Duplicate Records',
        'Severity': 'HIGH',
        'Description': f'{len(duplicate_ids):,} records có ID trùng lặp trong cùng platform',
        'Recommendation': 'Cần xử lý các bản ghi trùng lặp'
    })

# Kiểm tra No Brand
no_brand_pct = (df['brand'] == 'No Brand').sum() / len(df) * 100
if no_brand_pct > 10:
    quality_report.append({
        'Issue': 'Missing Brand Information',
        'Severity': 'MEDIUM',
        'Description': f'{no_brand_pct:.1f}% sản phẩm không có thông tin brand',
        'Recommendation': 'Có thể cần extract brand từ tên sản phẩm'
    })

# In báo cáo
print("\nCÁC VẤN ĐỀ CHẤT LƯỢNG DỮ LIỆU:")
print("-" * 80)
for i, issue in enumerate(quality_report, 1):
    severity_color = {
        'HIGH': '🔴',
        'MEDIUM': '🟡',
        'LOW': '🟢'
    }
    print(
        f"{i}. {severity_color[issue['Severity']]} [{issue['Severity']}] {issue['Issue']}")
    print(f"   Mô tả: {issue['Description']}")
    print(f"   Khuyến nghị: {issue['Recommendation']}")
    print()

# Lưu báo cáo vào file
with open(OUTPUT_DIR / 'data_quality_report.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("BÁO CÁO CHẤT LƯỢNG DỮ LIỆU\n".center(80))
    f.write("=" * 80 + "\n\n")

    f.write(f"Tổng số records: {len(df):,}\n")
    f.write(f"Tổng số columns: {len(df.columns)}\n")
    f.write(f"Platforms: {', '.join(df['platform'].unique())}\n")
    f.write(
        f"Thời gian: {df['crawl_date'].min()} - {df['crawl_date'].max()}\n\n")

    f.write("CÁC VẤN ĐỀ CHẤT LƯỢNG DỮ LIỆU:\n")
    f.write("-" * 80 + "\n")
    for i, issue in enumerate(quality_report, 1):
        f.write(f"{i}. [{issue['Severity']}] {issue['Issue']}\n")
        f.write(f"   Mô tả: {issue['Description']}\n")
        f.write(f"   Khuyến nghị: {issue['Recommendation']}\n\n")

    f.write("\n" + "=" * 80 + "\n")
    f.write("CHI TIẾT MISSING VALUES:\n")
    f.write("-" * 80 + "\n")
    f.write(null_analysis.to_string(index=False))

print(f"\n✓ Đã lưu báo cáo: data_quality_report.txt")

# Tổng kết
print("\n" + "=" * 80)
print("HOÀN THÀNH".center(80))
print("=" * 80)
print(f"\n✓ Đã tạo 10 biểu đồ trực quan hóa")
print(f"✓ Đã tạo báo cáo chất lượng dữ liệu")
print(f"✓ Tất cả files đã được lưu trong thư mục: {OUTPUT_DIR}")
print("\nCÁC FILE ĐÃ TẠO:")
print("  1. 01_null_values_analysis.png - Phân tích giá trị NULL")
print("  2. 02_platform_distribution.png - Phân bố theo platform")
print("  3. 03_category_distribution.png - Phân bố theo category")
print("  4. 04_price_distribution.png - Phân tích giá")
print("  5. 05_rating_review_analysis.png - Phân tích rating và review")
print("  6. 06_discount_analysis.png - Phân tích discount")
print("  7. 07_brand_analysis.png - Phân tích brand")
print("  8. 08_location_analysis.png - Phân tích location")
print("  9. 09_top_quantity_sold.png - Phân tích quantity sold")
print("  8. 10_seller_analysis.png - Phân tích seller")
print("  10. data_quality_report.txt - Báo cáo chất lượng dữ liệu")
print("\n" + "=" * 80)
