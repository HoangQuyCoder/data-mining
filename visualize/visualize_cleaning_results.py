import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Thiết lập style cho biểu đồ
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_raw_data(input_file):
    """Load dữ liệu raw"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)


def load_cleaned_data(input_file):
    """Load dữ liệu cleaned"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)


def extract_price(price_str):
    """Trích xuất giá từ chuỗi"""
    if pd.isna(price_str) or price_str is None:
        return None
    if isinstance(price_str, (int, float)):
        return float(price_str)

    price_str = str(price_str).replace('₫', '').strip()
    price_str = price_str.replace('.', '').replace(',', '.')

    try:
        return float(price_str)
    except:
        return None


def create_comparison_figure(df_raw, df_cleaned, output_dir):
    """Tạo hình so sánh dữ liệu trước và sau cleaning"""

    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('So Sánh Dữ Liệu Trước và Sau Cleaning',
                 fontsize=18, fontweight='bold', y=0.995)

    # 1. So sánh số lượng records
    ax1 = plt.subplot(3, 3, 1)
    categories = ['Raw Data', 'Cleaned Data']
    counts = [len(df_raw), len(df_cleaned)]
    colors = ['#ff6b6b', '#51cf66']
    bars = ax1.bar(categories, counts, color=colors,
                   alpha=0.7, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Số Records', fontsize=11, fontweight='bold')
    ax1.set_title('1. Tổng Số Records', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, max(counts) * 1.1)

    # Thêm giá trị trên thanh
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                 f'{count:,}', ha='center', va='bottom', fontweight='bold')

    # Thêm % loại bỏ
    removed_pct = ((len(df_raw) - len(df_cleaned)) / len(df_raw)) * 100
    ax1.text(0.5, max(counts) * 0.8, f'Loại bỏ: {removed_pct:.1f}%',
             ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    # 2. So sánh số cột
    ax2 = plt.subplot(3, 3, 2)
    col_counts = [len(df_raw.columns), len(df_cleaned.columns)]
    bars = ax2.bar(categories, col_counts, color=colors,
                   alpha=0.7, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Số Cột', fontsize=11, fontweight='bold')
    ax2.set_title('2. Số Cột', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, max(col_counts) * 1.1)

    for bar, count in zip(bars, col_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{count}', ha='center', va='bottom', fontweight='bold')

    # 3. So sánh dữ liệu thiếu
    ax3 = plt.subplot(3, 3, 3)
    missing_raw = df_raw.isnull().sum().sum()
    missing_cleaned = df_cleaned.isnull().sum().sum()
    missing_pct_raw = (missing_raw / (len(df_raw) * len(df_raw.columns))) * 100
    missing_pct_cleaned = (
        missing_cleaned / (len(df_cleaned) * len(df_cleaned.columns))) * 100

    x = np.arange(2)
    width = 0.35
    bars1 = ax3.bar(x - width/2, [missing_pct_raw, 0],
                    width, label='Raw', color='#ff6b6b', alpha=0.7)
    bars2 = ax3.bar(x + width/2, [0, missing_pct_cleaned],
                    width, label='Cleaned', color='#51cf66', alpha=0.7)
    ax3.set_ylabel('% Dữ Liệu Thiếu', fontsize=11, fontweight='bold')
    ax3.set_title('3. Dữ Liệu Thiếu (%)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend()

    # 4. So sánh phân phối giá (nếu có cột price)
    ax4 = plt.subplot(3, 3, 4)
    price_cols = [col for col in df_raw.columns if 'price' in col.lower()]
    cleaned_price_col = [
        col for col in df_cleaned.columns if 'price' in col.lower()]
    raw_prices = None
    cleaned_prices = None
    if price_cols:
        price_col = price_cols[0]
        raw_prices = df_raw[price_col].apply(extract_price).dropna()

        if cleaned_price_col:
            cleaned_prices = df_cleaned[cleaned_price_col[0]].dropna()

            ax4.boxplot([raw_prices, cleaned_prices])
            ax4.set_xticklabels(['Raw', 'Cleaned'])
            ax4.set_ylabel('Giá (VNĐ)', fontsize=11, fontweight='bold')
            ax4.set_title('4. Phân Phối Giá', fontsize=12, fontweight='bold')
            ax4.ticklabel_format(style='plain', axis='y')

    # 5. So sánh giá trị trung bình
    ax5 = plt.subplot(3, 3, 5)
    if price_cols and cleaned_price_col and raw_prices is not None and cleaned_prices is not None:
        avg_raw = raw_prices.mean()
        avg_cleaned = cleaned_prices.mean()

        bars = ax5.bar(['Raw', 'Cleaned'], [avg_raw, avg_cleaned],
                       color=['#ff6b6b', '#51cf66'], alpha=0.7, edgecolor='black', linewidth=2)
        ax5.set_ylabel('Giá Trung Bình (VNĐ)', fontsize=11, fontweight='bold')
        ax5.set_title('5. Giá Trung Bình', fontsize=12, fontweight='bold')

        for bar, val in zip(bars, [avg_raw, avg_cleaned]):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                     f'{val:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # 6. So sánh phân phối theo platform (nếu có)
    ax6 = plt.subplot(3, 3, 6)
    if 'platform' in df_cleaned.columns:
        platform_dist = df_cleaned['platform'].value_counts()
        colors_platform = plt.cm.get_cmap('Set3')(
            np.linspace(0, 1, len(platform_dist)))
        ax6.pie(platform_dist.values, labels=platform_dist.index, autopct='%1.1f%%',
                colors=colors_platform.tolist(), startangle=90)
        ax6.set_title('6. Phân Phối Platform', fontsize=12, fontweight='bold')

    # 7. So sánh phân phối theo category (top 5)
    ax7 = plt.subplot(3, 3, 7)
    if 'category' in df_cleaned.columns:
        top_categories = df_cleaned['category'].value_counts().head(5)
        ax7.barh(range(len(top_categories)), top_categories.values,
                 color='#4ecdc4', alpha=0.7, edgecolor='black')
        ax7.set_yticks(range(len(top_categories)))
        ax7.set_yticklabels(top_categories.index, fontsize=9)
        ax7.set_xlabel('Số Records', fontsize=11, fontweight='bold')
        ax7.set_title('7. Top 5 Categories', fontsize=12, fontweight='bold')

        for i, v in enumerate(top_categories.values):
            ax7.text(v, i, f' {v:,}', va='center', fontweight='bold')

    # 8. So sánh rating (nếu có)
    ax8 = plt.subplot(3, 3, 8)
    rating_cols = [
        col for col in df_cleaned.columns if 'rating' in col.lower()]
    if rating_cols:
        ratings = df_cleaned[rating_cols[0]].dropna()
        ax8.hist(ratings, bins=20, color='#95e1d3',
                 alpha=0.7, edgecolor='black')
        ax8.axvline(ratings.mean(), color='red', linestyle='--',
                    linewidth=2, label=f'Trung bình: {ratings.mean():.2f}')
        ax8.set_xlabel('Rating', fontsize=11, fontweight='bold')
        ax8.set_ylabel('Số Sản Phẩm', fontsize=11, fontweight='bold')
        ax8.set_title('8. Phân Phối Rating', fontsize=12, fontweight='bold')
        ax8.legend()

    # 9. Thống kê tóm tắt dạng bảng
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')

    summary_data = []
    summary_data.append(['Chỉ Số', 'Raw Data', 'Cleaned Data'])
    summary_data.append(
        ['Tổng Records', f"{len(df_raw):,}", f"{len(df_cleaned):,}"])
    summary_data.append(
        ['Số Cột', f"{len(df_raw.columns)}", f"{len(df_cleaned.columns)}"])
    summary_data.append(
        ['% Dữ liệu thiếu', f"{missing_pct_raw:.2f}%", f"{missing_pct_cleaned:.2f}%"])

    if price_cols and cleaned_price_col and raw_prices is not None and cleaned_prices is not None:
        summary_data.append(
            ['Giá Min', f"{raw_prices.min():,.0f}", f"{cleaned_prices.min():,.0f}"])
        summary_data.append(
            ['Giá Max', f"{raw_prices.max():,.0f}", f"{cleaned_prices.max():,.0f}"])
        summary_data.append(
            ['Giá TB', f"{raw_prices.mean():,.0f}", f"{cleaned_prices.mean():,.0f}"])

    table = ax9.table(cellText=summary_data, cellLoc='center', loc='center',
                      colWidths=[0.35, 0.32, 0.32])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Định dạng header
    for i in range(3):
        table[(0, i)].set_facecolor('#4ecdc4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Định dạng hàng
    for i in range(1, len(summary_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    ax9.set_title('9. Thống Kê Tóm Tắt', fontsize=12,
                  fontweight='bold', pad=20)

    plt.tight_layout()

    # Lưu hình
    output_file = os.path.join(output_dir, 'cleaning_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Hình so sánh đã lưu: {output_file}")

    return output_file


def create_detailed_statistics(df_cleaned, output_dir):
    """Tạo hình chi tiết thống kê dữ liệu cleaned"""

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Thống Kê Chi Tiết Dữ Liệu Cleaned',
                 fontsize=18, fontweight='bold', y=0.995)

    # 1. Brand phân phối (top 10)
    ax1 = plt.subplot(2, 3, 1)
    if 'brand' in df_cleaned.columns:
        top_brands = df_cleaned['brand'].value_counts().head(10)
        ax1.barh(range(len(top_brands)), top_brands.values,
                 color='#f38181', alpha=0.7, edgecolor='black')
        ax1.set_yticks(range(len(top_brands)))
        ax1.set_yticklabels(top_brands.index, fontsize=9)
        ax1.set_xlabel('Số Sản Phẩm', fontsize=11, fontweight='bold')
        ax1.set_title('Top 10 Brands', fontsize=12, fontweight='bold')

        for i, v in enumerate(top_brands.values):
            ax1.text(v, i, f' {v:,}', va='center',
                     fontweight='bold', fontsize=8)

    # 2. Quality category distribution
    ax2 = plt.subplot(2, 3, 2)
    if 'quality_category' in df_cleaned.columns:
        quality_dist = df_cleaned['quality_category'].value_counts()
        colors_quality = ['#ff6b6b', '#ffd93d',
                          '#6bcf7f', '#4d96ff', '#a78bfa']
        ax2.pie(quality_dist.values, labels=quality_dist.index, autopct='%1.1f%%',
                colors=colors_quality, startangle=90)
        ax2.set_title('Phân Loại Chất Lượng', fontsize=12, fontweight='bold')

    # 3. Popularity category distribution
    ax3 = plt.subplot(2, 3, 3)
    if 'popularity_category' in df_cleaned.columns:
        popularity_dist = df_cleaned['popularity_category'].value_counts()
        # Sắp xếp theo thứ tự
        order = ['Very Popular', 'Popular', 'Moderate', 'Low', 'Very Low']
        popularity_dist = popularity_dist.reindex(
            [x for x in order if x in popularity_dist.index])

        colors_pop = ['#51cf66', '#94d82d', '#ffd43b', '#ff922b', '#ff6b6b']
        ax3.bar(range(len(popularity_dist)), popularity_dist.values,
                color=colors_pop, alpha=0.7, edgecolor='black')
        ax3.set_xticks(range(len(popularity_dist)))
        ax3.set_xticklabels(popularity_dist.index,
                            rotation=45, ha='right', fontsize=9)
        ax3.set_ylabel('Số Sản Phẩm', fontsize=11, fontweight='bold')
        ax3.set_title('Phân Loại Độ Phổ Biến', fontsize=12, fontweight='bold')

        for i, v in enumerate(popularity_dist.values):
            ax3.text(i, v, f'{v:,}', ha='center',
                     va='bottom', fontweight='bold', fontsize=9)

    # 4. Discount rate distribution
    ax4 = plt.subplot(2, 3, 4)
    if 'discount_rate' in df_cleaned.columns:
        discounts = df_cleaned['discount_rate'].dropna()
        ax4.hist(discounts, bins=30, color='#74c0fc',
                 alpha=0.7, edgecolor='black')
        ax4.axvline(discounts.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Trung bình: {discounts.mean():.1f}%')
        ax4.set_xlabel('Discount Rate (%)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Số Sản Phẩm', fontsize=11, fontweight='bold')
        ax4.set_title('Phân Phối Tỷ Lệ Giảm Giá',
                      fontsize=12, fontweight='bold')
        ax4.legend()

    # 5. Reviews distribution
    ax5 = plt.subplot(2, 3, 5)
    if 'num_reviews' in df_cleaned.columns:
        reviews = df_cleaned['num_reviews'].dropna()
        ax5.hist(reviews, bins=30, color='#b197fc',
                 alpha=0.7, edgecolor='black')
        ax5.axvline(reviews.mean(), color='red', linestyle='--', linewidth=2,
                    label=f'Trung bình: {reviews.mean():.0f}')
        ax5.set_xlabel('Số Reviews', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Số Sản Phẩm', fontsize=11, fontweight='bold')
        ax5.set_title('Phân Phối Số Reviews', fontsize=12, fontweight='bold')
        ax5.legend()

    # 6. Seller location (top 10)
    ax6 = plt.subplot(2, 3, 6)
    if 'seller_location' in df_cleaned.columns:
        top_locations = df_cleaned['seller_location'].value_counts().head(10)
        ax6.barh(range(len(top_locations)), top_locations.values,
                 color='#a8e6cf', alpha=0.7, edgecolor='black')
        ax6.set_yticks(range(len(top_locations)))
        ax6.set_yticklabels(top_locations.index, fontsize=9)
        ax6.set_xlabel('Số Sản Phẩm', fontsize=11, fontweight='bold')
        ax6.set_title('Top 10 Seller Locations',
                      fontsize=12, fontweight='bold')

        for i, v in enumerate(top_locations.values):
            ax6.text(v, i, f' {v:,}', va='center',
                     fontweight='bold', fontsize=8)

    plt.tight_layout()

    output_file = os.path.join(output_dir, 'detailed_statistics.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Hình thống kê chi tiết đã lưu: {output_file}")

    return output_file


def create_summary_report(df_raw, df_cleaned, output_dir):
    """Tạo file báo cáo text chi tiết"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""
{'='*80}
BÁO CÁO CLEANING DỮ LIỆU
{'='*80}
Thời gian: {timestamp}

1. TÓRE TẮT
{'-'*80}
  • Tổng records trước cleaning: {len(df_raw):,}
  • Tổng records sau cleaning: {len(df_cleaned):,}
  • Số records loại bỏ: {len(df_raw) - len(df_cleaned):,}
  • Phần trăm loại bỏ: {((len(df_raw) - len(df_cleaned)) / len(df_raw)) * 100:.2f}%
  
  • Số cột trước cleaning: {len(df_raw.columns)}
  • Số cột sau cleaning: {len(df_cleaned.columns)}

2. CHẤT LƯỢNG DỮ LIỆU
{'-'*80}
Dữ liệu thiếu (raw):
"""

    # Thêm dữ liệu thiếu từ raw
    missing_raw = df_raw.isnull().sum()
    missing_raw = missing_raw[missing_raw > 0].sort_values(ascending=False)
    if len(missing_raw) > 0:
        for col, count in missing_raw.items():
            pct = (count / len(df_raw)) * 100
            report += f"  • {col}: {count:,} ({pct:.2f}%)\n"
    else:
        report += "  • Không có dữ liệu thiếu\n"

    report += f"\nDữ liệu thiếu (cleaned):\n"
    missing_cleaned = df_cleaned.isnull().sum()
    missing_cleaned = missing_cleaned[missing_cleaned > 0].sort_values(
        ascending=False)
    if len(missing_cleaned) > 0:
        for col, count in missing_cleaned.items():
            pct = (count / len(df_cleaned)) * 100
            report += f"  • {col}: {count:,} ({pct:.2f}%)\n"
    else:
        report += "  • Không có dữ liệu thiếu\n"

    # Thống kê giá
    price_cols = [col for col in df_raw.columns if 'price' in col.lower()]
    if price_cols:
        price_col = price_cols[0]
        raw_prices = df_raw[price_col].apply(
            lambda x: float(str(x).replace(
                '₫', '').replace('.', '').replace(',', '.'))
            if pd.notna(x) else None
        ).dropna()

        cleaned_price_col = [
            col for col in df_cleaned.columns if 'price' in col.lower()]
        if cleaned_price_col:
            cleaned_prices = df_cleaned[cleaned_price_col[0]].dropna()

            report += f"""
3. THỐNG KÊ GIÁ
{'-'*80}
Raw Data:
  • Min: {raw_prices.min():,.0f} VNĐ
  • Max: {raw_prices.max():,.0f} VNĐ
  • Trung bình: {raw_prices.mean():,.0f} VNĐ
  • Median: {raw_prices.median():,.0f} VNĐ
  • Std Dev: {raw_prices.std():,.0f} VNĐ

Cleaned Data:
  • Min: {cleaned_prices.min():,.0f} VNĐ
  • Max: {cleaned_prices.max():,.0f} VNĐ
  • Trung bình: {cleaned_prices.mean():,.0f} VNĐ
  • Median: {cleaned_prices.median():,.0f} VNĐ
  • Std Dev: {cleaned_prices.std():,.0f} VNĐ
"""

    # Thống kê platform
    if 'platform' in df_cleaned.columns:
        report += f"""
4. PHÂN PHỐI THEO PLATFORM
{'-'*80}
"""
        platform_dist = df_cleaned['platform'].value_counts()
        for platform, count in platform_dist.items():
            pct = (count / len(df_cleaned)) * 100
            report += f"  • {platform}: {count:,} ({pct:.2f}%)\n"

    # Thống kê category
    if 'category' in df_cleaned.columns:
        report += f"""
5. TOP 10 CATEGORIES
{'-'*80}
"""
        top_categories = df_cleaned['category'].value_counts().head(10)
        for i, (category, count) in enumerate(top_categories.items(), 1):
            pct = (count / len(df_cleaned)) * 100
            report += f"  {i}. {category}: {count:,} ({pct:.2f}%)\n"

    # Thống kê brand
    if 'brand' in df_cleaned.columns:
        report += f"""
6. TOP 10 BRANDS
{'-'*80}
"""
        top_brands = df_cleaned['brand'].value_counts().head(10)
        for i, (brand, count) in enumerate(top_brands.items(), 1):
            pct = (count / len(df_cleaned)) * 100
            report += f"  {i}. {brand}: {count:,} ({pct:.2f}%)\n"

    # Thống kê rating
    if 'rating_average' in df_cleaned.columns:
        ratings = df_cleaned['rating_average'].dropna()
        report += f"""
7. THỐNG KÊ RATING
{'-'*80}
  • Trung bình: {ratings.mean():.2f}
  • Median: {ratings.median():.2f}
  • Min: {ratings.min():.2f}
  • Max: {ratings.max():.2f}
  • Std Dev: {ratings.std():.2f}
"""

    # Quality category
    if 'quality_category' in df_cleaned.columns:
        report += f"""
8. PHÂN LOẠI CHẤT LƯỢNG
{'-'*80}
"""
        quality_dist = df_cleaned['quality_category'].value_counts()
        for quality, count in quality_dist.items():
            pct = (count / len(df_cleaned)) * 100
            report += f"  • {quality}: {count:,} ({pct:.2f}%)\n"

    # Popularity category
    if 'popularity_category' in df_cleaned.columns:
        report += f"""
9. PHÂN LOẠI ĐỘ PHỔ BIẾN
{'-'*80}
"""
        popularity_dist = df_cleaned['popularity_category'].value_counts()
        order = ['Very Popular', 'Popular', 'Moderate', 'Low', 'Very Low']
        for pop in order:
            if pop in popularity_dist.index:
                count = popularity_dist[pop]
                pct = (count / len(df_cleaned)) * 100
                report += f"  • {pop}: {count:,} ({pct:.2f}%)\n"

    report += f"""
10. KẾT LUẬN
{'-'*80}
  • Dữ liệu đã được làm sạch thành công
  • Loại bỏ {len(df_raw) - len(df_cleaned):,} records không hợp lệ
  • Chuẩn hóa giá trị và xử lý dữ liệu thiếu
  • Dữ liệu sẵn sàng cho phân tích tiếp theo

{'='*80}
"""

    output_file = os.path.join(output_dir, 'cleaning_report.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"✓ Báo cáo text đã lưu: {output_file}")
    return output_file


def main():
    """Chương trình chính"""

    # Đường dẫn
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_file = os.path.join(base, 'data/raw/merged_raw_data.json')
    cleaned_file = os.path.join(base, 'data/clean/merged_cleaned_data.json')
    output_dir = os.path.join(base, 'data/visualizations')

    # Tạo thư mục output
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("📊 TRỰC QUAN HÓA KẾT QUẢ CLEANING")
    print("="*80 + "\n")

    # Load dữ liệu
    print("📂 Đang tải dữ liệu...")
    df_raw = load_raw_data(raw_file)
    df_cleaned = load_cleaned_data(cleaned_file)
    print(
        f"✓ Raw data: {len(df_raw):,} records\n✓ Cleaned data: {len(df_cleaned):,} records\n")

    # Tạo hình so sánh
    print("🎨 Tạo hình so sánh...")
    create_comparison_figure(df_raw, df_cleaned, output_dir)

    # Tạo hình thống kê chi tiết
    print("📈 Tạo hình thống kê chi tiết...")
    create_detailed_statistics(df_cleaned, output_dir)

    # Tạo báo cáo text
    print("📝 Tạo báo cáo text...")
    create_summary_report(df_raw, df_cleaned, output_dir)

    print("\n" + "="*80)
    print("✅ HOÀN THÀNH!")
    print(f"📁 Tất cả kết quả đã lưu trong: {output_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
