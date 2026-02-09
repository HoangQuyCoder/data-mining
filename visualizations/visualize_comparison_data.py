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
    ax1.set_title('Tổng Số Records', fontsize=12, fontweight='bold')
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
    ax2.set_title('Số Cột', fontsize=12, fontweight='bold')
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
    ax3.set_title('Dữ Liệu Thiếu (%)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend()

    # 4. So sánh phân phối giá (nếu có cột price)
    ax4 = plt.subplot(3, 3, 4)
    price_cols = [col for col in df_raw.columns if 'price' in col.lower()]
    cleaned_price_col = [
        col for col in df_cleaned.columns if 'current_price' in col.lower()]
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
            ax4.set_title('Phân Phối Giá', fontsize=12, fontweight='bold')
            ax4.ticklabel_format(style='plain', axis='y')

    # 5. So sánh giá trị trung bình
    ax5 = plt.subplot(3, 3, 5)
    if price_cols and cleaned_price_col and raw_prices is not None and cleaned_prices is not None:
        avg_raw = raw_prices.mean()
        avg_cleaned = cleaned_prices.mean()

        bars = ax5.bar(['Raw', 'Cleaned'], [avg_raw, avg_cleaned],
                       color=['#ff6b6b', '#51cf66'], alpha=0.7, edgecolor='black', linewidth=2)
        ax5.set_ylabel('Giá Trung Bình (VNĐ)', fontsize=11, fontweight='bold')
        ax5.set_title('Giá Trung Bình', fontsize=12, fontweight='bold')

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
        ax6.set_title('Phân Phối Platform', fontsize=12, fontweight='bold')

    # 7. So sánh phân phối theo category (top 5)
    ax7 = plt.subplot(3, 3, 7)
    if 'category' in df_cleaned.columns:
        top_categories = df_cleaned['category'].value_counts().head(5)
        ax7.barh(range(len(top_categories)), top_categories.values,
                 color='#4ecdc4', alpha=0.7, edgecolor='black')
        ax7.set_yticks(range(len(top_categories)))
        ax7.set_yticklabels(top_categories.index, fontsize=9)
        ax7.set_xlabel('Số Records', fontsize=11, fontweight='bold')
        ax7.set_title('Top 5 Categories', fontsize=12, fontweight='bold')

        for i, v in enumerate(top_categories.values):
            ax7.text(v, i, f' {v:,}', va='center', fontweight='bold')

    # 8. So sánh rating (nếu có)
    ax8 = plt.subplot(3, 3, 8)
    rating_cols = [
        col for col in df_cleaned.columns if 'rating_average' in col.lower()]
    if rating_cols:
        ratings = df_cleaned[rating_cols[0]].dropna()
        ax8.hist(ratings, bins=20, color='#95e1d3',
                 alpha=0.7, edgecolor='black')
        ax8.axvline(ratings.mean(), color='red', linestyle='--',
                    linewidth=2, label=f'Trung bình: {ratings.mean():.2f}')
        ax8.set_xlabel('Rating', fontsize=11, fontweight='bold')
        ax8.set_ylabel('Số Sản Phẩm', fontsize=11, fontweight='bold')
        ax8.set_title('Phân Phối Rating', fontsize=12, fontweight='bold')
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

    ax9.set_title('Thống Kê Tóm Tắt', fontsize=12,
                  fontweight='bold', pad=20)

    plt.tight_layout()

    # Lưu hình
    output_file = os.path.join(output_dir, 'cleaning_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Hình so sánh đã lưu: {output_file}")

    return output_file


def create_quantity_sold_comparison(df_raw, df_cleaned, output_dir):
    """Tạo trực quan hóa so sánh quantity sold"""
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('So Sánh Quantity Sold - Trước và Sau Cleaning',
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Tìm cột quantity_sold
    qty_cols_raw = [col for col in df_raw.columns if 'quantity_sold' in col.lower()]
    qty_cols_cleaned = [col for col in df_cleaned.columns if 'quantity_sold' in col.lower()]
    
    if not qty_cols_raw or not qty_cols_cleaned:
        print("⚠ Không tìm thấy cột quantity_sold")
        plt.close()
        return None
    
    qty_raw = df_raw[qty_cols_raw[0]].dropna()
    qty_cleaned = df_cleaned[qty_cols_cleaned[0]].dropna()
    
    # 1. Box plot so sánh
    ax1 = plt.subplot(2, 3, 1)
    bp = ax1.boxplot([qty_raw, qty_cleaned], tick_labels=['Raw', 'Cleaned'],
                      patch_artist=True, showfliers=True)
    bp['boxes'][0].set_facecolor('#ff6b6b')
    bp['boxes'][1].set_facecolor('#51cf66')
    ax1.set_ylabel('Quantity Sold', fontsize=11, fontweight='bold')
    ax1.set_title('Box Plot - Quantity Sold', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogram so sánh
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(qty_raw, bins=50, alpha=0.5, label='Raw', color='#ff6b6b', edgecolor='black')
    ax2.hist(qty_cleaned, bins=50, alpha=0.5, label='Cleaned', color='#51cf66', edgecolor='black')
    ax2.set_xlabel('Quantity Sold', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Histogram - Quantity Sold', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Thống kê so sánh
    ax3 = plt.subplot(2, 3, 3)
    stats_data = [
        ['Metric', 'Raw', 'Cleaned'],
        ['Count', f'{len(qty_raw):,}', f'{len(qty_cleaned):,}'],
        ['Mean', f'{qty_raw.mean():,.1f}', f'{qty_cleaned.mean():,.1f}'],
        ['Median', f'{qty_raw.median():,.1f}', f'{qty_cleaned.median():,.1f}'],
        ['Std Dev', f'{qty_raw.std():,.1f}', f'{qty_cleaned.std():,.1f}'],
        ['Min', f'{qty_raw.min():,.0f}', f'{qty_cleaned.min():,.0f}'],
        ['Max', f'{qty_raw.max():,.0f}', f'{qty_cleaned.max():,.0f}'],
    ]
    ax3.axis('off')
    table = ax3.table(cellText=stats_data, cellLoc='center', loc='center',
                      colWidths=[0.35, 0.32, 0.32])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#4ecdc4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(stats_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax3.set_title('Thống Kê Quantity Sold', fontsize=12, fontweight='bold', pad=20)
    
    # 4. Phân phối theo khoảng (Cleaned data)
    ax4 = plt.subplot(2, 3, 4)
    bins_ranges = [0, 100, 500, 1000, 5000, 10000, qty_cleaned.max()]
    labels = ['0-100', '100-500', '500-1K', '1K-5K', '5K-10K', f'10K+']
    qty_binned = pd.cut(qty_cleaned, bins=bins_ranges, labels=labels, include_lowest=True)
    qty_dist = qty_binned.value_counts().sort_index()
    
    colors_gradient = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(qty_dist)))
    bars = ax4.bar(range(len(qty_dist)), qty_dist.values, color=colors_gradient, edgecolor='black')
    ax4.set_xticks(range(len(qty_dist)))
    ax4.set_xticklabels(qty_dist.index, rotation=45, ha='right')
    ax4.set_ylabel('Số Sản Phẩm', fontsize=11, fontweight='bold')
    ax4.set_title('Phân Phối Theo Khoảng (Cleaned)', fontsize=12, fontweight='bold')
    
    for bar, val in zip(bars, qty_dist.values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 5. Top 10 sản phẩm bán chạy nhất (Cleaned)
    ax5 = plt.subplot(2, 3, 5)
    if 'name' in df_cleaned.columns:
        top_products = df_cleaned.nlargest(10, qty_cols_cleaned[0])[['name', qty_cols_cleaned[0]]]
        product_names = [name[:30] + '...' if len(name) > 30 else name for name in top_products['name']]
        
        bars = ax5.barh(range(len(top_products)), top_products[qty_cols_cleaned[0]].values,
                        color='#ff6b6b', alpha=0.7, edgecolor='black')
        ax5.set_yticks(range(len(top_products)))
        ax5.set_yticklabels(product_names, fontsize=8)
        ax5.set_xlabel('Quantity Sold', fontsize=11, fontweight='bold')
        ax5.set_title('Top 10 Sản Phẩm Bán Chạy', fontsize=12, fontweight='bold')
        ax5.invert_yaxis()
        
        for i, (bar, val) in enumerate(zip(bars, top_products[qty_cols_cleaned[0]].values)):
            width = bar.get_width()
            ax5.text(width, i, f' {val:,.0f}', va='center', fontsize=8, fontweight='bold')
    
    # 6. Outliers removed
    ax6 = plt.subplot(2, 3, 6)
    outliers_removed = len(qty_raw) - len(qty_cleaned)
    outliers_pct = (outliers_removed / len(qty_raw)) * 100
    
    data = [len(qty_cleaned), outliers_removed]
    labels_pie = [f'Retained\n{len(qty_cleaned):,}', f'Removed\n{outliers_removed:,}']
    colors_pie = ['#51cf66', '#ff6b6b']
    
    ax6.pie(data, labels=labels_pie, autopct='%1.1f%%',
                                         colors=colors_pie, startangle=90,
                                         textprops={'fontsize': 10, 'fontweight': 'bold'})
    
    ax6.set_title(f'Records Removed ({outliers_pct:.1f}%)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'quantity_sold_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Quantity Sold comparison saved: {output_file}")
    plt.close()
    
    return output_file


def create_rating_comparison(df_raw, df_cleaned, output_dir):
    """Tạo trực quan hóa so sánh rating"""
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('So Sánh Rating - Trước và Sau Cleaning',
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Tìm cột rating
    rating_cols_raw = [col for col in df_raw.columns if 'rating' in col.lower() and 'average' in col.lower()]
    rating_cols_cleaned = [col for col in df_cleaned.columns if 'rating' in col.lower() and 'average' in col.lower()]
    
    if not rating_cols_raw or not rating_cols_cleaned:
        print("⚠ Không tìm thấy cột rating")
        plt.close()
        return None
    
    rating_raw = df_raw[rating_cols_raw[0]].dropna()
    rating_cleaned = df_cleaned[rating_cols_cleaned[0]].dropna()
    
    # 1. Histogram so sánh
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(rating_raw, bins=20, alpha=0.6, label='Raw', color='#ff6b6b', edgecolor='black')
    ax1.hist(rating_cleaned, bins=20, alpha=0.6, label='Cleaned', color='#51cf66', edgecolor='black')
    ax1.axvline(rating_raw.mean(), color='red', linestyle='--', linewidth=2, label=f'Raw Mean: {rating_raw.mean():.2f}')
    ax1.axvline(rating_cleaned.mean(), color='green', linestyle='--', linewidth=2, label=f'Cleaned Mean: {rating_cleaned.mean():.2f}')
    ax1.set_xlabel('Rating', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Histogram - Rating Distribution', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot
    ax2 = plt.subplot(2, 3, 2)
    bp = ax2.boxplot([rating_raw, rating_cleaned], tick_labels=['Raw', 'Cleaned'],
                      patch_artist=True, showfliers=True)
    bp['boxes'][0].set_facecolor('#ff6b6b')
    bp['boxes'][1].set_facecolor('#51cf66')
    ax2.set_ylabel('Rating', fontsize=11, fontweight='bold')
    ax2.set_title('Box Plot - Rating', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Thống kê
    ax3 = plt.subplot(2, 3, 3)
    stats_data = [
        ['Metric', 'Raw', 'Cleaned'],
        ['Count', f'{len(rating_raw):,}', f'{len(rating_cleaned):,}'],
        ['Mean', f'{rating_raw.mean():.2f}', f'{rating_cleaned.mean():.2f}'],
        ['Median', f'{rating_raw.median():.2f}', f'{rating_cleaned.median():.2f}'],
        ['Std Dev', f'{rating_raw.std():.2f}', f'{rating_cleaned.std():.2f}'],
        ['Min', f'{rating_raw.min():.2f}', f'{rating_cleaned.min():.2f}'],
        ['Max', f'{rating_raw.max():.2f}', f'{rating_cleaned.max():.2f}'],
    ]
    ax3.axis('off')
    table = ax3.table(cellText=stats_data, cellLoc='center', loc='center',
                      colWidths=[0.35, 0.32, 0.32])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#4ecdc4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(stats_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax3.set_title('Thống Kê Rating', fontsize=12, fontweight='bold', pad=20)
    
    # 4. Phân phối theo khoảng rating (Cleaned)
    ax4 = plt.subplot(2, 3, 4)
    rating_ranges = [0, 1, 2, 3, 4, 5]
    rating_labels = ['0-1', '1-2', '2-3', '3-4', '4-5']
    rating_binned = pd.cut(rating_cleaned, bins=rating_ranges, labels=rating_labels, include_lowest=True)
    rating_dist = rating_binned.value_counts().sort_index()
    
    colors_rating = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#27ae60']
    bars = ax4.bar(range(len(rating_dist)), rating_dist.values, color=colors_rating, edgecolor='black')
    ax4.set_xticks(range(len(rating_dist)))
    ax4.set_xticklabels(rating_dist.index, rotation=0)
    ax4.set_ylabel('Số Sản Phẩm', fontsize=11, fontweight='bold')
    ax4.set_title('Phân Phối Theo Khoảng Rating (Cleaned)', fontsize=12, fontweight='bold')
    
    for bar, val in zip(bars, rating_dist.values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 5. Quality category distribution (nếu có)
    ax5 = plt.subplot(2, 3, 5)
    if 'quality_category' in df_cleaned.columns:
        quality_dist = df_cleaned['quality_category'].value_counts()
        cmap = plt.cm.get_cmap('RdYlGn')
        colors_quality = [tuple(c) for c in cmap(np.linspace(0.2, 0.8, len(quality_dist)))]

        ax5.pie(quality_dist.values, labels=quality_dist.index,
                                             autopct='%1.1f%%', colors=colors_quality,
                                             startangle=90, textprops={'fontsize': 9, 'fontweight': 'bold'})
        
        ax5.set_title('Phân Loại Chất Lượng', fontsize=12, fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'No quality_category\navailable', ha='center', va='center',
                 fontsize=12, transform=ax5.transAxes)
        ax5.axis('off')
    
    # 6. Top rated products
    ax6 = plt.subplot(2, 3, 6)
    if 'name' in df_cleaned.columns:
        top_rated = df_cleaned.nlargest(10, rating_cols_cleaned[0])[['name', rating_cols_cleaned[0]]]
        product_names = [name[:30] + '...' if len(name) > 30 else name for name in top_rated['name']]
        
        bars = ax6.barh(range(len(top_rated)), top_rated[rating_cols_cleaned[0]].values,
                        color='#27ae60', alpha=0.7, edgecolor='black')
        ax6.set_yticks(range(len(top_rated)))
        ax6.set_yticklabels(product_names, fontsize=8)
        ax6.set_xlabel('Rating', fontsize=11, fontweight='bold')
        ax6.set_title('Top 10 Sản Phẩm Rating Cao Nhất', fontsize=12, fontweight='bold')
        ax6.set_xlim(0, 5.5)
        ax6.invert_yaxis()
        
        for i, (bar, val) in enumerate(zip(bars, top_rated[rating_cols_cleaned[0]].values)):
            width = bar.get_width()
            ax6.text(width, i, f' {val:.2f}', va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'rating_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Rating comparison saved: {output_file}")
    plt.close()
    
    return output_file


def create_review_comparison(df_raw, df_cleaned, output_dir):
    """Tạo trực quan hóa so sánh review count"""
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('So Sánh Review Count - Trước và Sau Cleaning',
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Tìm cột review count
    review_cols_raw = [col for col in df_raw.columns if 'review_count' in col.lower() ]
    review_cols_cleaned = [col for col in df_cleaned.columns if 'num_reviews' in col.lower() ]
    
    if not review_cols_raw or not review_cols_cleaned:
        print("⚠ Không tìm thấy cột review_count")
        plt.close()
        return None
    
    review_raw = df_raw[review_cols_raw[0]].dropna()
    review_cleaned = df_cleaned[review_cols_cleaned[0]].dropna()
    
    # 1. Box plot
    ax1 = plt.subplot(2, 3, 1)
    bp = ax1.boxplot([review_raw, review_cleaned], tick_labels=['Raw', 'Cleaned'],
                      patch_artist=True, showfliers=True)
    bp['boxes'][0].set_facecolor('#ff6b6b')
    bp['boxes'][1].set_facecolor('#51cf66')
    ax1.set_ylabel('Review Count', fontsize=11, fontweight='bold')
    ax1.set_title('Box Plot - Review Count', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogram
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(review_raw, bins=50, alpha=0.5, label='Raw', color='#ff6b6b', edgecolor='black')
    ax2.hist(review_cleaned, bins=50, alpha=0.5, label='Cleaned', color='#51cf66', edgecolor='black')
    ax2.set_xlabel('Review Count', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title('Histogram - Review Count', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Thống kê
    ax3 = plt.subplot(2, 3, 3)
    stats_data = [
        ['Metric', 'Raw', 'Cleaned'],
        ['Count', f'{len(review_raw):,}', f'{len(review_cleaned):,}'],
        ['Mean', f'{review_raw.mean():,.1f}', f'{review_cleaned.mean():,.1f}'],
        ['Median', f'{review_raw.median():,.1f}', f'{review_cleaned.median():,.1f}'],
        ['Std Dev', f'{review_raw.std():,.1f}', f'{review_cleaned.std():,.1f}'],
        ['Min', f'{review_raw.min():,.0f}', f'{review_cleaned.min():,.0f}'],
        ['Max', f'{review_raw.max():,.0f}', f'{review_cleaned.max():,.0f}'],
    ]
    ax3.axis('off')
    table = ax3.table(cellText=stats_data, cellLoc='center', loc='center',
                      colWidths=[0.35, 0.32, 0.32])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#4ecdc4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(stats_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax3.set_title('Thống Kê Review Count', fontsize=12, fontweight='bold', pad=20)
    
    # 4. Phân phối theo khoảng
    ax4 = plt.subplot(2, 3, 4)
    bins_ranges = [0, 10, 50, 100, 500, 1000, review_cleaned.max()]
    labels = ['0-10', '10-50', '50-100', '100-500', '500-1K', '1K+']
    review_binned = pd.cut(review_cleaned, bins=bins_ranges, labels=labels, include_lowest=True)
    review_dist = review_binned.value_counts().sort_index()
    
    colors_gradient = plt.cm.get_cmap('plasma')(np.linspace(0, 1, len(review_dist)))
    bars = ax4.bar(range(len(review_dist)), review_dist.values, color=colors_gradient, edgecolor='black')
    ax4.set_xticks(range(len(review_dist)))
    ax4.set_xticklabels(review_dist.index, rotation=45, ha='right')
    ax4.set_ylabel('Số Sản Phẩm', fontsize=11, fontweight='bold')
    ax4.set_title('Phân Phối Theo Khoảng (Cleaned)', fontsize=12, fontweight='bold')
    
    for bar, val in zip(bars, review_dist.values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 5. Top products by review count
    ax5 = plt.subplot(2, 3, 5)
    if 'name' in df_cleaned.columns:
        top_reviewed = df_cleaned.nlargest(10, review_cols_cleaned[0])[['name', review_cols_cleaned[0]]]
        product_names = [name[:30] + '...' if len(name) > 30 else name for name in top_reviewed['name']]
        
        bars = ax5.barh(range(len(top_reviewed)), top_reviewed[review_cols_cleaned[0]].values,
                        color='#3498db', alpha=0.7, edgecolor='black')
        ax5.set_yticks(range(len(top_reviewed)))
        ax5.set_yticklabels(product_names, fontsize=8)
        ax5.set_xlabel('Review Count', fontsize=11, fontweight='bold')
        ax5.set_title('Top 10 Sản Phẩm Nhiều Review Nhất', fontsize=12, fontweight='bold')
        ax5.invert_yaxis()
        
        for i, (bar, val) in enumerate(zip(bars, top_reviewed[review_cols_cleaned[0]].values)):
            width = bar.get_width()
            ax5.text(width, i, f' {val:,.0f}', va='center', fontsize=8, fontweight='bold')
    
    # 6. Correlation with rating (nếu có)
    ax6 = plt.subplot(2, 3, 6)
    rating_cols = [col for col in df_cleaned.columns if 'rating' in col.lower() and 'average' in col.lower()]
    if rating_cols:
        scatter_data = df_cleaned[[review_cols_cleaned[0], rating_cols[0]]].dropna()
        ax6.scatter(scatter_data[review_cols_cleaned[0]], scatter_data[rating_cols[0]],
                    alpha=0.5, c='#9b59b6', edgecolors='black', s=30)
        ax6.set_xlabel('Review Count', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Rating', fontsize=11, fontweight='bold')
        ax6.set_title('Correlation: Review Count vs Rating', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # Tính correlation
        corr = scatter_data.corr().iloc[0, 1]
        ax6.text(0.05, 0.95, f'Correlation: {corr:.3f}',
                 transform=ax6.transAxes, fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                 verticalalignment='top')
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'review_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Review comparison saved: {output_file}")
    plt.close()
    
    return output_file


def create_discount_comparison(df_raw, df_cleaned, output_dir):
    """Tạo trực quan hóa so sánh discount"""
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('So Sánh Discount - Trước và Sau Cleaning',
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Tìm cột discount
    discount_cols_raw = [col for col in df_raw.columns if 'discount_rate' in col.lower()]
    discount_cols_cleaned = [col for col in df_cleaned.columns if 'discount_rate' in col.lower()]
    
    if not discount_cols_raw or not discount_cols_cleaned:
        print("⚠ Không tìm thấy cột discount_rate")
        plt.close()
        return None
    
    discount_raw = df_raw[discount_cols_raw[0]].dropna()
    discount_cleaned = df_cleaned[discount_cols_cleaned[0]].dropna()
    
    # 1. Histogram
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(discount_raw, bins=30, alpha=0.6, label='Raw', color='#ff6b6b', edgecolor='black')
    ax1.hist(discount_cleaned, bins=30, alpha=0.6, label='Cleaned', color='#51cf66', edgecolor='black')
    ax1.set_xlabel('Discount Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title('Histogram - Discount Rate', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plot
    ax2 = plt.subplot(2, 3, 2)
    bp = ax2.boxplot([discount_raw, discount_cleaned], tick_labels=['Raw', 'Cleaned'],
                      patch_artist=True, showfliers=True)
    bp['boxes'][0].set_facecolor('#ff6b6b')
    bp['boxes'][1].set_facecolor('#51cf66')
    ax2.set_ylabel('Discount Rate (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Box Plot - Discount Rate', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. Thống kê
    ax3 = plt.subplot(2, 3, 3)
    stats_data = [
        ['Metric', 'Raw', 'Cleaned'],
        ['Count', f'{len(discount_raw):,}', f'{len(discount_cleaned):,}'],
        ['Mean', f'{discount_raw.mean():.1f}%', f'{discount_cleaned.mean():.1f}%'],
        ['Median', f'{discount_raw.median():.1f}%', f'{discount_cleaned.median():.1f}%'],
        ['Std Dev', f'{discount_raw.std():.1f}%', f'{discount_cleaned.std():.1f}%'],
        ['Min', f'{discount_raw.min():.0f}%', f'{discount_cleaned.min():.0f}%'],
        ['Max', f'{discount_raw.max():.0f}%', f'{discount_cleaned.max():.0f}%'],
    ]
    ax3.axis('off')
    table = ax3.table(cellText=stats_data, cellLoc='center', loc='center',
                      colWidths=[0.35, 0.32, 0.32])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#4ecdc4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(stats_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax3.set_title('Thống Kê Discount Rate', fontsize=12, fontweight='bold', pad=20)
    
    # 4. Phân phối theo khoảng discount
    ax4 = plt.subplot(2, 3, 4)
    discount_ranges = [0, 10, 20, 30, 40, 50, 100]
    discount_labels = ['0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50%+']
    discount_binned = pd.cut(discount_cleaned, bins=discount_ranges, labels=discount_labels, include_lowest=True)
    discount_dist = discount_binned.value_counts().sort_index()
    
    colors_discount = plt.cm.get_cmap('Reds')(np.linspace(0.3, 0.9, len(discount_dist)))
    bars = ax4.bar(range(len(discount_dist)), discount_dist.values, color=colors_discount, edgecolor='black')
    ax4.set_xticks(range(len(discount_dist)))
    ax4.set_xticklabels(discount_dist.index, rotation=45, ha='right')
    ax4.set_ylabel('Số Sản Phẩm', fontsize=11, fontweight='bold')
    ax4.set_title('Phân Phối Theo Khoảng Discount (Cleaned)', fontsize=12, fontweight='bold')
    
    for bar, val in zip(bars, discount_dist.values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:,}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 5. Products with highest discount
    ax5 = plt.subplot(2, 3, 5)
    if 'name' in df_cleaned.columns:
        top_discount = df_cleaned.nlargest(10, discount_cols_cleaned[0])[['name', discount_cols_cleaned[0]]]
        product_names = [name[:30] + '...' if len(name) > 30 else name for name in top_discount['name']]
        
        bars = ax5.barh(range(len(top_discount)), top_discount[discount_cols_cleaned[0]].values,
                        color='#e74c3c', alpha=0.7, edgecolor='black')
        ax5.set_yticks(range(len(top_discount)))
        ax5.set_yticklabels(product_names, fontsize=8)
        ax5.set_xlabel('Discount Rate (%)', fontsize=11, fontweight='bold')
        ax5.set_title('Top 10 Sản Phẩm Giảm Giá Cao Nhất', fontsize=12, fontweight='bold')
        ax5.invert_yaxis()
        
        for i, (bar, val) in enumerate(zip(bars, top_discount[discount_cols_cleaned[0]].values)):
            width = bar.get_width()
            ax5.text(width, i, f' {val:.0f}%', va='center', fontsize=8, fontweight='bold')
    
    # 6. Pie chart - có discount vs không discount
    ax6 = plt.subplot(2, 3, 6)
    has_discount = (discount_cleaned > 0).sum()
    no_discount = (discount_cleaned == 0).sum()
    
    data = [has_discount, no_discount]
    labels_pie = [f'Có Giảm Giá\n{has_discount:,}', f'Không Giảm Giá\n{no_discount:,}']
    colors_pie = ['#e74c3c', '#95a5a6']
    
    ax6.pie(data, labels=labels_pie, autopct='%1.1f%%',
                                         colors=colors_pie, startangle=90,
                                         textprops={'fontsize': 10, 'fontweight': 'bold'})
    ax6.set_title('Tỷ Lệ Sản Phẩm Có/Không Giảm Giá', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'discount_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Discount comparison saved: {output_file}")
    plt.close()
    
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
    raw_file = os.path.join(base, 'data/preliminary/merged_preliminary_data.json')
    cleaned_file = os.path.join(base, 'data/clean/merged_cleaned_data.json')
    output_dir = os.path.join(base, 'data/visualizations/compare')

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

    # Tạo hình so sánh tổng quan
    print("🎨 Tạo hình so sánh tổng quan...")
    create_comparison_figure(df_raw, df_cleaned, output_dir)

    # Tạo trực quan hóa chi tiết cho từng metric
    print("\n📈 Tạo trực quan hóa chi tiết...")
    
    print("  → Quantity Sold comparison...")
    create_quantity_sold_comparison(df_raw, df_cleaned, output_dir)
    
    print("  → Rating comparison...")
    create_rating_comparison(df_raw, df_cleaned, output_dir)
    
    print("  → Review Count comparison...")
    create_review_comparison(df_raw, df_cleaned, output_dir)
    
    print("  → Discount comparison...")
    # create_discount_comparison(df_raw, df_cleaned, output_dir)

    # Tạo báo cáo text
    print("\n📝 Tạo báo cáo text...")
    create_summary_report(df_raw, df_cleaned, output_dir)

    print("\n" + "="*80)
    print("✅ HOÀN THÀNH!")
    print(f"📁 Tất cả kết quả đã lưu trong: {output_dir}")
    print("="*80 + "\n")



if __name__ == "__main__":
    main()
