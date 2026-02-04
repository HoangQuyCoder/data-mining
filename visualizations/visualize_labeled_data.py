import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 180
plt.rcParams['figure.figsize'] = (12, 6)

COLORS = {
    'Hot Trend':    '#FF6B6B',
    'Best Seller':  '#4ECDC4',
    'Best Deal':    '#45B7D1',
    'Normal':       '#96CEB4'
}

LABEL_ORDER = ['Hot Trend', 'Best Seller', 'Best Deal', 'Normal']


def load_labeled_data(file_path=None):
    """Load file JSON đã labeling"""
    if file_path is None:
        # Đường dẫn mặc định theo cấu trúc dự án
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(
            base_dir, 'data', 'transformation', 'labeled_data.json')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}\n"
                                "Hãy kiểm tra đường dẫn hoặc chạy labeling trước.")

    print(f"Đang đọc dữ liệu từ: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"→ Đọc được {len(df):,} sản phẩm | {len(df.columns)} cột")
    return df


def ensure_output_folder() -> str:
    """Tạo thư mục lưu ảnh trong data/visualizations/label"""

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'data', 'visualizations', 'label')

    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def plot_label_distribution(df, folder):
    """1. Phân bố nhãn - Pie + Bar + %"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    counts = df['label'].value_counts()
    percentages = counts / len(df) * 100

    # Pie chart
    explode = [0.04] * len(counts)
    wedges, texts, autotexts = ax1.pie(
        percentages,
        labels=[f"{lbl}\n{perc:.1f}%" for lbl,
                perc in zip(counts.index, percentages)],
        autopct='',
        startangle=90,
        explode=explode,
        colors=[COLORS.get(l, '#cccccc') for l in counts.index],
        shadow=True,
        textprops={'fontsize': 11}
    )
    ax1.set_title("Tỷ lệ phân bố nhãn sản phẩm",
                  fontsize=14, fontweight='bold', pad=20)

    # Bar chart
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index,  palette=[COLORS.get(l, '#cccccc') for l in counts.index],
                order=LABEL_ORDER, legend=False, ax=ax2)
    ax2.set_title("Số lượng sản phẩm theo nhãn",
                  fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylabel("Số lượng", fontsize=12)
    ax2.set_xlabel("")

    # Thêm text lên bar
    total = len(df)
    for i, v in enumerate(counts.reindex(LABEL_ORDER)):
        ax2.text(i, v + total * 0.004, f"{v:,}\n({v/total*100:.1f}%)",
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(folder, "01_phan_bo_nhan.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu: {save_path}")


def plot_key_metrics_by_label(df, folder):
    """2. So sánh các chỉ số chính theo nhãn"""
    metrics = [
        'trend_momentum', 'engagement_score', 'popularity_score',
        'sales_velocity_normalized', 'deal_quality_score', 'value_score'
    ]

    # Chỉ lấy cột tồn tại
    metrics = [m for m in metrics if m in df.columns]
    if not metrics:
        print("Không tìm thấy cột metric nào để vẽ so sánh.")
        return

    n = len(metrics)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4.2 * n), squeeze=False)

    for i, metric in enumerate(metrics):
        # Boxplot
        sns.boxplot(data=df, x='label', y=metric, hue='label', order=LABEL_ORDER,
                    palette=[COLORS.get(l, '#cccccc') for l in LABEL_ORDER], legend=False,
                    ax=axes[i, 0], showfliers=True, flierprops={'marker': 'o', 'markersize': 3})
        axes[i, 0].set_title(f"Phân bố: {metric}", fontsize=13)
        axes[i, 0].set_ylabel(metric)
        axes[i, 0].set_xlabel("")

        # Bar mean
        means = df.groupby('label')[metric].mean().reindex(LABEL_ORDER)
        sns.barplot(x=means.index, y=means.values, hue=means.index,
                    palette=[COLORS.get(l, '#cccccc') for l in LABEL_ORDER],
                    legend=False,
                    ax=axes[i, 1])
        axes[i, 1].set_title(f"Trung bình: {metric}", fontsize=13)
        axes[i, 1].set_ylabel("Giá trị trung bình")
        axes[i, 1].set_xlabel("")

        # Giá trị trên bar
        for j, val in enumerate(means):
            axes[i, 1].text(j, val + max(means) * 0.015, f"{val:.2f}",
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_path = os.path.join(folder, "02_so_sanh_chi_so_chinh.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu: {save_path}")


def plot_scatter_trend_engagement(df, folder, sample_size=8000):
    """3. Scatter Trend Momentum vs Engagement Score"""
    if 'trend_momentum' not in df.columns or 'engagement_score' not in df.columns:
        print("Thiếu cột 'trend_momentum' hoặc 'engagement_score' → bỏ qua scatter plot.")
        return

    plot_df = df.sample(min(len(df), sample_size), random_state=42)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=plot_df,
        x='trend_momentum',
        y='engagement_score',
        hue='label',
        hue_order=LABEL_ORDER,
        palette=[COLORS.get(l, '#cccccc') for l in LABEL_ORDER],
        alpha=0.65,
        s=45,
        edgecolor='none'
    )
    plt.title("Trend Momentum vs Engagement Score theo nhãn",
              fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Trend Momentum", fontsize=12)
    plt.ylabel("Engagement Score", fontsize=12)
    plt.legend(title="Nhãn", bbox_to_anchor=(
        1.02, 1), loc='upper left', fontsize=10)
    plt.grid(alpha=0.3)

    save_path = os.path.join(folder, "03_scatter_trend_vs_engagement.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Đã lưu: {save_path}")


if __name__ == "__main__":
    try:
        df = load_labeled_data()

        output_folder = ensure_output_folder()
        print(f"\nLưu tất cả biểu đồ vào thư mục: {output_folder}\n")

        plot_label_distribution(df, output_folder)
        plot_key_metrics_by_label(df, output_folder)
        plot_scatter_trend_engagement(df, output_folder)

        print("\n" + "="*60)
        print("HOÀN THÀNH – Tất cả biểu đồ đã được lưu thành file PNG")
        print("="*60)

    except Exception as e:
        print("\nLỖI:", str(e))
