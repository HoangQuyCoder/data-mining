# ==========================================
# Visualization for Recommendation
# CRISP-DM | User Behavior Mining
# ==========================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
OUTPUT_DIR = "figures"


# ---------- Load Data ----------
def load_data(filename="labeled_data.json"):
    print("[INFO] Loading labeled data...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_json(path)

    # đảm bảo kiểu số
    numeric_cols = [
        "current_price", "quantity_sold_value",
        "rating_average", "discount_percent",
        "popularity_score"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


# ---------- 1. Label Distribution ----------
def plot_label_distribution(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="label")
    plt.title("Người dùng quan tâm đến nhóm sản phẩm nào")
    plt.xlabel("Nhãn sản phẩm")
    plt.ylabel("Số lượng sản phẩm")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_label_distribution.png", dpi=300)
    plt.close()


# ---------- 2. Price Impact (BAR) ----------
def plot_price_vs_sold(df):
    df["price_range"] = pd.qcut(
        df["current_price"], 3,
        labels=["Giá thấp", "Giá trung bình", "Giá cao"]
    )

    pivot = df.groupby("price_range")["quantity_sold_value"].mean().reset_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=pivot, x="price_range", y="quantity_sold_value")
    plt.title("Ảnh hưởng của giá đến hành vi mua")
    plt.xlabel("Khoảng giá")
    plt.ylabel("Số lượng bán trung bình")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_price_vs_sold_bar.png", dpi=300)
    plt.close()


# ---------- 3. Rating Impact (BAR) ----------
def plot_rating_vs_sold(df):
    df["rating_group"] = pd.cut(
        df["rating_average"],
        bins=[0, 3, 4, 5],
        labels=["Rating thấp", "Rating khá", "Rating cao"]
    )

    pivot = df.groupby("rating_group")["quantity_sold_value"].mean().reset_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=pivot, x="rating_group", y="quantity_sold_value")
    plt.title("Rating ảnh hưởng đến quyết định mua")
    plt.xlabel("Nhóm rating")
    plt.ylabel("Số lượng bán trung bình")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_rating_vs_sold_bar.png", dpi=300)
    plt.close()


# ---------- 4. Discount Impact (BAR) ----------
def plot_discount_vs_sold(df):
    df["discount_group"] = pd.cut(
        df["discount_percent"],
        bins=[0, 10, 30, 100],
        labels=["Giảm thấp", "Giảm vừa", "Giảm cao"]
    )

    pivot = df.groupby("discount_group")["quantity_sold_value"].mean().reset_index()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=pivot, x="discount_group", y="quantity_sold_value")
    plt.title("Ảnh hưởng của giảm giá đến hành vi mua")
    plt.xlabel("Mức giảm giá")
    plt.ylabel("Số lượng bán trung bình")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_discount_vs_sold_bar.png", dpi=300)
    plt.close()


# ---------- 5. Popularity Score ----------
def plot_popularity_score(df):
    top = df.sort_values("popularity_score", ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=top, x="popularity_score", y="product_name")
    plt.title("Top sản phẩm có độ phổ biến cao (Candidate Recommendation)")
    plt.xlabel("Popularity Score")
    plt.ylabel("Sản phẩm")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_top_popularity.png", dpi=300)
    plt.close()


# ---------- 6. Top Products per Label ----------
def plot_top_products_by_label(df):
    top_df = (
        df.sort_values("quantity_sold_value", ascending=False)
          .groupby("label")
          .head(3)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=top_df,
        x="quantity_sold_value",
        y="product_name",
        hue="label"
    )
    plt.title("Top sản phẩm đề xuất theo từng nhóm người dùng")
    plt.xlabel("Số lượng bán")
    plt.ylabel("Sản phẩm")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_top_products_by_label.png", dpi=300)
    plt.close()


# ---------- 7. User Behavior Summary (BAR) ----------
def plot_user_behavior_summary(df):
    df["price_range"] = pd.qcut(
        df["current_price"], 3,
        labels=["Giá thấp", "Giá trung bình", "Giá cao"]
    )

    pivot = (
        df.groupby(["label", "price_range"])["quantity_sold_value"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=pivot,
        x="price_range",
        y="quantity_sold_value",
        hue="label"
    )

    plt.title("Hành vi mua theo khoảng giá và nhóm người dùng")
    plt.xlabel("Khoảng giá")
    plt.ylabel("Số lượng bán trung bình")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/07_user_behavior_bar.png", dpi=300)
    plt.close()


# ---------- Main ----------
def main():
    print("[INFO] Visualization script started")
    ensure_output_dir()
    df = load_data()

    plot_label_distribution(df)
    plot_price_vs_sold(df)
    plot_rating_vs_sold(df)
    plot_discount_vs_sold(df)
    plot_popularity_score(df)
    plot_top_products_by_label(df)
    plot_user_behavior_summary(df)

    print("[DONE] All BAR visualizations saved to 'figures/' folder")


if __name__ == "__main__":
    main()
