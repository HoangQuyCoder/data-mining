import requests
import time
import random
import datetime
import logging
from crawl.base_crawler import crawl_all_generic, get_fresh_cookies
from crawl.settings import (
    TIKI_RAW_DIR,
    TIKI_CATEGORY_DIR,
    MAX_PAGES,
    SLEEP_MIN,
    SLEEP_MAX
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

TIKI_CATEGORIES = [
    {"name": "Nhà Sách Tiki", "urlKey": "nha-sach-tiki", "category": "8322"},
    {"name": "Nhà Cửa - Đời sống", "urlKey": "nha-cua-doi-song", "category": "1883"},
    {"name": "Điện Thoại - Máy Tính Bảng",
        "urlKey": "dien-thoai-may-tinh-bang", "category": "1789"},
    {"name": "Đồ Chơi - Mẹ & Bé", "urlKey": "do-choi-me-be", "category": "2549"},
    {"name": "Thiết Bị Số - Phụ Kiện Số",
        "urlKey": "thiet-bi-so-phu-kien-so", "category": "1815"},
    {"name": "Điện Gia Dụng", "urlKey": "dien-gia-dung", "category": "20824"},
    {"name": "Làm Đẹp - Sức Khỏe", "urlKey": "lam-dep-suc-khoe", "category": "1520"},
    {"name": "Ô Tô - Xe Máy - Xe Đạp",
        "urlKey": "o-to-xe-may-xe-dap", "category": "21346"},
    {"name": "Thời Trang Nữ", "urlKey": "thoi-trang-nu", "category": "931"},
    {"name": "Bách Hóa Online", "urlKey": "bach-hoa-online", "category": "4384"},
    {"name": "Thể Thao - Dã Ngoại", "urlKey": "the-thao-da-ngoai", "category": "1975"},
    {"name": "Thời Trang Nam", "urlKey": "thoi-trang-nam", "category": "915"},
    {"name": "Laptop - Máy Vi Tính - Linh Kiện",
        "urlKey": "laptop-may-vi-tinh-linh-kien", "category": "1846"},
    {"name": "Giày Dép Nam", "urlKey": "giay-dep-nam", "category": "1686"},
    {"name": "Điện Tử - Điện Lạnh", "urlKey": "dien-tu-dien-lanh", "category": "4221"},
    {"name": "Giày Dép Nữ", "urlKey": "giay-dep-nu", "category": "1703"},
    {"name": "Máy Ảnh - Máy Quay Phim", "urlKey": "may-anh", "category": "1801"},
    {"name": "Phụ kiện thời trang",
        "urlKey": "phu-kien-thoi-trang", "category": "27498"},
    {"name": "Đồng hồ và Trang sức",
        "urlKey": "dong-ho-va-trang-suc", "category": "8371"},
    {"name": "Balo và Vali", "urlKey": "balo-va-vali", "category": "6000"},
    {"name": "Túi thời trang nữ", "urlKey": "tui-thoi-trang-nu", "category": "976"},
    {"name": "Túi thời trang nam", "urlKey": "tui-thoi-trang-nam", "category": "27616"},
    {"name": "Chăm sóc nhà cửa", "urlKey": "cham-soc-nha-cua", "category": "15078"},
]


def get_fresh_cookies_tiki():
    return get_fresh_cookies(
        url="https://tiki.vn",
        headless=True,
        scroll=False,
        wait_time=10
    )


def crawl_category_tiki(cat, cookies, max_pages, retries=2):
    BASE_URL = "https://tiki.vn/api/personalish/v1/blocks/listings"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Referer": f"https://tiki.vn/{cat['urlKey']}/c{cat['category']}",
        "X-Requested-With": "XMLHttpRequest",
    }

    products = []
    params = {
        "limit": 48,
        "sort": "top_seller",
        "urlKey": cat["urlKey"],
        "category": cat["category"],
        "page": 1,
    }
    PAGE_SLEEP_MIN = SLEEP_MIN / 3
    PAGE_SLEEP_MAX = SLEEP_MAX / 3

    for page in range(1, max_pages + 1):
        params["page"] = page
        attempt = 0
        success = False
        while attempt < retries and not success:
            try:
                resp = requests.get(BASE_URL, headers=HEADERS,
                                    cookies=cookies, params=params, timeout=20)
                if resp.status_code != 200:
                    logging.warning(
                        f"{cat['name']} trang {page}: HTTP {resp.status_code}")
                    break

                data = resp.json()
                items = data.get("listings") or data.get("data", []) or []
                if not isinstance(items, list):
                    logging.warning(
                        f"{cat['name']} trang {page}: Listings không phải list")
                    break

                logging.info(
                    f"{cat['name']} - Trang {page}: {len(items)} sản phẩm")

                # products.append(items)
                for item in items:
                    if not item or not isinstance(item, dict):
                        continue  # Bỏ qua item None hoặc không phải dict

                    qs = item.get("quantity_sold", {})
                    seller = item.get("seller") or {}
                    impression_info = item.get("visible_impression_info") or {}
                    amplitude = impression_info.get("amplitude") or {}

                    url_path = item.get('url_path', '')
                    if url_path and not url_path.startswith('/'):
                        url_path = '/' + url_path

                    product = {
                        "crawl_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "platform": "Tiki",
                        "category_name": cat["name"],
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "price": item.get("price"),
                        "original_price": item.get("original_price"),
                        "discount": item.get("discount"),
                        "discount_rate": item.get("discount_rate"),
                        "rating_average": item.get("rating_average"),
                        "review_count": item.get("review_count"),
                        "quantity_sold_value": qs.get("value") if isinstance(qs, dict) else None,
                        "quantity_sold_text": qs.get("text") if isinstance(qs, dict) else None,
                        "brand": item.get("brand_name"),
                        "location": amplitude.get("origin") if isinstance(amplitude, dict) else None,
                        "seller_name": seller.get("name") if isinstance(seller, dict) else None,
                        "url": "https://tiki.vn{url_path}",
                    }
                    products.append(product)

                success = True
                time.sleep(random.uniform(PAGE_SLEEP_MIN, PAGE_SLEEP_MAX))

            except Exception as e:
                attempt += 1
                logging.error(
                    f"{cat['name']} trang {page} (thử {attempt}): {e}")
                time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

        if not success:
            break

    return products


if __name__ == "__main__":
    crawl_all_generic(
        platform_name="Tiki",
        categories=TIKI_CATEGORIES,
        crawl_category_func=crawl_category_tiki,
        get_cookies_func=get_fresh_cookies_tiki,
        output_dir=TIKI_RAW_DIR,
        category_dir=TIKI_CATEGORY_DIR,
        max_pages=MAX_PAGES,
        retries=2,
        sleep_min=SLEEP_MIN,
        sleep_max=SLEEP_MAX,
        file_prefix="tiki"
    )
