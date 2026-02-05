import requests
import time
import random
import datetime
import logging
from base_crawler import crawl_all_generic, get_fresh_cookies
from settings import (
    TIKI_RAW_DIR,
    TIKI_CATEGORY_DIR,
    TIKI_CATEGORIES,
    MAX_PAGES,
    SLEEP_MIN,
    SLEEP_MAX
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


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
        retry_count = 0
        success = False

        while retry_count < retries and not success:
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
                        continue  

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
                retry_count += 1
                logging.error(
                    f"{cat['name']} trang {page} (thử {retry_count}): {e}")
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
