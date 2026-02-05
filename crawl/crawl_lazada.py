import requests
import time
import random
import re
import datetime
import logging
from base_crawler import crawl_all_generic, get_fresh_cookies
from settings import (
    LAZADA_RAW_DIR,
    LAZADA_CATEGORY_DIR,
    LAZADA_CATEGORIES,
    MAX_PAGES,
    SLEEP_MAX,
    SLEEP_MIN
)


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def get_fresh_cookies_lazada(path: str):
    return get_fresh_cookies(
        url=f"https://www.lazada.vn/{path}",
        headless=False,
        scroll=True,
        wait_time=10
    )


def crawl_category_lazada(cat, cookies, max_pages, retries=2):
    path = cat["path"]
    name = cat["name"]
    BASE_URL = f"https://www.lazada.vn/{path}/"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "vi-VN,vi;q=0.9",
        "Referer": f"https://www.lazada.vn/{path}/",
        "X-Requested-With": "XMLHttpRequest"
    }

    products = []
    page = 1
    if not cookies:
        cookies = get_fresh_cookies_lazada(path)

    if not cookies:
        logging.error(f"Không lấy được cookie cho {name} → bỏ qua")
        return products

    PAGE_SLEEP_MIN = SLEEP_MIN / 3
    PAGE_SLEEP_MAX = SLEEP_MAX / 3

    for page in range(1, max_pages + 1):
        params = {"ajax": "true", "page": page}
        retry_count = 0
        success = False

        while retry_count < retries and not success:
            try:
                resp = requests.get(
                    BASE_URL,
                    headers=HEADERS,
                    cookies=cookies,
                    params=params,
                    timeout=20
                )

                if resp.status_code != 200:
                    logging.warning(
                        f"{name} page {page}: HTTP {resp.status_code} → refresh cookie")
                    cookies = get_fresh_cookies_lazada(path)
                    retry_count += 1
                    time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))
                    continue

                data = resp.json()
                items = data.get("mods", {}).get("listItems", [])
                if not items:
                    logging.info(f"{name} - Hết dữ liệu ở page {page}")
                    break

                logging.info(f"{name} - Page {page}: {len(items)} sản phẩm")

                # products.append(items)
                for item in items:
                    if not item or not isinstance(item, dict):
                        continue

                    sold_cnt_show = item.get("itemSoldCntShow")
                    sold_text = (sold_cnt_show or "").strip(
                    ) if sold_cnt_show else ""
                    sold_value = None

                    if sold_text:
                        numbers = re.findall(
                            r'\d+', sold_text.replace(',', ''))
                        if numbers:
                            sold_value = int(numbers[0])
                            if 'k' in sold_text.lower() or 'K' in sold_text:
                                sold_value *= 1000

                    item_url = item.get("itemUrl")
                    full_url = f"https:{item_url}" if item_url else None

                    product = {
                        "crawl_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                        "platform": "Lazada",
                        "category_name": name,
                        "id": item.get("itemId"),
                        "name": item.get("name"),
                        "price": item.get("priceShow") or item.get("price"),
                        "original_price": item.get("originalPriceShow") or item.get("originalPrice"),
                        "discount_rate": item.get("discount"),
                        "rating_average": item.get("ratingScore"),
                        "review_count": item.get("review"),
                        "quantity_sold_value": sold_value,
                        "quantity_sold_text": sold_text,
                        "brand": item.get("brandName"),
                        "location": item.get("location"),
                        "seller_name": item.get("sellerName"),
                        "url": full_url
                    }
                    products.append(product)

                success = True
                time.sleep(random.uniform(PAGE_SLEEP_MIN, PAGE_SLEEP_MAX))

            except Exception as e:
                retry_count += 1
                logging.error(f"{name} page {page}: {e}")
                time.sleep(random.uniform(SLEEP_MIN, SLEEP_MAX))

        if not success:
            break

    return products


if __name__ == "__main__":
    crawl_all_generic(
        platform_name="Lazada",
        categories=LAZADA_CATEGORIES,
        crawl_category_func=crawl_category_lazada,
        get_cookies_func=None,
        output_dir=LAZADA_RAW_DIR,
        category_dir=LAZADA_CATEGORY_DIR,
        max_pages=MAX_PAGES,
        retries=2,
        sleep_min=SLEEP_MIN,
        sleep_max=SLEEP_MAX,
        file_prefix="lazada"
    )
