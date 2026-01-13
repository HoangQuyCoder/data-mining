import os
import time
import random
import pandas as pd
import datetime
import logging
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


def get_fresh_cookies(
    url: str = "https://tiki.vn",
    headless: bool = True,
    scroll: bool = False,
    wait_time: int = 10,
) -> dict:
    """
    Hàm chung lấy cookies từ một URL bất kỳ.

    Args:
        url: URL cần lấy cookies (mặc định tiki.vn)
        headless: Chế độ headless (mặc định True)
        scroll: Có scroll trang hay không (mặc định False)
        wait_time: Thời gian chờ tải trang (mặc định 10 giây)

    Returns:
        Dict cookies hoặc {} nếu lỗi
    """
    options = Options()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
    )

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    try:
        driver.get(url)
        time.sleep(wait_time)

        if scroll:
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight / 2);")
            time.sleep(5)

        cookies = {c['name']: c['value'] for c in driver.get_cookies()}
        logging.info(
            f"✅ Lấy cookies thành công: {len(cookies)} cookies từ {url}")
        return cookies
    except Exception as e:
        logging.error(f"Lỗi lấy cookies từ {url}: {e}")
        return {}
    finally:
        driver.quit()


def crawl_all_generic(
    *,
    platform_name: str,
    categories: list,
    crawl_category_func,
    output_dir: str,
    get_cookies_func=None,
    category_dir: str | None = None,
    max_pages: int = 20,
    retries: int = 2,
    sleep_min: int = 8,
    sleep_max: int = 15,
    file_prefix: str | None = None,
):
    """
    Generic crawl_all template

    Lưu dữ liệu vào:
    - File tổng hợp tất cả danh mục: output_dir
    - File riêng từng danh mục: category_dir (nếu được cung cấp)
    """

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    all_products = []

    cookies = None
    if get_cookies_func:
        cookies = get_cookies_func()

    logging.info(f"🚀 Bắt đầu crawl {platform_name}")

    os.makedirs(output_dir, exist_ok=True)
    if category_dir:
        os.makedirs(category_dir, exist_ok=True)

    prefix = file_prefix or platform_name.lower()

    for cat in tqdm(categories, desc=f"Danh mục {platform_name}"):
        try:
            prods = crawl_category_func(
                cat,
                cookies=cookies,
                max_pages=max_pages,
                retries=retries
            )

            if prods:
                all_products.extend(prods)

                # Lưu từng danh mục vào file riêng
                if category_dir:
                    cat_name = cat.get("name", "unknown").replace(
                        "/", "_").replace(" ", "_")
                    cat_filename = os.path.join(
                        category_dir,
                        f"{prefix}_{cat_name}_{timestamp}.json"
                    )
                    pd.DataFrame(prods).to_json(
                        cat_filename,
                        orient="records",
                        force_ascii=False,
                        indent=2
                    )
                    logging.info(
                        f"{cat['name']}: Lưu {len(prods)} sản phẩm → {cat_filename}")

        except Exception as e:
            logging.error(f"❌ Lỗi category {cat}: {e}")

        time.sleep(random.uniform(sleep_min, sleep_max))

    if not all_products:
        logging.warning(f"{platform_name}: Không có dữ liệu")
        return

    # Lưu file tổng hợp tất cả danh mục
    filename = os.path.join(
        output_dir,
        f"{prefix}_all_{timestamp}.json"
    )

    pd.DataFrame(all_products).to_json(
        filename,
        orient="records",
        force_ascii=False,
        indent=2
    )

    logging.info(
        f"{platform_name}: Lưu {len(all_products)} sản phẩm (tổng) → {filename}"
    )
