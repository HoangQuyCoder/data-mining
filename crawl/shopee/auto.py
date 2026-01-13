from DrissionPage import ChromiumPage, ChromiumOptions
import time
import random
import math
import logging
import datetime
from crawl.base_crawler import crawl_all_generic
from crawl.settings import (
    SHOPEE_RAW_DIR,
    SHOPEE_CATEGORY_DIR,
    MAX_PAGES,
    SLEEP_MIN,
    SLEEP_MAX
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# --- DANH MỤC SHOPEE ---
SHOPEE_CATEGORIES = [
    {"name": "Điện thoại", "url": "https://shopee.vn/search?keyword=Điện%20thoại"},
    {"name": "Laptop", "url": "https://shopee.vn/search?keyword=Laptop"},
    {"name": "Thời trang nữ", "url": "https://shopee.vn/search?keyword=Thời%20trang%20nữ"},
    {"name": "Thời trang nam", "url": "https://shopee.vn/search?keyword=Thời%20trang%20nam"},
    {"name": "Giày dép", "url": "https://shopee.vn/search?keyword=Giày%20dép"},
    {"name": "Túi xách", "url": "https://shopee.vn/search?keyword=Túi%20xách"},
    {"name": "Đồng hồ", "url": "https://shopee.vn/search?keyword=Đồng%20hồ"},
    {"name": "Trang sức", "url": "https://shopee.vn/search?keyword=Trang%20sức"},
    {"name": "Mỹ phẩm", "url": "https://shopee.vn/search?keyword=Mỹ%20phẩm"},
    {"name": "Chăm sóc da", "url": "https://shopee.vn/search?keyword=Chăm%20sóc%20da"},
    {"name": "Máy ảnh", "url": "https://shopee.vn/search?keyword=Máy%20ảnh"},
    {"name": "Máy tính bảng", "url": "https://shopee.vn/search?keyword=Máy%20tính%20bảng"},
    {"name": "Headphone", "url": "https://shopee.vn/search?keyword=Headphone"},
    {"name": "Loa", "url": "https://shopee.vn/search?keyword=Loa"},
    {"name": "Phụ kiện điện thoại",
        "url": "https://shopee.vn/search?keyword=Phụ%20kiện%20điện%20thoại"},
    {"name": "Sách", "url": "https://shopee.vn/search?keyword=Sách"},
]

# --- HÀM DELAY THÔNG MINH ---


def smart_delay(action_type='normal'):
    """
    Tạo delay ngẫu nhiên giống người dùng thật
    """
    delays = {
        'quick': (1.5, 3),
        'normal': (3, 7),
        'careful': (5, 12),
        'wait': (10, 20)
    }

    min_delay, max_delay = delays.get(action_type, (3, 7))
    time.sleep(random.uniform(min_delay, max_delay))


# --- INIT BROWSER ---
# --- GLOBAL BROWSER INSTANCE ---
GLOBAL_PAGE = None


def get_browser_instance():
    global GLOBAL_PAGE
    if GLOBAL_PAGE:
        return GLOBAL_PAGE

    co = ChromiumOptions()
    co.set_user_agent(
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    co.set_argument('--disable-blink-features=AutomationControlled')
    co.set_argument('--no-sandbox')
    co.set_argument('--disable-dev-shm-usage')
    co.set_argument('--start-maximized')
    # Use a specific user data folder to save login session if desired
    co.set_user_data_path(r'./chrome_profile')

    GLOBAL_PAGE = ChromiumPage(addr_or_opts=co)

    # --- LOGIN WAIT LOGIC ---
    print("\n⚠️ QUAN TRỌNG - HÃY XỬ LÝ TRƯỚC KHI TIẾP TỤC:")
    print("   1. Đăng nhập tài khoản Shopee (BẮT BUỘC ĐỂ ÍT BỊ CAPTCHA HƠN!)")
    print("   2. Hoàn thành captcha/xác minh nếu có")
    print("   3. Chương trình sẽ tự động tiếp tục sau 20 giây...")
    GLOBAL_PAGE.get('https://shopee.vn')
    time.sleep(20)  # Time for manual login

    return GLOBAL_PAGE


def close_browser():
    """Đóng trình duyệt Chrome"""
    global GLOBAL_PAGE
    if GLOBAL_PAGE:
        try:
            GLOBAL_PAGE.quit()
            GLOBAL_PAGE = None
            logging.info("✅ Đã đóng Chrome")
        except Exception as e:
            logging.error(f"❌ Lỗi đóng Chrome: {e}")


def crawl_category_shopee(cat, cookies=None, max_pages=5, retries=2):
    url = cat['url']
    name = cat['name']

    try:
        page = get_browser_instance()
    except Exception as e:
        logging.error(f"❌ Lỗi khởi tạo trình duyệt: {e}")
        return []

    products = []

    logging.info(f"🚀 Bắt đầu crawl Shopee: {name}")

    try:
        if page.url != 'https://shopee.vn/':
            page.get('https://shopee.vn')
        time.sleep(2)

        # Start listener
        page.listen.start('search_items')

        page.get(url)
        logging.info(f"✅ Đã vào URL: {url}")
        time.sleep(5)

    except Exception as e:
        logging.error(f"❌ Lỗi truy cập {name}: {e}")
        page.quit()
        return []

    for page_num in range(1, max_pages + 1):
        logging.info(f"--- Đang xử lý Trang {page_num} ({name}) ---")

        # Check chặn
        if page.ele('text:Trang không khả dụng', timeout=1) or page.ele('text:Traffic Error', timeout=1):
            logging.warning("🛑 BỊ CHẶN! Hãy xử lý captcha thủ công.")
            input("👉 Nhấn Enter sau khi xử lý để tiếp tục...")
            page.listen.start('search_items')
            page.refresh()
            time.sleep(5)

        # Scroll
        page.scroll.to_bottom()
        smart_delay('quick')

        # Lấy items từ listener
        found_items_in_page = 0
        try:
            for packet in page.listen.steps(timeout=8):
                try:
                    body = packet.response.body
                    if not isinstance(body, dict):
                        continue

                    items = body['items']

                    for item in items:
                        basic = item.get('item_basic', item)
                        itemid = basic.get('itemid')
                        shopid = basic.get('shopid')

                        raw_rating = basic.get(
                            'item_rating', {}).get('rating_star', 0)
                        try:
                            r = float(
                                raw_rating) if raw_rating is not None else 0.0
                            rating_star_val = math.floor(r * 10) / 10
                        except Exception:
                            rating_star_val = raw_rating

                        price = (basic.get('price', 0) or 0) / 100000

                        product = {
                            "crawl_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                            "platform": "Shopee",
                            "category_name": name,
                            'id': str(itemid),
                            'itemid': str(itemid),
                            'shopid': str(shopid),
                            'name': basic.get('name', 'N/A'),
                            'price': price,
                            'original_price': (basic.get('price_before_discount', 0) or 0) / 100000,
                            'discount': basic.get('discount', ''),
                            'rating_average': rating_star_val,
                            'review_count': basic.get('cmt_count'),
                            'liked_count': basic.get('liked_count', 0),
                            'quantity_sold_value': basic.get('sold', 0),
                            'quantity_historical_sold_value': basic.get('historical_sold', 0),
                            'brand': str(basic.get('brand')),
                            'location': basic.get('shop_location', 'N/A'),
                            'seller_name': basic.get('shop_name'),
                            'url': f"https://shopee.vn/product/{shopid}/{itemid}"
                        }

                        # Check duplicate
                        if not any(p['id'] == product['id'] for p in products):
                            products.append(product)
                            found_items_in_page += 1

                except Exception as e:
                    logging.warning(f"⚠️ Lỗi đọc gói tin: {e}")
                    continue

        except Exception as e:
            logging.error(f"❌ Lỗi listener: {e}")

        logging.info(f"-> Tìm thấy {found_items_in_page} sản phẩm mới.")

        if found_items_in_page == 0:
            logging.warning(
                "⚠️ Không tìm thấy sản phẩm. Có thể hết trang hoặc lỗi.")
            if page_num == 1:
                pass

        # Next page
        if page_num < max_pages:
            try:
                btn_next = page.ele(
                    'css:.shopee-icon-button--right:not(.shopee-button-disabled)', timeout=2)
                if btn_next:
                    btn_next.click()
                    smart_delay('careful')
                else:
                    logging.info("🛑 Không thấy nút Next hoặc đã hết trang.")
                    break
            except Exception as e:
                logging.error(f"🛑 Lỗi next page: {e}")
                break

    return products


if __name__ == "__main__":
    try:
        crawl_all_generic(
            platform_name="Shopee",
            categories=SHOPEE_CATEGORIES,
            crawl_category_func=crawl_category_shopee,
            get_cookies_func=None,
            output_dir=SHOPEE_RAW_DIR,
            category_dir=SHOPEE_CATEGORY_DIR,
            max_pages=MAX_PAGES,
            retries=2,
            sleep_min=SLEEP_MIN,
            sleep_max=SLEEP_MAX,
            file_prefix="shopee"
        )
    finally:
        close_browser()
