import os

# =========================
# PROJECT ROOT
# =========================
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
# BASE_DIR = project_root


# =========================
# DATA DIRECTORIES
# =========================
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

# Lazada
LAZADA_RAW_DIR = os.path.join(RAW_DATA_DIR, "lazada")
LAZADA_CATEGORY_DIR = os.path.join(LAZADA_RAW_DIR, "categories")
LAZADA_CATEGORIES = [
    {"name": "Điện thoại di động", "path": "dien-thoai-di-dong"},
    {"name": "Máy tính bảng", "path": "may-tinh-bang"},
    {"name": "Laptop", "path": "laptop"},
    {"name": "Pin sạc dự phòng", "path": "pin-sac-du-phong"},
    {"name": "Tai nghe không dây", "path": "shop-wireless-earbuds"},
    {"name": "Máy ảnh máy quay phim", "path": "may-anh-may-quay-phim"},
    {"name": "Tủ lạnh", "path": "tu-lanh"},
    {"name": "Máy giặt", "path": "may-giat"},
    {"name": "Máy lạnh", "path": "may-lanh"},
    {"name": "Áo phông & Áo ba lỗ", "path": "shop-t-shirts-&-tanks"},
    {"name": "Quần jeans", "path": "shop-men-jeans"},
    {"name": "Dưỡng da & Serum", "path": "duong-da-va-serum"},
    {"name": "Son thỏi", "path": "son-thoi"},
    {"name": "Bách hóa online", "path": "bach-hoa-online"},
    {"name": "Phụ kiện làm thơm phòng", "path": "do-dung-lam-thom-phong"},
    {"name": "Giường", "path": "giuong"},
    {"name": "Bóng đá", "path": "bong-da"},
    {"name": "Máy chạy bộ", "path": "may-chay-bo"},
    {"name": "Bikini", "path": "bikini-2"},
    {"name": "Búp bê cho bé", "path": "bup-be-cho-be"},
    {"name": "Xe máy", "path": "xe-may"},
]

# Tiki
TIKI_RAW_DIR = os.path.join(RAW_DATA_DIR, "tiki")
TIKI_CATEGORY_DIR = os.path.join(TIKI_RAW_DIR, "categories")
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


# Shopee
SHOPEE_RAW_DIR = os.path.join(RAW_DATA_DIR, "shopee")
SHOPEE_CATEGORY_DIR = os.path.join(SHOPEE_RAW_DIR, "categories")
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

# =========================
# ENSURE DIRECTORIES EXIST
# =========================
os.makedirs(LAZADA_CATEGORY_DIR, exist_ok=True)
os.makedirs(TIKI_CATEGORY_DIR, exist_ok=True)
os.makedirs(SHOPEE_CATEGORY_DIR, exist_ok=True)


# CRAWL SETTINGS
MAX_PAGES = 20
SLEEP_MIN = 10
SLEEP_MAX = 20
