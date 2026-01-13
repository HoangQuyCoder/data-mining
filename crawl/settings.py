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

#Tiki
TIKI_RAW_DIR= os.path.join(RAW_DATA_DIR, "tiki")
TIKI_CATEGORY_DIR = os.path.join(TIKI_RAW_DIR, "categories")

# Shopee
SHOPEE_RAW_DIR = os.path.join(RAW_DATA_DIR, "shopee")
SHOPEE_CATEGORY_DIR = os.path.join(SHOPEE_RAW_DIR, "categories")

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



