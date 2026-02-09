# Data Mining Pipeline - Hướng Dẫn Sử Dụng

## 📋 Tổng Quan

Pipeline tự động thực hiện toàn bộ quy trình data mining cho phân vùng sản phẩm e-commerce:

1. **Stage 1: Data Crawling** - Thu thập dữ liệu từ Shopee, Tiki, Lazada
2. **Stage 2: Data Cleaning** - Làm sạch và chuẩn hóa dữ liệu
3. **Stage 3: Feature Engineering** - Trích xuất và tạo features
4. **Stage 4: Labeling** - Gán nhãn phân vùng sản phẩm
5. **Stage 5: Encoding** - Mã hóa dữ liệu và chia train/test

## 🚀 Cách Sử Dụng

### 1. Chạy Toàn Bộ Pipeline (Bao Gồm Crawl)

```bash
python main.py --full
```

### 2. Chạy Toàn Bộ Pipeline (Bỏ Qua Crawl)

Nếu bạn đã có dữ liệu crawl sẵn:

```bash
python main.py --full --skip-crawl
```

### 3. Chạy Một Phần Pipeline

Chạy từ stage 2 đến stage 5 (clean → encoding):

```bash
python main.py --partial --start 2 --end 5
```

### 4. Chạy Từng Stage Riêng Lẻ

#### Stage 1: Crawl Data

```bash
# Crawl tất cả platforms
python main.py --crawl

# Crawl chỉ Shopee và Tiki
python main.py --crawl --platforms shopee tiki

# Crawl với số trang tùy chỉnh
python main.py --crawl --max-pages 10
```

#### Stage 2: Clean Data

```bash
python main.py --clean
```

#### Stage 3: Feature Engineering

```bash
# Với visualizations
python main.py --feature

# Không tạo visualizations
python main.py --feature --no-visualize
```

#### Stage 4: Labeling

```bash
python main.py --label
```

#### Stage 5: Encoding

```bash
python main.py --encode
```

## 📊 Luồng Dữ Liệu

```
Raw Data (Shopee, Tiki, Lazada)
    ↓
data/preliminary/merged_preliminary_data.json
    ↓ [Stage 2: Clean]
data/clean/cleaned_merged_data.json
    ↓ [Stage 3: Feature Engineering]
data/transformation/engineered_features.json
    ↓ [Stage 4: Labeling]
data/transformation/labeled_data.json
    ↓ [Stage 5: Encoding]
data/transformation/encoded_data.json
data/transformation/encoders/ (encoder objects)
```

## ⚙️ Tùy Chỉnh Cấu Hình

Bạn có thể chỉnh sửa các tham số trong class `PipelineConfig` trong file `main.py`:

```python
# Crawl settings
self.max_pages = 5  # Số trang tối đa cho mỗi category
self.sleep_min = 2  # Thời gian chờ tối thiểu (giây)
self.sleep_max = 5  # Thời gian chờ tối đa (giây)

# Labeling settings
self.use_model = True  # Sử dụng ML model
self.prob_threshold = 0.70  # Ngưỡng xác suất
self.min_seed_per_class = 50  # Số seed tối thiểu mỗi class
self.model_type = 'random_forest'  # Loại model

# Encoding settings
self.test_size = 0.2  # Tỷ lệ test set (20%)
```

## 📝 Logs

Pipeline tự động ghi log vào:
- **Console**: Hiển thị real-time
- **File**: `pipeline.log` (trong thư mục gốc)

## 🎯 Kết Quả

Sau khi chạy xong pipeline, bạn sẽ có:

### 1. Cleaned Data
- File: `data/clean/cleaned_merged_data.json`
- Dữ liệu đã được làm sạch, chuẩn hóa

### 2. Engineered Features
- File: `data/transformation/engineered_features.json`
- Các features đã được tạo:
  - `popularity_score`: Điểm phổ biến (0-100)
  - `engagement_score`: Điểm tương tác (0-100)
  - `value_score`: Điểm giá trị (0-100)
  - `deal_quality_score`: Chất lượng deal (0-100)
  - `trend_momentum`: Momentum xu hướng
  - Và nhiều features khác...

### 3. Labeled Data
- File: `data/transformation/labeled_data.json`
- 4 nhãn phân vùng:
  - 🔥 **Hot Trend**: Sản phẩm đang viral
  - 🏆 **Best Seller**: Sản phẩm bán chạy
  - 💰 **Best Deal**: Ưu đãi tốt nhất
  - 📦 **Normal**: Sản phẩm thông thường

### 4. Encoded Data
- File: `data/transformation/encoded_data.json`
- Encoders: `data/transformation/encoders/`
- Dữ liệu đã được mã hóa, sẵn sàng cho ML models

## 🔧 Xử Lý Lỗi

### Lỗi: "Merged file not found"
**Giải pháp**: Đảm bảo bạn đã có dữ liệu trong `data/preliminary/` hoặc chạy stage crawl trước.

### Lỗi: "Cleaned file not found"
**Giải pháp**: Chạy stage 2 (clean) trước khi chạy stage 3.

### Lỗi: "Engineered file not found"
**Giải pháp**: Chạy stage 3 (feature engineering) trước khi chạy stage 4.

### Lỗi: "Labeled file not found"
**Giải pháp**: Chạy stage 4 (labeling) trước khi chạy stage 5.

## 💡 Tips

1. **Lần đầu chạy**: Sử dụng `--full` để chạy toàn bộ pipeline
2. **Thử nghiệm**: Sử dụng `--max-pages 2` để crawl ít dữ liệu hơn
3. **Debug**: Chạy từng stage riêng lẻ để dễ kiểm tra
4. **Production**: Tăng `max_pages` để có nhiều dữ liệu hơn

## 📞 Hỗ Trợ

Nếu gặp vấn đề, kiểm tra:
1. File log: `pipeline.log`
2. Console output
3. Đảm bảo tất cả dependencies đã được cài đặt: `pip install -r requirements.txt`

## 🎓 Ví Dụ Workflow

### Workflow 1: Lần Đầu Chạy (Full Pipeline)
```bash
# Chạy toàn bộ từ crawl đến encoding
python main.py --full --max-pages 3
```

### Workflow 2: Đã Có Dữ Liệu Raw
```bash
# Bỏ qua crawl, chạy từ clean đến encoding
python main.py --full --skip-crawl
```

### Workflow 3: Chỉ Cập Nhật Features và Labels
```bash
# Chạy từ stage 3 đến 5
python main.py --partial --start 3 --end 5
```

### Workflow 4: Thử Nghiệm Labeling
```bash
# Chỉ chạy labeling để xem kết quả
python main.py --label
```

### Workflow 5: Crawl Thêm Dữ Liệu
```bash
# Crawl thêm từ Lazada
python main.py --crawl --platforms lazada --max-pages 10

# Sau đó merge và clean lại
python main.py --partial --start 2 --end 5
```

---

**Happy Data Mining! 🚀**
