import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

class DataQualityValidator:
    """Lớp kiểm tra chất lượng dữ liệu đã làm sạch"""
    
    def __init__(self, cleaned_file):
        self.cleaned_file = cleaned_file
        # self.df = None
        self.issues = []
        self.warnings = []
        self.score = 100  # Điểm chất lượng bắt đầu từ 100
        
    def load_data(self):
        """Đọc dữ liệu đã làm sạch"""
        print("=" * 80)
        print("KIỂM TRA CHẤT LƯỢNG DỮ LIỆU SAU KHI LÀM SẠCH".center(80))
        print("=" * 80)
        print(f"\n📂 Đang đọc file: {self.cleaned_file}")
        
        with open(self.cleaned_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.df = pd.DataFrame(data)

        print(f"✓ Đã đọc {len(self.df):,} records với {len(self.df.columns)} columns\n")
        
    def check_schema(self):
        """Kiểm tra schema - các cột bắt buộc phải có"""
        print("\n[1] KIỂM TRA SCHEMA")
        print("-" * 80)
        
        required_columns = [
            'id', 'crawl_date', 'platform', 'category', 'product_name',
            'current_price', 'discount_rate', 'rating_average', 
            'num_reviews', 'quantity_sold', 'brand', 'seller_location'
        ]
        
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        
        if missing_columns:
            self.issues.append(f"❌ CRITICAL: Thiếu các cột bắt buộc: {missing_columns}")
            self.score -= 20
            print(f"   ❌ Thiếu cột: {missing_columns}")
        else:
            print("   ✅ Tất cả các cột bắt buộc đều có")
        
        print(f"   📋 Tổng số cột: {len(self.df.columns)}")
        print(f"   📋 Danh sách cột: {list(self.df.columns)}")
        
    def check_null_values(self):
        """Kiểm tra giá trị NULL"""
        print("\n[2] KIỂM TRA GIÁ TRỊ NULL/MISSING")
        print("-" * 80)
        
        null_counts = self.df.isnull().sum()
        null_percentages = (null_counts / len(self.df) * 100).round(2)
        
        critical_columns = ['id', 'platform', 'product_name', 'current_price']
        problematic = []
        
        for col in critical_columns:
            if col in self.df.columns:
                null_pct = null_percentages[col]
                if null_pct > 0:
                    problematic.append(f"{col} ({null_pct}%)")
                    self.issues.append(f"❌ CRITICAL: Cột '{col}' có {null_pct}% giá trị NULL")
                    self.score -= 15
        
        if problematic:
            print(f"   ❌ Cột quan trọng có NULL: {', '.join(problematic)}")
        else:
            print("   ✅ Không có NULL trong các cột quan trọng")
        
        # Kiểm tra các cột khác
        other_nulls = null_counts[null_counts > 0]
        if len(other_nulls) > 0:
            print(f"\n   ⚠️  Các cột khác có NULL:")
            for col, count in other_nulls.items():
                pct = null_percentages.log[col]
                print(f"      - {col}: {count:,} ({pct}%)")
                if pct > 50:
                    self.warnings.append(f"⚠️  Cột '{col}' có {pct}% NULL")
                    self.score -= 2
        else:
            print("   ✅ PERFECT: Không có giá trị NULL trong toàn bộ dataset!")
    
    def check_data_types(self):
        """Kiểm tra kiểu dữ liệu"""
        print("\n[3] KIỂM TRA KIỂU DỮ LIỆU")
        print("-" * 80)
        
        type_checks = {
            'id': ['int64', 'object', 'str'],
            'current_price': ['float64', 'int64'],
            'discount_rate': ['float64', 'int64'],
            'rating_average': ['float64', 'int64'],
            'num_reviews': ['float64', 'int64'],
            'quantity_sold': ['float64', 'int64']
        }
        
        all_correct = True
        for col, expected_types in type_checks.items():
            if col in self.df.columns:
                actual_type = str(self.df[col].dtype)
                if actual_type not in expected_types:
                    print(f"   ⚠️  {col}: {actual_type} (mong đợi: {expected_types})")
                    self.warnings.append(f"Cột '{col}' có kiểu {actual_type}")
                    all_correct = False
                else:
                    print(f"   ✅ {col}: {actual_type}")
        
        if all_correct:
            print("   ✅ Tất cả các cột số có kiểu dữ liệu đúng")
    
    def check_value_ranges(self):
        """Kiểm tra giá trị hợp lệ"""
        print("\n[4] KIỂM TRA KHOẢNG GIÁ TRỊ HỢP LỆ")
        print("-" * 80)
        
        # Kiểm tra giá
        if 'current_price' in self.df.columns:
            invalid_prices = self.df[
                (self.df['current_price'].notna()) & 
                (self.df['current_price'] <= 0)
            ]
            if len(invalid_prices) > 0:
                self.issues.append(f"❌ CRITICAL: {len(invalid_prices)} sản phẩm có giá <= 0")
                self.score -= 20
                print(f"   ❌ Giá không hợp lệ: {len(invalid_prices)} records")
            else:
                print(f"   ✅ Giá: MIN={self.df['current_price'].min():,.0f}, MAX={self.df['current_price'].max():,.0f}")
        
        # Kiểm tra rating
        if 'rating_average' in self.df.columns:
            invalid_ratings = self.df[
                (self.df['rating_average'].notna()) & 
                ((self.df['rating_average'] < 0) | (self.df['rating_average'] > 5))
            ]
            if len(invalid_ratings) > 0:
                self.issues.append(f"❌ {len(invalid_ratings)} rating ngoài khoảng 0-5")
                self.score -= 10
                print(f"   ❌ Rating không hợp lệ: {len(invalid_ratings)} records")
            else:
                print(f"   ✅ Rating: MIN={self.df['rating_average'].min():.2f}, MAX={self.df['rating_average'].max():.2f}")
        
        # Kiểm tra discount
        if 'discount_rate' in self.df.columns:
            invalid_discounts = self.df[
                (self.df['discount_rate'].notna()) & 
                ((self.df['discount_rate'] < 0) | (self.df['discount_rate'] > 100))
            ]
            if len(invalid_discounts) > 0:
                self.warnings.append(f"⚠️  {len(invalid_discounts)} discount ngoài khoảng 0-100")
                self.score -= 5
                print(f"   ⚠️  Discount không hợp lệ: {len(invalid_discounts)} records")
            else:
                print(f"   ✅ Discount: MIN={self.df['discount_rate'].min():.1f}%, MAX={self.df['discount_rate'].max():.1f}%")
        
        # Kiểm tra số âm trong các cột quantity
        if 'num_reviews' in self.df.columns:
            negative_reviews = self.df[(self.df['num_reviews'] < 0)]
            if len(negative_reviews) > 0:
                self.issues.append(f"❌ {len(negative_reviews)} có num_reviews < 0")
                self.score -= 10
                print(f"   ❌ Num reviews âm: {len(negative_reviews)} records")
            else:
                print(f"   ✅ Num reviews: MIN={self.df['num_reviews'].min():.0f}, MAX={self.df['num_reviews'].max():,.0f}")
        
        if 'quantity_sold' in self.df.columns:
            negative_sold = self.df[(self.df['quantity_sold'] < 0)]
            if len(negative_sold) > 0:
                self.issues.append(f"❌ {len(negative_sold)} có quantity_sold < 0")
                self.score -= 10
                print(f"   ❌ Quantity sold âm: {len(negative_sold)} records")
            else:
                print(f"   ✅ Quantity sold: MIN={self.df['quantity_sold'].min():.0f}, MAX={self.df['quantity_sold'].max():,.0f}")
    
    def check_duplicates(self):
        """Kiểm tra bản ghi trùng lặp"""
        print("\n[5] KIỂM TRA BẢN GHI TRÙNG LẶP")
        print("-" * 80)
        
        if 'id' in self.df.columns and 'platform' in self.df.columns:
            duplicates = self.df[self.df.duplicated(subset=['id', 'platform'], keep=False)]
            if len(duplicates) > 0:
                self.issues.append(f"❌ {len(duplicates)} bản ghi trùng lặp (id + platform)")
                self.score -= 15
                print(f"   ❌ Có {len(duplicates)} bản ghi trùng lặp")
            else:
                print("   ✅ Không có bản ghi trùng lặp")
        else:
            print("   ⚠️  Không thể kiểm tra (thiếu cột id hoặc platform)")
    
    def check_data_consistency(self):
        """Kiểm tra tính nhất quán của dữ liệu"""
        print("\n[6] KIỂM TRA TÍNH NHẤT QUÁN")
        print("-" * 80)
        
        # Kiểm tra platform values
        if 'platform' in self.df.columns:
            valid_platforms = ['Lazada', 'Tiki', 'Shopee']
            invalid_platforms = self.df[~self.df['platform'].isin(valid_platforms)]
            if len(invalid_platforms) > 0:
                self.warnings.append(f"⚠️  {len(invalid_platforms)} records có platform không hợp lệ")
                self.score -= 5
                print(f"   ⚠️  Platform không hợp lệ: {len(invalid_platforms)} records")
                print(f"      Các giá trị: {invalid_platforms['platform'].unique()}")
            else:
                print(f"   ✅ Platform values hợp lệ: {self.df['platform'].unique().tolist()}")
        
        # Kiểm tra tính nhất quán giá
        if 'current_price' in self.df.columns and 'discount_rate' in self.df.columns:
            # Nếu có discount mà giá không thay đổi
            suspicious = self.df[
                (self.df['discount_rate'] > 0) & 
                (self.df['current_price'] == self.df['current_price'])
            ]
            print(f"   ℹ️  {len(suspicious)} sản phẩm có discount (để kiểm tra thủ công nếu cần)")
        
        # Kiểm tra tên sản phẩm
        if 'product_name' in self.df.columns:
            empty_names = self.df[(self.df['product_name'].isna()) | (self.df['product_name'] == '')]
            if len(empty_names) > 0:
                self.issues.append(f"❌ {len(empty_names)} sản phẩm không có tên")
                self.score -= 10
                print(f"   ❌ Tên sản phẩm trống: {len(empty_names)} records")
            else:
                print("   ✅ Tất cả sản phẩm đều có tên")
    
    def check_transformation_readiness(self):
        """Kiểm tra sẵn sàng cho transformation"""
        print("\n[7] KIỂM TRA SẴN SÀNG CHO TRANSFORMATION")
        print("-" * 80)
        
        readiness_checks = []
        
        # 1. Các cột cần thiết cho transformation
        required_for_transform = ['current_price', 'rating_average', 'num_reviews', 'quantity_sold']
        all_present = all(col in self.df.columns for col in required_for_transform)
        
        if all_present:
            print("   ✅ Tất cả cột cần thiết cho transformation đều có")
            readiness_checks.append(True)
        else:
            missing = [col for col in required_for_transform if col not in self.df.columns]
            print(f"   ❌ Thiếu cột cho transformation: {missing}")
            readiness_checks.append(False)
            self.score -= 15
        
        # 2. Kiểm tra phân bố dữ liệu
        print(f"\n   📊 Phân bố dữ liệu:")
        if 'platform' in self.df.columns:
            platform_counts = self.df['platform'].value_counts()
            print(f"      Platform distribution:")
            for platform, count in platform_counts.items():
                pct = count / len(self.df) * 100
                print(f"         {platform}: {count:,} ({pct:.1f}%)")
            
            # Cảnh báo nếu phân bố quá lệch
            min_pct = platform_counts.min() / len(self.df) * 100
            if min_pct < 10:
                self.warnings.append(f"⚠️  Phân bố platform không đồng đều (min: {min_pct:.1f}%)")
                print(f"      ⚠️  Phân bố không đồng đều")
                readiness_checks.append(False)
            else:
                readiness_checks.append(True)
        
        # 3. Kiểm tra đủ dữ liệu cho phân tích
        min_records = 1000
        if len(self.df) >= min_records:
            print(f"\n   ✅ Đủ dữ liệu cho phân tích ({len(self.df):,} >= {min_records:,})")
            readiness_checks.append(True)
        else:
            print(f"\n   ⚠️  Dữ liệu ít ({len(self.df):,} < {min_records:,})")
            readiness_checks.append(False)
        
        # 4. Kiểm tra chất lượng features
        if 'rating_average' in self.df.columns:
            rating_coverage = (self.df['rating_average'].notna().sum() / len(self.df) * 100)
            print(f"\n   📈 Coverage của features:")
            print(f"      Rating average: {rating_coverage:.1f}%")
            
            if rating_coverage < 50:
                self.warnings.append(f"⚠️  Rating coverage thấp ({rating_coverage:.1f}%)")
        
        return all(readiness_checks)
    
    def generate_report(self):
        """Tạo báo cáo tổng hợp"""
        print("\n" + "=" * 80)
        print("BÁO CÁO CHẤT LƯỢNG DỮ LIỆU".center(80))
        print("=" * 80)
        
        # Điểm chất lượng
        print(f"\n📊 ĐIỂM CHẤT LƯỢNG: {self.score}/100")
        
        if self.score >= 90:
            quality_level = "XUẤT SẮC ⭐⭐⭐⭐⭐"
            recommendation = "Dữ liệu đã sẵn sàng cho transformation!"
        elif self.score >= 75:
            quality_level = "TốT ⭐⭐⭐⭐"
            recommendation = "Dữ liệu tốt, có thể tiến hành transformation với một số lưu ý."
        elif self.score >= 60:
            quality_level = "TRUNG BÌNH ⭐⭐⭐"
            recommendation = "Nên xem xét sửa một số vấn đề trước khi transformation."
        else:
            quality_level = "CẦN CẢI THIỆN ⭐⭐"
            recommendation = "CẦN XỬ LÝ các vấn đề nghiêm trọng trước khi transformation!"
        
        print(f"   Mức độ: {quality_level}")
        print(f"   Khuyến nghị: {recommendation}")
        
        # Các vấn đề nghiêm trọng
        if self.issues:
            print(f"\n🚨 CÁC VẤN ĐỀ NGHIÊM TRỌNG ({len(self.issues)}):")
            for i, issue in enumerate(self.issues, 1):
                print(f"   {i}. {issue}")
        else:
            print(f"\n✅ Không có vấn đề nghiêm trọng!")
        
        # Cảnh báo
        if self.warnings:
            print(f"\n⚠️  CÁC CẢNH BÁO ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")
        else:
            print(f"\n✅ Không có cảnh báo!")
        
        # Thống kê tóm tắt
        print(f"\n📈 THỐNG KÊ TÓM TẮT:")
        print(f"   - Tổng số records: {len(self.df):,}")
        print(f"   - Tổng số columns: {len(self.df.columns)}")
        
        if 'current_price' in self.df.columns:
            print(f"   - Giá trung bình: {self.df['current_price'].mean():,.0f} VNĐ")
            print(f"   - Giá median: {self.df['current_price'].median():,.0f} VNĐ")
        
        if 'rating_average' in self.df.columns:
            print(f"   - Rating trung bình: {self.df['rating_average'].mean():.2f}/5.0")
        
        if 'platform' in self.df.columns:
            print(f"   - Số platforms: {self.df['platform'].nunique()}")
        
        if 'category' in self.df.columns:
            print(f"   - Số categories: {self.df['category'].nunique()}")
        
        print("\n" + "=" * 80)
        
        # Lưu báo cáo
        self.save_report()
        
        return self.score >= 60  # True nếu đạt điểm tối thiểu
    
    def save_report(self):
        """Lưu báo cáo ra file"""
        output_dir = Path(self.cleaned_file).parent
        report_file = output_dir / 'data_quality_validation_report.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("BÁO CÁO KIỂM TRA CHẤT LƯỢNG DỮ LIỆU SAU KHI LÀM SẠCH\n".center(80))
            f.write("=" * 80 + "\n\n")
            f.write(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"File: {self.cleaned_file}\n")
            f.write(f"Tổng records: {len(self.df):,}\n")
            f.write(f"Điểm chất lượng: {self.score}/100\n\n")
            
            if self.issues:
                f.write("VẤN ĐỀ NGHIÊM TRỌNG:\n")
                f.write("-" * 80 + "\n")
                for i, issue in enumerate(self.issues, 1):
                    f.write(f"{i}. {issue}\n")
                f.write("\n")
            
            if self.warnings:
                f.write("CẢNH BÁO:\n")
                f.write("-" * 80 + "\n")
                for i, warning in enumerate(self.warnings, 1):
                    f.write(f"{i}. {warning}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
        
        print(f"\n💾 Báo cáo đã được lưu: {report_file}")
    
    def validate(self):
        """Thực hiện toàn bộ quá trình validation"""
        self.load_data()
        self.check_schema()
        self.check_null_values()
        self.check_data_types()
        self.check_value_ranges()
        self.check_duplicates()
        self.check_data_consistency()
        ready = self.check_transformation_readiness()
        result = self.generate_report()
        
        return result and ready


if __name__ == "__main__":
    # Đường dẫn file cleaned data
    base = Path(__file__).resolve().parents[1]
    cleaned_file = base / 'data' / 'clean' / 'merged_cleaned_data.json'
    
    # Kiểm tra file có tồn tại không
    if not cleaned_file.exists():
        print(f"❌ File không tồn tại: {cleaned_file}")
        print(f"   Vui lòng chạy clean_merged_data.py trước!")
        exit(1)
    
    # Thực hiện validation
    validator = DataQualityValidator(cleaned_file)
    is_ready = validator.validate()
    
    if is_ready:
        print("\n" + "🎉 " * 20)
        print("DỮ LIỆU ĐÃ SẴN SÀNG CHO TRANSFORMATION!".center(80))
        print("🎉 " * 20)
        exit(0)
    else:
        print("\n" + "⚠️ " * 20)
        print("VUI LÒNG XỬ LÝ CÁC VẤN ĐỀ TRƯỚC KHI TRANSFORMATION!".center(80))
        print("⚠️ " * 20)
        exit(1)
