"""
Quick Demo Script - Test Pipeline với dữ liệu mẫu
=================================================

Script này giúp test pipeline nhanh chóng mà không cần crawl dữ liệu thật.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from main import DataMiningPipeline, PipelineConfig

def demo_pipeline():
    """Demo pipeline với các stage từ 2-5 (giả sử đã có dữ liệu)"""
    
    print("=" * 80)
    print("DEMO: Data Mining Pipeline")
    print("=" * 80)
    print("\nScript này sẽ chạy pipeline từ Stage 2 đến Stage 5")
    print("(Giả sử bạn đã có dữ liệu trong data/preliminary/)")
    print("\nNếu chưa có dữ liệu, hãy chạy:")
    print("  python main.py --crawl --platforms shopee --max-pages 2")
    print("=" * 80)
    
    # Confirm
    response = input("\nTiếp tục? (y/n): ")
    if response.lower() != 'y':
        print("Đã hủy.")
        return
    
    # Create config
    config = PipelineConfig()
    config.max_pages = 2  # Giảm số trang cho demo
    
    # Create pipeline
    pipeline = DataMiningPipeline(config)
    
    # Run partial pipeline (stage 2-5)
    print("\nBắt đầu chạy pipeline...\n")
    success = pipeline.run_partial_pipeline(start_stage=2, end_stage=5)
    
    if success:
        print("\n" + "=" * 80)
        print("DEMO HOÀN THÀNH!")
        print("=" * 80)
        print("\nKiểm tra kết quả tại:")
        print(f"  - Cleaned: {config.cleaned_file}")
        print(f"  - Features: {config.engineered_file}")
        print(f"  - Labels: {config.labeled_file}")
        print(f"  - Encoded: {config.encoded_file}")
        print("\nTip: Mở các file JSON này để xem kết quả!")
    else:
        print("\nDemo thất bại. Kiểm tra log để biết chi tiết.")

if __name__ == "__main__":
    demo_pipeline()
