"""
Data Mining Pipeline - Automated E-commerce Product Segmentation
=================================================================

Pipeline tự động thực hiện toàn bộ quy trình data mining:
1. Crawl dữ liệu từ Shopee, Tiki, Lazada
2. Clean & Normalize dữ liệu
3. Feature Engineering
4. Labeling (Product Segmentation)
5. Encoding & Train/Test Split
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime
from typing import Optional, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# PIPELINE CONFIGURATION
# ============================================================================


class PipelineConfig:
    """Configuration cho toàn bộ pipeline"""

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        # Directories
        self.data_dir = os.path.join(self.base_dir, 'data')
        self.raw_dir = os.path.join(self.data_dir, 'raw')
        self.preliminary_dir = os.path.join(self.data_dir, 'preliminary')
        self.clean_dir = os.path.join(self.data_dir, 'clean')
        self.transformation_dir = os.path.join(self.data_dir, 'transformation')

        # Output files
        self.merged_raw_file = os.path.join(
            self.preliminary_dir, 'merged_preliminary_data.json')
        self.cleaned_file = os.path.join(
            self.clean_dir, 'cleaned_merged_data.json')
        self.engineered_file = os.path.join(
            self.transformation_dir, 'engineered_features.json')
        self.labeled_file = os.path.join(
            self.transformation_dir, 'labeled_data.json')
        self.encoded_file = os.path.join(
            self.transformation_dir, 'encoded_data.json')
        self.encoder_dir = os.path.join(self.transformation_dir, 'encoders')

        # Crawl settings
        self.max_pages = 5  # Số trang tối đa cho mỗi category
        self.sleep_min = 2
        self.sleep_max = 5

        # Transformation settings
        self.use_model = True  # Sử dụng ML model cho labeling
        self.prob_threshold = 0.70
        self.min_seed_per_class = 50
        self.model_type = 'random_forest'
        self.test_size = 0.2

        # Ensure directories exist
        self._create_directories()

    def _create_directories(self):
        """Tạo các thư mục cần thiết"""
        for directory in [self.data_dir, self.raw_dir, self.preliminary_dir,
                          self.clean_dir, self.transformation_dir, self.encoder_dir]:
            os.makedirs(directory, exist_ok=True)


# ============================================================================
# PIPELINE STAGES
# ============================================================================

class DataMiningPipeline:
    """Main pipeline class điều phối toàn bộ quy trình"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stats = {
            'start_time': datetime.now(),
            'stages_completed': [],
            'stages_failed': [],
            'data_counts': {}
        }

    # ========================================================================
    # STAGE 1: DATA CRAWLING
    # ========================================================================

    def stage_1_crawl_data(self, platforms: list[str] | None = None):
        """
        Stage 1: Crawl dữ liệu từ các e-commerce platforms

        Parameters:
            platforms: List các platform cần crawl ['shopee', 'tiki', 'lazada']
                      Nếu None, crawl tất cả
        """
        logger.info("=" * 80)
        logger.info("STAGE 1: DATA CRAWLING")
        logger.info("=" * 80)

        if platforms is None:
            platforms = ['shopee', 'tiki', 'lazada']

        try:
            # Import crawl modules
            sys.path.insert(0, os.path.join(self.config.base_dir, 'crawl'))

            for platform in platforms:
                logger.info(f"\n📥 Crawling data from {platform.upper()}...")

                try:
                    if platform.lower() == 'shopee':
                        from crawl.crawl_shopee import crawl_category_shopee, close_browser
                        from crawl.base_crawler import crawl_all_generic
                        from crawl.settings import SHOPEE_CATEGORIES, SHOPEE_RAW_DIR, SHOPEE_CATEGORY_DIR

                        crawl_all_generic(
                            platform_name="Shopee",
                            categories=SHOPEE_CATEGORIES,
                            crawl_category_func=crawl_category_shopee,
                            get_cookies_func=None,
                            output_dir=SHOPEE_RAW_DIR,
                            category_dir=SHOPEE_CATEGORY_DIR,
                            max_pages=self.config.max_pages,
                            retries=2,
                            sleep_min=self.config.sleep_min,
                            sleep_max=self.config.sleep_max,
                            file_prefix="shopee"
                        )
                        close_browser()

                    elif platform.lower() == 'tiki':
                        from crawl.crawl_tiki import crawl_category_tiki, get_fresh_cookies_tiki
                        from crawl.base_crawler import crawl_all_generic
                        from crawl.settings import TIKI_CATEGORIES, TIKI_RAW_DIR, TIKI_CATEGORY_DIR

                        crawl_all_generic(
                            platform_name="Tiki",
                            categories=TIKI_CATEGORIES,
                            crawl_category_func=crawl_category_tiki,
                            get_cookies_func=get_fresh_cookies_tiki,
                            output_dir=TIKI_RAW_DIR,
                            category_dir=TIKI_CATEGORY_DIR,
                            max_pages=self.config.max_pages,
                            retries=2,
                            sleep_min=self.config.sleep_min,
                            sleep_max=self.config.sleep_max,
                            file_prefix="tiki"
                        )

                    elif platform.lower() == 'lazada':
                        from crawl.crawl_lazada import crawl_category_lazada
                        from crawl.base_crawler import crawl_all_generic
                        from crawl.settings import LAZADA_CATEGORIES, LAZADA_RAW_DIR, LAZADA_CATEGORY_DIR

                        crawl_all_generic(
                            platform_name="Lazada",
                            categories=LAZADA_CATEGORIES,
                            crawl_category_func=crawl_category_lazada,
                            get_cookies_func=None,
                            output_dir=LAZADA_RAW_DIR,
                            category_dir=LAZADA_CATEGORY_DIR,
                            max_pages=self.config.max_pages,
                            retries=2,
                            sleep_min=self.config.sleep_min,
                            sleep_max=self.config.sleep_max,
                            file_prefix="lazada"
                        )

                    logger.info(
                        f"Successfully crawled data from {platform.upper()}")

                except Exception as e:
                    logger.error(f"Failed to crawl {platform.upper()}: {e}")
                    self.stats['stages_failed'].append(f"crawl_{platform}")
                    continue

            self.stats['stages_completed'].append('stage_1_crawl')
            logger.info("\nSTAGE 1 COMPLETED: Data Crawling")
            return True

        except Exception as e:
            logger.error(f"STAGE 1 FAILED: {e}")
            self.stats['stages_failed'].append('stage_1_crawl')
            return False

    # ========================================================================
    # STAGE 2: DATA CLEANING
    # ========================================================================

    def stage_2_clean_data(self):
        """
        Stage 2: Clean và normalize dữ liệu
        - Merge tất cả file JSON từ preliminary
        - Normalize format từ các platform khác nhau
        - Handle outliers, missing values
        - Standardize data types
        """
        logger.info("=" * 80)
        logger.info("STAGE 2: DATA CLEANING & NORMALIZATION")
        logger.info("=" * 80)

        try:
            # Import clean module
            sys.path.insert(0, os.path.join(self.config.base_dir, 'clean'))
            from clean.clean_data import DataCleaner

            # Check if merged file exists
            if not os.path.exists(self.config.merged_raw_file):
                logger.warning(
                    f"Merged file not found: {self.config.merged_raw_file}")
                logger.info(
                    "Merging JSON files from preliminary directory...")

                # Merge JSON files
                sys.path.insert(0, os.path.join(
                    self.config.base_dir, 'common'))
                from common.utils import merge_json_files

                total = merge_json_files(
                    input_dir=self.config.preliminary_dir,
                    output_file=self.config.merged_raw_file
                )
                logger.info(f"✅ Merged {total} records")

            # Clean data
            logger.info(
                f"\nCleaning data from: {self.config.merged_raw_file}")
            cleaner = DataCleaner(
                input_file=self.config.merged_raw_file,
                output_file=self.config.cleaned_file
            )

            df_cleaned = cleaner.clean()

            self.stats['data_counts']['raw'] = len(
                cleaner.df_original) if hasattr(cleaner, 'df_original') else 0 # type: ignore
            self.stats['data_counts']['cleaned'] = len(df_cleaned)

            logger.info(
                f"\nCleaned data saved to: {self.config.cleaned_file}")
            logger.info(
                f"Records: {self.stats['data_counts']['raw']} - {self.stats['data_counts']['cleaned']}")

            self.stats['stages_completed'].append('stage_2_clean')
            logger.info("\nSTAGE 2 COMPLETED: Data Cleaning")
            return True

        except Exception as e:
            logger.error(f"STAGE 2 FAILED: {e}")
            self.stats['stages_failed'].append('stage_2_clean')
            import traceback
            traceback.print_exc()
            return False

    # ========================================================================
    # STAGE 3: FEATURE ENGINEERING
    # ========================================================================

    def stage_3_feature_engineering(self, visualize: bool = True):
        """
        Stage 3: Feature Engineering
        - Tạo các features mới từ dữ liệu đã clean
        - Tính toán scores: popularity, engagement, value, deal quality
        - Tạo categorical features
        """
        logger.info("=" * 80)
        logger.info("STAGE 3: FEATURE ENGINEERING")
        logger.info("=" * 80)

        try:
            # Check if cleaned file exists
            if not os.path.exists(self.config.cleaned_file):
                logger.error(
                    f"Cleaned file not found: {self.config.cleaned_file}")
                logger.error("Please run stage_2_clean_data first!")
                return False

            # Import transformation module
            sys.path.insert(0, os.path.join(
                self.config.base_dir, 'transformation'))
            from transformation.feature_engineering import FeatureEngineer
            import pandas as pd

            # Load cleaned data
            logger.info(
                f"Loading cleaned data from: {self.config.cleaned_file}")
            with open(self.config.cleaned_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)

            logger.info(f"Loaded {len(df)} records")

            # Feature engineering
            logger.info("\nEngineering features...")
            engineer = FeatureEngineer(df)
            df_engineered = engineer.engineer_features(visualize=visualize)

            # Save engineered features
            logger.info(
                f"\nSaving engineered features to: {self.config.engineered_file}")
            df_engineered.to_json(
                self.config.engineered_file,
                orient='records',
                force_ascii=False,
                indent=2
            )

            self.stats['data_counts']['engineered'] = len(df_engineered)

            logger.info(f"Engineered features saved")
            logger.info(
                f"Features created: {len(df_engineered.columns)} columns")

            self.stats['stages_completed'].append(
                'stage_3_feature_engineering')
            logger.info("\nSTAGE 3 COMPLETED: Feature Engineering")
            return True

        except Exception as e:
            logger.error(f"STAGE 3 FAILED: {e}")
            self.stats['stages_failed'].append('stage_3_feature_engineering')
            import traceback
            traceback.print_exc()
            return False

    # ========================================================================
    # STAGE 4: LABELING
    # ========================================================================

    def stage_4_labeling(self):
        """
        Stage 4: Product Segmentation Labeling
        - Gán nhãn cho sản phẩm: Hot Trend, Best Seller, Best Deal, Normal
        - Sử dụng hybrid approach: rule-based + ML model
        """
        logger.info("=" * 80)
        logger.info("STAGE 4: PRODUCT LABELING")
        logger.info("=" * 80)

        try:
            # Check if engineered file exists
            if not os.path.exists(self.config.engineered_file):
                logger.error(
                    f"Engineered file not found: {self.config.engineered_file}")
                logger.error("Please run stage_3_feature_engineering first!")
                return False

            # Import labeling module
            sys.path.insert(0, os.path.join(
                self.config.base_dir, 'transformation'))
            from transformation.labeling import create_labeling

            # Create labels
            logger.info(
                f"Loading engineered features from: {self.config.engineered_file}")
            logger.info(f"  Creating labels with:")
            logger.info(f"   - Use Model: {self.config.use_model}")
            logger.info(f"   - Model Type: {self.config.model_type}")
            logger.info(
                f"   - Probability Threshold: {self.config.prob_threshold}")
            logger.info(
                f"   - Min Seeds per Class: {self.config.min_seed_per_class}")

            df_labeled, stats = create_labeling(
                input_file=self.config.engineered_file,
                output_file=self.config.labeled_file,
                use_model=self.config.use_model,
                prob_threshold=self.config.prob_threshold,
                min_seed_per_class=self.config.min_seed_per_class,
                model_type=self.config.model_type
            )
            label_counts = df_labeled['label'].value_counts().to_dict() # type: ignore
            self.stats['data_counts']['labeled'] = len(df_labeled) # type: ignore
            self.stats['label_distribution'] = label_counts

            logger.info(
                f"\nLabeled data saved to: {self.config.labeled_file}")
            logger.info(f"Label distribution:")
            # for label, count in stats.items():
            #     logger.info(f"   - {label}: {count}")

            self.stats['stages_completed'].append('stage_4_labeling')
            logger.info("\nSTAGE 4 COMPLETED: Product Labeling")
            return True

        except Exception as e:
            logger.error(f"STAGE 4 FAILED: {e}")
            self.stats['stages_failed'].append('stage_4_labeling')
            import traceback
            traceback.print_exc()
            return False

    # ========================================================================
    # STAGE 5: ENCODING
    # ========================================================================

    def stage_5_encoding(self):
        """
        Stage 5: Data Encoding & Train/Test Split
        - One-hot encoding cho categorical features
        - Standardization cho numerical features
        - Label encoding cho target
        - Split train/test sets
        """
        logger.info("=" * 80)
        logger.info("STAGE 5: DATA ENCODING & TRAIN/TEST SPLIT")
        logger.info("=" * 80)

        try:
            # Check if labeled file exists
            if not os.path.exists(self.config.labeled_file):
                logger.error(
                    f"Labeled file not found: {self.config.labeled_file}")
                logger.error("Please run stage_4_labeling first!")
                return False

            # Import encoding module
            sys.path.insert(0, os.path.join(
                self.config.base_dir, 'transformation'))
            from transformation.encoding import create_encoding

            # Create encoding
            logger.info(
                f"Loading labeled data from: {self.config.labeled_file}")
            logger.info(
                f"Encoding data with test_size={self.config.test_size}")

            X_train, X_test, y_train, y_test, info = create_encoding(
                input_file=self.config.labeled_file,
                output_file=self.config.encoded_file,
                encoder_dir=self.config.encoder_dir,
                test_size=self.config.test_size
            )

            self.stats['data_counts']['train'] = len(X_train)
            self.stats['data_counts']['test'] = len(X_test) # type: ignore
            self.stats['encoding_info'] = info

            logger.info(
                f"\nEncoded data saved to: {self.config.encoded_file}")
            logger.info(f"Encoders saved to: {self.config.encoder_dir}")
            logger.info(f"Train/Test split:")
            logger.info(f"   - Train: {len(X_train)} samples")
            logger.info(f"   - Test: {len(X_test)} samples") # type: ignore
            logger.info(
                f"   - Features: {X_train.shape[1] if hasattr(X_train, 'shape') else 'N/A'}")

            self.stats['stages_completed'].append('stage_5_encoding')
            logger.info("\nSTAGE 5 COMPLETED: Data Encoding")
            return True

        except Exception as e:
            logger.error(f"STAGE 5 FAILED: {e}")
            self.stats['stages_failed'].append('stage_5_encoding')
            import traceback
            traceback.print_exc()
            return False

    # ========================================================================
    # PIPELINE EXECUTION
    # ========================================================================

    def run_full_pipeline(self, skip_crawl: bool = False, platforms: list[str] | None = None):
        """
        Chạy toàn bộ pipeline từ đầu đến cuối

        Parameters:
            skip_crawl: Bỏ qua stage crawl (nếu đã có dữ liệu)
            platforms: List platforms cần crawl (nếu không skip)
        """
        logger.info("\n" + "=" * 80)
        logger.info("STARTING DATA MINING PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Start time: {self.stats['start_time']}")
        logger.info("=" * 80 + "\n")

        # Stage 1: Crawl (optional)
        if not skip_crawl:
            if not self.stage_1_crawl_data(platforms):
                logger.error("Pipeline stopped due to crawl failure")
                return False
        else:
            logger.info("⏭️  Skipping Stage 1: Data Crawling")

        # Stage 2: Clean
        if not self.stage_2_clean_data():
            logger.error("Pipeline stopped due to cleaning failure")
            return False

        # Stage 3: Feature Engineering
        if not self.stage_3_feature_engineering():
            logger.error("Pipeline stopped due to feature engineering failure")
            return False

        # Stage 4: Labeling
        if not self.stage_4_labeling():
            logger.error("Pipeline stopped due to labeling failure")
            return False

        # Stage 5: Encoding
        if not self.stage_5_encoding():
            logger.error("Pipeline stopped due to encoding failure")
            return False

        # Pipeline completed
        self.stats['end_time'] = datetime.now()
        self.stats['duration'] = (
            self.stats['end_time'] - self.stats['start_time']).total_seconds()

        self._print_final_summary()

        return True

    def run_partial_pipeline(self, start_stage: int = 2, end_stage: int = 5):
        """
        Chạy một phần pipeline

        Parameters:
            start_stage: Stage bắt đầu (1-5)
            end_stage: Stage kết thúc (1-5)
        """
        logger.info("\n" + "=" * 80)
        logger.info(
            f"RUNNING PARTIAL PIPELINE (Stage {start_stage} - {end_stage})")
        logger.info("=" * 80)
        logger.info(f"Start time: {self.stats['start_time']}")
        logger.info("=" * 80 + "\n")

        stages = {
            1: self.stage_1_crawl_data,
            2: self.stage_2_clean_data,
            3: self.stage_3_feature_engineering,
            4: self.stage_4_labeling,
            5: self.stage_5_encoding
        }

        for stage_num in range(start_stage, end_stage + 1):
            if stage_num in stages:
                if stage_num == 1:
                    if not stages[stage_num]():
                        logger.error(f"Pipeline stopped at stage {stage_num}")
                        return False
                else:
                    if not stages[stage_num]():
                        logger.error(f"Pipeline stopped at stage {stage_num}")
                        return False

        self.stats['end_time'] = datetime.now()
        self.stats['duration'] = (
            self.stats['end_time'] - self.stats['start_time']).total_seconds()

        self._print_final_summary()

        return True

    def _print_final_summary(self):
        """In tóm tắt kết quả pipeline"""
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Start time: {self.stats['start_time']}")
        logger.info(f"End time: {self.stats['end_time']}")
        logger.info(f"Duration: {self.stats['duration']:.2f} seconds")
        logger.info("\nData Flow:")
        for key, value in self.stats['data_counts'].items():
            logger.info(f"   - {key.capitalize()}: {value:,} records")

        if 'label_distribution' in self.stats:
            logger.info("\nLabel Distribution:")
            for label, info in self.stats['label_distribution'].items():
                logger.info(f"   - {label}:")

        logger.info(
            f"\nStages Completed: {len(self.stats['stages_completed'])}")
        for stage in self.stats['stages_completed']:
            logger.info(f" {stage}")

        if self.stats['stages_failed']:
            logger.info(
                f"\nStages Failed: {len(self.stats['stages_failed'])}")
            for stage in self.stats['stages_failed']:
                logger.info(f"   ✗ {stage}")

        logger.info("\nOutput Files:")
        logger.info(f"   - Cleaned: {self.config.cleaned_file}")
        logger.info(f"   - Engineered: {self.config.engineered_file}")
        logger.info(f"   - Labeled: {self.config.labeled_file}")
        logger.info(f"   - Encoded: {self.config.encoded_file}")
        logger.info(f"   - Encoders: {self.config.encoder_dir}")
        logger.info("=" * 80 + "\n")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function với command-line interface"""
    parser = argparse.ArgumentParser(
        description='Data Mining Pipeline - E-commerce Product Segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Chạy toàn bộ pipeline (bao gồm crawl)
  python main.py --full
  
  # Chạy toàn bộ pipeline, bỏ qua crawl
  python main.py --full --skip-crawl
  
  # Chỉ crawl từ Shopee và Tiki
  python main.py --crawl --platforms shopee tiki
  
  # Chạy từ stage 2 đến 5 (clean → encoding)
  python main.py --partial --start 2 --end 5
  
  # Chỉ chạy stage clean
  python main.py --clean
  
  # Chỉ chạy stage feature engineering
  python main.py --feature
  
  # Chỉ chạy stage labeling
  python main.py --label
  
  # Chỉ chạy stage encoding
  python main.py --encode
        """
    )

    # Pipeline modes
    parser.add_argument('--full', action='store_true',
                        help='Chạy toàn bộ pipeline')
    parser.add_argument('--partial', action='store_true',
                        help='Chạy một phần pipeline')
    parser.add_argument('--start', type=int, default=2,
                        help='Stage bắt đầu (1-5) cho partial mode')
    parser.add_argument('--end', type=int, default=5,
                        help='Stage kết thúc (1-5) cho partial mode')

    # Individual stages
    parser.add_argument('--crawl', action='store_true',
                        help='Chỉ chạy stage 1: Crawl data')
    parser.add_argument('--clean', action='store_true',
                        help='Chỉ chạy stage 2: Clean data')
    parser.add_argument('--feature', action='store_true',
                        help='Chỉ chạy stage 3: Feature engineering')
    parser.add_argument('--label', action='store_true',
                        help='Chỉ chạy stage 4: Labeling')
    parser.add_argument('--encode', action='store_true',
                        help='Chỉ chạy stage 5: Encoding')

    # Options
    parser.add_argument('--skip-crawl', action='store_true',
                        help='Bỏ qua stage crawl trong full mode')
    parser.add_argument('--platforms', nargs='+',
                        choices=['shopee', 'tiki', 'lazada'],
                        help='Platforms cần crawl')
    parser.add_argument('--max-pages', type=int, default=5,
                        help='Số trang tối đa cho mỗi category')
    parser.add_argument('--no-visualize', action='store_true',
                        help='Không tạo visualizations trong feature engineering')

    args = parser.parse_args()

    # Create config
    config = PipelineConfig()
    config.max_pages = args.max_pages

    # Create pipeline
    pipeline = DataMiningPipeline(config)

    # Execute based on arguments
    try:
        if args.full:
            # Full pipeline
            success = pipeline.run_full_pipeline(
                skip_crawl=args.skip_crawl,
                platforms=args.platforms
            )
        elif args.partial:
            # Partial pipeline
            success = pipeline.run_partial_pipeline(
                start_stage=args.start,
                end_stage=args.end
            )
        elif args.crawl:
            # Only crawl
            success = pipeline.stage_1_crawl_data(platforms=args.platforms)
        elif args.clean:
            # Only clean
            success = pipeline.stage_2_clean_data()
        elif args.feature:
            # Only feature engineering
            success = pipeline.stage_3_feature_engineering(
                visualize=not args.no_visualize
            )
        elif args.label:
            # Only labeling
            success = pipeline.stage_4_labeling()
        elif args.encode:
            # Only encoding
            success = pipeline.stage_5_encoding()
        else:
            # No arguments - show help
            parser.print_help()
            return

        if success:
            logger.info("\nPipeline execution completed successfully!")
            sys.exit(0)
        else:
            logger.error("\nPipeline execution failed!")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nPipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
