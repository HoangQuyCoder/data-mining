import pandas as pd
import numpy as np
import json
import os
import argparse
from typing import Any, Dict, Tuple, Optional


class ProductLabeler:
    """
    Labeling Module cho phân vùng sản phẩm (Product Segmentation)

    Target Categories (4 danh mục):
    1. 🔥 Hot Trend: Sản phẩm đang viral, trending, tăng trưởng nhanh
    2. 🏆 Best Seller: Sản phẩm bán chạy nhất, ổn định
    3. 💰 Best Deal: Sản phẩm có ưu đãi tốt, giá trị cao
    4. 📦 Normal: Sản phẩm bình thường
    """

    def __init__(self, df: pd.DataFrame):
        """
        Khởi tạo Product Labeler

        Parameters:
        - df: DataFrame đã được feature engineering
        """
        self.df = df.copy()
        self.label_stats = {}
        self.thresholds = {}

        # Validate required features
        self._validate_features()

    def _validate_features(self):
        """Kiểm tra các features bắt buộc"""
        required_features = {
            'hot_trend': ['trend_momentum', 'engagement_score', 'product_age'],
            'best_seller': ['popularity_score', 'sales_velocity_normalized', 'quality_tier'],
            'best_deal': ['deal_quality_score', 'value_score', 'discount_intensity']
        }

        missing = {}
        for category, features in required_features.items():
            missing_features = [
                f for f in features if f not in self.df.columns]
            if missing_features:
                missing[category] = missing_features

        if missing:
            print("⚠️  WARNING: Thiếu một số features quan trọng:")
            for category, features in missing.items():
                print(f"   {category}: {features}")
            print("   → Labeling có thể kém chính xác hơn!")

    def create_labels(self, use_model: bool = True) -> pd.DataFrame:
        """
        Main method để tạo labels (hybrid approach)

        Parameters:
        - use_model: Có sử dụng ML model hay không (True = hybrid, False = pure rule-based)

        Returns:
        - DataFrame với labels
        """
        return self.create_labels_with_params(use_model=use_model)

    def create_labels_with_params(
        self,
        prob_threshold: float = 0.70,
        min_seed_per_class: int = 50,
        model_type: str = 'random_forest',
        use_model: bool = True
    ) -> pd.DataFrame:
        """
        Hybrid labeling với customizable parameters

        Pipeline:
        1. Tính thresholds từ data
        2. Gán seed labels (high precision rules)
        3. Train model từ seeds (nếu use_model=True)
        4. Predict cho phần còn lại
        5. Fallback sang rule-based nếu model fail
        """

        print("🏷️  BẮT ĐẦU LABELING (HYBRID: RULE SEEDS + AI MODEL)")
        print("=" * 80)

        # Step 1: Calculate thresholds
        print("\n✓ Bước 1: Tính toán thresholds...")
        self._calculate_thresholds()

        # Step 2: Assign seed labels
        print("✓ Bước 2: Gán seed labels (high precision rules)...")
        self._assign_seed_labels()

        # Step 3: Train & predict with model (if enabled)
        success = False
        if use_model:
            print("✓ Bước 3: Train & Predict với ML model...")
            try:
                success = self._train_and_predict_from_seeds(
                    prob_threshold=prob_threshold,
                    min_seed_per_class=min_seed_per_class,
                    model_type=model_type
                )
            except ImportError as e:
                print(f"\n⚠️  {e}")
                success = False

        # Step 4: Fallback to full rule-based if needed
        if not success:
            print("\n⚠️  Fallback sang rule-based labeling đầy đủ...")
            self._assign_labels_rule_based()
            self.df['label_source'] = 'rule_full'

        # Summary
        print("\n" + "=" * 80)
        print("✅ HOÀN THÀNH LABELING")
        print(f"   Total products: {len(self.df):,}")

        print("\n   Distribution:")
        label_counts = self.df['label'].value_counts()
        self.label_stats = label_counts.to_dict()
        for label, count in label_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"     - {label}: {count:,} ({percentage:.1f}%)")

        if 'label_source' in self.df.columns:
            print("\n   Label source:")
            for src, cnt in self.df['label_source'].value_counts().items():
                print(f"     - {src}: {cnt:,} ({cnt/len(self.df)*100:.1f}%)")

        return self.df

    def _calculate_thresholds(self):
        """
        Tính thresholds dựa trên percentiles của features 
        """

        # Hot Trend thresholds - nới lỏng
        if 'trend_momentum' in self.df.columns:
            trend_p70 = self.df['trend_momentum'].quantile(0.70)
            trend_p75 = self.df['trend_momentum'].quantile(0.75)
            trend_p80 = self.df['trend_momentum'].quantile(0.80)
        else:
            trend_p70 = trend_p75 = trend_p80 = 0

        if 'engagement_score' in self.df.columns:
            engagement_p65 = self.df['engagement_score'].quantile(0.65)
            engagement_p70 = self.df['engagement_score'].quantile(0.70)
        else:
            engagement_p65 = engagement_p70 = 0

        # Best Seller thresholds - nới lỏng
        if 'popularity_score' in self.df.columns:
            popularity_p55 = self.df['popularity_score'].quantile(0.55)
            popularity_p60 = self.df['popularity_score'].quantile(0.60)
            popularity_p65 = self.df['popularity_score'].quantile(0.65)
            popularity_p70 = self.df['popularity_score'].quantile(0.70)
        else:
            popularity_p55 = popularity_p60 = popularity_p65 = popularity_p70 = 0

        if 'sales_velocity_normalized' in self.df.columns:
            sales_vel_p65 = self.df['sales_velocity_normalized'].quantile(0.65)
            sales_vel_p70 = self.df['sales_velocity_normalized'].quantile(0.70)
        else:
            sales_vel_p65 = sales_vel_p70 = 0

        # Best Deal thresholds - nới lỏng nhẹ
        if 'deal_quality_score' in self.df.columns:
            deal_p70 = self.df['deal_quality_score'].quantile(0.70)
            deal_p75 = self.df['deal_quality_score'].quantile(0.75)
            deal_p80 = self.df['deal_quality_score'].quantile(0.80)
            deal_p85 = self.df['deal_quality_score'].quantile(0.85)
        else:
            deal_p70 = deal_p75 = deal_p80 = deal_p85 = 0

        if 'value_score' in self.df.columns:
            value_p65 = self.df['value_score'].quantile(0.65)
            value_p70 = self.df['value_score'].quantile(0.70)
            value_p75 = self.df['value_score'].quantile(0.75)
        else:
            value_p65 = value_p70 = value_p75 = 0

        self.thresholds = {
            'hot_trend': {
                'trend_momentum_p70': trend_p70,
                'trend_momentum_p75': trend_p75,
                'trend_momentum_p80': trend_p80,
                'engagement_score_p65': engagement_p65,
                'engagement_score_p70': engagement_p70,
                'new_age_categories': ['Brand New', 'New', 'Recent']
            },
            'best_seller': {
                'popularity_score_p55': popularity_p55,
                'popularity_score_p60': popularity_p60,
                'popularity_score_p65': popularity_p65,
                'popularity_score_p70': popularity_p70,
                'sales_velocity_p65': sales_vel_p65,
                'sales_velocity_p70': sales_vel_p70,
                'high_quality_tiers': ['Premium', 'High', 'Good', 'Average']
            },
            'best_deal': {
                'deal_quality_score_p70': deal_p70,
                'deal_quality_score_p75': deal_p75,
                'deal_quality_score_p80': deal_p80,
                'deal_quality_score_p85': deal_p85,
                'value_score_p65': value_p65,
                'value_score_p70': value_p70,
                'value_score_p75': value_p75,
                'strong_discount_levels': ['Aggressive', 'Heavy', 'Moderate']
            }
        }

        # In thresholds mới
        print(f"\n   🔥 Hot Trend (nới lỏng):")
        print(f"      trend_momentum >= {trend_p75:.2f} (P75)")
        print(f"      engagement_score >= {engagement_p65:.2f} (P65)")

        print(f"\n   🏆 Best Seller (nới lỏng):")
        print(f"      popularity_score >= {popularity_p70:.2f} (P70)")
        print(f"      sales_velocity >= {sales_vel_p65:.2f} (P65)")

        print(f"\n   💰 Best Deal (nới lỏng nhẹ):")
        print(f"      deal_quality_score >= {deal_p75:.2f} (P75)")
        print(f"      value_score >= {value_p65:.2f} (P65)")
        print(f"      Chấp nhận Aggressive/Heavy/Moderate")
        print(f"      ⚠️  HARD GATE: vẫn chỉ Aggressive/Heavy/Moderate discount")
        print(f"      ⚠️  Best Deal vẫn capped ~30% tổng")

    def _assign_seed_labels(self):
        """
        Gán seed labels với high precision rules - đã nới lỏng ngưỡng theo phiên bản mới
        """
        def seed_rule(row):
            # Extract features
            trend_momentum = row.get('trend_momentum', np.nan)
            engagement_score = row.get('engagement_score', np.nan)
            product_age = row.get('product_age', None)

            popularity_score = row.get('popularity_score', np.nan)
            sales_velocity = row.get('sales_velocity_normalized', np.nan)
            quality_tier = row.get('quality_tier', None)

            deal_quality_score = row.get('deal_quality_score', np.nan)
            value_score = row.get('value_score', np.nan)
            discount_intensity = row.get('discount_intensity', None)

            # Get thresholds
            ht = self.thresholds['hot_trend']
            bs = self.thresholds['best_seller']
            bd = self.thresholds['best_deal']

            # 🔥 SEED FOR HOT TREND
            if pd.notna(trend_momentum):
                if trend_momentum >= ht['trend_momentum_p80']:
                    return 'Hot Trend', f"seed: momentum>={ht['trend_momentum_p80']:.1f} (P80)"

                if (trend_momentum >= ht['trend_momentum_p75'] and
                        product_age in ['Brand New', 'New']):
                    return 'Hot Trend', f"seed: momentum>={ht['trend_momentum_p75']:.1f} & brand_new/new"

            # 🏆 SEED FOR BEST SELLER
            if pd.notna(popularity_score):
                # Tier cao: trước P80 + Premium/High → giờ P70 + Premium/High
                if (popularity_score >= bs['popularity_score_p70'] and
                        quality_tier in ['Premium', 'High']):
                    return 'Best Seller', f"seed: popularity>={bs['popularity_score_p70']:.1f} & high_quality"

                # Tier trung: trước P70 + quality ok → giờ P65 + quality ok
                if (popularity_score >= bs['popularity_score_p65'] and
                    sales_velocity >= bs['sales_velocity_p65'] and
                        quality_tier in bs['high_quality_tiers']):
                    return 'Best Seller', f"seed: popularity>={bs['popularity_score_p65']:.1f} & quality ok"

                # Tier thấp: trước P60 → giờ P55
                if popularity_score >= bs['popularity_score_p55']:
                    return 'Best Seller', f"seed: popularity>={bs['popularity_score_p55']:.1f}"

            # 💰 SEED FOR BEST DEAL (strict + discount gate)
            if pd.notna(deal_quality_score):
                # Trước P90 → giờ P85
                if deal_quality_score >= bd['deal_quality_score_p85']:
                    return 'Best Deal', f"seed: deal_quality>={bd['deal_quality_score_p85']:.1f} (P85)"

                # Trước P85 + discount mạnh → giờ P80 + discount ok
                if (deal_quality_score >= bd['deal_quality_score_p80'] and
                        discount_intensity in ['Aggressive', 'Heavy', 'Moderate']):
                    return 'Best Deal', f"seed: deal_quality>={bd['deal_quality_score_p80']:.1f} & aggressive/heavy/moderate"

                # Trước P80 + value P75 → giờ P75 + value P70
                if (deal_quality_score >= bd['deal_quality_score_p75'] and
                        value_score >= bd['value_score_p70']):
                    return 'Best Deal', f"seed: deal_quality>={bd['deal_quality_score_p75']:.1f} & value>={bd['value_score_p70']:.1f}"

            # No seed
            return np.nan, np.nan

        seed = self.df.apply(seed_rule, axis=1, result_type='expand')
        seed.columns = ['seed_label', 'seed_reason']
        self.df[['seed_label', 'seed_reason']] = seed
        self.df['label_source'] = np.where(
            self.df['seed_label'].notna(), 'rule_seed', 'unlabeled')

        # In thống kê seed
        counts = self.df['seed_label'].value_counts(dropna=True)
        print("\n   Seed distribution (sau khi nới lỏng):")
        if len(counts) == 0:
            print("     (no seed labels)")
        else:
            total_seeds = counts.sum()
            for label, count in counts.items():
                pct = count / len(self.df) * 100
                print(f"     - {label}: {count:,} ({pct:.1f}% of total)")
            print(
                f"     → Total seeds: {total_seeds:,} ({total_seeds/len(self.df)*100:.1f}%)")

    def _assign_labels_rule_based(self):
        """
        Gán labels bằng rule-based logic đầy đủ 
        Priority order giữ nguyên: Hot Trend > Best Seller > Best Deal > Normal
        """

        def categorize_product(row):
            # Extract features
            trend_momentum = row.get('trend_momentum', 0)
            engagement_score = row.get('engagement_score', 0)
            product_age = row.get('product_age', '')

            popularity_score = row.get('popularity_score', 0)
            sales_velocity = row.get('sales_velocity_normalized', 0)
            quality_tier = row.get('quality_tier', '')

            deal_quality_score = row.get('deal_quality_score', 0)
            value_score = row.get('value_score', 0)
            discount_intensity = row.get('discount_intensity', '')

            # Get thresholds
            ht = self.thresholds['hot_trend']
            bs = self.thresholds['best_seller']
            bd = self.thresholds['best_deal']

            # ========================================
            # Priority 1: 🔥 HOT TREND
            # ========================================
            # Tier 1: Very hot – trước P85 → giờ P80
            if trend_momentum >= ht['trend_momentum_p80']:
                return 'Hot Trend'

            # Tier 2: Hot with new age – trước P80 → giờ P75
            if (trend_momentum >= ht['trend_momentum_p75'] and
                    product_age in ht['new_age_categories']):
                return 'Hot Trend'

            # Tier 3: High engagement + very new
            if (engagement_score >= ht['engagement_score_p70'] and
                    product_age in ['Brand New', 'New']):
                return 'Hot Trend'

            # Tier 4: Very high engagement (giữ buffer)
            if engagement_score >= ht['engagement_score_p70'] + 4:  # giảm buffer nhẹ
                return 'Hot Trend'

            # ========================================
            # Priority 2: 🏆 BEST SELLER
            # ========================================
            # Tier 1: Top seller – trước P80 + Premium/High → giờ P70 + Premium/High
            if (popularity_score >= bs['popularity_score_p70'] and
                    quality_tier in ['Premium', 'High']):
                return 'Best Seller'

            # Tier 2: Strong seller – trước P70 → giờ P65
            if (popularity_score >= bs['popularity_score_p65'] and
                    quality_tier in bs['high_quality_tiers']):
                return 'Best Seller'

            # Tier 3: Solid seller – trước P60 → giờ P55
            if popularity_score >= bs['popularity_score_p55']:
                return 'Best Seller'

            # ========================================
            # Priority 3: 💰 BEST DEAL
            # ========================================
            # Hard gate discount – vẫn giữ Aggressive/Heavy/Moderate
            if discount_intensity not in ['Aggressive', 'Heavy', 'Moderate']:
                return 'Normal'

            # Tier 1: Excellent deal – trước P90 → giờ P85
            if deal_quality_score >= bd['deal_quality_score_p85']:
                return 'Best Deal'

            # Tier 2: Great deal – trước P85 + value P80 → giờ P80 + value P75
            if (deal_quality_score >= bd['deal_quality_score_p80'] and
                    value_score >= bd['value_score_p75']):
                return 'Best Deal'

            # Tier 3: Good deal – trước P80 + value P75 → giờ P75 + value P70
            if (deal_quality_score >= bd['deal_quality_score_p75'] and
                value_score >= bd['value_score_p70'] and
                    discount_intensity in ['Aggressive', 'Heavy', 'Moderate']):
                return 'Best Deal'

            # Tier 4: Decent deal – trước P75 + value P70 → giờ P70 + value P65
            if (deal_quality_score >= bd['deal_quality_score_p70'] and
                value_score >= bd['value_score_p65'] and
                    discount_intensity in ['Aggressive', 'Heavy', 'Moderate']):
                return 'Best Deal'

            # ========================================
            # Default: 📦 NORMAL
            # ========================================
            return 'Normal'

        self.df['label'] = self.df.apply(categorize_product, axis=1)

        # In phân bố
        print(f"\n   Distribution (rule-based sau nới lỏng):")
        label_counts = self.df['label'].value_counts()
        self.label_stats = label_counts.to_dict()

        for label, count in label_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"     - {label}: {count:,} ({percentage:.1f}%)")

    def _train_and_predict_from_seeds(
        self,
        prob_threshold: float,
        min_seed_per_class: int,
        model_type: str
    ) -> bool:
        """
        Train ML model từ seed labels và predict cho unlabeled samples

        Returns:
            bool: True nếu thành công, False nếu fail
        """

        try:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, classification_report
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
        except Exception:
            raise ImportError(
                "Hybrid mode cần scikit-learn. Cài đặt: pip install scikit-learn\n"
                "Hoặc dùng --no-model để chạy pure rule-based."
            )

        # Check if we have seeds
        if 'seed_label' not in self.df.columns:
            print("   ⚠️  Không có seed labels")
            return False

        seed_df = self.df[self.df['seed_label'].notna()].copy()
        if len(seed_df) == 0:
            print("   ⚠️  Không có seed samples")
            return False

        # Check minimum seeds per class
        counts = seed_df['seed_label'].value_counts()
        valid_classes = [c for c, n in counts.items() if n >=
                         min_seed_per_class]

        if len(valid_classes) < 2:
            print(
                f"   ⚠️  Không đủ seeds (cần ≥{min_seed_per_class} mỗi class, chỉ có {len(valid_classes)} classes đủ)")
            return False

        # Filter to valid classes only
        seed_df = seed_df[seed_df['seed_label'].isin(valid_classes)].copy()

        # Feature selection - prioritize v2.0 features
        numeric_features = [
            # Core scores
            'popularity_score', 'engagement_score', 'trend_momentum',
            'value_score', 'deal_quality_score',
            # Velocities
            'sales_velocity', 'sales_velocity_normalized',
            'review_velocity', 'review_velocity_normalized',
            # Context
            'discount_score', 'category_popularity_rank', 'category_price_percentile',
            # Raw
            'quantity_sold', 'rating_average', 'num_reviews',
            'current_price', 'discount_rate', 'absolute_saving',
            'days_active'
        ]

        categorical_features = [
            'popularity_category', 'discount_intensity', 'quality_tier',
            'price_segment', 'product_age'
        ]

        # Filter to existing columns
        numeric_cols = [c for c in numeric_features if c in self.df.columns]
        cat_cols = [c for c in categorical_features if c in self.df.columns]

        if len(numeric_cols) == 0:
            print("   ⚠️  Không có numeric features")
            return False

        print(
            f"   Features: {len(numeric_cols)} numeric + {len(cat_cols)} categorical")

        # Build feature matrix
        def build_X(df: pd.DataFrame) -> pd.DataFrame:
            # Numeric features
            X_num = df[numeric_cols].copy() if len(
                numeric_cols) else pd.DataFrame(index=df.index)
            X_num = X_num.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            # Categorical features
            X_cat = df[cat_cols].copy() if len(
                cat_cols) else pd.DataFrame(index=df.index)
            for c in X_cat.columns:
                X_cat[c] = X_cat[c].astype('object').fillna('Unknown')

            # One-hot encode categorical
            if len(X_cat.columns) > 0:
                X_cat = pd.get_dummies(X_cat, columns=list(
                    X_cat.columns), dummy_na=False)

            # Combine
            X = pd.concat([X_num, X_cat], axis=1)
            return X

        # Build training data
        X = build_X(seed_df)
        y = seed_df['seed_label'].astype(str)

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Select model
        model_type = (model_type or '').lower().strip()
        if model_type in ['logreg', 'logistic', 'logistic_regression']:
            model = LogisticRegression(
                max_iter=1000,
                multi_class='auto',
                random_state=42
            )
            model_name = 'LogisticRegression'
        else:
            model = RandomForestClassifier(
                n_estimators=400,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight='balanced_subsample',
                n_jobs=-1
            )
            model_name = 'RandomForest'

        # Train
        model.fit(X_train, y_train)

        # Validate
        val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, val_pred)

        print(f"\n   Model: {model_name}")
        print(f"   Seeds: {len(seed_df):,} samples")
        print(f"   Train: {len(X_train):,} | Val: {len(X_val):,}")
        print(f"   Val Accuracy: {val_acc:.3f}")

        # Print class-wise performance
        if len(valid_classes) <= 4:  # Only if not too many classes
            print("\n   Per-class validation:")
            report_dict = classification_report(
                y_val, val_pred, output_dict=True, zero_division=0)
            for cls in valid_classes:
                cls_str = str(cls)
                cls_report = report_dict.get(cls_str, {})  # type: ignore
                if isinstance(cls_report, dict):
                    precision = cls_report.get('precision', 0.0)
                    recall = cls_report.get('recall', 0.0)
                    print(f"     {cls}: P={precision:.2f} R={recall:.2f}")

        # Predict for unlabeled
        unlabeled_idx = self.df[self.df['seed_label'].isna()].index

        if len(unlabeled_idx) == 0:
            # All samples have seeds
            self.df['label'] = self.df['seed_label']
            self.df['label_source'] = 'rule_seed'
            print("\n   ✓ All samples already have seeds")
            return True

        # Build X for all data
        X_all = build_X(self.df)
        # Align columns with training data
        X_all = X_all.reindex(columns=X.columns, fill_value=0.0)

        # Predict with probability threshold
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X_all.loc[unlabeled_idx])
            classes = list(model.classes_)
            best_idx = prob.argmax(axis=1)
            best_prob = prob.max(axis=1)
            pred_labels = [classes[i] for i in best_idx]

            # Apply probability threshold (raised from 0.55 → 0.70 for stricter gating)
            pred_labels = np.where(
                best_prob >= prob_threshold, pred_labels, 'Normal')

            # ================================================================
            # POST-PREDICTION CLASS CAP (bias guardrail)
            # Best Deal capped at 30% of total dataset.
            # Sản phẩm vượt cap (sorted by confidence asc) → reassign Normal.
            # ================================================================
            total_records = len(self.df)
            max_best_deal_ratio = 0.30  # max 30% of total
            max_best_deal_count = int(total_records * max_best_deal_ratio)

            # Count Best Deal from seeds
            seed_best_deal = (self.df['seed_label'] == 'Best Deal').sum()
            remaining_bd_quota = max(0, max_best_deal_count - seed_best_deal)

            # Among predictions, find all Best Deal predictions with their confidence
            bd_mask = np.array([p == 'Best Deal' for p in pred_labels])
            bd_count = bd_mask.sum()

            if bd_count > remaining_bd_quota:
                # Sort by confidence ascending → keep only top-confidence ones
                bd_indices_in_pred = np.where(bd_mask)[0]
                bd_confidences = best_prob[bd_indices_in_pred]
                # argsort ascending → first elements are lowest confidence
                sorted_order = np.argsort(bd_confidences)
                # Reassign excess (lowest confidence) to Normal
                excess_count = bd_count - remaining_bd_quota
                excess_positions = bd_indices_in_pred[sorted_order[:excess_count]]
                pred_labels_arr = np.array(pred_labels, dtype=object)
                pred_labels_arr[excess_positions] = 'Normal'
                pred_labels = list(pred_labels_arr)
                print(
                    f"\n   ⚠️  Best Deal cap applied: {bd_count} → {remaining_bd_quota} (reassigned {excess_count} → Normal)")

            # Stats on predictions
            high_conf_count = (best_prob >= prob_threshold).sum()
            print(f"\n   Unlabeled predictions:")
            print(f"     Total: {len(unlabeled_idx):,}")
            print(
                f"     High confidence (≥{prob_threshold}): {high_conf_count:,} ({high_conf_count/len(unlabeled_idx)*100:.1f}%)")
            print(
                f"     Low confidence → Normal: {len(unlabeled_idx) - high_conf_count:,}")
        else:
            pred_labels = model.predict(X_all.loc[unlabeled_idx])

        # Merge predictions
        self.df['label'] = self.df['seed_label']
        self.df.loc[unlabeled_idx, 'label'] = pred_labels
        self.df['label'] = self.df['label'].fillna('Normal')

        # Mark label sources
        self.df.loc[unlabeled_idx, 'label_source'] = 'model'
        self.df.loc[self.df['seed_label'].notna(),
                    'label_source'] = 'rule_seed'

        return True

    def get_label_statistics(self) -> Dict:
        """Lấy thống kê labels"""
        return self.label_stats

    def get_thresholds(self) -> Dict:
        """Lấy thresholds đã tính"""
        return self.thresholds

    def get_dataframe(self) -> pd.DataFrame:
        """Trả về DataFrame với labels"""
        return self.df


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def create_labeling(
    input_file: str,
    output_file: Optional[str] = None,
    use_model: bool = True,
    prob_threshold: float = 0.70,
    min_seed_per_class: int = 50,
    model_type: str = 'random_forest'
) -> pd.DataFrame:
    """
    Main function cho labeling

    Parameters:
    - input_file: Path to engineered features JSON
    - output_file: Path to output labeled data JSON (default: auto)
    - use_model: Use ML model or pure rule-based
    - prob_threshold: Probability threshold for model predictions
    - min_seed_per_class: Minimum seeds per class to train model
    - model_type: 'random_forest' or 'logistic_regression'

    Returns:
    - DataFrame with labels
    """

    print("\n" + "=" * 80)
    print("🎯 PRODUCT LABELING")
    print("   🔥 Hot Trend | 🏆 Best Seller | 💰 Best Deal | 📦 Normal")
    print("=" * 80)

    # Load data
    print("\n📂 Loading data...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    print(f"✓ Loaded {len(df):,} records")
    print(f"✓ Columns: {len(df.columns)}")

    # Labeling
    labeler = ProductLabeler(df)
    df_labeled = labeler.create_labels_with_params(
        prob_threshold=prob_threshold,
        min_seed_per_class=min_seed_per_class,
        model_type=model_type,
        use_model=use_model
    )

    # Save
    if output_file is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(base, 'data/transformation')
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'labeled_data.json')

    print(f"\n💾 Saving to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(df_labeled.to_dict('records'),
                  f, ensure_ascii=False, indent=2)

    print(f"✅ Saved {len(df_labeled):,} records")

    # Detailed statistics
    print("\n" + "=" * 80)
    print("📊 DETAILED STATISTICS")
    print("=" * 80)

    for label in ['Hot Trend', 'Best Seller', 'Best Deal', 'Normal']:
        label_data = df_labeled[df_labeled['label'] == label]
        count = len(label_data)

        if count == 0:
            continue

        percentage = count / len(df_labeled) * 100
        print(f"\n{label}:")
        print(f"  Count: {count:,} ({percentage:.1f}%)")

        if label == 'Hot Trend' and 'trend_momentum' in label_data.columns:
            print(
                f"  Avg trend_momentum: {label_data['trend_momentum'].mean():.2f}")
            print(
                f"  Avg engagement_score: {label_data['engagement_score'].mean():.2f}")
            if 'product_age' in label_data.columns:
                age_dist = label_data['product_age'].value_counts().head(3)
                print(f"  Top ages: {age_dist.to_dict()}")

        elif label == 'Best Seller' and 'popularity_score' in label_data.columns:
            print(
                f"  Avg popularity_score: {label_data['popularity_score'].mean():.2f}")
            print(
                f"  Avg quantity_sold: {label_data['quantity_sold'].mean():,.0f}")
            if 'quality_tier' in label_data.columns:
                quality_dist = label_data['quality_tier'].value_counts(
                ).to_dict()
                print(f"  Quality tiers: {quality_dist}")

        elif label == 'Best Deal' and 'deal_quality_score' in label_data.columns:
            print(
                f"  Avg deal_quality_score: {label_data['deal_quality_score'].mean():.2f}")
            print(f"  Avg value_score: {label_data['value_score'].mean():.2f}")
            if 'discount_intensity' in label_data.columns:
                discount_dist = label_data['discount_intensity'].value_counts(
                ).to_dict()
                print(f"  Discount levels: {discount_dist}")

    print("\n" + "=" * 80)
    print("✅ LABELING COMPLETED")
    print("=" * 80 + "\n")

    stats = {
        "total": len(df_labeled),
        "distribution": df_labeled['label'].value_counts().to_dict()
    }

    return df_labeled, stats # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Product Labeling with v2.0 Features"
    )
    parser.add_argument(
        '--input', type=str, default=None,
        help='Path to engineered_features.json'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path to output labeled_data.json'
    )
    parser.add_argument(
        '--prob-threshold', type=float, default=0.70,
        help='Probability threshold for model predictions (0.0-1.0)'
    )
    parser.add_argument(
        '--min-seed-per-class', type=int, default=50,
        help='Minimum seed samples per class to train model'
    )
    parser.add_argument(
        '--model', type=str, default='random_forest',
        choices=['random_forest', 'logistic_regression'],
        help='Model type for hybrid labeling'
    )
    parser.add_argument(
        '--no-model', action='store_true',
        help='Use pure rule-based labeling (no ML model)'
    )

    args = parser.parse_args()

    # Default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)

    input_file = args.input or os.path.join(
        base_dir, 'data', 'transformation', 'engineered_features.json'
    )
    output_file = args.output or os.path.join(
        base_dir, 'data', 'transformation', 'labeled_data.json'
    )

    if not os.path.exists(input_file):
        raise FileNotFoundError(
            f"Input file not found: {input_file}\n"
            "Please run feature_engineering.py first to generate engineered_features.json"
        )

    # Run labeling
    df_result = create_labeling(
        input_file=input_file,
        output_file=output_file,
        use_model=(not args.no_model),
        prob_threshold=args.prob_threshold,
        min_seed_per_class=args.min_seed_per_class,
        model_type=args.model
    )

    print(f"\n📋 Final shape: {df_result.shape}")
    # Show first 10 columns
    print(f"📋 Columns: {list(df_result.columns)[:10]}...")
