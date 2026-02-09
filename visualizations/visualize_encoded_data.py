import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class EncodedDataVisualizer:
    """
    Trực quan hóa dữ liệu sau khi encoding
    
    Visualizations:
    1. Feature distributions (scaled numerical features)
    2. Label distribution
    3. Feature correlation heatmap
    4. One-hot encoded features analysis
    5. Train/Test split comparison
    6. Feature importance by label
    """

    def __init__(self, encoded_data_file: str):
        """
        Khởi tạo visualizer
        
        Parameters:
        - encoded_data_file: đường dẫn đến file encoded_data.json
        """
        self.encoded_data_file = encoded_data_file
        # self.data = None
        # self.X_train = None
        # self.X_test = None
        # self.y_train = None
        # self.y_test = None
        # self.feature_names = None
        # self.feature_info = None
        
        # Load data
        self._load_data()

    def _load_data(self):
        """Load encoded data từ JSON file"""
        print("📂 Loading encoded data...")
        
        with open(self.encoded_data_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Extract components
        self.X_train = pd.DataFrame(self.data['X_train'])
        self.y_train = np.array(self.data['y_train']) if self.data['y_train'] else None
        
        if self.data['X_test']:
            self.X_test = pd.DataFrame(self.data['X_test'])
            self.y_test = np.array(self.data['y_test']) if self.data['y_test'] else None
        
        self.feature_names = self.data.get('feature_names', [])
        self.feature_info = self.data.get('feature_info', {})
        
        print(f"✓ Loaded data:")
        print(f"  - Train samples: {len(self.X_train):,}")
        if self.X_test is not None:
            print(f"  - Test samples: {len(self.X_test):,}")
        print(f"  - Features: {len(self.feature_names):,}")
        print(f"  - Labels: {len(np.unique(self.y_train)) if self.y_train is not None else 0}")

    def visualize_all(self, output_dir: Optional[str] = None):
        """Tạo tất cả các visualizations"""
        
        if output_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base, 'data/visualizations/encoded')
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n📊 Creating visualizations → {output_dir}")
        print("=" * 80)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        
        # 1. Label distribution
        if self.y_train is not None:
            print("\n1️⃣ Visualizing label distribution...")
            self._plot_label_distribution(output_dir)
        
        # 2. Feature type distribution
        print("2️⃣ Visualizing feature type distribution...")
        self._plot_feature_types(output_dir)
        
        # 3. Scaled numerical features distribution
        print("3️⃣ Visualizing scaled numerical features...")
        self._plot_scaled_features(output_dir)
        
        # 4. One-hot encoded features
        print("4️⃣ Analyzing one-hot encoded features...")
        self._plot_onehot_features(output_dir)
        
        # 5. Feature correlation heatmap
        print("5️⃣ Creating correlation heatmap...")
        self._plot_correlation_heatmap(output_dir)
        
        # 6. Train/Test comparison
        if self.X_test is not None:
            print("6️⃣ Comparing train/test distributions...")
            self._plot_train_test_comparison(output_dir)
        
        # 7. Feature statistics by label
        if self.y_train is not None:
            print("7️⃣ Analyzing features by label...")
            self._plot_features_by_label(output_dir)
        
        # 8. Data quality report
        print("8️⃣ Generating data quality report...")
        self._generate_quality_report(output_dir)
        
        print("\n" + "=" * 80)
        print(f"✅ All visualizations saved to: {output_dir}")
        print("=" * 80)

    def _plot_label_distribution(self, output_dir: str):
        """Visualize label distribution"""
        
        # Get label encoder mapping
        label_mapping = self.feature_info.get('label', {}).get('encoding', {})
        inverse_mapping = {v: k for k, v in label_mapping.items()}
        
        # Count labels
        y_train = np.asarray(self.y_train)

        unique, counts = np.unique(y_train, return_counts=True)

        label_names = [inverse_mapping.get(int(l), f"Label {l}") for l in unique]
        
        # Create figure with 2 subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = axes[0].bar(label_names, counts, color=colors[:len(label_names)])
        axes[0].set_title('Label Distribution (Train Set)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Label')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(count):,}\n({count/len(np.asarray(self.y_train))*100:.1f}%)',
                        ha='center', va='bottom', fontsize=10)
        
        # Pie chart
        axes[1].pie(counts, labels=label_names, autopct='%1.1f%%',
                   colors=colors[:len(label_names)], startangle=90)
        axes[1].set_title('Label Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '01_label_distribution.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # If test set exists, compare train/test label distribution
        if self.y_test is not None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Train
            y_train = np.asarray(self.y_train)
            unique_train, counts_train = np.unique(y_train, return_counts=True)
            label_names_train = [inverse_mapping.get(int(l), f"Label {l}") for l in unique_train]
            axes[0].bar(label_names_train, counts_train, color=colors[:len(label_names_train)])
            axes[0].set_title('Train Set Labels', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Label')
            axes[0].set_ylabel('Count')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Test
            unique_test, counts_test = np.unique(self.y_test, return_counts=True)
            label_names_test = [inverse_mapping.get(int(l), f"Label {l}") for l in unique_test]
            axes[1].bar(label_names_test, counts_test, color=colors[:len(label_names_test)])
            axes[1].set_title('Test Set Labels', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Label')
            axes[1].set_ylabel('Count')
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '02_train_test_labels.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()

    def _plot_feature_types(self, output_dir: str):
        """Visualize feature type distribution"""
        
        # Categorize features
        scaled_features = [f for f in self.feature_names if f.endswith('_scaled')]
        onehot_features = [f for f in self.feature_names if any(
            cat in f for cat in ['popularity_category_', 'price_segment_', 
                                'quality_tier_', 'discount_intensity_', 'product_age_']
        )]
        other_features = [f for f in self.feature_names 
                         if f not in scaled_features and f not in onehot_features]
        
        # Create pie chart
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sizes = [len(scaled_features), len(onehot_features), len(other_features)]
        labels = [f'Scaled Numerical\n({len(scaled_features)})', 
                 f'One-Hot Encoded\n({len(onehot_features)})',
                 f'Other\n({len(other_features)})']
        colors = ['#3498db', '#e74c3c', '#95a5a6']
        
        pie_result = ax.pie(
            sizes,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        autotexts = pie_result[2] if len(pie_result) == 3 else []

        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        ax.set_title('Feature Type Distribution', fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '03_feature_types.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_scaled_features(self, output_dir: str):
        """Visualize scaled numerical features"""
        
        scaled_features = [f for f in self.feature_names if f.endswith('_scaled')]
        
        if len(scaled_features) == 0:
            print("   ⚠️ No scaled features found")
            return
        
        # Plot distributions
        n_features = len(scaled_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(scaled_features):
            data = self.X_train[feature]
            
            axes[idx].hist(data, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
            axes[idx].set_title(feature.replace('_scaled', ''), fontsize=10, fontweight='bold')
            axes[idx].set_xlabel('Scaled Value')
            axes[idx].set_ylabel('Frequency')
            
            # Add statistics
            mean_val = data.mean()
            std_val = data.std()
            axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                            label=f'μ={mean_val:.2f}')
            axes[idx].axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5,
                            label=f'σ={std_val:.2f}')
            axes[idx].axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5)
            axes[idx].legend(fontsize=8)
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Scaled Numerical Features Distribution', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '04_scaled_features.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_onehot_features(self, output_dir: str):
        """Analyze one-hot encoded features"""
        
        # Find one-hot feature groups
        categorical_features = self.feature_info.get('categorical_features', [])
        
        if len(categorical_features) == 0:
            print("   ⚠️ No categorical features found")
            return
        
        fig, axes = plt.subplots(len(categorical_features), 1, 
                                figsize=(12, len(categorical_features) * 3))
        
        if len(categorical_features) == 1:
            axes = [axes]
        
        for idx, cat_feature in enumerate(categorical_features):
            # Find all one-hot columns for this feature
            onehot_cols = [f for f in self.feature_names if f.startswith(f'{cat_feature}_')]
            
            if len(onehot_cols) == 0:
                continue
            
            # Count occurrences (sum of 1s in each column)
            counts = [self.X_train[col].sum() for col in onehot_cols]
            labels = [col.replace(f'{cat_feature}_', '') for col in onehot_cols]
            
            # Plot
            bars = axes[idx].bar(range(len(labels)), counts, color='teal', alpha=0.7)
            axes[idx].set_title(f'{cat_feature} Distribution', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Category')
            axes[idx].set_ylabel('Count')
            axes[idx].set_xticks(range(len(labels)))
            axes[idx].set_xticklabels(labels, rotation=45, ha='right')
            
            # Add count labels
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{int(count):,}\n({count/len(self.X_train)*100:.1f}%)',
                             ha='center', va='bottom', fontsize=9)
        
        plt.suptitle('One-Hot Encoded Features Distribution', 
                    fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '05_onehot_features.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_correlation_heatmap(self, output_dir: str):
        """Create correlation heatmap for scaled features"""
        
        scaled_features = [f for f in self.feature_names if f.endswith('_scaled')]
        
        if len(scaled_features) < 2:
            print("   ⚠️ Not enough scaled features for correlation")
            return
        
        # Calculate correlation
        corr_matrix = self.X_train[scaled_features].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Clean feature names for display
        clean_names = [f.replace('_scaled', '') for f in scaled_features]
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5,
                   xticklabels=clean_names, yticklabels=clean_names,
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        ax.set_title('Feature Correlation Heatmap (Scaled Features)', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '06_correlation_heatmap.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_train_test_comparison(self, output_dir: str):
        """Compare train and test set distributions"""
        
        scaled_features = [f for f in self.feature_names if f.endswith('_scaled')][:6]  # Top 6
        
        if len(scaled_features) == 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(scaled_features):
            train_data = self.X_train[feature]
            test_data = self.X_test[feature]
            
            axes[idx].hist(train_data, bins=30, alpha=0.6, label='Train', color='blue')
            axes[idx].hist(test_data, bins=30, alpha=0.6, label='Test', color='orange')
            axes[idx].set_title(feature.replace('_scaled', ''), fontsize=10, fontweight='bold')
            axes[idx].set_xlabel('Value')
            axes[idx].set_ylabel('Frequency')
            axes[idx].legend()
        
        plt.suptitle('Train vs Test Distribution Comparison', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '07_train_test_comparison.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_features_by_label(self, output_dir: str):
        """Analyze feature distributions by label"""
        
        # Get label mapping
        label_mapping = self.feature_info.get('label', {}).get('encoding', {})
        inverse_mapping = {v: k for k, v in label_mapping.items()}
        
        # Select top scaled features
        scaled_features = [f for f in self.feature_names if f.endswith('_scaled')][:6]
        
        if len(scaled_features) == 0:
            return
        
        # Create DataFrame with labels
        df_analysis = self.X_train[scaled_features].copy()
        df_analysis['label'] = [inverse_mapping.get(int(l), f"Label {l}") 
                                for l in np.asarray(self.y_train)]
        
        # Create box plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(scaled_features):
            df_analysis.boxplot(column=feature, by='label', ax=axes[idx])
            axes[idx].set_title(feature.replace('_scaled', ''), fontsize=10, fontweight='bold')
            axes[idx].set_xlabel('Label')
            axes[idx].set_ylabel('Scaled Value')
            plt.sca(axes[idx])
            plt.xticks(rotation=45)
        
        plt.suptitle('Feature Distributions by Label', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '08_features_by_label.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()

    def _generate_quality_report(self, output_dir: str):
        """Generate data quality report"""
        
        report_path = os.path.join(output_dir, 'encoding_quality_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ENCODED DATA QUALITY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Dataset info
            f.write("📊 DATASET INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Train samples: {len(self.X_train):,}\n")
            if self.X_test is not None:
                f.write(f"Test samples: {len(self.X_test):,}\n")
            f.write(f"Total features: {len(self.feature_names):,}\n")
            f.write(f"Labels: {len(np.unique(self.y_train)) if self.y_train is not None else 0}\n\n")
            
            # Feature breakdown
            f.write("🔢 FEATURE BREAKDOWN\n")
            f.write("-" * 80 + "\n")
            
            categorical_features = self.feature_info.get('categorical_features', [])
            numerical_features = self.feature_info.get('numerical_features', [])
            
            f.write(f"Original categorical features: {len(categorical_features)}\n")
            for feat in categorical_features:
                f.write(f"  - {feat}\n")
            
            f.write(f"\nOriginal numerical features: {len(numerical_features)}\n")
            for feat in numerical_features:
                f.write(f"  - {feat}\n")
            
            # Encoded features
            scaled_features = [f for f in self.feature_names if f.endswith('_scaled')]
            onehot_features = [f for f in self.feature_names if any(
                cat in f for cat in ['_' + cat + '_' for cat in categorical_features]
            )]
            
            f.write(f"\nScaled features: {len(scaled_features)}\n")
            f.write(f"One-hot encoded features: {len(onehot_features)}\n\n")
            
            # Data quality checks
            f.write("✅ DATA QUALITY CHECKS\n")
            f.write("-" * 80 + "\n")
            
            # Check for missing values
            missing_train = self.X_train.isnull().sum().sum()
            f.write(f"Missing values (train): {missing_train}\n")
            
            if self.X_test is not None:
                missing_test = self.X_test.isnull().sum().sum()
                f.write(f"Missing values (test): {missing_test}\n")
            
            # Check for infinite values
            inf_train = np.isinf(self.X_train.select_dtypes(include=[np.number])).sum().sum()
            f.write(f"Infinite values (train): {inf_train}\n")
            
            # Label distribution
            if self.y_train is not None:
                f.write("\n📈 LABEL DISTRIBUTION\n")
                f.write("-" * 80 + "\n")
                
                label_mapping = self.feature_info.get('label', {}).get('encoding', {})
                inverse_mapping = {v: k for k, v in label_mapping.items()}
                
                unique, counts = np.unique(self.y_train, return_counts=True)
                for label_id, count in zip(unique, counts):
                    label_name = inverse_mapping.get(int(label_id), f"Label {label_id}")
                    percentage = count / len(self.y_train) * 100
                    f.write(f"{label_name}: {count:,} ({percentage:.2f}%)\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("✅ Report generated successfully\n")
            f.write("=" * 80 + "\n")
        
        print(f"   ✓ Quality report saved: {report_path}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def visualize_encoded_data(
    encoded_data_file: Optional[str] = None,
    output_dir: Optional[str] = None
):
    """
    Main function để visualize encoded data
    
    Parameters:
    - encoded_data_file: đường dẫn đến encoded_data.json
    - output_dir: thư mục output cho visualizations
    """
    
    if encoded_data_file is None:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        encoded_data_file = os.path.join(base, 'data/transformation/encoded_data.json')
    
    print("\n" + "=" * 80)
    print("📊 ENCODED DATA VISUALIZATION")
    print("=" * 80)
    
    # Create visualizer
    visualizer = EncodedDataVisualizer(encoded_data_file)
    
    # Generate all visualizations
    visualizer.visualize_all(output_dir)
    
    print("\n✅ Visualization completed!")


if __name__ == "__main__":
    visualize_encoded_data()
