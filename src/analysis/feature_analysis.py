import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import VarianceThreshold
import sys
import os

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scraper.feature_scraper import FeatureScraper

class FeatureAnalyzer:
    def __init__(self):
        self.data = pd.DataFrame()
        self.continuous_features = None
        self.categorical_features = None
        self.corr = None

    def identify_feature_types(self):
        # Ensure that categorical_feature_names contains only valid feature names
        categorical_feature_names = ["CEO", "CFO", "COO", "Dir", "Pres", "VP", "TenPercent", \
                                    "CDL_DOJI", "CDL_HAMMER", "CDL_ENGULFING", \
                                    "Sector_Basic Materials", "Sector_Communication Services", "Sector_Consumer Cyclical", \
                                    "Sector_Consumer Defensive", "Sector_Energy", "Sector_Financial Services", \
                                    "Sector_Healthcare", "Sector_Industrials", "Sector_Real Estate", \
                                    "Sector_Technology", "Sector_Utilities"]

        # Identify categorical features based on user input
        self.categorical_features = self.data[categorical_feature_names]

        # All other features are considered continuous
        continuous_feature_names = [feature for feature in self.data.columns if feature not in categorical_feature_names]
        self.continuous_features = self.data[continuous_feature_names]

    def calculate_correlations(self):
        """Calculate Pearson and Spearman correlations."""
        self.pearson_corr = self.data.corr(method='pearson').abs()
        self.spearman_corr = self.data.corr(method='spearman').abs()

        # Initialize the final correlation matrix
        self.corr = pd.DataFrame(index=self.data.columns, columns=self.data.columns)

        # Fill the final correlation matrix based on the feature types
        for col1 in self.data.columns:
            for col2 in self.data.columns:
                if col1 == col2:
                    self.corr.at[col1, col2] = np.nan
                elif col1 in self.continuous_features.columns and col2 in self.continuous_features.columns:
                    self.corr.at[col1, col2] = self.pearson_corr.at[col1, col2]
                else:
                    self.corr.at[col1, col2] = self.spearman_corr.at[col1, col2]

    def plot_correlation_heatmap(self):
        """Plot a heatmap of the correlation matrix."""
        # Mask the lower triangle
        mask = np.tril(np.ones_like(self.corr, dtype=bool))

        plt.figure(figsize=(20, 20), dpi=300)  # Adjusted size for larger plots
        sns.heatmap(self.corr.astype(float), mask=mask, annot=False, cmap='coolwarm', square=True, cbar_kws={"shrink": .75})
        plt.title('Feature Correlation Heatmap')
        
        # Adjust layout to ensure everything fits
        plt.tight_layout()

        # Add grid lines
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Save the plot as a PNG file
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        os.makedirs(data_dir, exist_ok=True)  # Create the data directory if it doesn't exist
        file_path = os.path.join(data_dir, 'output/feature_correlation_heatmap.png')
        plt.savefig(file_path)
        plt.close()
        print("Feature correlation heatmap saved at",file_path)


    def plot_sorted_correlations(self):
        """Plot sorted correlations as a barplot."""
        # Remove the diagonal and lower triangle to avoid duplicate pairs
        upper_triangle_mask = np.triu(np.ones(self.corr.shape), k=1).astype(bool)
        sorted_corr = self.corr.where(upper_triangle_mask).unstack().dropna().sort_values(ascending=False)

        # Filter to only show correlations greater than the threshold (e.g., 0.8)
        sorted_corr = sorted_corr[sorted_corr > 0.8]

        # Increase the figure size and DPI
        plt.figure(figsize=(14, 10), dpi=300)  # Larger figure size and increased DPI for better resolution
        sorted_corr.plot(kind='bar', width=0.8)

        plt.title('Sorted Feature Correlations Above 0.8')

        # Set the y-axis to start from the cutoff (e.g., 0.8)
        plt.ylim(0.8, 1.0)

        # Add more y-ticks between 0.8 and 1.0 for better granularity
        plt.yticks(np.arange(0.8, 1.05, 0.05))

        # Add grid lines with more granularity
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

        # Adjust layout to ensure everything fits
        plt.tight_layout()

        # Save the plot as a PNG file
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        os.makedirs(data_dir, exist_ok=True)  # Create the data directory if it doesn't exist
        file_path = os.path.join(data_dir, 'output/sorted_correlations.png')
        plt.savefig(file_path, dpi=300)  # Save with increased DPI
        plt.close()
        print("Sorted feature correlation barplot saved at", file_path)


    def drop_highly_correlated_features(self, threshold=0.9):
        """Drop features with correlation higher than the threshold."""
        # Combine both Pearson and Spearman correlations into a single matrix
        corr_matrix = self.corr  # This combined matrix was calculated earlier

        # Remove the lower triangle and the diagonal to avoid duplicate pairs
        upper_triangle_mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        upper = corr_matrix.where(upper_triangle_mask)

        # Identify columns to drop based on the threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        # Drop the identified columns from the data
        self.data.drop(columns=to_drop, axis=1, inplace=True)
        
        # Remove the dropped columns from the correlation matrix
        self.corr.drop(index=to_drop, columns=to_drop, inplace=True)
        
        self.continuous_features.drop(columns=[col for col in to_drop if col in self.continuous_features.columns], axis=1, inplace=True)
        self.categorical_features.drop(columns=[col for col in to_drop if col in self.categorical_features.columns], axis=1, inplace=True)
        
        print(f"Dropped {len(to_drop)} highly correlated features: {to_drop}")

    def clip_continuous_features(self):
        """Clip continuous features at the 1st and 99th percentiles."""
        for column in self.continuous_features:
            lower_bound = self.data[column].quantile(0.01)
            upper_bound = self.data[column].quantile(0.99)
            self.data[column] = self.data[column].clip(lower=lower_bound, upper=upper_bound)
        print("Clipped continuous features at the 1st and 99th percentiles.")

    def normalize_continuous_features(self):
        """Apply Min-Max Normalization to continuous features."""
        for column in self.continuous_features:
            min_value = self.data[column].min()
            max_value = self.data[column].max()
            self.data[column] = (self.data[column] - min_value) / (max_value - min_value)
        print("Applied Min-Max Normalization to continuous features.")

    def filter_low_variance_features(self, variance_threshold=0.02, categorical_threshold=0.02):
        """
        Remove features with low variance for continuous features and rare categories for categorical features.
        
        Parameters:
        - variance_threshold: The threshold below which variance is considered too low for continuous features.
        - categorical_threshold: The threshold for the least frequent class proportion below which categorical features are removed.
        """
        # Features that are already normalized and can thus be mistaken for having low variance
        normalized_features = ['SMA_10', 'SMA_50', 'EMA_10', 'EMA_50', 'Bollinger_Upper', 'Bollinger_Lower', 'ATR_14', 'SAR', \
                                'OBV', 'Cumulative_Alpha', 'Rolling_Alpha_30', 'Jensen_Alpha'\
                                'Operating_Cash_Flow_to_Market_Cap', 'Investing_Cash_Flow_to_Market_Cap', 'Financing_Cash_Flow_to_Market_Cap', \
                                'Net_Income_to_Market_Cap', 'Total_Assets_to_Market_Cap', 'Total_Liabilities_to_Market_Cap', \
                                'Total_Equity_to_Market_Cap', 'Average_Volume_to_Market_Cap', 'Free_Cash_Flow_to_Market_Cap', \
                                '52_Week_High_Normalized', '52_Week_Low_Normalized']
        
        # Remove low variance continuous features, ignoring normalized features
        non_normalized_continuous_features = [col for col in self.continuous_features.columns if col not in normalized_features]

        if non_normalized_continuous_features:
            selector = VarianceThreshold(threshold=variance_threshold)
            selector.fit(self.continuous_features[non_normalized_continuous_features])
            low_variance_features = self.continuous_features[non_normalized_continuous_features].columns[~selector.get_support()]

            # Drop low variance continuous features from data and continuous_features list
            self.data.drop(columns=low_variance_features, inplace=True)
            self.continuous_features = self.continuous_features.drop(columns=low_variance_features, axis=1)
            print(f"Dropped {len(low_variance_features)} low variance continuous features: {low_variance_features.tolist()}")

        # Remove rare categorical features
        rare_categorical_features = []

        for col in self.categorical_features.columns:
            # Calculate the frequency of the least frequent class
            min_class_freq = min(self.data[col].mean(), 1 - self.data[col].mean())

            # If this frequency is less than the threshold, mark for removal
            if min_class_freq < categorical_threshold:
                rare_categorical_features.append(col)

        # Drop rare categorical features from data and categorical_features list
        if rare_categorical_features:
            self.data.drop(columns=rare_categorical_features, inplace=True)
            self.categorical_features = self.categorical_features.drop(columns=rare_categorical_features, axis=1)
        
        print(f"Dropped {len(rare_categorical_features)} rare categorical features: {rare_categorical_features}")

        
    def load_features(self, file_path='processed/features_full.xlsx'):
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        file_path = os.path.join(data_dir, file_path)

        if os.path.exists(file_path):
            try:
                self.data = pd.read_excel(file_path)
                print(f"Sheet successfully loaded from {file_path}.")
            except Exception as e:
                print(f"Failed to load sheet from {file_path}: {e}")
        else:
            print(f"File '{file_path}' does not exist.")
            
    def save_to_excel(self, file_path='output.xlsx'):
        """Save the self.data DataFrame to an Excel file."""
        data_dir = os.path.join(os.path.dirname(__file__), '../../data')
        os.makedirs(data_dir, exist_ok=True)  # Create the data directory if it doesn't exist
        
        file_path = os.path.join(data_dir, file_path)
        if not self.data.empty:
            try:
                self.data.to_excel(file_path, index=False)
                print(f"Data successfully saved to {file_path}.")
            except Exception as e:
                print(f"Failed to save data to Excel: {e}")
        else:
            print("No data to save.")

    def run_analysis(self):
        """Run the full analysis pipeline."""
        self.data.drop(columns=['Ticker', 'Filing Date'], inplace=True)
        self.identify_feature_types()
        self.filter_low_variance_features()
        self.clip_continuous_features()
        self.normalize_continuous_features()
        self.calculate_correlations()
        self.plot_correlation_heatmap()
        self.plot_sorted_correlations()
        self.drop_highly_correlated_features()
        self.save_to_excel('processed/features_processed.xlsx')
        
if __name__ == "__main__":    
    # Perform feature analysis
    analyzer = FeatureAnalyzer()
    analyzer.load_features()
    analyzer.run_analysis()