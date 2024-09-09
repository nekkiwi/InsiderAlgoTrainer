import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import chi2_contingency, pointbiserialr

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import chi2_contingency, pointbiserialr

# Helper functions for loading, saving, and identifying feature types

def load_feature_data(file_path):
    """Load the feature data from an Excel file and extract Ticker and Filing Date."""
    data_dir = os.path.join(os.path.dirname(__file__), '../../../data')
    file_path = os.path.join(data_dir, file_path)
    if os.path.exists(file_path):
        try:
            data = pd.read_excel(file_path)
            print(f"- Sheet successfully loaded from {file_path}.")
            return data
        except Exception as e:
            print(f"- Failed to load sheet from {file_path}: {e}")
            return None
    else:
        print(f"- File '{file_path}' does not exist.")
        return None

def save_normalization_params(normalization_params, file_path):
    """Save the normalization parameters (min and max values) to a CSV file."""
    normalization_df = pd.DataFrame.from_dict(normalization_params, orient='index', columns=['min', 'max'])
    data_dir = os.path.join(os.path.dirname(__file__), '../../../data')
    file_path = os.path.join(data_dir, file_path)
    normalization_df.to_excel(file_path)
    print(f"- Normalization parameters saved to {file_path}.")

def get_ticker_filing_dates(data):
    """Extract Ticker and Filing Date."""
    ticker_filing_dates = data[['Ticker', 'Filing Date']].copy()
    ticker_filing_dates['Filing Date'] = ticker_filing_dates['Filing Date'].dt.strftime('%d/%m/%Y %H:%M')
    return ticker_filing_dates

def save_feature_data(data, ticker_filing_dates, file_path):
    """Save the processed feature data."""
    data_dir = os.path.join(os.path.dirname(__file__), '../../../data')
    file_path = os.path.join(data_dir, file_path)
    ticker_filing_dates['Filing Date'] = pd.to_datetime(ticker_filing_dates['Filing Date'], dayfirst=True, errors='coerce')
    ticker_filing_dates.dropna(subset=['Filing Date'], inplace=True)
    ticker_filing_dates['Filing Date'] = ticker_filing_dates['Filing Date'].dt.strftime('%d/%m/%Y %H:%M')
    final_data = pd.concat([ticker_filing_dates, data], axis=1)

    if not final_data.empty:
        try:
            final_data.to_excel(file_path, index=False)
            print(f"- Data successfully saved to {file_path}.")
        except Exception as e:
            print(f"- Failed to save data to Excel: {e}")
    else:
        print("- No data to save.")
    return final_data

def normalize_continuous_features(data, continuous_features):
    """Apply Min-Max Normalization to continuous features and return the normalization parameters."""
    normalization_params = {}

    for column in continuous_features:
        min_value = data[column].min()
        max_value = data[column].max()

        if max_value - min_value == 0:
            print(f"- Warning: {column} has zero variance. Skipping normalization.")
            continue

        # Save the min and max values for future use
        normalization_params[column] = {'min': min_value, 'max': max_value}

        # Apply normalization
        data[column] = (data[column] - min_value) / (max_value - min_value)

    print("- Applied Min-Max Normalization to continuous features.")
    return data, normalization_params

def identify_feature_types(data):
    """Identify categorical and continuous features."""
    categorical_feature_names = ["CEO", "CFO", "COO", "Dir", "Pres", "VP", "TenPercent", 
                                 "CDL_DOJI", "CDL_HAMMER", "CDL_ENGULFING", 
                                 "Sector_Basic Materials", "Sector_Communication Services", 
                                 "Sector_Consumer Cyclical", "Sector_Consumer Defensive", 
                                 "Sector_Energy", "Sector_Financial Services", 
                                 "Sector_Healthcare", "Sector_Industrials", "Sector_Real Estate", 
                                 "Sector_Technology", "Sector_Utilities"]

    categorical_features = data[[feature for feature in categorical_feature_names if feature in data.columns]]
    continuous_feature_names = [feature for feature in data.columns if feature not in categorical_feature_names]
    continuous_features = data[continuous_feature_names]

    return categorical_features, continuous_features

# Filtering and clipping functions
def filter_low_variance_features(data, continuous_features, categorical_features, variance_threshold=0.02, categorical_threshold=0.02):
    """Filter out low variance continuous features and rare categorical features."""
    non_normalized_continuous_features = [col for col in continuous_features.columns]

    # Filter low variance continuous features
    if non_normalized_continuous_features:
        selector = VarianceThreshold(threshold=variance_threshold)
        selector.fit(continuous_features[non_normalized_continuous_features])
        low_variance_features = continuous_features[non_normalized_continuous_features].columns[~selector.get_support()]
        data.drop(columns=low_variance_features, inplace=True)
        continuous_features = continuous_features.drop(columns=low_variance_features, axis=1)
        print(f"- Dropped {len(low_variance_features)} low variance continuous features: {low_variance_features.tolist()}")

    # Filter rare categorical features
    rare_categorical_features = []
    for col in categorical_features.columns:
        min_class_freq = min(data[col].mean(), 1 - data[col].mean())
        if min_class_freq < categorical_threshold:
            rare_categorical_features.append(col)

    if rare_categorical_features:
        data.drop(columns=rare_categorical_features, inplace=True)
        categorical_features = categorical_features.drop(columns=rare_categorical_features, axis=1)
        print(f"- Dropped {len(rare_categorical_features)} rare categorical features: {rare_categorical_features}")

    # Return the updated data and feature sets
    return data, continuous_features, categorical_features

def clip_continuous_features(data, continuous_features, lower=0.01, upper=0.99):
    """Clip continuous features at the specified lower and upper percentiles."""
    for column in continuous_features:
        if column in data.columns:
            lower_bound = data[column].quantile(lower)
            upper_bound = data[column].quantile(upper)
            data[column] = data[column].clip(lower=lower_bound, upper=upper_bound).infer_objects(copy=False)
    print("- Clipped continuous features at the 1st and 99th percentiles.")
    return data

def drop_highly_correlated_features(data, corr_matrix, threshold=0.9):
    """Drop highly correlated features with a correlation above the threshold."""
    upper_triangle_mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    sorted_corr = corr_matrix.where(upper_triangle_mask).unstack().dropna().abs().sort_values(ascending=False)
    to_drop = [feature[1] for feature in sorted_corr.index if sorted_corr[feature] > threshold]
    data.drop(columns=to_drop, inplace=True)
    updated_corr_matrix = data.corr()
    print(f"- Dropped {len(to_drop)} highly correlated features: {to_drop}")
    return data, updated_corr_matrix

# Correlation calculation

def calculate_cramers_v(data, categorical_features):
    """Calculate Cramér's V for all pairs of categorical features."""
    n = len(categorical_features.columns)
    cramers_matrix = pd.DataFrame(np.zeros((n, n)), index=categorical_features.columns, columns=categorical_features.columns)
    for col1 in categorical_features.columns:
        for col2 in categorical_features.columns:
            confusion_matrix = pd.crosstab(data[col1], data[col2])
            cramers_matrix.at[col1, col2] = cramers_v(confusion_matrix)
    return cramers_matrix

def cramers_v(confusion_matrix):
    """Calculate Cramer's V statistic for categorical-categorical correlation."""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(r - 1, k - 1))))

def point_biserial_correlation(data, continuous_col, binary_col):
    """Calculate Point-Biserial correlation between a continuous feature and a binary categorical feature."""
    return pointbiserialr(data[binary_col], data[continuous_col])[0]

def hybrid_correlation_matrix(data, continuous_features, categorical_features):
    """Calculate a hybrid correlation matrix for continuous and categorical features."""
    columns = continuous_features.columns.tolist() + categorical_features.columns.tolist()
    hybrid_corr = pd.DataFrame(np.nan, index=columns, columns=columns)

    pearson_corr = continuous_features.corr(method='pearson').abs()
    hybrid_corr.loc[continuous_features.columns, continuous_features.columns] = pearson_corr

    cramers_v_matrix = calculate_cramers_v(data, categorical_features)
    hybrid_corr.loc[categorical_features.columns, categorical_features.columns] = cramers_v_matrix

    for cont_col in continuous_features.columns:
        for cat_col in categorical_features.columns:
            if data[cat_col].nunique() == 2:
                hybrid_corr.at[cont_col, cat_col] = point_biserial_correlation(data, cont_col, cat_col)
                hybrid_corr.at[cat_col, cont_col] = hybrid_corr.at[cont_col, cat_col]

    return hybrid_corr

# Plotting functions

def plot_correlation_heatmap(corr_matrix, output_dir):
    """Plot a heatmap of the correlation matrix and save it as a PNG file."""
    data_dir = os.path.join(os.path.dirname(__file__), '../../../data')
    file_path = os.path.join(data_dir, output_dir)
    
    mask = np.tril(np.ones_like(corr_matrix, dtype=bool))

    plt.figure(figsize=(20, 20), dpi=300)
    sns.heatmap(corr_matrix.astype(float), mask=mask, annot=False, cmap='coolwarm', square=True, cbar_kws={"shrink": .75},vmin=-1, vmax=1)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    file_path = os.path.join(file_path, 'feature_correlation_heatmap.png')
    plt.savefig(file_path)
    plt.close()
    print("- Feature correlation heatmap saved at", file_path)

def plot_sorted_correlations(corr_matrix, output_dir):
    """Plot sorted correlations as a barplot and save it as a PNG file."""
    data_dir = os.path.join(os.path.dirname(__file__), '../../../data')
    file_path = os.path.join(data_dir, output_dir)
    
    upper_triangle_mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    sorted_corr = corr_matrix.where(upper_triangle_mask).unstack().dropna().abs().sort_values(ascending=False)

    sorted_corr = sorted_corr[sorted_corr > 0.8]

    plt.figure(figsize=(14, 10), dpi=300)
    sorted_corr.plot(kind='bar', width=0.8)
    plt.title('Sorted Feature Correlations Above 0.8')
    plt.ylim(0.8, 1.0)
    plt.yticks(np.arange(0.8, 1.05, 0.05))
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()

    file_path = os.path.join(file_path, 'sorted_correlations.png')
    plt.savefig(file_path, dpi=300)
    plt.close()
    print("- Sorted feature correlation barplot saved at", file_path)
