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

def get_ticker_filing_dates(data):
    """Extract Ticker and Filing Date."""
    ticker_filing_dates = data[['Ticker', 'Filing Date']].copy()
    ticker_filing_dates['Filing Date'] = ticker_filing_dates['Filing Date'].dt.strftime('%d/%m/%Y %H:%M')
    return ticker_filing_dates

def save_feature_data(data, ticker_filing_dates, file_path, train):
    """Save the processed feature data."""
    data_dir = os.path.join(os.path.dirname(__file__), '../../../data')
    ticker_filing_dates['Filing Date'] = pd.to_datetime(ticker_filing_dates['Filing Date'], dayfirst=True, errors='coerce')
    ticker_filing_dates.dropna(subset=['Filing Date'], inplace=True)
    ticker_filing_dates['Filing Date'] = ticker_filing_dates['Filing Date'].dt.strftime('%d/%m/%Y %H:%M')
    final_data = pd.concat([ticker_filing_dates, data], axis=1)

    if train:
        file_path = os.path.join(data_dir, file_path)
        if not final_data.empty:
            try:
                final_data.to_excel(file_path, index=False)
                print(f"- Data successfully saved to {file_path}.")
            except Exception as e:
                print(f"- Failed to save data to Excel: {e}")
        else:
            print("- No data to save.")
    return final_data

def identify_feature_types(df):
    """
    Very simple split:
      - Categorical: columns whose set of non-null values is exactly {0,1}, 
                     or whose dtype is object/category/bool.
      - Continuous:  all other numeric columns.
      - Everything else: categorical.
    Returns (categorical_cols, continuous_cols).
    """
    categorical_cols = []
    continuous_cols  = []
    
    for col in df.columns:
        ser = df[col]
        # drop nulls for the value‐check
        vals = set(ser.dropna().unique())
        
        # 1) 0/1‐only → categorical
        if vals == {0, 1}:
            categorical_cols.append(col)
        
        else:
            continuous_cols.append(col)
    
    return categorical_cols, continuous_cols

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

def drop_highly_correlated_features(data, corr_matrix, threshold=0.9):
    """
    Drop highly correlated features using a robust, iterative approach.
    It identifies all pairs of features with a correlation above the threshold,
    and for each pair, it removes one feature, ensuring that a feature marked
    for deletion isn't used to justify deleting another.
    """
    # Get the absolute value of the correlation matrix
    corr_matrix_abs = corr_matrix.abs()
    
    # Select the upper triangle of the correlation matrix to avoid duplicate pairs
    upper = corr_matrix_abs.where(np.triu(np.ones(corr_matrix_abs.shape), k=1).astype(bool))
    
    # Find all pairs of features with a correlation greater than the threshold
    highly_correlated_pairs = upper.unstack().dropna()
    highly_correlated_pairs = highly_correlated_pairs[highly_correlated_pairs > threshold]
    
    # Sort by correlation value to handle the strongest correlations first
    sorted_corr_pairs = highly_correlated_pairs.sort_values(ascending=False)
    
    # A set to keep track of features we have decided to drop
    to_drop = set()
    
    for (feat1, feat2), corr_value in sorted_corr_pairs.items():
        # If neither feature is already in our drop set...
        if feat1 not in to_drop and feat2 not in to_drop:
            # ...we decide to drop one. A simple heuristic is to drop the second feature.
            # This ensures that for any correlated pair, only one feature is removed.
            to_drop.add(feat2)
            
    # Drop the identified columns from the dataframe
    if to_drop:
        original_cols = set(data.columns)
        data.drop(columns=list(to_drop), inplace=True)
        # Identify which columns were actually dropped
        dropped_cols = list(original_cols - set(data.columns))
        print(f"- Dropped {len(dropped_cols)} highly correlated features: {dropped_cols}")
    else:
        print("- No features dropped due to high correlation.")
        
    categorical_features, continuous_features = identify_feature_types(data)
        
    # Recalculate the correlation matrix with the remaining features for consistency
    # (Note: This is computationally more expensive but guarantees accuracy. 
    # For speed, you could also drop rows/columns from the existing matrix).
    updated_corr_matrix = hybrid_correlation_matrix(data, 
                                                   data[data.columns.intersection(continuous_features)], 
                                                   data[data.columns.intersection(categorical_features)])

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


def engineer_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers new, more powerful features from the existing feature set.

    Args:
        df (pd.DataFrame): The input DataFrame from features_final.xlsx.

    Returns:
        pd.DataFrame: The DataFrame with new, engineered features added.
    """
    print("- Engineering new features...")
    
    # Use a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Define a small epsilon to prevent division-by-zero errors
    epsilon = 1e-6

    # --- Strategy 1: Interaction Features (Role x Value) ---
    # These features isolate the transaction value for the most important roles.
    df['CEO_Buy_Value'] = df['Value'] * df['CEO']
    df['CFO_Buy_Value'] = df['Value'] * df['CFO']
    df['Pres_Buy_Value'] = df['Value'] * df['Pres']
    
    # --- Strategy 2: Consolidated Insider Importance Score ---
    # This creates a single powerful feature summarizing the insider's rank.
    df['Insider_Importance_Score'] = (
        3 * df['CEO'] + 
        3 * df['Pres'] + 
        2 * df['CFO'] + 
        1 * df['Dir']
    )

    # --- Strategy 3: Ratio and Momentum Features ---
    # Ratio of recent purchases to sales. High values indicate strong buying pressure.
    # df['Purchase_Sale_Ratio_Quarter'] = df['num_purchases_quarter'] / (df['num_sales_quarter'] + epsilon)

    # Normalizes the transaction value by the company's market cap.
    df['Value_to_MarketCap'] = df['Value'] / (df['Market_Cap'] + epsilon)

    # Captures "buying the dip" vs. "buying at new highs".
    df['Distance_from_52W_High'] = 1 - df['52_Week_High_Normalized']
    
    # Ensure 'Filing Date' is a datetime object for date operations
    df['Filing Date'] = pd.to_datetime(df['Filing Date'], dayfirst=True)
    
    # 1. Day of Year
    df['Day_Of_Year'] = df['Filing Date'].dt.dayofyear

    # 2. Days of Quarter
    quarter_start_months = [1, 4, 7, 10]
    
    def days_since_quarter_start(date):
        # Determine the start month of the quarter for the given date
        start_month = max(m for m in quarter_start_months if m <= date.month)
        # Create the timestamp for the first day of that quarter
        quarter_start_date = pd.Timestamp(year=date.year, month=start_month, day=1)
        # Calculate the number of days passed
        return (date - quarter_start_date).days + 1

    df['Days_Of_Quarter'] = df['Filing Date'].apply(days_since_quarter_start)
    
    new_features_list = [
        'CEO_Buy_Value', 'CFO_Buy_Value', 'Pres_Buy_Value', 
        'Insider_Importance_Score', 'Value_to_MarketCap', 
        'Distance_from_52W_High', 'Day_Of_Year', 'Days_Of_Quarter'
    ]
    print(f"- Successfully added {len(new_features_list)} new features.")
    
    return df
