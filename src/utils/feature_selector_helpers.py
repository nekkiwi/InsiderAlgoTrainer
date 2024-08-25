import pandas as pd
from sklearn.feature_selection import f_classif, chi2, f_regression

def select_features(X, y, p_threshold=0.1):
    """Select features based on the type of the target and feature types."""
    categorical_features = X.columns.intersection([
        "CEO", "CFO", "COO", "Dir", "Pres", "VP", "TenPercent", 
        "CDL_DOJI", "CDL_HAMMER", "CDL_ENGULFING", 
        "Sector_Basic Materials", "Sector_Communication Services", 
        "Sector_Consumer Cyclical", "Sector_Consumer Defensive", 
        "Sector_Energy", "Sector_Financial Services", 
        "Sector_Healthcare", "Sector_Industrials", "Sector_Real Estate", 
        "Sector_Technology", "Sector_Utilities"
    ])

    scores = pd.Series(index=X.columns, dtype=float)
    p_values = pd.Series(index=X.columns, dtype=float)
    
    for feature in X.columns:
        if feature in categorical_features:
            if y.nunique() <= 2:  # Binary target (categorical)
                score, p_value = chi2(X[[feature]], y)
            else:  # Continuous target
                score, p_value = f_regression(X[[feature]], y)
        else:
            if y.nunique() <= 2:  # Binary target (categorical)
                score, p_value = f_classif(X[[feature]], y)
            else:  # Continuous target
                score, p_value = f_regression(X[[feature]], y)
        
        scores[feature] = score[0]  # Extract the first element since score is a 1D array
        p_values[feature] = p_value[0]  # Extract the first element since p_value is a 1D array

    # Filter features based on p-value threshold
    selected_features = p_values[p_values < p_threshold].index.tolist()

    return selected_features, scores, p_values
