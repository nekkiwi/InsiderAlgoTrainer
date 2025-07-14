# In src/analysis/utils/feature_selector_helpers.py

import os
import re
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def select_features_for_fold(X: pd.DataFrame, y: pd.Series, top_n: int, seed: int) -> list:
    """
    Selects the top N features based on feature importance from a RandomForest model.
    
    --- FIX: The function now accepts a 'seed' and uses it to initialize the
    internal model, ensuring the feature selection varies across runs. ---
    """
    if X.empty:
        return []

    # Initialize a model with the provided seed to rank features
    feature_ranker = RandomForestClassifier(
        n_estimators=100, 
        random_state=seed,  # Use the passed-in seed
        n_jobs=-1
    )
    
    feature_ranker.fit(X, y)
    
    importances_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_ranker.feature_importances_
    })
    
    # Select the top N features based on importance
    top_features = importances_df.sort_values(by='Importance', ascending=False).head(top_n)
    
    return top_features['Feature'].tolist()

def get_sorted_strategy_keys(strategy_keys: list) -> list:
    """
    Sorts strategy keys chronologically by time horizon (1d, 1w, 1m),
    then by threshold.
    """
    def sort_key(key):
        match = re.search(r'_((\d+)([dmwy]))_binary_(\d+)pct', key)
        if match:
            horizon, num_str, unit, threshold_str = match.groups()
            num = int(num_str)
            threshold = int(threshold_str)
            # Define a multiplier for each time unit to create a comparable value
            unit_multiplier = {'d': 1, 'w': 7, 'm': 30, 'y': 365}
            time_value = num * unit_multiplier.get(unit, 0)
            return (time_value, threshold)
        return (float('inf'), 0) # Fallback for non-matching keys

    return sorted(strategy_keys, key=sort_key)

def select_features_with_model(features_df, targets_df_dict, raw_target_name, threshold_pct, top_n):
    """
    Selects features by dynamically creating a binary target from a raw continuous target.
    """
    if raw_target_name not in targets_df_dict:
        return [], None
    X = features_df.drop(columns=['Ticker', 'Filing Date'], errors='ignore')
    y_raw_df = targets_df_dict[raw_target_name]
    merged_data = pd.merge(features_df, y_raw_df, on=['Ticker', 'Filing Date'], how='inner')
    if merged_data.empty or raw_target_name not in merged_data.columns:
        return [], None
    y_binary = (merged_data[raw_target_name] >= (threshold_pct / 100.0)).astype(int)
    y_binary.dropna(inplace=True)
    X_aligned = merged_data.loc[y_binary.index, X.columns]
    if y_binary.nunique() < 2 or y_binary.value_counts().min() < 5:
        return [], None
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    model.fit(X_aligned, y_binary)
    importances = pd.Series(model.feature_importances_, index=X_aligned.columns)
    non_zero_importances = importances[importances > 0]
    if non_zero_importances.empty:
        return [], None
    top_features = non_zero_importances.nlargest(top_n)
    return top_features.index.tolist(), top_features

def save_feature_selection_results(all_scores: dict, output_dir: str):
    """
    Saves feature selection results into a single Excel file with
    separate sheets for 'return' and 'alpha' features.
    """
    output_file = os.path.join(output_dir, "all_selected_features.xlsx")
    os.makedirs(output_dir, exist_ok=True)
    
    return_scores = {k: v for k, v in all_scores.items() if k.startswith('return')}
    alpha_scores = {k: v for k, v in all_scores.items() if k.startswith('alpha')}

    with pd.ExcelWriter(output_file) as writer:
        if return_scores:
            sorted_return_keys = get_sorted_strategy_keys(list(return_scores.keys()))
            return_df = pd.DataFrame()
            for key in sorted_return_keys:
                df = return_scores[key].reset_index().rename(columns={'index': 'Feature', 0: key})
                return_df = pd.merge(return_df, df, on='Feature', how='outer') if not return_df.empty else df
            return_df.fillna(0, inplace=True)
            return_df.to_excel(writer, sheet_name='Return_Features', index=False)
            print("- Return feature importances saved to 'Return_Features' sheet.")

        if alpha_scores:
            sorted_alpha_keys = get_sorted_strategy_keys(list(alpha_scores.keys()))
            alpha_df = pd.DataFrame()
            for key in sorted_alpha_keys:
                df = alpha_scores[key].reset_index().rename(columns={'index': 'Feature', 0: key})
                alpha_df = pd.merge(alpha_df, df, on='Feature', how='outer') if not alpha_df.empty else df
            alpha_df.fillna(0, inplace=True)
            alpha_df.to_excel(writer, sheet_name='Alpha_Features', index=False)
            print("- Alpha feature importances saved to 'Alpha_Features' sheet.")
            
    print(f"\n- Consolidated feature importance table saved to: {output_file}")

def create_feature_heatmap(scores: dict, category: str, output_dir: str):
    """
    Creates a consolidated heatmap where only selected features (those with
    non-zero importance) are colored.
    """
    if not scores:
        print(f"[INFO] No scores provided for '{category}' heatmap. Skipping.")
        return
    
    sorted_keys = get_sorted_strategy_keys(list(scores.keys()))
    
    # Create the base DataFrame with all feature scores
    heatmap_df = pd.concat(scores, axis=1).T.reindex(sorted_keys).dropna(how='all', axis=1).fillna(0)
    
    # Filter out features that were never selected in any strategy to keep the plot clean
    heatmap_df = heatmap_df.loc[:, (heatmap_df != 0).any(axis=0)]
    
    if heatmap_df.empty:
        print(f"[INFO] No features with non-zero importance found for '{category}' heatmap. Skipping.")
        return

    # --- THIS IS THE KEY CHANGE ---
    # Replace all zero-importance scores with NaN (Not a Number).
    # Seaborn's heatmap will render NaN values as blank (uncolored) cells,
    # effectively hiding the non-selected features from the visualization.
    heatmap_df.replace(0, np.nan, inplace=True)

    # Normalize the remaining (non-NaN) scores to sum to 1 for better comparison
    heatmap_df = heatmap_df.div(heatmap_df.sum(axis=1), axis=0)

    # Plotting
    plt.figure(figsize=(max(12, heatmap_df.shape[1] * 0.6), max(8, heatmap_df.shape[0] * 0.5)))
    
    # The heatmap will now only apply color to the cells that have a numeric value
    sns.heatmap(
        heatmap_df, 
        annot=True, 
        fmt=".2f", 
        cmap="viridis", 
        linewidths=.5, 
        cbar=True
    )
    
    plt.title(f'Selected Feature Importance for {category.title()} Targets', fontsize=16)
    plt.ylabel('Target Strategy', fontsize=12)
    plt.xlabel('Features', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"selected_{category}_feature_importance.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"- Selected feature importance heatmap for '{category}' saved to: {out_path}")