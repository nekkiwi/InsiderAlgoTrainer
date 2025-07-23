# In src/training/utils/train_helpers.py
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
from sklearn.metrics import matthews_corrcoef
from lightgbm import LGBMClassifier
from tqdm import tqdm

def apply_tp_sl(
    events_df, 
    tp_sl=(0.10, 0.05),
    high_sheet=None, 
    low_sheet=None, 
    timepoint_bdays=5,
    verbose=False
):
    """
    Args:
        events_df: DataFrame (trades to test), expects Ticker, Filing Date
        tp_sl: tuple (tp, sl)
        high_sheet: DataFrame, must match on Ticker and Filing Date, columns Day_0, Day_1, ...
        low_sheet: likewise
        timepoint_bdays: The number of days in the window (e.g., 5 for 5d)
        verbose: If True, prints detailed step-by-step information for each trade.
    Returns:
        pd.Series: realized returns per trade (TP/SL logic applied)
    """
    tp, sl = tp_sl
    realized_returns = []

    # Wrap the iterator with tqdm for a progress bar
    iterator = tqdm(events_df.iterrows(), total=len(events_df), desc="    Applying TP/SL", leave=False)
    
    for idx, row in iterator:
        ticker = row["Ticker"]
        filing_date = row["Filing Date"].strftime('%Y-%m-%d') # Format date for printing
        
        if verbose: print(f"\n--- Processing: {ticker} on {filing_date} ---")

        # Find the corresponding high and low price movements for the event
        match_high = high_sheet[(high_sheet["Ticker"] == ticker) & (high_sheet["Filing Date"] == row["Filing Date"])]
        match_low = low_sheet[(low_sheet["Ticker"] == ticker) & (low_sheet["Filing Date"] == row["Filing Date"])]
        
        exit_return = None
        
        # If no price data is found for this event, it will be handled as a fallback
        if match_high.empty or match_low.empty:
            if verbose: print("  - No price data found in high/low sheets.")
        else:
            # Check each day within the window for a TP or SL trigger
            for d in range(timepoint_bdays):
                col = f"Day_{d}"
                high_val = match_high[col].iloc[0] if col in match_high.columns else np.nan
                low_val = match_low[col].iloc[0] if col in match_low.columns else np.nan

                if verbose:
                    high_str = f"{high_val:.2%}" if pd.notna(high_val) else "N/A"
                    low_str = f"{low_val:.2%}" if pd.notna(low_val) else "N/A"
                    print(f"  - Day {d}: High={high_str}, Low={low_str}")

                # Check for TP hit
                if pd.notna(high_val) and high_val >= tp:
                    exit_return = tp
                    if verbose: print(f"  >>> TAKE-PROFIT hit on Day {d} at {tp:.2%}")
                    break
                # Check for SL hit
                if pd.notna(low_val) and low_val <= -sl:
                    exit_return = -sl
                    if verbose: print(f"  >>> STOP-LOSS hit on Day {d} at {-sl:.2%}")
                    break
        
        # If no TP or SL was triggered after checking all days
        if exit_return is None:
            fallback_col_name = f'return_{timepoint_bdays}d'
            fallback_value = row.get(fallback_col_name, np.nan)
            exit_return = fallback_value
            if verbose:
                fallback_str = f"{fallback_value:.2%}" if pd.notna(fallback_value) else "NaN"
                print(f"  >>> No TP/SL hit. Using FALLBACK return: {fallback_str}")
                
        realized_returns.append(exit_return)

    return pd.Series(realized_returns, index=events_df.index)

def select_features_for_fold(X: pd.DataFrame, y: pd.Series, top_n: int, seed: int) -> list:
    """
    Selects the top N features based on feature importance from a LightGBM model.
    
    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        top_n (int): Number of top features to select.
        seed (int): Random seed to ensure reproducibility.
        
    Returns:
        List[str]: List of top_n feature names.
    """
    if X.empty:
        return []
    
    # Initialize LightGBM model with the provided seed
    feature_ranker = LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=seed, n_jobs=-1, verbosity=-1)
    
    # Fit the model to compute feature importances
    feature_ranker.fit(X, y)
    
    # Build DataFrame of feature importances
    importances_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_ranker.feature_importances_
    })
    
    # Select the top N features
    top_features = (
        importances_df
        .sort_values(by='Importance', ascending=False)
        .head(top_n)
    )
    
    return top_features['Feature'].tolist()

def annualize_sharpe_ratio(
    returns: pd.Series, 
    risk_free_rate: float = 0.0, 
    periods_per_year: int = 252
) -> float:
    """
    Calculates the annualized Sharpe ratio with robust safety checks.

    This function prevents numerical instability that leads to infinitely large
    or "crazy" numbers by checking for a near-zero standard deviation.

    Args:
        returns (pd.Series): A series of returns (e.g., daily, weekly).
        risk_free_rate (float): The annualized risk-free rate.
        periods_per_year (int): The number of trading periods in a year
                                (e.g., 252 for daily, 52 for weekly).

    Returns:
        float: The annualized Sharpe ratio, or np.nan if the calculation
               is statistically unstable (e.g., fewer than 2 returns or
               zero volatility).
    """
    # --- Safety Check 1: Insufficient Data ---
    # A Sharpe ratio requires at least 2 data points to calculate std dev.
    if returns.empty or len(returns) < 2:
        return np.nan

    # --- Safety Check 2: Near-Zero Volatility (The Core Fix) ---
    # Check if the standard deviation is negligibly small to prevent division by zero
    # or massive numbers from floating point inaccuracies.
    std_dev = returns.std()
    if std_dev < 1e-9:  # A small epsilon to check for effective zero
        return np.nan

    # If checks pass, proceed with the calculation
    # Adjust the risk-free rate to match the period of the returns
    periodic_risk_free_rate = risk_free_rate / periods_per_year
    excess_returns = returns - periodic_risk_free_rate
    
    # Calculate the periodic Sharpe ratio
    sharpe = np.mean(excess_returns) / std_dev
    
    # Annualize the result
    return sharpe * np.sqrt(periods_per_year)

def adjusted_sharpe_ratio(sharpe: float, num_signals: int, target_signals: int = 100) -> float:
    """Adjusts the Sharpe Ratio based on the number of signals."""
    if pd.isna(sharpe) or num_signals <= 0:
        return 0.0
    adjustment_factor = min(1.0, np.sqrt(num_signals / target_signals))
    return sharpe * adjustment_factor

def find_optimal_threshold(predicted_returns: pd.Series, actual_returns: pd.Series) -> dict:
    """Finds the regressor output threshold that maximizes a chosen metric on a validation set."""
    best_score = -np.inf
    best_threshold = np.nan
    
    for percentile in range(1, 100):
        threshold = np.percentile(predicted_returns, percentile)
        final_selection_returns = actual_returns[predicted_returns >= threshold]
        
        if len(final_selection_returns) < 10: continue

        sharpe = annualize_sharpe_ratio(final_selection_returns)
        adj_sharpe = adjusted_sharpe_ratio(sharpe, len(final_selection_returns))
        
        current_score = adj_sharpe
            
        if pd.notna(current_score) and current_score > best_score:
            best_score = current_score
            best_threshold = threshold
            
    return {'validation_score': best_score, 'optimal_threshold': best_threshold}

def load_data(features_file, targets_file):
    """Loads and prepares features and targets."""
    features_df = pd.read_excel(features_file)
    targets_dict_raw = pd.read_excel(targets_file, sheet_name=None)
    features_df['Filing Date'] = pd.to_datetime(features_df['Filing Date'], dayfirst=True)
    targets_dict = {}
    for sheet_name, df in targets_dict_raw.items():
        df['Filing Date'] = pd.to_datetime(df['Filing Date'], dayfirst=True)
        targets_dict[sheet_name] = df
    return features_df, targets_dict

def load_selected_features(sel_file, category, strategy_name, top_n):
    """Loads top N features for a strategy."""
    sheet_name = f"{category.title()}_Features"
    try:
        df = pd.read_excel(sel_file, sheet_name=sheet_name)
        if strategy_name not in df.columns: return []
        scores = df[['Feature', strategy_name]].copy().dropna(subset=[strategy_name])
        top = scores[scores[strategy_name] > 0].sort_values(by=strategy_name, ascending=False)
        return top['Feature'].head(top_n).tolist()
    except (FileNotFoundError, ValueError): return []

def evaluate_test_fold(
    classifier, regressor, optimal_threshold, X_ts_sel, y_bin_ts, y_cont_ts, category,
    high_sheet=None, low_sheet=None, # Changed from stock_data_final_file
    window_days=20, tp_sl=(0.10, 0.05), X_meta_ts=None
) -> dict:
    """
    Evaluates a single test fold, with and without take-profit/stop-loss.
    If stock_data_final_file is provided, will also calculate TP/SL-constrained results.
    """
    # Use only the selected features for prediction
    print(f"    [EVAL] Starting evaluation on {len(X_ts_sel)} test samples.")
    if X_ts_sel.empty or optimal_threshold is np.nan:
        print("    [EVAL] Skipping: Test set is empty or optimal threshold is NaN.")
        return None
    
    buy_signals = classifier.predict(X_ts_sel)
    num_buy_signals = buy_signals.sum()
    print(f"    [EVAL] Classifier generated {num_buy_signals} buy signals (out of {len(X_ts_sel)}).")
    if num_buy_signals == 0:
        print("    [EVAL] Skipping: No buy signals generated.")
        return None

    pos_class_idx = X_ts_sel.index[buy_signals == 1]
    neg_class_idx = X_ts_sel.index[buy_signals == 0]
    returns_pos_classifier = y_cont_ts.loc[pos_class_idx]
    returns_neg_classifier = y_cont_ts.loc[neg_class_idx]

    _, p_val_classifier = mannwhitneyu(returns_pos_classifier, returns_neg_classifier, alternative='greater')
    sharpe_classifier = annualize_sharpe_ratio(returns_pos_classifier)
    adj_sharpe_classifier = adjusted_sharpe_ratio(sharpe_classifier, len(returns_pos_classifier))

    print(f"    [EVAL] Classifier-only -> Mean Return: {returns_pos_classifier.mean():.4f}, Adj Sharpe: {adj_sharpe_classifier:.3f}")
    if regressor is None or returns_pos_classifier.empty:
        print("    [EVAL] Skipping regressor stage: No regressor model or no positive signals.")
        return None

    print(f"    [EVAL] Scoring {len(pos_class_idx)} positive signals with regressor.")
    predicted_returns = pd.Series(regressor.predict(X_ts_sel.loc[pos_class_idx]), index=pos_class_idx)
    final_selection_idx = predicted_returns[predicted_returns >= optimal_threshold].index
    print(f"    [EVAL] Applying threshold >= {optimal_threshold:.4f}: {len(final_selection_idx)} signals remain.")
    final_returns = y_cont_ts.loc[final_selection_idx]
    if final_returns.empty:
        print("    [EVAL] No signals passed the regressor threshold. Final strategy has 0 trades.")

    _, p_val_final = (np.nan, np.nan)
    if not final_returns.empty and not returns_neg_classifier.empty:
        _, p_val_final = mannwhitneyu(final_returns, returns_neg_classifier, alternative='greater')

    metrics = {}

    sharpe_final = annualize_sharpe_ratio(final_returns)
    adj_sharpe_final = adjusted_sharpe_ratio(sharpe_final, len(final_returns))
    
    print(f"    [EVAL] Final Strategy (no TP/SL) -> Mean Return: {final_returns.mean():.4f}, Adj Sharpe: {adj_sharpe_final:.3f}")
    
    metrics.update({
        'Adjusted Sharpe (Final)': adj_sharpe_final,
        'Adjusted Sharpe (Classifier)': adj_sharpe_classifier,
        'Sharpe Ratio (Final)': sharpe_final,
        'Sharpe Ratio (Classifier)': sharpe_classifier,
        'P-Value (Final vs Neg)': p_val_final,
        'P-Value (Classifier vs Neg)': p_val_classifier,
        'Num Signals (Final)': len(final_returns),
        'Num Signals (Classifier)': len(pos_class_idx),
        'Optimal Threshold': optimal_threshold,
        'Median Return (Final Strategy)': final_returns.median(),
        'Median Return (Classifier)': returns_pos_classifier.median(),
        'Mean Return (Final Strategy)': final_returns.mean(),
        'Mean Return (Classifier)': returns_pos_classifier.mean(),
        'MCC (Classifier)': matthews_corrcoef(y_bin_ts, buy_signals)
    })

    # ---- START of TP/SL calculation ----
    if high_sheet is not None and low_sheet is not None and len(final_selection_idx) > 0:
        print(f"    [EVAL] Applying TP/SL ({tp_sl[0]}/{tp_sl[1]}) to {len(final_selection_idx)} final signals...")
        if X_meta_ts is None:
            print("    [ERROR] X_meta_ts is required for TP/SL evaluation but was not provided.")
            return metrics

        # Create the events_df from the metadata corresponding to the final signal indices
        events_df = X_meta_ts.loc[final_selection_idx].copy()
        
        # Add fallback close-close returns if present (e.g. 'return_20d')
        fallback_col = f'return_{window_days}d'
        if fallback_col in y_cont_ts:
            events_df[fallback_col] = y_cont_ts.loc[final_selection_idx].values

        tp_sl_returns = apply_tp_sl(
            events_df,
            tp_sl=tp_sl,
            high_sheet=high_sheet,
            low_sheet=low_sheet,
            timepoint_bdays=window_days
        )
        
        p_val_tp_sl = np.nan
        clean_tp_sl_returns = tp_sl_returns.dropna()
        print(f"    [EVAL] Calculated {len(clean_tp_sl_returns)} returns after applying TP/SL.")
        
        # Only perform the test if both series have data
        if not clean_tp_sl_returns.empty and not returns_neg_classifier.empty:
            _, p_val_tp_sl = mannwhitneyu(clean_tp_sl_returns, returns_neg_classifier, alternative='greater')
        
        sharpe_tp_sl = annualize_sharpe_ratio(tp_sl_returns)
        adj_sharpe_tp_sl = adjusted_sharpe_ratio(sharpe_tp_sl, len(tp_sl_returns.dropna()))
        
        print(f"    [EVAL] TP/SL Strategy -> Mean Return: {tp_sl_returns.mean():.4f}, Adj Sharpe: {adj_sharpe_tp_sl:.3f}")
        
        metrics.update({
            'Sharpe Ratio (TP/SL)': sharpe_tp_sl,
            'Median Return (TP/SL)': tp_sl_returns.median(),
            'Mean Return (TP/SL)': tp_sl_returns.mean(),
            'Adjusted Sharpe (TP/SL)': adj_sharpe_tp_sl,
            'P-Value (TP/SL vs Neg)': p_val_tp_sl # <-- Added here
        })

    return metrics
    
def convert_timepoints_to_bdays(timepoints: list) -> dict:
    """
    Converts a list of timepoint strings (e.g., '5d', '2w') into a dictionary
    mapping each timepoint to its equivalent in business days.
    """
    converted = {}
    for tp in timepoints:
        match = re.match(r"(\d+)([dwmy])", tp.lower())
        if not match: raise ValueError(f"Invalid timepoint format: '{tp}'")
        num, unit = int(match.group(1)), match.group(2)
        if unit == 'd': converted[tp] = num
        elif unit == 'w': converted[tp] = num * 5
        elif unit == 'm': converted[tp] = num * 21 # More precise average
        elif unit == 'y': converted[tp] = num * 252 # More precise average
        else: raise ValueError(f"Unknown time unit: '{unit}'")
    return converted
    
def save_strategy_results(results_df, stats_dir, model_name, category):
    """
    Saves strategy results, now including separate parameter columns for the
    classifier and the regressor in the output summary.
    """
    if results_df.empty:
        print("[INFO] Results DataFrame is empty. No file will be saved.")
        return

    os.makedirs(stats_dir, exist_ok=True)
    # The filename now includes the model name and tuning status
    output_path = os.path.join(stats_dir, f"{model_name}_{category}_walk_forward_results.xlsx")

    # --- UPDATED: Group by the separate classifier and regressor parameters ---
    group_cols = [
        'Timepoint', 'Threshold', 'Fold', 'Model', 'TP', 'SL'
    ]
    
    # --- UPDATED: Add TP/SL metrics to the display list ---
    display_cols = [
        'Adjusted Sharpe (Final)',
        'Adjusted Sharpe (TP/SL)', # New
        'P-Value (Final vs Neg)',
        'P-Value (TP/SL vs Neg)',
        'Sharpe Ratio (Final)',
        'Sharpe Ratio (TP/SL)',
        'Adjusted Sharpe (Classifier)',
        'Sharpe Ratio (Classifier)',
        'P-Value (Classifier vs Neg)',
        'Num Signals (Final)',
        'Num Signals (Classifier)',
        'Optimal Threshold',
        'Median Return (Final Strategy)',
        'Median Return (TP/SL)', # New
        'Median Return (Classifier)',
        'Mean Return (Final Strategy)',
        'Mean Return (TP/SL)', # New
        'Mean Return (Classifier)',
        'MCC (Classifier)'
    ]
    
    valid_group_cols = [col for col in group_cols if col in results_df.columns]
    valid_display_cols = [col for col in display_cols if col in results_df.columns]    
    
    if not valid_group_cols:
        print("[ERROR] No valid grouping columns found in results. Cannot create summary.")
        return
    
    # Calculate mean and std grouped by the detailed model configuration
    mean_df = results_df.groupby(valid_group_cols, dropna=False)[valid_display_cols].mean().reset_index()
    std_df = results_df.groupby(valid_group_cols, dropna=False)[valid_display_cols].std().reset_index()

    mean_df.columns = [col if col in valid_group_cols else f"{col} (Mean)" for col in mean_df.columns]
    std_df.columns = [col if col in valid_group_cols else f"{col} (Std)" for col in std_df.columns]

    summary_df = pd.merge(mean_df, std_df, on=valid_group_cols, how='left')
    summary_df.sort_values(by=valid_group_cols, inplace=True)

    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            summary_df.to_excel(writer, sheet_name='Per-Fold Summary', index=False)
            results_df.to_excel(writer, sheet_name='All Results (Raw)', index=False)
        print(f"\n- Walk-forward strategy results saved to {output_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save Excel file: {e}")