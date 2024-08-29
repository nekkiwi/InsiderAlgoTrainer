import os
import pandas as pd

def filter_jumps(self, max_jump=1):
    """Remove rows with jumps (consecutive differences) higher than a specified threshold."""
    # Drop Ticker and Filing Date columns to focus on return data
    returns_data = self.return_df.drop(columns=['Ticker', 'Filing Date'])
    
    # Calculate consecutive differences and identify rows with jumps > specified threshold
    jumps = returns_data.diff(axis=1).abs().dropna(axis=1)
    mask = (jumps <= max_jump).all(axis=1)
    
    # Filter the return_df to remove those rows
    self.return_df = self.return_df[mask]
    # print(f"Removed {len(mask) - mask.sum()} rows with jumps greater than {int(max_jump*100)}%.")

def save_summary_statistics(return_df, output_dir, filtered=False):
    """Save summary statistics (min, 25%, median, mean, 75%, max) for each timestep to an Excel sheet."""
    returns_data = return_df.drop(columns=['Ticker', 'Filing Date'])

    summary_stats = returns_data.describe(percentiles=[0.25, 0.5, 0.75]).T
    summary_stats = summary_stats[['min', '25%', '50%', 'mean', '75%', 'max']]*100

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'stock_returns_summary_stats.xlsx')
    sheet_name = 'Filtered' if filtered else 'Original'
    with pd.ExcelWriter(output_path, mode='w') as writer:
        summary_stats.to_excel(writer, sheet_name=sheet_name, index_label='Day')
    # print(f"Summary statistics saved to {output_path} in sheet {sheet_name}")
