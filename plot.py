#!/usr/bin/env python3
import os
import pandas as pd
from src.training.utils.train_helpers import plot_final_summary

def main():
    # Base stats folder where your .xlsx lives
    stats_dir   = os.path.abspath(os.path.join('data', 'training', 'stats'))
    # Where we’ll dump the plots
    summary_dir = os.path.join(stats_dir, 'summary')
    os.makedirs(summary_dir, exist_ok=True)

    # Choose your target category & metric
    category     = 'alpha'             # e.g. 'Revenue', 'EPS', etc.
    optimize_for = 'adjusted_sharpe'   # or 'sharpe_ratio', etc.

    # Grab all Excel files in stats_dir
    excels = [f for f in os.listdir(stats_dir) if f.lower().endswith('.xlsx')]
    if not excels:
        print(f"No .xlsx files found in {stats_dir}")
        return

    for fname in excels:
        path = os.path.join(stats_dir, fname)
        print(f"Loading results from {path} …")
        # read the first sheet by default
        df = pd.read_excel(path)
        # call your shared plotting helper
        plot_final_summary(df, summary_dir, category, optimize_for)
        print(f" ✅ Plotted summary for {fname}")

if __name__ == '__main__':
    main()
