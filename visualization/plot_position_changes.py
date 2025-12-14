"""
Plot quantitative strategy position changes over time
Visualize dynamic position adjustments (0-100%) and compare with Bitcoin price
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from config import TRAINED_WEIGHTS, ROLLING_Z_WINDOW
from strategies.quant_rf import load_btc_data, compute_factors, rolling_standardize, weights_to_positions, backtest

# Set font
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def run_backtest(csv_path, weights):
    """Run backtest and return results with positions"""
    df = load_btc_data(Path(csv_path))
    df, factor_cols = compute_factors(df)
    df, z_cols = rolling_standardize(df, factor_cols, window=ROLLING_Z_WINDOW)
    positions = weights_to_positions(df, z_cols, weights, max_position=1.0)
    results = backtest(df, positions, tc_bps=15.0)
    results['positions'] = positions
    return results, df

def plot_position_changes():
    """Plot position changes over time with Bitcoin price"""
    
    # Run backtests
    print("Running training set backtest...")
    train_results, train_df = run_backtest('data/bitcoin_train_2010_2020 copy.csv', TRAINED_WEIGHTS)
    
    print("Running test set backtest...")
    test_results, test_df = run_backtest('data/bitcoin_test_2023_2024 copy.csv', TRAINED_WEIGHTS)
    
    # Create figure with 2 rows (training and test)
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot training set
    plot_single_period(axes[0], train_df, train_results, 'Training Period (2010-2020)')
    
    # Plot test set
    plot_single_period(axes[1], test_df, test_results, 'Test Period (2023-2024)')
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'visualization/position_changes.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Position changes chart saved: {output_path}")
    
    # Print statistics
    print_position_statistics(train_results, test_results)

def plot_single_period(ax, price_df, results, title):
    """Plot position changes for a single period"""
    
    # Create twin axis for position
    ax2 = ax.twinx()
    
    # Get positions series (aligned with dataframe)
    positions_series = results['positions']
    valid_positions = positions_series.dropna()
    
    # Plot Bitcoin price (left y-axis)
    ax.plot(price_df.index, price_df['Close'], color='#2C3E50', linewidth=1.5, 
            label='Bitcoin Price', alpha=0.7)
    ax.set_ylabel('Bitcoin Price (USD)', fontsize=11, fontweight='bold', color='#2C3E50')
    ax.tick_params(axis='y', labelcolor='#2C3E50')
    ax.set_yscale('log')  # Use log scale for price
    ax.grid(True, alpha=0.2)
    
    # Plot position (right y-axis)
    positions_pct = valid_positions * 100  # Convert to percentage
    
    # Fill area under position curve
    ax2.fill_between(valid_positions.index, 0, positions_pct, alpha=0.3, color='#3498DB', label='BTC Position')
    ax2.plot(valid_positions.index, positions_pct, color='#3498DB', linewidth=2, label='BTC Position (%)')
    
    ax2.set_ylabel('BTC Position (%)', fontsize=11, fontweight='bold', color='#3498DB')
    ax2.set_ylim(-5, 105)
    ax2.tick_params(axis='y', labelcolor='#3498DB')
    ax2.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50% Level')
    
    # Set title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Add legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)
    
    # Format x-axis
    ax.tick_params(axis='x', rotation=45)

def print_position_statistics(train_results, test_results):
    """Print position change statistics"""
    
    print("\n" + "="*70)
    print("Position Change Statistics")
    print("="*70)
    
    for period_name, results in [("Training (2010-2020)", train_results), 
                                   ("Test (2023-2024)", test_results)]:
        positions = np.array(results['positions'])
        position_pct = positions * 100
        
        # Calculate position changes
        position_changes = np.diff(positions)
        num_changes = np.sum(np.abs(position_changes) > 0.01)  # Threshold: 1%
        
        print(f"\n📊 {period_name}:")
        print(f"   • Average Position: {position_pct.mean():.1f}%")
        print(f"   • Max Position: {position_pct.max():.1f}%")
        print(f"   • Min Position: {position_pct.min():.1f}%")
        print(f"   • Std Dev: {position_pct.std():.1f}%")
        print(f"   • Days with >50% position: {np.sum(positions > 0.5)} ({np.sum(positions > 0.5)/len(positions)*100:.1f}%)")
        print(f"   • Days with <50% position: {np.sum(positions < 0.5)} ({np.sum(positions < 0.5)/len(positions)*100:.1f}%)")
        print(f"   • Position changes (>1%): {num_changes}")
        print(f"   • Total trading days: {len(positions)}")
    
    print("="*70)

if __name__ == '__main__':
    plot_position_changes()
