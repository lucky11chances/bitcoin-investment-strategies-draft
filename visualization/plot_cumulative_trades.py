"""
Plot cumulative trade count over time
Visualize trading frequency and turnover rate
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

def plot_cumulative_trades():
    """Plot cumulative trade count over time"""
    
    # Run backtests
    print("Running training set backtest...")
    train_results, train_df = run_backtest('data/bitcoin_train_2010_2020 copy.csv', TRAINED_WEIGHTS)
    
    print("Running test set backtest...")
    test_results, test_df = run_backtest('data/bitcoin_test_2023_2024 copy.csv', TRAINED_WEIGHTS)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot training period
    plot_single_period(ax1, train_df, train_results, 'Training Period (2010-2020)')
    
    # Plot test period
    plot_single_period(ax2, test_df, test_results, 'Test Period (2023-2024)')
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'visualization/cumulative_trades.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Cumulative trades chart saved: {output_path}")
    
    # Print statistics
    print_trade_statistics(train_df, train_results, test_df, test_results)

def plot_single_period(ax, price_df, results, title):
    """Plot cumulative trades for a single period"""
    
    # Get positions series (aligned with dataframe)
    positions_series = results['positions']
    valid_positions = positions_series.dropna()
    positions = valid_positions.values
    
    # Calculate trades (any position change > 1%)
    position_changes = np.abs(np.diff(positions))
    trades = position_changes > 0.01
    
    # Calculate cumulative trades
    cumulative_trades = np.zeros(len(positions))
    cumulative_trades[1:] = np.cumsum(trades)
    
    # Create twin axis
    ax2 = ax.twinx()
    
    # Plot cumulative trades (left y-axis)
    ax.plot(valid_positions.index, cumulative_trades, color='#E74C3C', linewidth=2.5, 
            label='Cumulative Trades', marker='o', markersize=2, 
            markevery=max(1, len(cumulative_trades)//50))
    ax.fill_between(valid_positions.index, 0, cumulative_trades, alpha=0.2, color='#E74C3C')
    
    ax.set_ylabel('Cumulative Trade Count', fontsize=11, fontweight='bold', color='#E74C3C')
    ax.tick_params(axis='y', labelcolor='#E74C3C')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot position on right y-axis (for context)
    position_pct = positions * 100
    ax2.plot(valid_positions.index, position_pct, color='#3498DB', linewidth=1, alpha=0.4, 
             linestyle='--', label='BTC Position (%)')
    ax2.set_ylabel('BTC Position (%)', fontsize=11, fontweight='bold', color='#3498DB')
    ax2.set_ylim(-5, 105)
    ax2.tick_params(axis='y', labelcolor='#3498DB')
    ax2.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.3)
    
    # Set title
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Add legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    
    # Format x-axis
    ax.tick_params(axis='x', rotation=45)
    
    # Add annotation for final trade count
    final_trades = int(cumulative_trades[-1])
    last_idx = valid_positions.index[-1]
    ax.annotate(f'Total: {final_trades} trades', 
                xy=(last_idx, cumulative_trades[-1]),
                xytext=(-80, 20), textcoords='offset points',
                fontsize=10, fontweight='bold', color='#E74C3C',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#E74C3C', linewidth=2),
                arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5))

def print_trade_statistics(train_df, train_results, test_df, test_results):
    """Print trading statistics"""
    
    print("\n" + "="*70)
    print("Trading Frequency Statistics")
    print("="*70)
    
    for period_name, df, results in [("Training (2010-2020)", train_df, train_results), 
                                      ("Test (2023-2024)", test_df, test_results)]:
        positions_series = results['positions']
        valid_positions = positions_series.dropna()
        positions = valid_positions.values
        
        # Calculate trades
        position_changes = np.abs(np.diff(positions))
        trades = position_changes > 0.01
        num_trades = np.sum(trades)
        
        # Calculate time period
        total_days = len(df)
        trading_days = len(positions)
        
        # Calculate years from index
        if hasattr(df.index, 'to_pydatetime'):
            years = (df.index[-1] - df.index[0]).days / 365.25
        else:
            # Fallback if index is not datetime
            years = trading_days / 365.25
        
        # Calculate turnover
        total_turnover = np.sum(position_changes)
        
        print(f"\n📊 {period_name}:")
        print(f"   • Total trades: {num_trades}")
        print(f"   • Trading days: {trading_days}")
        print(f"   • Time period: {years:.2f} years")
        print(f"   • Trades per year: {num_trades/years:.1f}")
        print(f"   • Average days between trades: {trading_days/max(num_trades, 1):.1f}")
        print(f"   • Total turnover: {total_turnover*100:.1f}% (position changes)")
        print(f"   • Trading frequency: {num_trades/trading_days*100:.2f}% (trades/days)")
        
        # Analyze trade sizes
        trade_sizes = position_changes[trades] * 100
        if len(trade_sizes) > 0:
            print(f"   • Average trade size: {trade_sizes.mean():.1f}%")
            print(f"   • Max trade size: {trade_sizes.max():.1f}%")
            print(f"   • Min trade size: {trade_sizes.min():.1f}%")
    
    print("="*70)

if __name__ == '__main__':
    plot_cumulative_trades()
