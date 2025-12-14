"""
Create animated GIF of portfolio value growth over time (Test Set)
Shows HODL, DCA, and Quant strategies with progressive animation
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

from config import TRAINED_WEIGHTS, ROLLING_Z_WINDOW, INITIAL_CAPITAL, DCA_MONTHLY_AMOUNT, DCA_NUM_MONTHS
from strategies.quant_rf import load_btc_data, compute_factors, rolling_standardize, weights_to_positions, backtest

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_and_prepare_data():
    """Load test data and compute all three strategies"""
    
    # Load data
    df = pd.read_csv('data/bitcoin_test_2023_2024 copy.csv')
    df['date'] = pd.to_datetime(df['Start'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # HODL Strategy
    initial_btc = INITIAL_CAPITAL / df['Close'].iloc[0]
    hodl_portfolio = initial_btc * df['Close']
    
    # DCA Strategy
    dca_portfolio = []
    total_btc = 0
    total_invested = 0
    
    for i, row in df.iterrows():
        # Check if it's time for monthly purchase
        months_elapsed = (row['date'] - df['date'].iloc[0]).days // 30
        
        if months_elapsed < DCA_NUM_MONTHS and i > 0:
            prev_months = (df['date'].iloc[i-1] - df['date'].iloc[0]).days // 30
            if months_elapsed > prev_months:
                # Buy more BTC
                btc_bought = DCA_MONTHLY_AMOUNT / row['Close']
                total_btc += btc_bought
                total_invested += DCA_MONTHLY_AMOUNT
        
        # Portfolio value
        portfolio_value = total_btc * row['Close']
        dca_portfolio.append(portfolio_value)
    
    # Quant Strategy
    btc_df = load_btc_data(Path('data/bitcoin_test_2023_2024 copy.csv'))
    btc_df, factor_cols = compute_factors(btc_df)
    btc_df, z_cols = rolling_standardize(btc_df, factor_cols, window=ROLLING_Z_WINDOW)
    positions = weights_to_positions(btc_df, z_cols, TRAINED_WEIGHTS, max_position=1.0)
    results = backtest(btc_df, positions, tc_bps=15.0)
    
    # Convert quant equity curve to portfolio value
    quant_equity = results['equity_curve'].values
    quant_portfolio = INITIAL_CAPITAL * quant_equity
    
    # Align quant portfolio with original df
    quant_aligned = np.full(len(df), np.nan)
    valid_indices = results['df'].index
    for i, idx in enumerate(valid_indices):
        if idx < len(quant_aligned):
            quant_aligned[idx] = quant_portfolio[i]
    
    return df, hodl_portfolio.values, np.array(dca_portfolio), quant_aligned

def create_animated_gif():
    """Create animated GIF showing portfolio growth"""
    
    print("Loading test set data and computing strategies...")
    df, hodl_values, dca_values, quant_values = load_and_prepare_data()
    
    # Install imageio if needed
    try:
        import imageio
    except ImportError:
        print("Installing imageio for GIF creation...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio"])
        import imageio
    
    dates = df['date'].values
    n_frames = 80  # Fewer frames for shorter test period
    frame_step = max(1, len(df) // n_frames)
    
    frames = []
    
    print(f"Generating {n_frames} frames...")
    
    for frame_idx in range(n_frames):
        # Calculate how much data to show
        end_idx = min((frame_idx + 1) * frame_step, len(df))
        
        if end_idx < 2:
            continue
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot data up to current frame
        current_dates = dates[:end_idx]
        
        # HODL
        ax.plot(current_dates, hodl_values[:end_idx], 
                color='#2E86AB', linewidth=2.5, label='HODL', alpha=0.9)
        
        # DCA
        ax.plot(current_dates, dca_values[:end_idx], 
                color='#A23B72', linewidth=2.5, label='DCA', alpha=0.9)
        
        # Quant (only where data is valid)
        valid_mask = ~np.isnan(quant_values[:end_idx])
        if valid_mask.any():
            ax.plot(current_dates[valid_mask], quant_values[:end_idx][valid_mask], 
                    color='#F18F01', linewidth=2.5, label='Quant', alpha=0.9)
        
        # Formatting
        ax.set_yscale('log')
        ax.set_ylabel('Portfolio Value (USD)', fontsize=13, fontweight='bold')
        ax.set_xlabel('Date', fontsize=13, fontweight='bold')
        ax.set_title('Bitcoin Investment Strategies - Test Period (2023-2024)', 
                     fontsize=16, fontweight='bold', pad=20)
        
        ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add progress indicator
        progress = end_idx / len(df) * 100
        ax.text(0.98, 0.02, f'Progress: {progress:.1f}%', 
                transform=ax.transAxes, fontsize=11, 
                ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
        
        # Add final values if at end
        if end_idx == len(df):
            final_hodl = hodl_values[-1]
            final_dca = dca_values[-1]
            final_quant = quant_values[~np.isnan(quant_values)][-1] if (~np.isnan(quant_values)).any() else 0
            
            ax.text(0.02, 0.98, 
                    f'Final Values:\n'
                    f'HODL: ${final_hodl:,.0f}\n'
                    f'DCA: ${final_dca:,.0f}\n'
                    f'Quant: ${final_quant:,.0f}',
                    transform=ax.transAxes, fontsize=10,
                    va='top', ha='left',
                    bbox=dict(boxstyle='round,pad=0.7', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        
        # Convert plot to image
        fig.canvas.draw()
        image = np.array(fig.canvas.renderer.buffer_rgba())
        image = image[:, :, :3]  # Remove alpha channel
        frames.append(image)
        
        plt.close(fig)
        
        # Print progress
        if (frame_idx + 1) % 10 == 0:
            print(f"  Generated frame {frame_idx + 1}/{n_frames}")
    
    # Add pause frames at the end
    print("Adding pause frames at end...")
    for _ in range(20):
        frames.append(frames[-1])
    
    # Save as GIF
    output_path = 'visualization/portfolio_value_test_animated.gif'
    print(f"Saving GIF to {output_path}...")
    imageio.mimsave(output_path, frames, fps=10, loop=0)
    
    print(f"\n✅ Animated GIF created: {output_path}")
    print(f"   Total frames: {len(frames)}")
    print(f"   Duration: ~{len(frames)/10:.1f} seconds per loop")
    print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

if __name__ == '__main__':
    create_animated_gif()
