#!/usr/bin/env python3
"""
Portfolio Value Visualization
Generate portfolio value curves for HODL, DCA, and Quantitative strategies
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from config import TRAIN_DATA_PATH, TEST_DATA_PATH, INITIAL_CAPITAL, DCA_MONTHLY_AMOUNT, TRAINED_WEIGHTS


def load_and_prepare_data(file_path):
    """Load and prepare Bitcoin data"""
    df = pd.read_csv(file_path)
    
    # Rename columns to lowercase for consistency
    df.columns = df.columns.str.lower()
    
    # Convert date column
    if 'start' in df.columns:
        df['date'] = pd.to_datetime(df['start'])
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    else:
        raise ValueError("No date column found in data")
    
    df = df.sort_values('date').reset_index(drop=True)
    
    return df

# Set Chinese font support
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def compute_hodl_portfolio(df, initial_capital):
    """Compute HODL portfolio value over time"""
    initial_price = df['close'].iloc[0]
    btc_held = initial_capital / initial_price
    portfolio_values = df['close'] * btc_held
    return portfolio_values


def compute_dca_portfolio(df, monthly_contribution, num_months=13):
    """Compute DCA portfolio value over time"""
    portfolio_values = []
    total_btc = 0
    contribution_count = 0
    
    for idx, row in df.iterrows():
        # Add monthly contribution (first day of each month)
        if contribution_count < num_months:
            if idx == 0 or df.loc[idx, 'date'].month != df.loc[idx-1, 'date'].month:
                total_btc += monthly_contribution / row['close']
                contribution_count += 1
        
        # Calculate current portfolio value
        portfolio_value = total_btc * row['close']
        portfolio_values.append(portfolio_value)
    
    return pd.Series(portfolio_values, index=df.index)


def compute_quant_portfolio(df, weights, initial_capital):
    """Compute Quantitative strategy portfolio value over time"""
    # Calculate factors
    df['ret_1d'] = df['close'].pct_change(1)
    df['ret_7d'] = df['close'].pct_change(7)
    df['ret_30d'] = df['close'].pct_change(30)
    df['vol_7d'] = df['ret_1d'].rolling(7).std()
    df['vol_30d'] = df['ret_1d'].rolling(30).std()
    df['mom_14d'] = df['close'].pct_change(14)
    df['rsi_14'] = compute_rsi(df['close'], 14)
    df['ma_ratio'] = df['close'] / df['close'].rolling(50).mean()
    df['volume_ma'] = df['volume'] / df['volume'].rolling(20).mean()
    df['price_range'] = (df['high'] - df['low']) / df['close']
    
    # Create factor matrix
    factors = df[['ret_7d', 'ret_30d', 'vol_7d', 'vol_30d', 'mom_14d', 
                   'rsi_14', 'ma_ratio', 'volume_ma', 'price_range']].copy()
    
    # Handle missing values
    factors = factors.fillna(0)
    
    # Normalize factors
    for col in factors.columns:
        mean = factors[col].mean()
        std = factors[col].std()
        if std > 0:
            factors[col] = (factors[col] - mean) / std
    
    # Add constant term
    factors.insert(0, 'const', 1.0)
    
    # Calculate scores and positions
    scores = factors.values @ weights
    positions = 1.0 / (1.0 + np.exp(-scores))
    positions = np.clip(positions, 0, 1)
    
    # Simulate portfolio
    portfolio_values = [initial_capital]
    cash = initial_capital
    btc_held = 0
    
    for i in range(1, len(df)):
        target_position = positions[i]
        current_price = df['close'].iloc[i]
        
        # Current portfolio value
        portfolio_value = cash + btc_held * current_price
        
        # Rebalance
        target_btc = (portfolio_value * target_position) / current_price
        btc_diff = target_btc - btc_held
        
        # Execute trade
        cash -= btc_diff * current_price
        btc_held = target_btc
        
        # Record portfolio value
        portfolio_value = cash + btc_held * current_price
        portfolio_values.append(portfolio_value)
    
    return pd.Series(portfolio_values, index=df.index)


def compute_rsi(prices, period=14):
    """Compute RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def plot_portfolio_comparison(df, dataset_name, output_path):
    """Plot portfolio value comparison for three strategies"""
    
    # Compute portfolio values
    print(f"Computing {dataset_name} portfolio values...")
    hodl_portfolio = compute_hodl_portfolio(df, INITIAL_CAPITAL)
    dca_portfolio = compute_dca_portfolio(df, DCA_MONTHLY_AMOUNT)
    quant_portfolio = compute_quant_portfolio(df, TRAINED_WEIGHTS, INITIAL_CAPITAL)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot three strategies
    ax.plot(df['date'], hodl_portfolio, label='HODL Strategy', linewidth=2.5, color='#2E86AB')
    ax.plot(df['date'], dca_portfolio, label='DCA Strategy', linewidth=2.5, color='#A23B72')
    ax.plot(df['date'], quant_portfolio, label='Quantitative Strategy', linewidth=2.5, color='#F18F01')
    
    # Formatting
    ax.set_xlabel('Date', fontsize=14, fontweight='bold')
    ax.set_ylabel('Portfolio Value ($)', fontsize=14, fontweight='bold')
    ax.set_title(f'Portfolio Value Comparison - {dataset_name}', fontsize=16, fontweight='bold', pad=20)
    
    # Format y-axis with currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {output_path}")
    
    # Display final values
    print(f"\n{dataset_name} Final Portfolio Values:")
    print(f"  HODL:        ${hodl_portfolio.iloc[-1]:,.2f}")
    print(f"  DCA:         ${dca_portfolio.iloc[-1]:,.2f}")
    print(f"  Quantitative: ${quant_portfolio.iloc[-1]:,.2f}")
    print()
    
    plt.close()


def main():
    """Main function to generate all plots"""
    print("=" * 80)
    print("Portfolio Value Visualization".center(80))
    print("=" * 80)
    print()
    
    # Create output directory
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading training data...")
    train_df = load_and_prepare_data(TRAIN_DATA_PATH)
    
    print("Loading test data...")
    test_df = load_and_prepare_data(TEST_DATA_PATH)
    
    # Generate plots
    print("\n" + "=" * 80)
    print("Generating Training Set Plot".center(80))
    print("=" * 80)
    plot_portfolio_comparison(
        train_df,
        "Training Set (2010-2020)",
        output_dir / "portfolio_value_training.png"
    )
    
    print("=" * 80)
    print("Generating Test Set Plot".center(80))
    print("=" * 80)
    plot_portfolio_comparison(
        test_df,
        "Test Set (2023-2024)",
        output_dir / "portfolio_value_test.png"
    )
    
    print("=" * 80)
    print("✅ All plots generated successfully!".center(80))
    print("=" * 80)


if __name__ == "__main__":
    main()
