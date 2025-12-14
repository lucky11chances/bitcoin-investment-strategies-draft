"""
Plot factor weights pie chart for quantitative strategy
Visualize the relative importance of 10 technical factors
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import TRAINED_WEIGHTS

# Set font
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# Factor names (English)
FACTOR_NAMES = [
    'MOM_20\n(20-day Momentum)',
    'MOM_60\n(60-day Momentum)',
    'MA_50_SPREAD\n(50-day MA Spread)',
    'MA_200_SPREAD\n(200-day MA Spread)',
    'VOL_20\n(20-day Volatility)',
    'ATR_PCT_14\n(14-day ATR)',
    'VOL_RATIO_20\n(Volume Ratio)',
    'PRICE_POS_60\n(60-day Price Position)',
    'CLOSE_POS\n(Intraday Close Position)',
    'RSI_14\n(14-day RSI)'
]

def plot_factor_weights():
    """Plot factor weights pie chart"""
    
    # Use absolute values to represent importance
    weights = np.array(TRAINED_WEIGHTS)
    abs_weights = np.abs(weights)
    
    # Calculate percentages
    total = abs_weights.sum()
    percentages = (abs_weights / total) * 100
    
    # Create color mapping: blue for positive weights, red for negative weights
    colors = []
    for w in weights:
        if w > 0:
            # Blue (bullish signal)
            colors.append('#4A90E2')
        else:
            # Red (bearish signal)
            colors.append('#E74C3C')
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left plot: Pie chart (by absolute value)
    wedges, texts, autotexts = ax1.pie(
        abs_weights,
        labels=FACTOR_NAMES,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 9},
        wedgeprops={'edgecolor': 'black', 'linewidth': 2.5}
    )
    
    # Beautify percentage text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(8)
    
    ax1.set_title('Factor Weight Distribution (by Absolute Value)\nBlue=Bullish Signal | Red=Bearish Signal', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # Right plot: Bar chart (showing positive/negative)
    x_pos = np.arange(len(FACTOR_NAMES))
    bar_colors = ['#4A90E2' if w > 0 else '#E74C3C' for w in weights]
    
    bars = ax2.barh(x_pos, weights, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels([name.replace('\n', ' ') for name in FACTOR_NAMES], fontsize=9)
    ax2.set_xlabel('Weight Value', fontsize=11, fontweight='bold')
    ax2.set_title('Factor Weight Bar Chart (with Direction)', fontsize=14, fontweight='bold', pad=20)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, weight) in enumerate(zip(bars, weights)):
        width = bar.get_width()
        label_x = width + 0.05 if width > 0 else width - 0.05
        ha = 'left' if width > 0 else 'right'
        ax2.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'{weight:.2f}',
                ha=ha, va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'visualization/factor_weights_en.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Factor weights chart saved: {output_path}")
    
    # Print weight statistics
    print("\n" + "="*60)
    print("Factor Weight Statistics")
    print("="*60)
    print(f"{'Factor Name':<30} {'Weight':>10} {'Abs Value':>10} {'Ratio':>10}")
    print("-"*60)
    
    for name, weight, abs_w, pct in zip(FACTOR_NAMES, weights, abs_weights, percentages):
        clean_name = name.replace('\n', ' ')
        print(f"{clean_name:<30} {weight:>10.3f} {abs_w:>10.3f} {pct:>9.1f}%")
    
    print("-"*60)
    print(f"{'Total Absolute Weight':<30} {'':<10} {total:>10.3f} {'100.0%':>10}")
    print("="*60)
    
    # Print key insights
    print("\n📊 Key Insights:")
    max_idx = abs_weights.argmax()
    print(f"   • Most important factor: {FACTOR_NAMES[max_idx].replace(chr(10), ' ')} ({percentages[max_idx]:.1f}%)")
    
    positive_count = (weights > 0).sum()
    negative_count = (weights < 0).sum()
    print(f"   • Bullish factors: {positive_count} | Bearish factors: {negative_count}")
    
    top3_idx = abs_weights.argsort()[-3:][::-1]
    print(f"   • Top 3 important factors:")
    for i, idx in enumerate(top3_idx, 1):
        direction = "Bullish" if weights[idx] > 0 else "Bearish"
        print(f"     {i}. {FACTOR_NAMES[idx].replace(chr(10), ' ')} ({direction}, {percentages[idx]:.1f}%)")

if __name__ == '__main__':
    plot_factor_weights()
