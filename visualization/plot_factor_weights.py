"""
绘制量化策略因子权重饼状图
展示10个技术因子的相对重要性
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import TRAINED_WEIGHTS

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 因子名称（中文）
FACTOR_NAMES = [
    'MOM_20\n(20日动量)',
    'MOM_60\n(60日动量)',
    'MA_50_SPREAD\n(50日均线价差)',
    'MA_200_SPREAD\n(200日均线价差)',
    'VOL_20\n(20日波动率)',
    'ATR_PCT_14\n(14日ATR)',
    'VOL_RATIO_20\n(成交量比率)',
    'PRICE_POS_60\n(60日价格位置)',
    'CLOSE_POS\n(日内收盘位置)',
    'RSI_14\n(14日RSI)'
]

def plot_factor_weights():
    """绘制因子权重饼状图"""
    
    # 使用权重的绝对值来表示重要性
    weights = np.array(TRAINED_WEIGHTS)
    abs_weights = np.abs(weights)
    
    # 计算百分比
    total = abs_weights.sum()
    percentages = (abs_weights / total) * 100
    
    # 创建颜色映射：正权重用蓝色系，负权重用红色系
    colors = []
    for w in weights:
        if w > 0:
            # 蓝色系（看多信号）
            colors.append('#4A90E2')
        else:
            # 红色系（看空信号）
            colors.append('#E74C3C')
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：饼状图（按绝对值）
    wedges, texts, autotexts = ax1.pie(
        abs_weights,
        labels=FACTOR_NAMES,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 9},
        wedgeprops={'edgecolor': 'black', 'linewidth': 2.5}
    )
    
    # 美化百分比文字
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(8)
    
    ax1.set_title('因子权重分布（按绝对值）\n蓝色=看多信号 | 红色=看空信号', 
                  fontsize=14, fontweight='bold', pad=20)
    
    # 右图：柱状图（显示正负）
    x_pos = np.arange(len(FACTOR_NAMES))
    bar_colors = ['#4A90E2' if w > 0 else '#E74C3C' for w in weights]
    
    bars = ax2.barh(x_pos, weights, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    ax2.set_yticks(x_pos)
    ax2.set_yticklabels([name.replace('\n', ' ') for name in FACTOR_NAMES], fontsize=9)
    ax2.set_xlabel('权重值', fontsize=11, fontweight='bold')
    ax2.set_title('因子权重柱状图（含正负方向）', fontsize=14, fontweight='bold', pad=20)
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    
    # 在柱子上标注具体数值
    for i, (bar, weight) in enumerate(zip(bars, weights)):
        width = bar.get_width()
        label_x = width + 0.05 if width > 0 else width - 0.05
        ha = 'left' if width > 0 else 'right'
        ax2.text(label_x, bar.get_y() + bar.get_height()/2, 
                f'{weight:.2f}',
                ha=ha, va='center', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图片
    output_path = 'visualization/factor_weights.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 因子权重图已保存: {output_path}")
    
    # 打印权重统计
    print("\n" + "="*60)
    print("因子权重统计")
    print("="*60)
    print(f"{'因子名称':<30} {'权重':>10} {'绝对值':>10} {'占比':>10}")
    print("-"*60)
    
    for name, weight, abs_w, pct in zip(FACTOR_NAMES, weights, abs_weights, percentages):
        clean_name = name.replace('\n', ' ')
        print(f"{clean_name:<30} {weight:>10.3f} {abs_w:>10.3f} {pct:>9.1f}%")
    
    print("-"*60)
    print(f"{'总权重绝对值和':<30} {'':<10} {total:>10.3f} {'100.0%':>10}")
    print("="*60)
    
    # 打印关键见解
    print("\n📊 关键见解:")
    max_idx = abs_weights.argmax()
    print(f"   • 最重要因子: {FACTOR_NAMES[max_idx].replace(chr(10), ' ')} (占比 {percentages[max_idx]:.1f}%)")
    
    positive_count = (weights > 0).sum()
    negative_count = (weights < 0).sum()
    print(f"   • 看多信号因子: {positive_count}个 | 看空信号因子: {negative_count}个")
    
    top3_idx = abs_weights.argsort()[-3:][::-1]
    print(f"   • 前三重要因子:")
    for i, idx in enumerate(top3_idx, 1):
        direction = "看多" if weights[idx] > 0 else "看空"
        print(f"     {i}. {FACTOR_NAMES[idx].replace(chr(10), ' ')} ({direction}, {percentages[idx]:.1f}%)")

if __name__ == '__main__':
    plot_factor_weights()
