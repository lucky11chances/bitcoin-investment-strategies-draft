# Bitcoin Investment Strategies: HODL vs DCA vs Quantitative

比特币投资策略对比分析项目，实现并对比三种经典投资策略在2010-2024年的表现。

## 🚀 快速开始

```bash
# 安装依赖
pip install pandas numpy scikit-learn

# 运行完整分析
python run.py
```

## 📊 三大策略

1. **HODL** (Buy and Hold) - 一次性投入$12,500，长期持有
2. **DCA** (Dollar-Cost Averaging) - 每月定投$1,000，共13个月
3. **Quantitative** - 基于10个技术因子的量化交易系统

## 📈 核心结果

### 测试集表现 (2023-2024)
| 策略 | Sharpe Ratio | 最终价值 | 表现 |
|------|-------------|---------|------|
| **DCA** | **3.04** | $31,328 | ✅ 最佳 |
| **HODL** | **3.26** | $46,009 | ✅ 最高收益 |
| Quant | 1.08 | $18,156 | ⚠️ 过拟合 |

### 关键发现
- ✅ **DCA策略**：稳健可靠，风险调整后收益优异
- ✅ **HODL策略**：绝对收益最高，Sharpe比率也很强
- ⚠️ **量化策略**：训练集Sharpe 1.98 → 测试集1.08，过拟合明显
- 📊 **高频交易成本**：量化策略年均280+次交易，换手率40,000%+

## 📊 可视化展示

项目包含多种专业可视化：

### 动态展示
- 📈 **Portfolio增长动画** (GIF)
  - 训练集 (2010-2020) 和测试集 (2023-2024)
  - 逐步展示三种策略的价值变化
  - 循环播放，直观对比

### 静态图表
- 📊 **Portfolio价值曲线** - 训练集和测试集对比
- 📍 **仓位变化时间序列** - 量化策略动态调仓可视化  
- 📈 **累积交易次数** - 展示交易频率和换手率
- 🥧 **因子权重分布** (中英双语) - 10个技术因子重要性

所有可视化文件位于 `visualization/` 文件夹

## 🗂️ 项目结构

```
bitcoin-investment-strategies/
├── src/                    # 源代码
│   ├── strategies/        # 策略实现
│   ├── config.py         # 全局配置
│   ├── metrics.py        # 性能指标
│   ├── utils.py          # 工具函数
│   └── main.py           # 主程序
├── data/                  # 数据文件
├── docs/                  # 详细文档
├── visualization/         # 可视化图表和脚本
│   ├── *.png             # 静态图表
│   ├── *.gif             # 动态GIF
│   └── plot_*.py         # 绘图脚本
└── run.py                # 入口点
```

## 📚 详细文档

完整的技术文档和分析报告请查看 [`docs/README.md`](docs/README.md)

包含：
- 策略详细实现说明
- 完整的性能对比分析
- 量化策略技术细节
- 代码架构设计文档

## 🛠️ 技术栈

- **Python 3.13**
- **pandas** - 数据处理
- **numpy** - 数值计算
- **matplotlib** - 数据可视化
- **imageio** - GIF动画生成

## 📄 许可证

MIT License

---

**项目作者**: lucky11chances  
**GitHub**: [bitcoin-investment-strategies](https://github.com/lucky11chances/bitcoin-investment-strategies)
