# Bitcoin Investment Strategies: HODL vs DCA vs Quantitative

比特币投资策略对比分析项目，实现并对比三种经典投资策略在2010-2024年的表现。

## 📊 策略概览

### 1. HODL策略 (Buy and Hold)
一次性投入$12,500，长期持有不操作。

### 2. DCA策略 (Dollar-Cost Averaging)
每月定投$1,000，共13个月（总投资$13,000）。

### 3. Quantitative策略
基于10个技术因子的量化交易系统，通过随机游走优化权重。

## 📈 核心结果

### 训练集 (2010-2020)
| 策略 | Sharpe Ratio | 最终价值 | 最大回撤 |
|------|-------------|---------|---------|
| HODL | 1.70 | $5.30B | -93.07% |
| DCA | 1.78 | $2.31B | -93.07% |
| Quant | 2.21 | $29.88B | - |

### 测试集 (2023-2024)
| 策略 | Sharpe Ratio | 收益率 | 最终价值 |
|------|-------------|--------|---------|
| DCA | 3.04 | +141% | $31,327.61 |
| Quant | -5.14 | -25.52% | $9,310.57 |

**关键发现**: 量化策略严重过拟合，测试集完全失效。

## 🗂️ 项目结构

```
.
├── hodl.py                          # HODL策略实现
├── dca.py                           # DCA策略实现（训练集）
├── dca_test.py                      # DCA策略测试
├── quant_rf.py                      # 量化策略（10因子+随机游走优化）
├── quant_test.py                    # 量化策略测试
├── bitcoin_train_2010_2020 copy.csv # 训练集数据
├── bitcoin_test_2023_2024 copy.csv  # 测试集数据
├── 策略对比报告.md                   # 完整对比分析报告
├── 量化策略技术总结.md               # 量化策略技术细节
└── DCA与HODL策略假设分析.md         # 基于代码的策略假设总结
```

## 🚀 快速开始

### 环境要求
- Python 3.8+
- pandas, numpy, scikit-learn

### 安装依赖
```bash
pip install pandas numpy scikit-learn
```

### 运行策略

```bash
# HODL策略
python hodl.py

# DCA策略（训练集）
python dca.py

# DCA策略（测试集）
python dca_test.py

# 量化策略（训练集）
python quant_rf.py

# 量化策略（测试集）
python quant_test.py
```

## 📝 文档说明

### 1. 策略对比报告.md
- 训练集与测试集详细表现对比
- 过拟合问题深度分析
- 投资建议与风险提示

### 2. 量化策略技术总结.md
- 10个技术因子详细说明
- 随机游走优化算法原理
- 权重经济含义解读
- 策略局限性与改进方向

### 3. DCA与HODL策略假设分析.md
- 基于代码实现提取的核心假设
- 策略参数设定与经济逻辑
- 共同假设与差异对比

## 🔬 量化策略技术细节

### 十大因子
1. **动量**: MOM_20, MOM_60
2. **均线价差**: MA_50_SPREAD, MA_200_SPREAD
3. **波动率**: VOL_20, ATR_PCT_14, VOL_RATIO_20
4. **价格位置**: PRICE_POS_60, CLOSE_POS
5. **市场周期**: POST_HALVING

### 优化方法
- 随机游走搜索（2000步迭代）
- 目标函数: 最大化Sharpe比率
- 滚动Z-score标准化（252天窗口）
- Sigmoid持仓映射

### 最终权重
```python
[1.70, -6.83, -1.58, 4.51, -0.69, 3.13, 2.07, 7.83, 3.29, -2.19]
```

## ⚠️ 风险提示

1. **过拟合风险**: 量化策略在测试集上Sharpe从2.21跌至-5.14
2. **历史数据局限**: 过往表现不代表未来收益
3. **市场风险**: 极端事件可能导致策略失效
4. **技术风险**: 区块链技术和监管环境变化

## 📊 核心指标

- **Sharpe Ratio**: 风险调整后收益
- **Sortino Ratio**: 下行风险调整收益
- **Max Drawdown**: 最大回撤
- **无风险利率**: 3% (年化)
- **资本成本**: 7% (HODL策略)

## 💡 核心结论

> 简单的DCA策略在长期表现和稳健性上显著优于过度优化的量化策略。

在比特币投资中：
- ✅ DCA策略稳健可靠，训练集和测试集均表现优异
- ⚠️ 量化策略需要严格的样本外验证，避免过拟合陷阱
- 📈 HODL适合风险承受力强且有大额资金的投资者

## 📚 参考资料

- Bitcoin Historical Price Data (2010-2024)
- Modern Portfolio Theory
- Technical Analysis Indicators
- Random Walk Optimization

## 🤝 贡献

欢迎提交Issue和Pull Request改进策略实现。

## 📄 许可

MIT License

---

**免责声明**: 本项目仅供学术研究和教育用途，不构成投资建议。投资有风险，入市需谨慎。
