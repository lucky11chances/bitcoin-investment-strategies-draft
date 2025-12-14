# HODL与DCA策略代码实现假设总结

## 策略参数设定

### HODL策略 (hodl.py)
```python
INITIAL_CAPITAL = 12_500.0        # 一次性投入$12,500
RISK_FREE_ANNUAL = 0.03           # 3%年化无风险利率（用于Sharpe/Sortino）
CAPITAL_COST_ANNUAL = 0.07        # 7%年化资本成本（机会成本）
DAYS_PER_YEAR = 365
```

### DCA策略 (dca.py)
```python
MONTHLY_CONTRIBUTION = 1_000.0    # 每月定投$1,000
TARGET_TOTAL = 12_500.0           # 总投资目标$12,500（即13个月）
RISK_FREE_ANNUAL = 0.03           # 3%年化无风险利率
DAYS_PER_YEAR = 365
```

---

## 一、HODL策略核心假设（基于代码实现）

### 假设1: 择时无效，一次性全仓买入最优
**代码体现**:
```python
buy_price = close.iloc[0]        # 在数据集第一天买入
btc_held = INITIAL_CAPITAL / buy_price  # 全部资金转换为BTC
```
- 不分批买入，不等待时机
- 假设第一天买入与任何其他时间买入无差异（长期看）

### 假设2: 持有期间零操作
**代码体现**:
```python
portfolio = close * btc_held     # 每日仅计算市值
daily_ret = portfolio.pct_change().dropna()  # 不主动交易
```
- BTC数量恒定不变（`btc_held`固定）
- 无止损、无止盈、无再平衡

### 假设3: 风险调整后收益的重要性
**代码体现**:
```python
rf_daily = (1.0 + RISK_FREE_ANNUAL) ** (1.0 / DAYS_PER_YEAR) - 1.0
excess_ret = daily_ret - rf_daily
sharpe = (excess_ret.mean() / vol) * math.sqrt(DAYS_PER_YEAR)
```
- 使用3%无风险利率计算超额收益
- 假设投资者关心"单位风险的超额回报"，而非绝对收益

### 假设4: 下行风险与上行波动不对称
**代码体现**:
```python
sortino = np.nan
downside = excess_ret[excess_ret < 0]  # 只计算负收益
down_vol = downside.std(ddof=1)
sortino = (excess_ret.mean() / down_vol) * math.sqrt(DAYS_PER_YEAR)
```
- 使用Sortino而非仅Sharpe，说明更关注下行风险
- 假设投资者对亏损的厌恶高于对盈利的偏好

### 假设5: 机会成本需要量化
**代码体现**:
```python
capital_cost = INITIAL_CAPITAL * (1.0 + CAPITAL_COST_ANNUAL) ** duration_years
net_profit_after_cost = final_value - capital_cost
```
- 假设资金有7%的年化机会成本（可能是借款利率或其他投资收益）
- 最终利润需扣除这笔隐性成本

### 假设6: 极端回撤可承受
**代码体现**:
```python
running_max = portfolio.cummax()
drawdown = portfolio / running_max - 1.0
max_drawdown = float(drawdown.min())  # 记录最大回撤但不触发任何操作
```
- 计算最大回撤仅用于事后统计，不设止损线
- 假设即使亏损90%+仍能坚持持有

---

## 二、DCA策略核心假设（基于代码实现）

### 假设1: 定期定额优于一次性投入
**代码体现**:
```python
monthly_first = df.resample("MS", on="Date").first().reset_index()
monthly_first["btc_bought"] = MONTHLY_CONTRIBUTION / monthly_first["Close"]
```
- 每月第一个交易日买入固定$1,000
- 假设分散买入时间能降低择时风险

### 假设2: 价格波动自动实现低买高卖
**代码体现**:
```python
btc_bought = MONTHLY_CONTRIBUTION / monthly_first["Close"]
```
- 固定美元金额 ÷ 当期价格 = 买入BTC数量
- **价格低时**：分母小 → 买入BTC多（自动抄底）
- **价格高时**：分母大 → 买入BTC少（自动减仓）
- 无需主观判断，价格自动调节仓位

### 假设3: 投资周期可预设
**代码体现**:
```python
max_months = int(np.ceil(TARGET_TOTAL / MONTHLY_CONTRIBUTION))  # 13个月
monthly_first = monthly_first.head(max_months)  # 精确执行13次
```
- 投资$12,500后停止，与HODL总投入相同
- 假设投资者有明确的资金规划和纪律

### 假设4: 初始零收益期需排除
**代码体现**:
```python
invested = merged[merged["cum_btc"] > 0].copy()  # 只在持有BTC后计算
daily_ret = invested["portfolio"].pct_change().dropna()
```
- 第一次买入前收益率为0，pct_change会生成inf/nan
- 假设收益计算应从有持仓开始

### 假设5: 累积持仓逐步增加
**代码体现**:
```python
monthly_first["cum_btc"] = monthly_first["btc_bought"].cumsum()
merged["cum_btc"] = merged["cum_btc"].ffill().fillna(0.0)  # 前向填充
```
- 每次买入的BTC永久持有，不卖出
- 持仓量随时间单调递增

### 假设6: 风险评估与HODL一致
**代码体现**:
```python
rf_daily = (1.0 + RISK_FREE_ANNUAL) ** (1.0 / DAYS_PER_YEAR) - 1.0
sharpe = (excess_ret.mean() / vol) * math.sqrt(DAYS_PER_YEAR)
sortino = (excess_ret.mean() / down_vol) * math.sqrt(DAYS_PER_YEAR)
```
- 使用相同的3%无风险利率
- 相同的Sharpe/Sortino计算逻辑
- 假设两种策略在同一风险框架下比较

---

## 三、两种策略的共同假设

### 1. 数据完整性假设
```python
# 两者都假设CSV文件存在且格式正确
df = pd.read_csv(path)
df[date_col] = pd.to_datetime(df[date_col])
```
- 日期列可自动识别（Date或Start）
- Close价格连续无缺失
- 数据按时间升序排列

### 2. 复利计算假设
```python
# 两者都将年化利率转换为日频
rf_daily = (1.0 + RISK_FREE_ANNUAL) ** (1.0 / 365) - 1.0
```
- 假设无风险利率按复利计息，而非简单利息
- 一年365天固定（不考虑闰年）

### 3. 收益率标准化假设
```python
# 两者都年化Sharpe/Sortino
sharpe = (excess_ret.mean() / vol) * math.sqrt(365)
```
- 假设日收益率可直接乘 √365 年化
- 隐含假设收益率服从IID（独立同分布）

### 4. 交易成本为零假设
```python
# 两个策略都没有扣除交易费用
btc_held = INITIAL_CAPITAL / buy_price  # HODL
btc_bought = MONTHLY_CONTRIBUTION / Close  # DCA
```
- 无手续费、滑点、税费
- 假设所有资金100%转换为BTC

### 5. 流动性无限假设
```python
# 两者都假设任何价格都能买入
portfolio = close * btc_held
```
- 市场深度足够，不会因买单推高价格
- 任何时间点都能以收盘价成交

---

## 四、策略差异的假设对比

| 维度 | HODL假设 | DCA假设 |
|------|---------|---------|
| **资金分配** | 假设有一次性大额资金可投入 | 假设有稳定月度现金流 |
| **择时能力** | 假设第一天买入等价于任何时点 | 假设无法择时，用时间分散风险 |
| **风险承受** | 假设可承受-93%最大回撤 | 假设分批买入可降低心理压力 |
| **操作纪律** | 假设买入后完全不操作 | 假设能严格执行13次定投 |
| **成本考量** | 考虑7%资本成本（机会成本） | 不计算资本成本，仅计算收益 |

---

## 五、代码中隐含的市场观

### HODL隐含观点
1. **长期持有优于频繁交易**（否则应设止损/止盈）
2. **波动是噪音，趋势是信号**（否则应回避高波动期）
3. **时间是最好的朋友**（持有时间越长越好）

### DCA隐含观点
1. **价格短期不可预测**（否则应集中在低点买入）
2. **平均成本法有效**（否则应一次性投入）
3. **纪律执行优于主观判断**（机械化执行月度买入）

---

## 总结

这两个策略的代码实现揭示了它们的核心假设：

**HODL = "我相信长期趋势向上，短期波动无关紧要，我有足够资金和耐心"**

**DCA = "我无法预测价格，但我有稳定现金流，用时间分散风险最安全"**

两者都假设：
- 比特币长期升值
- 主动择时无效
- 简单执行优于复杂策略
- 交易成本可忽略
- 市场流动性充足

差异在于：
- HODL假设投资者有一次性资金且风险承受力极强
- DCA假设投资者更关注心理舒适度和风险平滑

---

**数据来源**: hodl.py 和 dca.py 代码实现  
**文档日期**: 2025年12月12日
