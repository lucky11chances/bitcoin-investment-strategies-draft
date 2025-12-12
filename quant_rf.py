import numpy as np
import pandas as pd

# =========================
# 1. 数据加载与预处理
# =========================

def load_btc_data(csv_path) -> pd.DataFrame:
    """
    读取 BTC 日线数据，要求至少包含：
    Date/Start, Open, High, Low, Close, Volume
    """
    df = pd.read_csv(csv_path)
    # 自动识别日期列
    date_col = None
    for candidate in ('Date', 'Start'):
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        raise ValueError("No date column found. Expect one of: Date, Start")
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: 'Date'})
    df = df.sort_values('Date').reset_index(drop=True)

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in input CSV.")

    # 日收益（log return）
    df['ret'] = np.log(df['Close'] / df['Close'].shift(1))
    return df


# =========================
# 2. 因子计算（10 个因子，周期全部 ≤ 90 天）
# =========================

def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    在原始 df 上添加 10 个因子列，返回带因子的 df。
    所有 rolling 窗口 ≤ 90 天。
    """
    eps = 1e-9

    # ----- 1) 动量与均线 -----

    # 移动平均
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_90'] = df['Close'].rolling(window=90).mean()   # CHANGED: 由 MA_200 改为 MA_90

    # MOM_20, MOM_60（都 ≤ 90）
    df['MOM_20'] = df['Close'] / df['Close'].shift(20) - 1.0
    df['MOM_60'] = df['Close'] / df['Close'].shift(60) - 1.0

    # MA_50_SPREAD, MA_90_SPREAD（都 ≤ 90）
    df['MA_50_SPREAD'] = (df['Close'] - df['MA_50']) / (df['MA_50'] + eps)
    df['MA_90_SPREAD'] = (df['Close'] - df['MA_90']) / (df['MA_90'] + eps)   # CHANGED: 用 90 天

    # ----- 2) 波动 / ATR -----

    # VOL_20（realized vol，基于 log return）
    df['VOL_20'] = df['ret'].rolling(window=20).std()

    # True Range
    prev_close = df['Close'].shift(1)
    tr1 = df['High'] - df['Low']
    tr2 = (df['High'] - prev_close).abs()
    tr3 = (df['Low'] - prev_close).abs()
    df['TR'] = np.maximum.reduce([tr1, tr2, tr3])

    # ATR_14
    df['ATR_14'] = df['TR'].rolling(window=14).mean()
    df['ATR_PCT_14'] = df['ATR_14'] / (df['Close'] + eps)

    # ----- 3) 成交量因子 -----

    df['VOL_MA_20'] = df['Volume'].rolling(window=20).mean()
    df['VOL_RATIO_20'] = df['Volume'] / (df['VOL_MA_20'] + eps)

    # ----- 4) 价格位置 / K 线结构 -----

    # 过去 60 日最高 & 最低（≤ 90）
    df['HIGH_60'] = df['High'].rolling(window=60).max()
    df['LOW_60'] = df['Low'].rolling(window=60).min()
    df['PRICE_POS_60'] = (df['Close'] - df['LOW_60']) / (df['HIGH_60'] - df['LOW_60'] + eps)

    # CLOSE_POS：当天收盘在 High-Low 区间的位置
    df['CLOSE_POS'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + eps)

    # ----- 5) 减半周期因子（缩短到减半后 90 天） -----

    # 已知减半日期（UTC 近似）
    halving_dates = [
        pd.Timestamp('2012-11-28'),
        pd.Timestamp('2016-07-09'),
        pd.Timestamp('2020-05-11'),
    ]

    # 对每一天，找到最近一次（过去的）减半日期
    last_halving = []
    for d in df['Date']:
        past_halvings = [h for h in halving_dates if h <= d]
        if len(past_halvings) == 0:
            last_halving.append(pd.NaT)
        else:
            last_halving.append(max(past_halvings))
    df['LAST_HALVING'] = last_halving

    # 天数差
    df['DAYS_SINCE_HALVING'] = (df['Date'] - df['LAST_HALVING']).dt.days

    # POST_HALVING：减半后 90 天内 = 1，否则 0
    df['POST_HALVING'] = np.where(
        (df['DAYS_SINCE_HALVING'] >= 0) & (df['DAYS_SINCE_HALVING'] <= 90),   # CHANGED: 365 -> 90
        1.0,
        0.0
    )

    # 最终 10 个因子列
    factor_cols = [
        'MOM_20',
        'MOM_60',
        'MA_50_SPREAD',
        'MA_90_SPREAD',   # CHANGED: 替代原来的 MA_200_SPREAD
        'VOL_20',
        'ATR_PCT_14',
        'VOL_RATIO_20',
        'PRICE_POS_60',
        'CLOSE_POS',
        'POST_HALVING',
    ]

    return df, factor_cols


# =========================
# 3. 因子标准化（滚动 Z-score，窗口改为 ≤ 90）
# =========================

def rolling_standardize(df: pd.DataFrame, factor_cols, window: int = 90) -> pd.DataFrame:
    """
    对每个因子做滚动 Z-score：
    z_t = (x_t - mean_{t-1}) / std_{t-1}
    使用 rolling().mean().shift(1) 来避免未来数据泄露。
    window 默认改为 90 天（原来是 252）。
    """
    for col in factor_cols:
        roll_mean = df[col].rolling(window=window).mean().shift(1)
        roll_std = df[col].rolling(window=window).std().shift(1)
        df[col + '_Z'] = (df[col] - roll_mean) / (roll_std + 1e-9)
    z_cols = [c + '_Z' for c in factor_cols]
    return df, z_cols


# =========================
# 4. 仓位生成 & 回测
# =========================

def weights_to_positions(df: pd.DataFrame, z_cols, weights: np.ndarray) -> pd.Series:
    """
    给定一组权重（长度 == len(z_cols)），生成 [0, 1] 区间的日仓位。
    使用加权和 + Sigmoid 映射到 [0,1]。
    """
    if len(weights) != len(z_cols):
        raise ValueError("Length of weights must match number of factor columns.")

    X = df[z_cols].values
    # 线性打分
    scores = np.dot(X, weights)
    # Sigmoid 映射到 (0,1)
    positions = 1.0 / (1.0 + np.exp(-scores))
    return pd.Series(positions, index=df.index, name='position')


def backtest(df: pd.DataFrame, positions: pd.Series, tc_bps: float = 5.0):
    """
    简单回测：
    - pos_t 用于从 t 到 t+1 的收益（用 ret_{t+1}）
    - 交易成本：每次仓位变动 |pos_t - pos_{t-1}| * tc  (bps)
    """
    df = df.copy()
    df['position'] = positions

    # 实际持仓用前一日信号（避免未来函数）
    df['position_shifted'] = df['position'].shift(1).fillna(0.0)

    # 仓位变化（用于成本）
    df['position_change'] = df['position_shifted'].diff().fillna(df['position_shifted'])

    # 交易成本（bps 转为 小数）
    tc = tc_bps / 10000.0
    df['cost'] = np.abs(df['position_change']) * tc

    # 策略日收益
    df['strategy_ret'] = df['position_shifted'] * df['ret'] - df['cost']

    # 去掉前期因子未定义的 NaN
    strat = df.dropna(subset=['strategy_ret'])

    if strat['strategy_ret'].std() == 0 or np.isnan(strat['strategy_ret'].std()):
        sharpe = 0.0
    else:
        avg_daily = strat['strategy_ret'].mean()
        std_daily = strat['strategy_ret'].std()
        sharpe = np.sqrt(252) * avg_daily / std_daily

    cum_return = np.exp(strat['strategy_ret'].cumsum()) - 1.0

    return {
        'sharpe': sharpe,
        'cum_return': cum_return.iloc[-1] if len(cum_return) > 0 else 0.0,
        'equity_curve': np.exp(strat['strategy_ret'].cumsum()),
        'df': strat
    }


# =========================
# 5. 随机漫步式权重搜索
# =========================

def random_walk_search(df: pd.DataFrame,
                       z_cols,
                       n_steps: int = 2000,
                       step_size: float = 0.2,
                       tc_bps: float = 5.0,
                       seed: int = 42):
    """
    简单的“随机漫步 + hill climbing”：
    - 初始化一组随机权重
    - 每一步对权重加一个高斯噪声
    - 若新权重的 Sharpe 更高，则接受，否则拒绝
    - 返回最佳权重和绩效
    """
    rng = np.random.default_rng(seed)
    n_factors = len(z_cols)

    # 初始化权重（小随机数）
    w = rng.normal(loc=0.0, scale=0.5, size=n_factors)

    # 初始评估
    pos = weights_to_positions(df, z_cols, w)
    res = backtest(df, pos, tc_bps=tc_bps)
    best_sharpe = res['sharpe']
    best_w = w.copy()

    print(f"[Init] Sharpe = {best_sharpe:.4f}")

    for step in range(1, n_steps + 1):
        # 提案：在当前权重基础上加噪声
        proposal = best_w + rng.normal(loc=0.0, scale=step_size, size=n_factors)

        pos_p = weights_to_positions(df, z_cols, proposal)
        res_p = backtest(df, pos_p, tc_bps=tc_bps)
        sharpe_p = res_p['sharpe']

        # 若更好，则接受
        if sharpe_p > best_sharpe:
            best_sharpe = sharpe_p
            best_w = proposal
            print(f"[Step {step}] New best Sharpe = {best_sharpe:.4f}")

    return best_w, best_sharpe


# =========================
# 6. 主程序示例
# =========================

if __name__ == "__main__":
    # BTC 训练数据文件路径
    from pathlib import Path
    CSV_PATH = Path(__file__).with_name("bitcoin_train_2010_2020 copy.csv")

    df = load_btc_data(CSV_PATH)
    df, factor_cols = compute_factors(df)

    # 标准化窗口改成 ≤ 90（这里用 90）
    df, z_cols = rolling_standardize(df, factor_cols, window=90)  # CHANGED: 252 -> 90

    # 丢掉因子完全 NaN 的前期
    df = df.dropna(subset=z_cols + ['ret']).reset_index(drop=True)

    print("Factor columns (Z-scored):", z_cols)

    # ==== 6.1 手动设一组权重，快速看一眼 Sharpe ====
    # 注意：因子顺序没变，只是 MA_200_SPREAD 换成了 MA_90_SPREAD
    manual_weights = np.array([0.8, 1.0, 0.5, 0.5, -0.3, -0.3, 0.2, 0.4, 0.3, 0.5])
    positions = weights_to_positions(df, z_cols, manual_weights)
    res_manual = backtest(df, positions, tc_bps=5.0)
    print(f"Manual weights Sharpe: {res_manual['sharpe']:.4f}, "
          f"CumReturn: {res_manual['cum_return']:.2f}")

    # ==== 6.2 用随机漫步搜索一组权重 ====
    best_w, best_sharpe = random_walk_search(
        df,
        z_cols,
        n_steps=2000,
        step_size=0.2,
        tc_bps=5.0,
        seed=42
    )
    print(f"\nBest weights after search: {best_w}")
    print(f"Best Sharpe: {best_sharpe:.4f}")

    # ==== 6.3 最终回测并输出 final value (USD) ====
    INITIAL_CAPITAL = 12_500.0
    final_pos = weights_to_positions(df, z_cols, best_w)
    final_res = backtest(df, final_pos, tc_bps=5.0)
    final_value = INITIAL_CAPITAL * (1.0 + final_res['cum_return'])
    print(f"\n=== Final Results ===")
    print(f"Sharpe Ratio: {final_res['sharpe']:.4f}")
    print(f"Cumulative Return: {final_res['cum_return']:.2%}")
    print(f"Final Value (USD): ${final_value:,.2f}")
