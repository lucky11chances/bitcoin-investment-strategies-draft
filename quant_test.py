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
# 2. 因子计算（10 个因子）
# =========================

def compute_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    在原始 df 上添加 10 个因子列，返回带因子的 df。
    """
    eps = 1e-9

    # ----- 1) 动量与均线 -----

    # 移动平均
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()

    # MOM_20, MOM_60
    df['MOM_20'] = df['Close'] / df['Close'].shift(20) - 1.0
    df['MOM_60'] = df['Close'] / df['Close'].shift(60) - 1.0

    # MA_50_SPREAD, MA_200_SPREAD
    df['MA_50_SPREAD'] = (df['Close'] - df['MA_50']) / (df['MA_50'] + eps)
    df['MA_200_SPREAD'] = (df['Close'] - df['MA_200']) / (df['MA_200'] + eps)

    # ----- 2) 波动 / ATR -----

    # log return 已经有 df['ret']

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

    # 过去 60 日最高 & 最低
    df['HIGH_60'] = df['High'].rolling(window=60).max()
    df['LOW_60'] = df['Low'].rolling(window=60).min()
    df['PRICE_POS_60'] = (df['Close'] - df['LOW_60']) / (df['HIGH_60'] - df['LOW_60'] + eps)

    # CLOSE_POS：当天收盘在 High-Low 区间的位置
    df['CLOSE_POS'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + eps)

    # ----- 5) 减半周期因子 -----

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

    # POST_HALVING：减半后一年内 = 1，否则 0（含没减半的时期）
    df['POST_HALVING'] = np.where(
        (df['DAYS_SINCE_HALVING'] >= 0) & (df['DAYS_SINCE_HALVING'] <= 365),
        1.0,
        0.0
    )

    # 返回时可以删掉中间辅助列
    factor_cols = [
        'MOM_20',
        'MOM_60',
        'MA_50_SPREAD',
        'MA_200_SPREAD',
        'VOL_20',
        'ATR_PCT_14',
        'VOL_RATIO_20',
        'PRICE_POS_60',
        'CLOSE_POS',
        'POST_HALVING',
    ]

    return df, factor_cols


# =========================
# 3. 因子标准化（滚动 Z-score，避免未来信息）
# =========================

def rolling_standardize(df: pd.DataFrame, factor_cols, window: int = 252) -> pd.DataFrame:
    """
    对每个因子做滚动 Z-score：
    z_t = (x_t - mean_{t-1}) / std_{t-1}
    使用 rolling().mean().shift(1) 来避免未来数据泄露。
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
# 5. 主程序 - 测试集 (2023-2024)
# =========================

if __name__ == "__main__":
    from pathlib import Path
    
    # 测试集数据
    TEST_CSV_PATH = Path(__file__).with_name("bitcoin_test_2023_2024 copy.csv")
    
    # 训练集得到的最优权重（来自 quant_rf.py）
    TRAINED_WEIGHTS = np.array([12.2064582, -4.34554553, -3.56066882, -1.8849945, 
                                -6.29258246, 4.03083788, 5.27031729, 8.93138152, 
                                4.56339456, 3.78368854])
    
    INITIAL_CAPITAL = 12_500.0

    print("=== Quant Strategy Test (2023-2024) ===")
    
    # 加载测试集
    df = load_btc_data(TEST_CSV_PATH)
    df, factor_cols = compute_factors(df)
    df, z_cols = rolling_standardize(df, factor_cols, window=252)
    
    # 丢掉因子完全 NaN 的前期
    df = df.dropna(subset=z_cols + ['ret']).reset_index(drop=True)
    
    print(f"Test data: {df['Date'].iloc[0].date()} to {df['Date'].iloc[-1].date()}")
    print(f"Number of trading days: {len(df)}")
    
    # 使用训练好的权重进行回测
    positions = weights_to_positions(df, z_cols, TRAINED_WEIGHTS)
    res = backtest(df, positions, tc_bps=5.0)
    
    final_value = INITIAL_CAPITAL * (1.0 + res['cum_return'])
    
    print(f"\nSharpe Ratio: {res['sharpe']:.4f}")
    print(f"Cumulative Return: {res['cum_return']:.2%}")
    print(f"Final Value (USD): ${final_value:,.2f}")
