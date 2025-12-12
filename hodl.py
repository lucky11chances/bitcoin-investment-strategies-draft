import math
from pathlib import Path

import numpy as np
import pandas as pd

# 默认读取与脚本同目录下的训练集文件
DATA_PATH = Path(__file__).with_name("bitcoin_train_2010_2020 copy.csv")

INITIAL_CAPITAL = 12_500.0
RISK_FREE_ANNUAL = 0.03        # 用于 Sharpe/Sortino 的无风险利率
CAPITAL_COST_ANNUAL = 0.07     # 资金成本（假设借款/机会成本）
DAYS_PER_YEAR = 365


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path}")

    df = pd.read_csv(path)

    # 自动识别日期列
    date_col = None
    for candidate in ("Date", "Start"):
        if candidate in df.columns:
            date_col = candidate
            break
    if date_col is None:
        raise ValueError("No date column found. Expect one of: Date, Start")

    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(date_col, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df.rename(columns={date_col: "Date"})


def compute_metrics(df: pd.DataFrame) -> dict:
    close = df["Close"]
    start_date = df.loc[0, "Date"]
    end_date = df.loc[df.index[-1], "Date"]

    buy_price = close.iloc[0]
    btc_held = INITIAL_CAPITAL / buy_price

    portfolio = close * btc_held
    daily_ret = portfolio.pct_change().dropna()

    # 按复利换算年 -> 日
    rf_daily = (1.0 + RISK_FREE_ANNUAL) ** (1.0 / DAYS_PER_YEAR) - 1.0
    excess_ret = daily_ret - rf_daily

    # Sharpe
    sharpe = np.nan
    vol = excess_ret.std(ddof=1)
    if vol > 0 and not np.isnan(vol):
        sharpe = (excess_ret.mean() / vol) * math.sqrt(DAYS_PER_YEAR)

    # Sortino
    sortino = np.nan
    downside = excess_ret[excess_ret < 0]
    down_vol = downside.std(ddof=1)
    if down_vol > 0 and not np.isnan(down_vol):
        sortino = (excess_ret.mean() / down_vol) * math.sqrt(DAYS_PER_YEAR)

    # 最大回撤（负数）
    running_max = portfolio.cummax()
    drawdown = portfolio / running_max - 1.0
    max_drawdown = float(drawdown.min())

    duration_years = (end_date - start_date).days / DAYS_PER_YEAR
    final_value = float(portfolio.iloc[-1])

    # 资金成本：用复利估算借 7% 的机会成本 / 借款成本
    capital_cost = INITIAL_CAPITAL * (1.0 + CAPITAL_COST_ANNUAL) ** duration_years
    net_profit_after_cost = final_value - capital_cost

    return {
        "start_date": start_date.date(),
        "end_date": end_date.date(),
        "buy_price": float(buy_price),
        "btc_held": float(btc_held),
        "final_value": final_value,
        "duration_years": duration_years,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
        "net_profit_after_7pct_cost": net_profit_after_cost,
    }


def main():
    df = load_data(DATA_PATH)
    metrics = compute_metrics(df)

    for key, val in metrics.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()
