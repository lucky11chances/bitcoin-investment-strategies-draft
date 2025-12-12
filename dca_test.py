import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).with_name("bitcoin_test_2023_2024 copy.csv")
MONTHLY_CONTRIBUTION = 1_000.0
TARGET_TOTAL = 12_500.0
RISK_FREE_ANNUAL = 0.03
DAYS_PER_YEAR = 365


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found at {path}")

    df = pd.read_csv(path)

    # Find a date column and normalize to "Date".
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


def build_contributions(df: pd.DataFrame) -> pd.DataFrame:
    max_months = int(np.ceil(TARGET_TOTAL / MONTHLY_CONTRIBUTION))
    monthly_first = df.resample("MS", on="Date").first().reset_index()
    if len(monthly_first) < max_months:
        raise ValueError(f"Not enough monthly data to invest {max_months} months; only {len(monthly_first)} months present.")

    monthly_first = monthly_first.head(max_months)
    monthly_first["btc_bought"] = MONTHLY_CONTRIBUTION / monthly_first["Close"]
    monthly_first["cum_btc"] = monthly_first["btc_bought"].cumsum()
    return monthly_first[["Date", "cum_btc"]]


def compute_metrics(df: pd.DataFrame, contribs: pd.DataFrame) -> dict:
    merged = df.merge(contribs, on="Date", how="left")
    merged["cum_btc"] = merged["cum_btc"].ffill().fillna(0.0)
    merged["portfolio"] = merged["Close"] * merged["cum_btc"]

    # 只在开始投入之后计算收益，避免 0->正数 的 pct_change 生成 inf
    invested = merged[merged["cum_btc"] > 0].copy()
    daily_ret = invested["portfolio"].pct_change().dropna()

    rf_daily = (1.0 + RISK_FREE_ANNUAL) ** (1.0 / DAYS_PER_YEAR) - 1.0
    excess_ret = daily_ret - rf_daily

    sharpe = np.nan
    vol = excess_ret.std(ddof=1)
    if vol > 0 and not np.isnan(vol):
        sharpe = (excess_ret.mean() / vol) * math.sqrt(DAYS_PER_YEAR)

    sortino = np.nan
    downside = excess_ret[excess_ret < 0]
    down_vol = downside.std(ddof=1)
    if down_vol > 0 and not np.isnan(down_vol):
        sortino = (excess_ret.mean() / down_vol) * math.sqrt(DAYS_PER_YEAR)

    running_max = merged["portfolio"].cummax()
    drawdown = merged["portfolio"] / running_max - 1.0
    max_drawdown = float(drawdown.min())

    start_date = merged.loc[merged["cum_btc"] > 0, "Date"].iloc[0]
    end_date = merged["Date"].iloc[-1]
    final_value = float(merged["portfolio"].iloc[-1])

    total_invested = MONTHLY_CONTRIBUTION * len(contribs)

    return {
        "start_date": start_date.date(),
        "end_date": end_date.date(),
        "months_invested": len(contribs),
        "total_invested": total_invested,
        "final_value": final_value,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown,
    }


def main():
    df = load_data(DATA_PATH)
    contribs = build_contributions(df)
    metrics = compute_metrics(df, contribs)

    print("=== DCA Test (2023-2024) ===")
    for key, val in metrics.items():
        print(f"{key}: {val}")


if __name__ == "__main__":
    main()
